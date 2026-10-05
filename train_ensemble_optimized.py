"""
train_ensemble_optimized.py
---------------------------
Accuracy-focused rewrite of train_ensemble_ultimate.py.

Full run (GPU):   python train_ensemble_optimized.py --device cuda
Full run (CPU):   python train_ensemble_optimized.py
Quick check:      python train_ensemble_optimized.py --sample 30000 --trials 3 --cat-trials 2 --folds 3
Ablation example: python train_ensemble_optimized.py --stat-features --out models/ablation_stat
"""
import argparse
import inspect
import json
import os
import platform
import time
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pricing_core import (
    BASE_CATEGORICAL, FEATURE_LABELS, EnsemblePricer, build_features, feature_columns,
    forward_target, inverse_target, normalize_labels, to_catboost_frame,
    to_tree_frame,
)

MAX_ITERS = 6000
EARLY_STOP = 200


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Indian Airlines.csv")
    p.add_argument("--out", default="models")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--target", choices=["log", "raw"], default="log",
                   help="Scale the base models train on. The blender always works in INR.")
    p.add_argument("--trials", type=int, default=40, help="Optuna trials for LightGBM and XGBoost")
    p.add_argument("--cat-trials", type=int, default=12, help="Optuna trials for CatBoost (slower)")
    p.add_argument("--folds", type=int, default=5, help="Folds for out-of-fold stacking")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate for LightGBM/XGBoost")
    p.add_argument("--cat-lr", type=float, default=0.08, help="Learning rate for CatBoost")
    p.add_argument("--no-flight", action="store_true", help="Do not use the flight code feature")
    p.add_argument("--no-interactions", action="store_true",
                   help="Drop the airline_route / route_stops / dep_arr interaction features")
    p.add_argument("--interval-level", type=float, default=0.8,
                   help="Coverage target for the fare range (0.8 = 80%% of fares inside)")
    p.add_argument("--no-intervals", action="store_true", help="Skip training the fare-range models")
    p.add_argument("--importance-rows", type=int, default=3000,
                   help="Test rows used to compute global feature importance (0 = skip)")
    p.add_argument("--stat-features", action="store_true",
                   help="Add lead_time_urgency and surge_probability (for ablation)")
    p.add_argument("--baselines", action="store_true", help="Also train Ridge and RandomForest baselines")
    p.add_argument("--no-full-refit", action="store_true",
                   help="Deploy the train-split models instead of refitting on all rows")
    p.add_argument("--sample", type=int, default=0, help="Use a random subset of rows (quick tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
        "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'model':<22}{'RMSE':>10}{'MAE':>10}{'R2':>9}{'MAPE%':>8}")
    for name, m in rows.items():
        print(f"{name:<22}{m['RMSE']:>10.1f}{m['MAE']:>10.1f}{m['R2']:>9.4f}{m['MAPE_%']:>8.2f}")


def scale_iters(best_iter, n_rows, n_ref):
    """Rounds found with early stopping on n_ref rows, scaled to n_rows."""
    return max(50, int(round(best_iter * n_rows / n_ref)))


def load_data(args):
    df = pd.read_csv(args.data)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "class" in df.columns:
        df = df[df["class"].astype(str).str.lower() == "economy"]
    df = normalize_labels(df)
    df = df[df["price"] > 0]
    before = len(df)
    df = df.drop_duplicates()
    print(f"   economy rows: {before:,}  (exact duplicates removed: {before - len(df):,})")
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed)
        print(f"   sampled down to {len(df):,} rows")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Base model factories
# ---------------------------------------------------------------------------
def make_lgb(params, n_iter, args):
    return lgb.LGBMRegressor(n_estimators=n_iter, learning_rate=args.lr, subsample_freq=1,
                             random_state=args.seed, n_jobs=-1, verbose=-1, **params)


def make_xgb(params, n_iter, args, early_stop=False):
    extra = {"early_stopping_rounds": EARLY_STOP} if early_stop else {}
    return xgb.XGBRegressor(n_estimators=n_iter, learning_rate=args.lr, tree_method="hist",
                            device=args.device, enable_categorical=True, max_cat_to_onehot=8,
                            eval_metric="rmse", random_state=args.seed, **params, **extra)


def make_cat(params, n_iter, args):
    return CatBoostRegressor(iterations=n_iter, learning_rate=args.cat_lr, loss_function="RMSE",
                             task_type="GPU" if args.device == "cuda" else "CPU",
                             random_seed=args.seed, verbose=0, allow_writing_files=False, **params)


def lgb_eval_kwargs(Xva, yva):
    """LightGBM >= 4.6 renamed eval_set to eval_X/eval_y; support both."""
    if "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters:
        return {"eval_X": Xva, "eval_y": yva}
    return {"eval_set": [(Xva, yva)]}


FRAME_KIND = {"lightgbm": "tree", "xgboost": "tree", "catboost": "catboost"}


def fit_base(name, params, n_iter, X, y_t, args, cat_cols):
    if name == "lightgbm":
        return make_lgb(params, n_iter, args).fit(X, y_t)
    if name == "xgboost":
        return make_xgb(params, n_iter, args).fit(X, y_t, verbose=False)
    return make_cat(params, n_iter, args).fit(X, y_t, cat_features=cat_cols)


# ---------------------------------------------------------------------------
# Optuna tuning (early stopping on a validation split taken from TRAIN only)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "lightgbm": {"num_leaves": 255, "min_child_samples": 20, "subsample": 0.9,
                 "colsample_bytree": 0.8, "reg_lambda": 1.0, "cat_smooth": 10.0,
                 "min_data_per_group": 50, "max_cat_threshold": 64},
    "xgboost": {"max_depth": 10, "min_child_weight": 5.0, "subsample": 0.9,
                "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "catboost": {"depth": 8, "l2_leaf_reg": 3.0, "random_strength": 1.0},
}


def suggest(name, trial):
    if name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 63, 1023, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 50.0, log=True),
            "min_data_per_group": trial.suggest_int("min_data_per_group", 5, 200, log=True),
            "max_cat_threshold": trial.suggest_int("max_cat_threshold", 16, 128, log=True),
        }
    if name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    return {
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
    }


def tune(name, Xtr, ytr_t, Xva, yva_t, yva_real, n_trials, args, cat_cols):
    def objective(trial):
        params = suggest(name, trial)
        if name == "lightgbm":
            m = make_lgb(params, MAX_ITERS, args)
            m.fit(Xtr, ytr_t, **lgb_eval_kwargs(Xva, yva_t),
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            best = m.best_iteration_ or MAX_ITERS
        elif name == "xgboost":
            m = make_xgb(params, MAX_ITERS, args, early_stop=True)
            m.fit(Xtr, ytr_t, eval_set=[(Xva, yva_t)], verbose=False)
            best = m.best_iteration + 1
        else:
            m = make_cat(params, MAX_ITERS, args)
            m.fit(Xtr, ytr_t, cat_features=cat_cols, eval_set=(Xva, yva_t),
                  early_stopping_rounds=EARLY_STOP, use_best_model=True)
            best = m.get_best_iteration() + 1
        trial.set_user_attr("best_iter", int(best))
        # Select on the metric we actually report: RMSE in rupees
        return np.sqrt(mean_squared_error(yva_real, inverse_target(m.predict(Xva), args.target)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial(DEFAULTS[name])  # sensible defaults are always evaluated
    t = time.time()
    total = max(1, n_trials)

    def progress(study, trial):
        print(f"      {name} trial {trial.number + 1}/{total}: RMSE Rs {trial.value:,.1f} "
              f"(best Rs {study.best_value:,.1f}) [{time.time() - t:,.0f}s]", flush=True)

    study.optimize(objective, n_trials=total, callbacks=[progress])
    best_iter = study.best_trial.user_attrs["best_iter"]
    print(f"   {name:<9} val RMSE Rs {study.best_value:,.1f} | rounds {best_iter} | "
          f"{time.time() - t:,.0f}s | {study.best_params}")
    return {"params": study.best_params, "best_iter": best_iter, "val_rmse": study.best_value}


# ---------------------------------------------------------------------------
# Baselines (same split, same features) so the report compares like with like
# ---------------------------------------------------------------------------
def run_baselines(Xtr, ytr, Xte, cat_cols, num_cols, seed):
    out = {}
    ridge = make_pipeline(ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)]), Ridge(alpha=1.0))
    out["ridge_linear"] = ridge.fit(Xtr, ytr).predict(Xte)

    rf = make_pipeline(ColumnTransformer([
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols)]),
        RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=seed))
    out["random_forest"] = rf.fit(Xtr, ytr).predict(Xte)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    print("1) Loading data")
    df = load_data(args)
    use_flight = not args.no_flight
    if use_flight and "flight" not in df.columns:
        print("   WARNING: no 'flight' column in the CSV, continuing without it")
        use_flight = False

    use_inter = not args.no_interactions
    cat_cols, num_cols = feature_columns(use_flight, args.stat_features, use_inter)
    X = build_features(df, use_flight, args.stat_features, use_inter)
    y = df["price"].to_numpy(float)
    # Category vocabularies carry no target information, so using all rows is leak-free
    categories = {c: sorted(X[c].unique().tolist()) for c in cat_cols}
    print(f"   features: {len(cat_cols)} categorical, {len(num_cols)} numeric | "
          + ", ".join(f"{c}={len(categories[c])}" for c in ["route", "airline_route"] + (["flight"] if use_flight else [])))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=args.seed)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    ytr_t = forward_target(ytr, args.target)
    frames_tr = {"tree": to_tree_frame(Xtr, cat_cols, categories), "catboost": to_catboost_frame(Xtr, cat_cols)}
    frames_te = {"tree": to_tree_frame(Xte, cat_cols, categories), "catboost": to_catboost_frame(Xte, cat_cols)}

    print("2) Hyperparameter search (validation split carved from the training set)")
    a, b = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=args.seed)
    n_ref = len(a)
    tuned = {}
    for name in ["lightgbm", "xgboost", "catboost"]:
        F = frames_tr[FRAME_KIND[name]]
        trials = args.cat_trials if name == "catboost" else args.trials
        tuned[name] = tune(name, F.iloc[a], ytr_t[a], F.iloc[b], ytr_t[b], ytr[b], trials, args, cat_cols)
    names = list(tuned)

    print(f"3) Out-of-fold predictions for the blender ({args.folds} folds)")
    oof = np.zeros((len(ytr), len(names)))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (f_tr, f_va) in enumerate(kf.split(ytr), 1):
        for j, name in enumerate(names):
            F = frames_tr[FRAME_KIND[name]]
            n_iter = scale_iters(tuned[name]["best_iter"], len(f_tr), n_ref)
            m = fit_base(name, tuned[name]["params"], n_iter, F.iloc[f_tr], ytr_t[f_tr], args, cat_cols)
            oof[f_va, j] = inverse_target(m.predict(F.iloc[f_va]), args.target)
        print(f"   fold {k}/{args.folds} done")

    # Non-negative linear blend in INR. The intercept/weights also correct the
    # downward bias from expm1(mean of log price).
    meta = LinearRegression(positive=True).fit(oof, ytr)
    print("   blend weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, meta.coef_))
          + f", intercept={meta.intercept_:.1f}")

    print("4) Held-out test evaluation")
    test_preds, split_models = {}, {}
    for name in names:
        n_iter = scale_iters(tuned[name]["best_iter"], len(ytr), n_ref)
        m = fit_base(name, tuned[name]["params"], n_iter, frames_tr[FRAME_KIND[name]], ytr_t, args, cat_cols)
        split_models[name] = m
        test_preds[name] = inverse_target(m.predict(frames_te[FRAME_KIND[name]]), args.target)
    base_matrix = np.column_stack([test_preds[n] for n in names])
    test_preds["simple_average"] = base_matrix.mean(axis=1)
    test_preds["stacked_ensemble"] = np.clip(meta.predict(base_matrix), y.min() * 0.5, None)

    if args.baselines:
        print("   training baselines (Ridge, RandomForest)...")
        test_preds.update(run_baselines(Xtr, ytr, Xte, cat_cols, num_cols, args.seed))

    results = {n: metrics(yte, p) for n, p in test_preds.items()}
    print_table("TEST SET RESULTS (INR)", results)

    ens = test_preds["stacked_ensemble"]
    buckets = pd.cut(Xte["days_left"], [0, 3, 7, 14, 30, 60], labels=["1-3", "4-7", "8-14", "15-30", "31+"])
    by_bucket = {str(bk): metrics(yte[mask], ens[mask]) for bk in buckets.cat.categories
                 if (mask := (buckets == bk).to_numpy()).sum() > 0}
    print_table("ENSEMBLE ERROR BY DAYS_LEFT", by_bucket)

    # ------------------------------------------------------------------
    # 4b) Fare range: LightGBM quantile models + split-conformal calibration
    # ------------------------------------------------------------------
    interval_info, q_correction = None, 0.0
    if not args.no_intervals:
        level = args.interval_level
        alphas = {"lower": (1 - level) / 2, "upper": 1 - (1 - level) / 2}
        lgb_params = tuned["lightgbm"]["params"]
        q_iters = tuned["lightgbm"]["best_iter"]
        Ft, Fe = frames_tr["tree"], frames_te["tree"]
        print(f"\n4b) Fare range models ({level:.0%} target coverage)")

        def fit_quantiles(F, y_t, n_rows):
            return {k: make_lgb({**lgb_params, "objective": "quantile", "alpha": a},
                                scale_iters(q_iters, n_rows, n_ref), args).fit(F, y_t)
                    for k, a in alphas.items()}

        # Calibrate on the tuning validation rows (never the test set)
        q_cal = fit_quantiles(Ft.iloc[a], ytr_t[a], len(a))
        lo_b, hi_b = q_cal["lower"].predict(Ft.iloc[b]), q_cal["upper"].predict(Ft.iloc[b])
        scores = np.maximum(lo_b - ytr_t[b], ytr_t[b] - hi_b)
        n_cal = len(scores)
        q_correction = float(np.quantile(scores, min(1.0, np.ceil((n_cal + 1) * level) / n_cal),
                                         method="higher"))

        q_split = fit_quantiles(Ft, ytr_t, len(ytr))
        lo_raw, hi_raw = q_split["lower"].predict(Fe), q_split["upper"].predict(Fe)
        def coverage(lo_t, hi_t):
            lo_p = np.minimum(inverse_target(lo_t, args.target), ens)
            hi_p = np.maximum(inverse_target(hi_t, args.target), ens)
            return float(np.mean((yte >= lo_p) & (yte <= hi_p))), float(np.mean(hi_p - lo_p))
        cov_raw, width_raw = coverage(lo_raw, hi_raw)
        cov_cal, width_cal = coverage(lo_raw - q_correction, hi_raw + q_correction)
        interval_info = {"target_level": level, "conformal_correction": q_correction,
                         "test_coverage_uncalibrated": cov_raw, "test_avg_width_uncalibrated": width_raw,
                         "test_coverage": cov_cal, "test_avg_width": width_cal}
        print(f"   uncalibrated: coverage {cov_raw:.1%}, avg width Rs {width_raw:,.0f}")
        print(f"   calibrated:   coverage {cov_cal:.1%}, avg width Rs {width_cal:,.0f}")

    # ------------------------------------------------------------------
    # 4c) Global feature importance (mean absolute rupee effect on test rows)
    # ------------------------------------------------------------------
    importance = []
    if args.importance_rows > 0:
        print("\n4c) Global feature importance")
        probe = EnsemblePricer(split_models, FRAME_KIND, meta, cat_cols, num_cols, categories,
                               args.target, use_flight, args.stat_features, {}, 0.0, use_inter)
        rows = Xte.sample(min(args.importance_rows, len(Xte)), random_state=args.seed)
        try:
            ex = probe.explain_matrix(rows)
            imp = ex["rupees"].abs().mean().sort_values(ascending=False)
            importance = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "mean_abs_rupees": float(v)}
                          for f, v in imp.items()]
            pd.DataFrame(importance).to_csv(os.path.join(args.out, "feature_importance.csv"), index=False)
            for r in importance:
                print(f"   {r['label']:<28} Rs {r['mean_abs_rupees']:>8,.0f}")
            print(f"   (models used: {', '.join(ex['models_used'])})")
        except Exception as e:
            print(f"   skipped: {e}")

    print("\n5) Building the deployable model")
    if args.no_full_refit:
        X_fit, y_fit = frames_tr, ytr_t
    else:
        X_fit = {"tree": to_tree_frame(X, cat_cols, categories), "catboost": to_catboost_frame(X, cat_cols)}
        y_fit = forward_target(y, args.target)
    final_models = {}
    for name in names:
        n_iter = scale_iters(tuned[name]["best_iter"], len(y_fit), n_ref)
        m = fit_base(name, tuned[name]["params"], n_iter, X_fit[FRAME_KIND[name]], y_fit, args, cat_cols)
        if name == "xgboost":
            m.set_params(device="cpu")  # the API server may not have a GPU
        final_models[name] = m
        print(f"   {name} refit with {n_iter} rounds on {len(y_fit):,} rows")

    final_quantiles = {}
    if interval_info:
        lgb_params = tuned["lightgbm"]["params"]
        for k, alpha in {"lower": (1 - args.interval_level) / 2,
                         "upper": 1 - (1 - args.interval_level) / 2}.items():
            n_iter = scale_iters(tuned["lightgbm"]["best_iter"], len(y_fit), n_ref)
            final_quantiles[k] = make_lgb({**lgb_params, "objective": "quantile", "alpha": alpha},
                                          n_iter, args).fit(X_fit["tree"], y_fit)
        print(f"   fare-range models refit on {len(y_fit):,} rows")

    mode = lambda s: s.mode().iat[0]
    group_cols = ["airline", "source_city", "destination_city"] + (["flight"] if use_flight else [])
    profiles = (df.groupby(group_cols).agg(
        departure_time=("departure_time", mode), arrival_time=("arrival_time", mode),
        stops=("stops", mode), duration=("duration", "median"), rows=("price", "size"))
        .reset_index())
    metadata = {
        "categories": {c: categories[c] for c in BASE_CATEGORICAL + (["flight"] if use_flight else [])},
        "flight_profiles": profiles.to_dict("records") if use_flight else [],
        "uses_flight_code": use_flight,
        "days_left_range": [int(df["days_left"].min()), int(df["days_left"].max())],
        "duration_range": [float(df["duration"].min()), float(df["duration"].max())],
        "training_rows": int(len(y_fit)),
        "test_metrics": results["stacked_ensemble"],
        "test_metrics_all_models": results,
        "target_transform": args.target,
        "interval": interval_info,
        "feature_importance": importance,
        "blend_weights": dict(zip(names, meta.coef_.tolist())),
        "model_version": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(),
                        **{mod.__name__: getattr(mod, "__version__", "unknown")
                           for mod in (lgb, xgb, pd, np, optuna)}},
    }
    pricer = EnsemblePricer(final_models, FRAME_KIND, meta, cat_cols, num_cols, categories,
                            args.target, use_flight, args.stat_features, metadata,
                            min_price=float(y.min() * 0.5), include_interactions=use_inter,
                            quantile_models=final_quantiles, interval_correction=q_correction,
                            interval_level=args.interval_level if interval_info else None)

    model_path = os.path.join(args.out, "pricing_ensemble.pkl")
    joblib.dump(pricer, model_path)
    report = {
        "args": vars(args), "features": {"categorical": cat_cols, "numeric": num_cols},
        "tuning": tuned, "tuning_rows": n_ref, "interval": interval_info,
        "feature_importance": importance, "model_version": metadata["model_version"],
        "blend_weights": dict(zip(names, meta.coef_.tolist())),
        "blend_intercept": float(meta.intercept_), "test_metrics": results,
        "test_metrics_by_days_left": by_bucket, "runtime_minutes": (time.time() - t_start) / 60,
    }
    with open(os.path.join(args.out, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"   saved {model_path} and training_report.json ({report['runtime_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
