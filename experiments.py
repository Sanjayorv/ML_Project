"""
experiments.py
--------------
Rigorous evaluation for the report. Run AFTER train_ensemble_optimized.py:
it reuses the tuned hyperparameters in models/training_report.json, so no
new hyperparameter search is needed.

Part A  generalisation:  random K-fold vs GroupKFold by flight code,
                         each with and without the flight feature.
                         Answers: "how good is the model on flights it has never seen?"
Part B  ablations:       the same train/test split as training, one design choice
                         changed at a time, all other settings fixed.

Usage:
  python experiments.py --device cuda                     # everything
  python experiments.py --part generalization --folds 3
  python experiments.py --part ablations --models lightgbm  # fast version
Outputs go to experiments/ (JSON, CSV, and a Markdown file with report-ready tables).
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from pricing_core import (build_features, feature_columns, forward_target, inverse_target,
                          to_catboost_frame, to_tree_frame)
from train_ensemble_optimized import FRAME_KIND, fit_base, load_data, metrics, scale_iters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="models/training_report.json")
    p.add_argument("--data", default=None, help="Defaults to the path used in training")
    p.add_argument("--out", default="experiments")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--part", choices=["all", "generalization", "ablations"], default="all")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="lightgbm,xgboost,catboost",
                   help="Comma-separated subset of base models to include")
    return p.parse_args()


class Experiment:
    def __init__(self, cli):
        with open(cli.report) as f:
            self.report = json.load(f)
        ta = self.report["args"]
        self.args = SimpleNamespace(
            data=cli.data or ta["data"], sample=ta.get("sample", 0), seed=ta["seed"],
            lr=ta["lr"], cat_lr=ta["cat_lr"], device=cli.device)
        self.tuned = self.report["tuning"]
        self.models = [m.strip() for m in cli.models.split(",") if m.strip() in self.tuned]
        if not self.models:
            raise SystemExit("None of the requested models are in the training report.")
        self.base = {
            "use_flight": not ta.get("no_flight", False),
            "stat": ta.get("stat_features", False),
            "inter": not ta.get("no_interactions", False),
            "target": ta["target"],
        }
        print("Loading data...")
        self.df = load_data(self.args)
        self.y = self.df["price"].to_numpy(float)
        self.n_ref = self.report.get("tuning_rows") or int(len(self.df) * 0.8 * 0.85)
        if "flight" not in self.df.columns:
            raise SystemExit("The dataset has no 'flight' column; grouped evaluation needs it.")

    def design(self, use_flight, stat, inter):
        cat_cols, num_cols = feature_columns(use_flight, stat, inter)
        X = build_features(self.df, use_flight, stat, inter)
        cats = {c: sorted(X[c].unique().tolist()) for c in cat_cols}
        frames = {"tree": to_tree_frame(X, cat_cols, cats), "catboost": to_catboost_frame(X, cat_cols)}
        return frames, cat_cols

    def fit_predict(self, frames, cat_cols, tr, te, target, label):
        y_t = forward_target(self.y[tr], target)
        preds = {}
        for name in self.models:
            t = time.time()
            F = frames[FRAME_KIND[name]]
            n_iter = scale_iters(self.tuned[name]["best_iter"], len(tr), self.n_ref)
            m = fit_base(name, self.tuned[name]["params"], n_iter, F.iloc[tr], y_t, self.args, cat_cols)
            preds[name] = inverse_target(m.predict(F.iloc[te]), target)
            print(f"      {label} {name}: {time.time() - t:,.0f}s", flush=True)
        preds["average"] = np.mean([preds[n] for n in self.models], axis=0)
        return preds

    # ---------------------------------------------------------------- Part A
    def generalization(self, n_folds):
        print(f"\nPART A: random vs grouped-by-flight {n_folds}-fold CV")
        groups = self.df["flight"].to_numpy()
        splitters = {
            "random": list(KFold(n_folds, shuffle=True, random_state=self.args.seed).split(self.y)),
            "grouped_by_flight": list(GroupKFold(n_folds).split(self.y, groups=groups)),
        }
        rows = []
        for use_flight in ([True, False] if self.base["use_flight"] else [False]):
            frames, cat_cols = self.design(use_flight, self.base["stat"], self.base["inter"])
            for split_name, folds in splitters.items():
                for k, (tr, te) in enumerate(folds, 1):
                    label = f"[{'flight' if use_flight else 'no-flight'} | {split_name} | fold {k}/{n_folds}]"
                    preds = self.fit_predict(frames, cat_cols, tr, te, self.base["target"], label)
                    for model, p in preds.items():
                        rows.append({"uses_flight_code": use_flight, "split": split_name, "fold": k,
                                     "model": model, **metrics(self.y[te], p)})
        detail = pd.DataFrame(rows)
        summary = (detail.groupby(["uses_flight_code", "split", "model"])[["RMSE", "MAE", "R2", "MAPE_%"]]
                   .agg(["mean", "std"]).round(4))
        return detail, summary

    # ---------------------------------------------------------------- Part B
    def ablations(self):
        print("\nPART B: ablation study (same split as training, one change at a time)")
        tr, te = train_test_split(np.arange(len(self.y)), test_size=0.2, random_state=self.args.seed)
        b = self.base
        variants = {"deployed_configuration": dict(b)}
        if b["use_flight"]:
            variants["without_flight_code"] = {**b, "use_flight": False}
        variants["without_interactions" if b["inter"] else "with_interactions"] = {**b, "inter": not b["inter"]}
        variants["without_poisson_features" if b["stat"] else "with_poisson_features"] = {**b, "stat": not b["stat"]}
        variants["raw_price_target" if b["target"] == "log" else "log_price_target"] = \
            {**b, "target": "raw" if b["target"] == "log" else "log"}

        rows = []
        for vname, v in variants.items():
            frames, cat_cols = self.design(v["use_flight"], v["stat"], v["inter"])
            preds = self.fit_predict(frames, cat_cols, tr, te, v["target"], f"[{vname}]")
            for model, p in preds.items():
                rows.append({"variant": vname, "model": model, **metrics(self.y[te], p)})
        table = pd.DataFrame(rows)
        ref = table[table.variant == "deployed_configuration"].set_index("model")["RMSE"]
        table["RMSE_change_vs_deployed"] = table.apply(lambda r: r["RMSE"] - ref[r["model"]], axis=1)
        return table.round(4)


def markdown_table(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    cli = parse_args()
    os.makedirs(cli.out, exist_ok=True)
    exp = Experiment(cli)
    t0 = time.time()
    md = ["# Experiment results", "",
          f"Base models: {', '.join(exp.models)}. 'average' = simple mean of those models "
          "(the learned blend is not refit per fold). Hyperparameters are fixed from the "
          f"training run `{exp.report.get('model_version', 'unknown')}`.", ""]
    out = {"models": exp.models, "base_configuration": exp.base}

    if cli.part in ("all", "generalization"):
        detail, summary = exp.generalization(cli.folds)
        detail.to_csv(os.path.join(cli.out, "generalization_folds.csv"), index=False)
        summary.to_csv(os.path.join(cli.out, "generalization_summary.csv"))
        flat = summary.reset_index()
        flat.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in flat.columns]
        avg = flat[flat.model == "average"]
        md += ["## A. Random vs grouped-by-flight cross-validation", "",
               "`grouped_by_flight` keeps every flight code entirely in either the training or the "
               "test folds, so it measures accuracy on flights the model has never seen.", "",
               markdown_table(avg, ["uses_flight_code", "split", "RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean", "MAPE_%_mean"],
                              {"RMSE_mean": "Rs {:,.0f}", "RMSE_std": "± {:,.0f}", "MAE_mean": "Rs {:,.0f}",
                               "R2_mean": "{:.4f}", "MAPE_%_mean": "{:.2f}%"}), ""]
        out["generalization_summary"] = json.loads(flat.to_json(orient="records"))
        print("\n" + avg.to_string(index=False))

    if cli.part in ("all", "ablations"):
        table = exp.ablations()
        table.to_csv(os.path.join(cli.out, "ablations.csv"), index=False)
        avg = table[table.model == "average"]
        md += ["## B. Ablation study", "",
               "Same 80/20 split and hyperparameters as training; each row changes one design choice. "
               "Positive RMSE change = worse than the deployed configuration.", "",
               markdown_table(avg, ["variant", "RMSE", "RMSE_change_vs_deployed", "MAE", "R2", "MAPE_%"],
                              {"RMSE": "Rs {:,.0f}", "RMSE_change_vs_deployed": "{:+,.0f}", "MAE": "Rs {:,.0f}",
                               "R2": "{:.4f}", "MAPE_%": "{:.2f}%"}), "",
               "Note: the raw/log target variant reuses boosting-round counts tuned for the "
               "deployed target, which slightly favours the deployed configuration.", ""]
        out["ablations"] = json.loads(table.to_json(orient="records"))
        print("\n" + avg.to_string(index=False))

    out["runtime_minutes"] = (time.time() - t0) / 60
    with open(os.path.join(cli.out, "experiments_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(cli.out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved to {cli.out}/ ({out['runtime_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
