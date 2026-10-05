"""
pricing_core.py
---------------
Shared code imported by BOTH the training script and the FastAPI server.

Keeping label cleaning and feature engineering in one module guarantees the
API builds exactly the same features the model was trained on, so there is
no train/serve skew (the source of the silent "IndiGo" vs "Indigo" bug).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE_CATEGORICAL = [
    "airline", "source_city", "destination_city",
    "departure_time", "arrival_time", "stops",
]
STOPS_TO_NUM = {"zero": 0, "one": 1, "two_or_more": 2}


# ---------------------------------------------------------------------------
# 1. Label normalisation (applied to training data AND every API request)
# ---------------------------------------------------------------------------
def _key(value) -> str:
    return re.sub(r"[^a-z0-9+]", "", str(value).lower())


_ALIASES = {
    "airline": {
        "indigo": "Indigo", "airindia": "Air_India", "gofirst": "GO_FIRST",
        "goair": "GO_FIRST", "spicejet": "SpiceJet", "vistara": "Vistara",
        "airasia": "AirAsia",
    },
    "city": {
        "delhi": "Delhi", "newdelhi": "Delhi", "mumbai": "Mumbai",
        "bombay": "Mumbai", "bangalore": "Bangalore", "bengaluru": "Bangalore",
        "kolkata": "Kolkata", "calcutta": "Kolkata", "hyderabad": "Hyderabad",
        "chennai": "Chennai", "madras": "Chennai",
    },
    "time": {
        "earlymorning": "Early_Morning", "morning": "Morning",
        "afternoon": "Afternoon", "evening": "Evening", "night": "Night",
        "latenight": "Late_Night",
    },
    "stops": {
        "zero": "zero", "0": "zero", "nonstop": "zero", "direct": "zero",
        "one": "one", "1": "one",
        "twoormore": "two_or_more", "morethantwo": "two_or_more",
        "two": "two_or_more", "2": "two_or_more", "2+": "two_or_more",
    },
}
_COLUMN_GROUP = {
    "airline": "airline", "source_city": "city", "destination_city": "city",
    "departure_time": "time", "arrival_time": "time", "stops": "stops",
}


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map spelling variants to one canonical label. Unknown values pass through."""
    df = df.copy()
    for col, group in _COLUMN_GROUP.items():
        if col in df.columns:
            mapping = _ALIASES[group]
            df[col] = df[col].map(lambda v, m=mapping: m.get(_key(v), str(v).strip()))
    if "flight" in df.columns:
        df["flight"] = (
            df["flight"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
            .replace({"": "UNKNOWN", "NONE": "UNKNOWN"})
        )
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
INTERACTIONS = ["airline_route", "route_stops", "dep_arr"]

FEATURE_LABELS = {
    "airline": "Airline", "source_city": "Origin city", "destination_city": "Destination city",
    "departure_time": "Departure time", "arrival_time": "Arrival time", "stops": "Stops (category)",
    "route": "Route", "airline_route": "Airline on this route", "route_stops": "Stops on this route",
    "dep_arr": "Departure/arrival pairing", "flight": "Flight number", "duration": "Duration",
    "days_left": "Days until departure", "stops_num": "Number of stops",
    "lead_time_urgency": "Lead-time urgency", "surge_probability": "Surge probability",
}


def feature_columns(use_flight: bool, include_stat_features: bool, include_interactions: bool = True):
    cat = BASE_CATEGORICAL + ["route"] + (INTERACTIONS if include_interactions else [])
    if use_flight:
        cat.append("flight")
    num = ["duration", "days_left", "stops_num"]
    if include_stat_features:
        num += ["lead_time_urgency", "surge_probability"]
    return cat, num


def build_features(df: pd.DataFrame, use_flight: bool = True,
                   include_stat_features: bool = False,
                   include_interactions: bool = True) -> pd.DataFrame:
    cat_cols, num_cols = feature_columns(use_flight, include_stat_features, include_interactions)
    X = pd.DataFrame(index=df.index)
    for c in BASE_CATEGORICAL:
        X[c] = df[c].astype(str)

    # Explicit interactions: fare levels depend on airline x route, route x stops
    X["route"] = X["source_city"] + "-" + X["destination_city"]
    X["airline_route"] = X["airline"] + "|" + X["route"]
    X["route_stops"] = X["route"] + "|" + X["stops"]
    X["dep_arr"] = X["departure_time"] + "|" + X["arrival_time"]
    if use_flight:
        X["flight"] = df["flight"].astype(str) if "flight" in df.columns else "UNKNOWN"

    X["duration"] = pd.to_numeric(df["duration"]).astype(float)
    X["days_left"] = pd.to_numeric(df["days_left"]).astype(float)
    X["stops_num"] = X["stops"].map(STOPS_TO_NUM).astype(float)

    if include_stat_features:
        X["lead_time_urgency"] = 1.0 / (X["days_left"] + 1.0)
        lam = 50.0 / (X["days_left"] + 1.0)
        X["surge_probability"] = poisson.sf(30, lam)  # = 1 - CDF(30)

    return X[cat_cols + num_cols]


def to_tree_frame(X: pd.DataFrame, cat_cols, categories) -> pd.DataFrame:
    """LightGBM/XGBoost: pandas 'category' dtype with a FIXED category list,
    so codes are identical at training and inference. Unseen values -> NaN."""
    X = X.copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return X


def to_catboost_frame(X: pd.DataFrame, cat_cols) -> pd.DataFrame:
    """CatBoost: plain strings; it applies ordered target statistics itself."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    return X


def forward_target(y, kind: str):
    y = np.asarray(y, dtype=float)
    return np.log1p(y) if kind == "log" else y


def inverse_target(p, kind: str):
    p = np.asarray(p, dtype=float)
    return np.expm1(p) if kind == "log" else p


# ---------------------------------------------------------------------------
# 3. Per-model SHAP contributions (native TreeSHAP in each library; no shap package)
# ---------------------------------------------------------------------------
def model_contributions(name, model, frame, cat_cols):
    """Returns an (n_rows, n_features + 1) array; the last column is the baseline."""
    if name == "lightgbm":
        return np.asarray(model.predict(frame, pred_contrib=True))
    if name == "xgboost":
        import xgboost as xgb
        dm = xgb.DMatrix(frame, enable_categorical=True)
        return np.asarray(model.get_booster().predict(dm, pred_contribs=True))
    if name == "catboost":
        from catboost import Pool
        pool = Pool(frame, cat_features=list(cat_cols))
        return np.asarray(model.get_feature_importance(pool, type="ShapValues"))
    raise ValueError(f"No contribution method for model '{name}'")


# ---------------------------------------------------------------------------
# 4. Deployable model object
# ---------------------------------------------------------------------------
class EnsemblePricer:
    """Base models -> INR predictions -> non-negative linear blender,
    plus optional conformalised quantile models for a fare range."""

    def __init__(self, base_models, frame_kinds, meta, cat_cols, num_cols,
                 categories, target_kind, use_flight, include_stat_features,
                 metadata, min_price, include_interactions=True,
                 quantile_models=None, interval_correction=0.0, interval_level=None):
        self.base_models = base_models      # {name: fitted model}
        self.frame_kinds = frame_kinds      # {name: "tree" | "catboost"}
        self.meta = meta
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.categories = categories
        self.target_kind = target_kind
        self.use_flight = use_flight
        self.include_stat_features = include_stat_features
        self.include_interactions = include_interactions
        self.metadata = metadata
        self.min_price = min_price
        self.quantile_models = quantile_models or {}   # {"lower": model, "upper": model}
        self.interval_correction = interval_correction
        self.interval_level = interval_level

    # ---- internals -------------------------------------------------------
    def _frame(self, X, kind):
        if kind == "tree":
            return to_tree_frame(X, self.cat_cols, self.categories)
        return to_catboost_frame(X, self.cat_cols)

    def features(self, df_raw):
        df = normalize_labels(df_raw)
        return build_features(df, self.use_flight, self.include_stat_features,
                              getattr(self, "include_interactions", True))

    def _point(self, X):
        comps = pd.DataFrame(index=X.index)
        for name, model in self.base_models.items():
            raw = model.predict(self._frame(X, self.frame_kinds[name]))
            comps[name] = inverse_target(raw, self.target_kind)
        blended = self.meta.predict(comps.to_numpy())
        return np.clip(blended, self.min_price, None), comps

    def _interval(self, X):
        q = getattr(self, "quantile_models", None)
        if not q:
            return None
        F = to_tree_frame(X, self.cat_cols, self.categories)
        lo = inverse_target(q["lower"].predict(F) - self.interval_correction, self.target_kind)
        hi = inverse_target(q["upper"].predict(F) + self.interval_correction, self.target_kind)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def blend_weights(self):
        names = list(self.base_models)
        coef = np.asarray(getattr(self.meta, "coef_", np.ones(len(names))), float)
        coef = np.clip(coef, 0, None)
        if coef.sum() <= 0:
            coef = np.ones(len(names))
        return dict(zip(names, coef / coef.sum()))

    def explain_matrix(self, X):
        """Blend-weighted SHAP values, converted to rupees.

        In log-price space each model is exactly additive. Rupee effects split the
        difference between the baseline fare and the explained fare in proportion
        to each feature's log contribution, so they sum exactly to that difference."""
        cols = list(X.columns)
        p = len(cols)
        phi = np.zeros((len(X), p))
        base = np.zeros(len(X))
        used, total = [], 0.0
        for name, w in self.blend_weights().items():
            if w <= 0:
                continue
            frame = self._frame(X, self.frame_kinds[name])
            try:
                c = model_contributions(name, self.base_models[name], frame, self.cat_cols)
            except Exception:
                continue
            phi += w * c[:, :p]
            base += w * c[:, p]
            used.append(name)
            total += w
        if not used:
            raise RuntimeError("None of the base models could produce contributions.")
        phi, base = phi / total, base / total

        if self.target_kind == "log":
            base_price = np.expm1(base)
            s = phi.sum(axis=1)
            explained = np.expm1(base + s)
            safe = np.where(np.abs(s) > 1e-6, s, 1.0)
            ratio = np.where(np.abs(s) > 1e-6, (explained - base_price) / safe, np.exp(base))
            rupees = phi * ratio[:, None]
            percent = np.expm1(phi) * 100
        else:
            base_price = base
            explained = base + phi.sum(axis=1)
            rupees = phi
            percent = phi / np.where(base == 0, 1, base)[:, None] * 100

        return {
            "rupees": pd.DataFrame(rupees, columns=cols, index=X.index),
            "percent": pd.DataFrame(percent, columns=cols, index=X.index),
            "baseline_price": base_price, "explained_price": explained, "models_used": used,
        }

    # ---- public API --------------------------------------------------------
    def predict_with_components(self, df_raw: pd.DataFrame):
        return self._point(self.features(df_raw))

    def predict(self, df_raw: pd.DataFrame):
        return self.predict_with_components(df_raw)[0]

    def predict_details(self, df_raw: pd.DataFrame, explain: bool = False, top_n: int = 8):
        X = self.features(df_raw)
        price, comps = self._point(X)
        interval = self._interval(X)
        expl, expl_error = None, None
        if explain:
            try:
                expl = self.explain_matrix(X)
            except Exception as e:  # explanation is optional; never fail the prediction
                expl_error = str(e)

        results = []
        for i in range(len(X)):
            rec = {
                "predicted_price": round(float(price[i]), 2),
                "base_model_prices": {k: round(float(v), 2) for k, v in comps.iloc[i].items()},
                "price_range": None,
            }
            if interval is not None:
                rec["price_range"] = {
                    "lower": round(float(min(interval[0][i], price[i])), 2),
                    "upper": round(float(max(interval[1][i], price[i])), 2),
                    "level": self.interval_level,
                }
            if explain:
                if expl is None:
                    rec["explanation"] = None
                    rec["explanation_error"] = expl_error
                else:
                    r, pc = expl["rupees"].iloc[i], expl["percent"].iloc[i]
                    order = r.abs().sort_values(ascending=False).index
                    top, rest = list(order[:top_n]), list(order[top_n:])
                    contributions = []
                    for f in top:
                        v = X.iloc[i][f]
                        contributions.append({
                            "feature": f, "label": FEATURE_LABELS.get(f, f),
                            "value": round(float(v), 2) if isinstance(v, (int, float, np.number)) else str(v),
                            "rupees": round(float(r[f]), 2), "percent": round(float(pc[f]), 2),
                        })
                    rec["explanation"] = {
                        "baseline_price": round(float(expl["baseline_price"][i]), 2),
                        "explained_price": round(float(expl["explained_price"][i]), 2),
                        "models_used": expl["models_used"],
                        "contributions": contributions,
                        "other_features_rupees": round(float(r[rest].sum()), 2) if rest else 0.0,
                    }
            results.append(rec)
        return results
