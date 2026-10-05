import os
import time
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pricing_core  # noqa: F401  (EnsemblePricer must be importable to unpickle the model)
from pricing_core import BASE_CATEGORICAL, normalize_labels
from strategy import optimise_price

MODEL_PATH = os.getenv("PRICING_MODEL_PATH", "models/pricing_ensemble.pkl")
MAX_BATCH = 500

app = FastAPI(
    title="AM5305 Airline Fare Prediction API",
    description="Economy fare prediction (stacked LightGBM + XGBoost + CatBoost) with fare ranges, "
                "SHAP explanations and an assumption-driven revenue strategy layer.",
    version="5.0.0",
)

model = None
KNOWN_FLIGHTS = {}


def set_model(new_model):
    """Install a model (also used by the tests to inject a small model)."""
    global model, KNOWN_FLIGHTS
    model = new_model
    KNOWN_FLIGHTS = {}
    if model is not None and model.use_flight:
        for p in model.metadata.get("flight_profiles", []):
            key = (p["airline"], p["source_city"], p["destination_city"])
            KNOWN_FLIGHTS.setdefault(key, set()).add(p["flight"])


set_model(joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None)


# ------------------------------------------------------------------ schemas
class FlightRequest(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    stops: str
    days_left: int = Field(..., ge=1, le=60)
    duration: float = Field(..., gt=0, le=60)
    flight: Optional[str] = None  # e.g. "UK-819"; strongly improves accuracy when known


class BatchRequest(BaseModel):
    items: List[FlightRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class StrategyRequest(BaseModel):
    flight: FlightRequest
    remaining_seats: int = Field(..., ge=1, le=500)
    expected_demand_at_market_fare: float = Field(..., gt=0, le=5000,
        description="Expected booking requests before departure if priced at the market fare")
    elasticity: float = Field(..., gt=0, le=6)
    min_multiplier: float = Field(0.5, gt=0, le=1)
    max_multiplier: float = Field(2.0, ge=1, le=5)
    stay_within_fare_range: bool = False


# ------------------------------------------------------------------ helpers
def require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training script first.")


def prepare(items: List[FlightRequest]):
    df = normalize_labels(pd.DataFrame([i.model_dump() for i in items]))
    cats = model.metadata["categories"]
    lo, hi = model.metadata["days_left_range"]
    warnings = []
    for idx, r in df.iterrows():
        where = f"item {idx}: " if len(df) > 1 else ""
        if r["source_city"] == r["destination_city"]:
            raise HTTPException(status_code=422, detail=f"{where}source_city and destination_city must differ.")
        invalid = {c: r[c] for c in BASE_CATEGORICAL if r[c] not in cats[c]}
        if invalid:
            raise HTTPException(status_code=422, detail={
                "message": f"{where}Unknown category values.", "invalid": invalid,
                "allowed": {c: cats[c] for c in invalid}})
        w = []
        if not lo <= r["days_left"] <= hi:
            w.append(f"days_left outside training range {lo}-{hi}; prediction is an extrapolation.")
        if model.use_flight:
            known = KNOWN_FLIGHTS.get((r["airline"], r["source_city"], r["destination_city"]), set())
            if r["flight"] not in known:
                w.append("Flight code not provided or not seen on this airline/route; accuracy is reduced.")
        warnings.append(w)
    return df, warnings


def run(items, explain=False):
    df, warnings = prepare(items)
    t0 = time.perf_counter()
    try:
        results = model.predict_details(df, explain=explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    latency_ms = (time.perf_counter() - t0) * 1000
    for rec, w in zip(results, warnings):
        rec["recommended_price"] = rec["predicted_price"]  # backward compatibility
        if explain and rec.get("explanation") is None:
            w = w + ["Explanation unavailable for this model."]
        rec["warnings"] = w
    return results, latency_ms


# ------------------------------------------------------------------ endpoints
@app.get("/")
def health_check():
    return {"status": "Active", "model_loaded": model is not None, "model_path": MODEL_PATH,
            "model_version": model.metadata.get("model_version") if model else None}


@app.get("/metadata")
def metadata():
    require_model()
    return model.metadata


@app.post("/predict")
def predict_price(request: FlightRequest, explain: bool = False):
    require_model()
    results, latency_ms = run([request], explain=explain)
    return {"currency": "INR", **results[0], "model_latency_ms": round(latency_ms, 2),
            "model_version": model.metadata.get("model_version")}


@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    require_model()
    results, latency_ms = run(request.items)
    return {"currency": "INR", "count": len(results), "results": results,
            "model_latency_ms": round(latency_ms, 2), "model_version": model.metadata.get("model_version")}


@app.post("/strategy")
def revenue_strategy(request: StrategyRequest):
    require_model()
    results, latency_ms = run([request.flight])
    market = results[0]
    bounds = {}
    if request.stay_within_fare_range:
        if not market["price_range"]:
            raise HTTPException(status_code=422, detail="This model has no fare range; untick stay_within_fare_range.")
        bounds = {"lower_bound": market["price_range"]["lower"], "upper_bound": market["price_range"]["upper"]}
    try:
        plan = optimise_price(market["predicted_price"], request.remaining_seats,
                              request.expected_demand_at_market_fare, request.elasticity,
                              request.min_multiplier, request.max_multiplier, **bounds)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"currency": "INR", "market_prediction": market, "strategy": plan,
            "model_latency_ms": round(latency_ms, 2), "model_version": model.metadata.get("model_version")}
