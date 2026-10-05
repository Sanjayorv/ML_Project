"""
benchmark_latency.py - measure real API latency for the report.
Start the API first (uvicorn app:app), then:  python benchmark_latency.py --n 200
Reports client round-trip time (what a user waits) and server model time.
"""
import argparse
import json
import random
import statistics
import time

import numpy as np
import requests


def pct(values, q):
    return float(np.percentile(values, q))


def summarise(name, values):
    s = {"median_ms": statistics.median(values), "p95_ms": pct(values, 95),
         "p99_ms": pct(values, 99), "mean_ms": statistics.mean(values), "n": len(values)}
    print(f"{name:<34} median {s['median_ms']:7.1f} | p95 {s['p95_ms']:7.1f} | p99 {s['p99_ms']:7.1f} ms")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="latency_report.json")
    args = ap.parse_args()

    meta = requests.get(f"{args.url}/metadata", timeout=10).json()
    lo, hi = meta["days_left_range"]
    rng = random.Random(42)
    profiles = meta.get("flight_profiles") or []

    def payload():
        if profiles:
            p = rng.choice(profiles)
            return {k: p[k] for k in ["airline", "source_city", "destination_city", "departure_time",
                                      "arrival_time", "stops", "duration", "flight"]} | {"days_left": rng.randint(lo, hi)}
        c = meta["categories"]
        src, dst = rng.sample(c["source_city"], 2)
        return {"airline": rng.choice(c["airline"]), "source_city": src, "destination_city": dst,
                "departure_time": rng.choice(c["departure_time"]), "arrival_time": rng.choice(c["arrival_time"]),
                "stops": rng.choice(c["stops"]), "duration": 2.5, "days_left": rng.randint(lo, hi)}

    session = requests.Session()
    for _ in range(args.warmup):
        session.post(f"{args.url}/predict", json=payload(), timeout=30)

    report = {"model_version": meta.get("model_version"), "requests": args.n}
    for label, params in [("single /predict", None), ("single /predict?explain=true", {"explain": "true"})]:
        client, server = [], []
        for _ in range(args.n):
            t = time.perf_counter()
            r = session.post(f"{args.url}/predict", json=payload(), params=params, timeout=30)
            client.append((time.perf_counter() - t) * 1000)
            r.raise_for_status()
            server.append(r.json()["model_latency_ms"])
        report[label] = {"client_round_trip": summarise(label + " (round trip)", client),
                         "server_model_time": summarise(label + " (model only)", server)}

    batch_client = []
    for _ in range(max(10, args.n // 10)):
        items = [payload() | {"days_left": d} for d in range(lo, hi + 1)]
        t = time.perf_counter()
        session.post(f"{args.url}/predict_batch", json={"items": items}, timeout=60).raise_for_status()
        batch_client.append((time.perf_counter() - t) * 1000)
    report[f"batch of {hi - lo + 1} (price curve)"] = summarise(f"batch x{hi - lo + 1} (round trip)", batch_client)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.out}. Measured on this machine, locally; network latency not included.")


if __name__ == "__main__":
    main()
