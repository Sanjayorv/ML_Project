"""
strategy.py
-----------
Turns a predicted MARKET fare into a revenue-maximising price under explicit,
user-supplied assumptions. Nothing here is learned from data.

Demand model (constant price elasticity):
    mu(p) = D0 * (p / p_market) ** (-elasticity)
    D0         = expected booking requests before departure if the fare equals the market fare
    elasticity = % drop in requests for a 1% price rise

Booking requests are random: N ~ Poisson(mu(p)). Only `seats` can be sold, so
    expected seats sold  S(p) = E[min(N, seats)] = sum_{k=0}^{seats-1} P(N > k)
    expected revenue     R(p) = p * S(p)
The optimal price maximises R(p) over a price grid.
"""
import numpy as np
from scipy.stats import poisson


def expected_seats_sold(mu, seats):
    """E[min(N, seats)] for N ~ Poisson(mu); vectorised over mu."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    k = np.arange(int(seats))
    return poisson.sf(k[None, :], mu[:, None]).sum(axis=1)


def _evaluate(prices, market_price, seats, demand_at_market, elasticity):
    mu = demand_at_market * (prices / market_price) ** (-elasticity)
    sold = expected_seats_sold(mu, seats)
    return mu, sold, prices * sold


def optimise_price(market_price, seats, demand_at_market, elasticity,
                   min_multiplier=0.5, max_multiplier=2.0,
                   lower_bound=None, upper_bound=None, n_grid=401, with_sensitivity=True):
    if market_price <= 0 or seats < 1 or demand_at_market <= 0 or elasticity <= 0:
        raise ValueError("market_price, seats, demand_at_market and elasticity must be positive")

    lo = market_price * min_multiplier
    hi = market_price * max_multiplier
    if lower_bound is not None:
        lo = max(lo, lower_bound)
    if upper_bound is not None:
        hi = min(hi, upper_bound)
    hi = max(hi, lo)

    prices = np.linspace(lo, hi, n_grid)
    mu, sold, revenue = _evaluate(prices, market_price, seats, demand_at_market, elasticity)
    i = int(np.argmax(revenue))

    _, sold_m, rev_m = _evaluate(np.array([market_price]), market_price, seats, demand_at_market, elasticity)
    at_boundary = hi > lo and i in (0, n_grid - 1)

    notes = []
    if at_boundary:
        side = "upper" if i == n_grid - 1 else "lower"
        notes.append(
            f"The optimum sits at the {side} edge of the search range, so the true optimum under these "
            "assumptions lies outside it. With elasticity at or below 1, raising prices always adds "
            "revenue until seats go unsold, so the result is driven by the range you allow.")
    if sold[i] / seats < 0.5:
        notes.append("Expected load factor at the optimum is below 50%: many seats would fly empty.")

    step = max(1, n_grid // 120)
    result = {
        "market_price": round(float(market_price), 2),
        "optimal_price": round(float(prices[i]), 2),
        "price_change_pct": round(float((prices[i] / market_price - 1) * 100), 2),
        "expected_booking_requests": round(float(mu[i]), 2),
        "expected_seats_sold": round(float(sold[i]), 2),
        "expected_load_factor": round(float(sold[i] / seats), 4),
        "expected_revenue": round(float(revenue[i]), 2),
        "market_expected_seats_sold": round(float(sold_m[0]), 2),
        "market_expected_revenue": round(float(rev_m[0]), 2),
        "revenue_uplift_pct": round(float((revenue[i] / rev_m[0] - 1) * 100), 2) if rev_m[0] > 0 else None,
        "search_range": [round(float(lo), 2), round(float(hi), 2)],
        "at_search_boundary": bool(at_boundary),
        "notes": notes,
        "curve": [{"price": round(float(p), 2), "expected_seats_sold": round(float(s), 2),
                   "expected_revenue": round(float(r), 2)}
                  for p, s, r in zip(prices[::step], sold[::step], revenue[::step])],
        "assumptions": {"seats": int(seats), "demand_at_market": float(demand_at_market),
                        "elasticity": float(elasticity),
                        "model": "constant elasticity demand, Poisson booking requests, capacity cap"},
    }
    if with_sensitivity:
        result["sensitivity"] = []
        for e in (0.5, 1.0, 1.5, 2.0, 3.0):
            r = optimise_price(market_price, seats, demand_at_market, e, min_multiplier, max_multiplier,
                               lower_bound, upper_bound, n_grid, with_sensitivity=False)
            result["sensitivity"].append({
                "elasticity": e, "optimal_price": r["optimal_price"],
                "expected_load_factor": r["expected_load_factor"],
                "expected_revenue": r["expected_revenue"], "at_search_boundary": r["at_search_boundary"]})
    return result
