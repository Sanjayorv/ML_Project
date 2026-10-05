import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("PRICING_API_URL", "http://127.0.0.1:8000")

INK = "#10243B"
TEAL = "#0E7C86"
AMBER = "#B4761A"
UP = "#A63A28"
DOWN = "#1F6F4A"
MUTED = "#5A6B7B"
LINE = "#DCE3E9"

st.set_page_config(page_title="Airline Fare Console | AM5305", page_icon="✈️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
[data-testid="stMetricValue"], .fare-figure, .fare-range, table, .stDataFrame {{ font-feature-settings: 'tnum' 1; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; font-weight: 600; }}
h1 {{ font-size: 1.75rem; margin-bottom: 0.15rem; }}
h3 {{ font-size: 1.05rem; margin-top: 1.4rem; }}
.itinerary {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 0.2rem; }}
.stamp {{ color: {MUTED}; font-size: 0.78rem; }}
.fare-block {{ border: 1px solid {LINE}; border-left: 3px solid {INK}; border-radius: 4px;
              padding: 1.1rem 1.3rem; background: #FFFFFF; }}
.fare-label {{ color: {MUTED}; font-size: 0.82rem; margin-bottom: 0.25rem; }}
.fare-figure {{ color: {INK}; font-size: 2.6rem; font-weight: 600; line-height: 1.05; }}
.fare-range {{ color: {MUTED}; font-size: 0.92rem; margin-top: 0.35rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.35rem 0; font-size: 0.95rem; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.82rem; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
.stDeployButton, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API layer
@st.cache_data(ttl=300)
def load_metadata():
    r = requests.get(f"{API_URL}/metadata", timeout=5)
    r.raise_for_status()
    return r.json()


def call_api(path, payload, params=None, timeout=30):
    r = None
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("The pricing server is not responding. Start it with `uvicorn app:app --reload`, then try again.")
    except requests.exceptions.Timeout:
        st.error("The pricing server took too long. Try again, or reduce the lead-time range.")
    except requests.exceptions.HTTPError:
        detail, code = None, "error"
        if r is not None:
            code = r.status_code
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
        st.error(f"The server rejected this request ({code}). {detail or 'No details returned.'}")
    return None


def remember(key, signature, value):
    st.session_state[key] = {"sig": signature, "value": value}


def recall(key, signature):
    item = st.session_state.get(key)
    return item["value"] if item and item["sig"] == signature else None


def rupees(x):
    return f"₹{x:,.0f}"


def style(chart, height=320):
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont="IBM Plex Sans", titleFont="IBM Plex Sans", labelColor=MUTED,
                            titleColor=MUTED, titleFontWeight=500, labelFontSize=11, titleFontSize=11,
                            grid=True, gridColor="#EDF1F4", domainColor=LINE, tickColor=LINE)
            .configure_legend(labelFont="IBM Plex Sans", titleFont="IBM Plex Sans", labelColor=MUTED))


def empty_state(message):
    st.markdown(f"<p style='color:{MUTED};padding:2.2rem 0;'>{message}</p>", unsafe_allow_html=True)


try:
    meta = load_metadata()
except requests.exceptions.ConnectionError:
    st.title("Airline Fare Console")
    st.error("The pricing server is not running. Start it with `uvicorn app:app --reload`, then reload this page.")
    st.stop()
except requests.exceptions.HTTPError:
    st.title("Airline Fare Console")
    st.error("The server is running but has no trained model. Run `python train_ensemble_optimized.py`, "
             "restart the server, then reload this page.")
    st.stop()

cats = meta["categories"]
PRETTY = {"Air_India": "Air India", "GO_FIRST": "Go First", "two_or_more": "two or more",
          "Early_Morning": "Early morning", "Late_Night": "Late night", "zero": "non-stop", "one": "one stop"}
fmt = lambda v: PRETTY.get(v, v)  # noqa: E731


def index_of(options, value, default=0):
    return options.index(value) if value in options else default


# ------------------------------------------------------------------ left rail: the flight
with st.sidebar:
    st.markdown("### Flight")
    airline = st.selectbox("Airline", cats["airline"], format_func=fmt)
    source_city = st.selectbox("From", cats["source_city"])
    destination_city = st.selectbox("To", [c for c in cats["destination_city"] if c != source_city])

    route_profiles = [p for p in meta.get("flight_profiles", [])
                      if p["source_city"] == source_city and p["destination_city"] == destination_city]
    profile, flight = None, None
    if meta.get("uses_flight_code"):
        matches = sorted((p for p in route_profiles if p["airline"] == airline), key=lambda p: -p["rows"])
        choice = st.selectbox("Flight number", ["Not specified"] + [p["flight"] for p in matches],
                              help="Choosing a real flight loads its usual schedule and gives the most accurate estimate.")
        if choice != "Not specified":
            flight = choice
            profile = next(p for p in matches if p["flight"] == choice)

    st.markdown("### Schedule")
    days_left = st.slider("Days until departure", int(meta["days_left_range"][0]), int(meta["days_left_range"][1]),
                          min(15, int(meta["days_left_range"][1])))
    departure_time = st.selectbox("Departs", cats["departure_time"], format_func=fmt,
                                  index=index_of(cats["departure_time"], profile["departure_time"]) if profile else 0)
    arrival_time = st.selectbox("Arrives", cats["arrival_time"], format_func=fmt,
                                index=index_of(cats["arrival_time"], profile["arrival_time"]) if profile else 0)
    stops = st.selectbox("Stops", cats["stops"], format_func=fmt,
                         index=index_of(cats["stops"], profile["stops"]) if profile else 0)
    d_lo, d_hi = meta["duration_range"]
    duration = st.slider("Duration (hours)", float(round(d_lo, 2)), float(round(d_hi, 2)),
                         min(max(float(profile["duration"]) if profile else 2.5, d_lo), d_hi), 0.05)

    if profile and (stops != profile["stops"] or departure_time != profile["departure_time"]
                    or arrival_time != profile["arrival_time"]):
        st.warning(f"{flight} does not usually run this schedule, so the estimate is less reliable.")
    st.markdown(f"<p class='stamp'>Model {meta.get('model_version', 'unknown')}<br/>"
                f"trained {str(meta.get('trained_at', 'unknown'))[:10]}</p>", unsafe_allow_html=True)

payload = {"airline": airline, "source_city": source_city, "destination_city": destination_city,
           "departure_time": departure_time, "arrival_time": arrival_time, "stops": stops,
           "days_left": days_left, "duration": duration, "flight": flight}
signature = tuple(sorted(payload.items()))
level = (meta.get("interval") or {}).get("target_level")

# ------------------------------------------------------------------ header
st.title("Airline Fare Console")
st.markdown(
    f"<p class='itinerary'>{fmt(airline)}{' ' + flight if flight else ''}, {source_city} to {destination_city}, "
    f"{fmt(departure_time).lower()} departure {fmt(stops)}, booked {days_left} "
    f"{'day' if days_left == 1 else 'days'} before departure.</p>", unsafe_allow_html=True)
st.caption("Economy fares for domestic Indian routes, estimated from historical booking data.")

tab_fare, tab_curve, tab_price, tab_model = st.tabs(
    ["Fare estimate", "Booking curve", "Price setting", "Model quality"])

# ------------------------------------------------------------------ tab 1
with tab_fare:
    if st.button("Estimate this fare", type="primary"):
        with st.spinner("Pricing the flight"):
            res = call_api("/predict", payload, params={"explain": "true"})
        if res:
            remember("prediction", signature, res)

    res = recall("prediction", signature)
    if not res:
        empty_state("Set up a flight on the left, then estimate its fare.")
    else:
        pr = res.get("price_range")
        range_line = (f"Most comparable fares fall between {rupees(pr['lower'])} and {rupees(pr['upper'])}"
                      if pr else "This model was trained without a fare range.")
        st.markdown(f"""<div class="fare-block">
          <div class="fare-label">Estimated market fare</div>
          <div class="fare-figure">{rupees(res['predicted_price'])}</div>
          <div class="fare-range">{range_line}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"<p class='stamp' style='margin-top:0.5rem'>Returned in "
                    f"{res['model_latency_ms']:.0f} ms</p>", unsafe_allow_html=True)
        for w in res.get("warnings", []):
            st.warning(w)

        ex = res.get("explanation")
        if ex:
            st.markdown("### What moves this fare")
            contrib = pd.DataFrame(ex["contributions"])
            contrib["factor"] = contrib["label"] + ": " + contrib["value"].astype(str)
            if abs(ex["other_features_rupees"]) >= 1:
                contrib = pd.concat([contrib, pd.DataFrame([{"factor": "Everything else",
                                                             "rupees": ex["other_features_rupees"], "percent": None}])])
            contrib["direction"] = contrib["rupees"].apply(lambda v: "Raises the fare" if v >= 0 else "Lowers the fare")
            chart = alt.Chart(contrib).mark_bar(height=16).encode(
                x=alt.X("rupees:Q", title="Effect on the fare (₹)"),
                y=alt.Y("factor:N", sort=None, title=None),
                color=alt.Color("direction:N", title=None,
                                scale=alt.Scale(domain=["Raises the fare", "Lowers the fare"], range=[UP, DOWN])),
                tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("rupees:Q", format=",.0f", title="₹"),
                         alt.Tooltip("percent:Q", format="+.1f", title="%")])
            st.altair_chart(style(chart, height=40 + 26 * len(contrib)), use_container_width=True)
            st.caption(
                f"Starting from an average fare of {rupees(ex['baseline_price'])}, these factors together reach "
                f"{rupees(ex['explained_price'])}. Figures are blend-weighted SHAP contributions from "
                f"{', '.join(ex['models_used'])}; the final estimate differs slightly because the blender adds an "
                "intercept. They show patterns the model learned, not causes.")

        with st.expander("Individual model estimates"):
            comp = pd.DataFrame({"Model": list(res["base_model_prices"]),
                                 "Estimate": [rupees(v) for v in res["base_model_prices"].values()]})
            st.dataframe(comp, hide_index=True, use_container_width=True)
            st.caption("Close agreement means the flight resembles the training data. A wide spread means "
                       "the estimate rests on less evidence.")

# ------------------------------------------------------------------ tab 2
with tab_curve:
    st.caption("How the estimated fare changes with booking lead time, and how carriers compare on this route.")
    if st.button("Plot the booking curve"):
        with st.spinner("Pricing every lead time"):
            t_lo, t_hi = int(meta["days_left_range"][0]), int(meta["days_left_range"][1])
            days = list(range(t_lo, t_hi + 1))
            curve = call_api("/predict_batch", {"items": [{**payload, "days_left": d} for d in days]})
            comp_items, comp_labels = [], []
            if meta.get("uses_flight_code"):
                for a in cats["airline"]:
                    ps = sorted((p for p in route_profiles if p["airline"] == a), key=lambda p: -p["rows"])
                    if ps:
                        p = ps[0]
                        comp_items.append({"airline": a, "source_city": source_city,
                                           "destination_city": destination_city, "departure_time": p["departure_time"],
                                           "arrival_time": p["arrival_time"], "stops": p["stops"],
                                           "duration": p["duration"], "flight": p["flight"], "days_left": days_left})
                        comp_labels.append(f"{fmt(a)} {p['flight']}")
            else:
                for a in cats["airline"]:
                    comp_items.append({**payload, "airline": a})
                    comp_labels.append(fmt(a))
            comp = call_api("/predict_batch", {"items": comp_items}) if comp_items else None
        if curve:
            remember("curve", signature, {"days": days, "curve": curve, "comp": comp, "labels": comp_labels})

    stored = recall("curve", signature)
    if not stored:
        empty_state("Plot the curve to see how this fare behaves as departure approaches.")
    else:
        rows = []
        for d, r in zip(stored["days"], stored["curve"]["results"]):
            pr = r.get("price_range") or {}
            rows.append({"days_left": d, "predicted": r["predicted_price"],
                         "lower": pr.get("lower"), "upper": pr.get("upper")})
        cdf = pd.DataFrame(rows)
        cheapest = cdf.loc[cdf["predicted"].idxmin()]
        near = cdf[cdf["predicted"] <= cheapest["predicted"] * 1.03]["days_left"]
        now = cdf.loc[cdf["days_left"] == days_left, "predicted"]

        k1, k2, k3 = st.columns(3)
        k1.metric("Cheapest lead time", f"{int(cheapest['days_left'])} days", rupees(cheapest["predicted"]),
                  delta_color="off")
        k2.metric("Within 3% of cheapest", f"{int(near.min())} to {int(near.max())} days")
        if len(now):
            k3.metric(f"At {days_left} days", rupees(now.iloc[0]),
                      f"{(now.iloc[0] / cheapest['predicted'] - 1) * 100:+.1f}% vs cheapest", delta_color="inverse")

        base = alt.Chart(cdf).encode(x=alt.X("days_left:Q", title="Days until departure",
                                             scale=alt.Scale(reverse=True)))
        layers = []
        if cdf["lower"].notna().all():
            layers.append(base.mark_area(opacity=0.14, color=TEAL).encode(y="lower:Q", y2="upper:Q"))
        layers.append(base.mark_line(color=INK, strokeWidth=2).encode(
            y=alt.Y("predicted:Q", title="Estimated fare (₹)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("days_left:Q", title="Days out"),
                     alt.Tooltip("predicted:Q", format=",.0f", title="Fare"),
                     alt.Tooltip("lower:Q", format=",.0f", title="Low"),
                     alt.Tooltip("upper:Q", format=",.0f", title="High")]))
        layers.append(alt.Chart(pd.DataFrame({"days_left": [days_left]}))
                      .mark_rule(color=AMBER, strokeDash=[4, 3]).encode(x="days_left:Q"))
        st.altair_chart(style(alt.layer(*layers)), use_container_width=True)
        st.caption("The shaded band is the likely fare range; the amber line marks the lead time you selected.")

        if stored["comp"]:
            st.markdown(f"### Carriers on {source_city} to {destination_city}")
            comp_df = pd.DataFrame([{"carrier": lab, "predicted": r["predicted_price"],
                                     "lower": (r.get("price_range") or {}).get("lower"),
                                     "upper": (r.get("price_range") or {}).get("upper")}
                                    for lab, r in zip(stored["labels"], stored["comp"]["results"])]).sort_values("predicted")
            bars = alt.Chart(comp_df).mark_bar(height=18, color=TEAL, opacity=0.85).encode(
                x=alt.X("predicted:Q", title="Estimated fare (₹)"),
                y=alt.Y("carrier:N", sort=None, title=None),
                tooltip=[alt.Tooltip("carrier:N", title="Carrier"), alt.Tooltip("predicted:Q", format=",.0f", title="Fare")])
            chart = bars
            if comp_df["lower"].notna().all():
                chart = bars + alt.Chart(comp_df).mark_rule(color=INK).encode(
                    x="lower:Q", x2="upper:Q", y=alt.Y("carrier:N", sort=None))
            st.altair_chart(style(chart, height=40 + 30 * len(comp_df)), use_container_width=True)
            st.caption(f"Each carrier is priced at {days_left} days out"
                       + (", using its most frequently flown service on this route."
                          if meta.get("uses_flight_code") else "."))

# ------------------------------------------------------------------ tab 3
with tab_price:
    st.caption("Turns the estimated market fare into a price that maximises expected revenue on a flight with "
               "seats left to sell. The three demand inputs are your assumptions, not model output.")
    s1, s2, s3 = st.columns(3)
    seats = s1.number_input("Seats still unsold", 1, 500, 60)
    demand = s2.number_input("Bookings expected at market fare", 1.0, 5000.0, 75.0, 5.0,
                             help="How many travellers would book before departure if you charged the estimated fare.")
    elasticity = s3.slider("Price sensitivity", 0.2, 4.0, 1.5, 0.1,
                           help="1.5 means a 1% price rise loses about 1.5% of bookings. Leisure routes are "
                                "usually more sensitive than business routes.")
    r1, r2, r3 = st.columns(3)
    min_mult = r1.slider("Lowest price to consider", 0.3, 1.0, 0.6, 0.05, help="As a multiple of the estimated fare.")
    max_mult = r2.slider("Highest price to consider", 1.0, 3.0, 1.8, 0.05, help="As a multiple of the estimated fare.")
    within = r3.checkbox("Keep within the likely fare range", value=False, disabled=not level,
                         help="Only consider prices the market actually charges for comparable flights.")

    strat_payload = {"flight": payload, "remaining_seats": int(seats),
                     "expected_demand_at_market_fare": float(demand), "elasticity": float(elasticity),
                     "min_multiplier": float(min_mult), "max_multiplier": float(max_mult),
                     "stay_within_fare_range": bool(within)}
    strat_sig = (signature, tuple(sorted((k, v) for k, v in strat_payload.items() if k != "flight")))

    if st.button("Find the best price", type="primary"):
        with st.spinner("Testing prices"):
            out = call_api("/strategy", strat_payload)
        if out:
            remember("strategy", strat_sig, out)

    out = recall("strategy", strat_sig)
    if not out:
        empty_state("Set your demand assumptions, then find the price that earns the most.")
    else:
        plan = out["strategy"]
        st.markdown(f"""<div class="fare-block" style="border-left-color:{AMBER}">
          <div class="fare-label">Recommended price under these assumptions</div>
          <div class="fare-figure">{rupees(plan['optimal_price'])}</div>
          <div class="fare-range">{plan['price_change_pct']:+.1f}% against the estimated market fare of
          {rupees(plan['market_price'])}</div></div>""", unsafe_allow_html=True)
        for note in plan["notes"]:
            st.warning(note)

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected revenue", rupees(plan["expected_revenue"]),
                  f"{plan['revenue_uplift_pct']:+.1f}% vs market fare" if plan["revenue_uplift_pct"] is not None else None)
        m2.metric("Seats expected to sell", f"{plan['expected_seats_sold']:.0f} of {int(seats)}")
        m3.metric("Load factor", f"{plan['expected_load_factor']:.0%}")

        curve_df = pd.DataFrame(plan["curve"])
        marks = pd.DataFrame({"price": [plan["market_price"], plan["optimal_price"]],
                              "label": ["Estimated market fare", "Recommended price"]})
        rev = alt.Chart(curve_df).mark_line(color=INK, strokeWidth=2).encode(
            x=alt.X("price:Q", title="Ticket price (₹)"),
            y=alt.Y("expected_revenue:Q", title="Expected revenue (₹)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("price:Q", format=",.0f", title="Price"),
                     alt.Tooltip("expected_revenue:Q", format=",.0f", title="Revenue"),
                     alt.Tooltip("expected_seats_sold:Q", format=".1f", title="Seats")])
        rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(
            x="price:Q", color=alt.Color("label:N", title=None,
                                         scale=alt.Scale(domain=["Estimated market fare", "Recommended price"],
                                                         range=[TEAL, AMBER])),
            tooltip=["label:N", alt.Tooltip("price:Q", format=",.0f")])
        st.altair_chart(style(rev + rules), use_container_width=True)

        st.markdown("### If your price sensitivity assumption is wrong")
        sens = pd.DataFrame(plan["sensitivity"]).rename(columns={
            "elasticity": "Price sensitivity", "optimal_price": "Best price",
            "expected_load_factor": "Load factor", "expected_revenue": "Expected revenue",
            "at_search_boundary": "Limited by your price range"})
        sens["Best price"] = sens["Best price"].map(rupees)
        sens["Expected revenue"] = sens["Expected revenue"].map(rupees)
        sens["Load factor"] = sens["Load factor"].map(lambda v: f"{v:.0%}")
        sens["Limited by your price range"] = sens["Limited by your price range"].map({True: "yes", False: "no"})
        st.dataframe(sens, hide_index=True, use_container_width=True)
        st.caption("A recommendation that shifts sharply across these rows depends more on your assumption "
                   "than on the data.")

        with st.expander("How the recommendation is calculated"):
            st.markdown(
                "Bookings at price *p*: **μ(p) = D₀ × (p / market fare)^(−sensitivity)**, arriving at random "
                "(Poisson). You cannot sell more than the seats that remain, so expected seats sold is "
                "**E[min(N, seats)]**, and expected revenue is **p × seats sold**, maximised across your price "
                "range.\n\nThe model supplies only the market fare and its range. Demand, sensitivity and the "
                "price range are yours, which is why the table above matters.")

# ------------------------------------------------------------------ tab 4
with tab_model:
    tm = meta.get("test_metrics", {})
    if tm:
        st.markdown("### Accuracy on flights held back from training")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Typical error (RMSE)", rupees(tm["RMSE"]))
        c2.metric("Average error (MAE)", rupees(tm["MAE"]))
        c3.metric("Variance explained (R²)", f"{tm['R2']:.3f}")
        c4.metric("Average error", f"{tm['MAPE_%']:.1f}%")

    if meta.get("test_metrics_all_models"):
        tbl = (pd.DataFrame(meta["test_metrics_all_models"]).T.reset_index()
               .rename(columns={"index": "model"}).sort_values("RMSE"))
        tbl["model"] = tbl["model"].str.replace("_", " ").str.capitalize()
        show = pd.DataFrame({"Model": tbl["model"], "RMSE": tbl["RMSE"].map(rupees), "MAE": tbl["MAE"].map(rupees),
                             "R²": tbl["R2"].map("{:.4f}".format), "MAPE": tbl["MAPE_%"].map("{:.2f}%".format)})
        st.dataframe(show, hide_index=True, use_container_width=True)

    iv = meta.get("interval")
    if iv:
        st.markdown("### Fare range reliability")
        j1, j2, j3 = st.columns(3)
        j1.metric("Range aims to cover", f"{iv['target_level']:.0%} of fares")
        j2.metric("Actually covered", f"{iv['test_coverage']:.1%}",
                  f"{(iv['test_coverage'] - iv['test_coverage_uncalibrated']) * 100:+.1f} points from calibration",
                  delta_color="off")
        j3.metric("Average range width", rupees(iv["test_avg_width"]))

    if meta.get("feature_importance"):
        st.markdown("### What the model relies on")
        imp = pd.DataFrame(meta["feature_importance"])
        st.altair_chart(style(alt.Chart(imp).mark_bar(height=16, color=TEAL, opacity=0.85).encode(
            x=alt.X("mean_abs_rupees:Q", title="Average effect on the fare (₹)"),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("label:N", title="Feature"),
                     alt.Tooltip("mean_abs_rupees:Q", format=",.0f", title="₹")]),
            height=40 + 24 * len(imp)), use_container_width=True)
        st.caption("Average size of each feature's effect across held-back flights. Save the chart from its menu "
                   "to use it in the report.")

    if meta.get("blend_weights"):
        st.markdown("### How the three models are combined")
        bw = pd.DataFrame({"model": list(meta["blend_weights"]), "weight": list(meta["blend_weights"].values())})
        st.altair_chart(style(alt.Chart(bw).mark_bar(height=16, color=INK, opacity=0.85).encode(
            x=alt.X("weight:Q", title="Weight in the blend"), y=alt.Y("model:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("weight:Q", format=".3f", title="Weight")]),
            height=130), use_container_width=True)

    with st.expander("Training run details"):
        st.json({k: meta.get(k) for k in ["model_version", "trained_at", "training_rows", "target_transform",
                                          "uses_flight_code", "days_left_range", "environment"]})

st.markdown(f"<hr style='border:none;border-top:1px solid {LINE};margin:2.5rem 0 1rem'/>"
            f"<p class='stamp'>AM5305 Machine Learning project. Midhunraj M N and Muthu Sanjay Muruganandam, "
            f"supervised by Mrs. G. S. Akila Gandhi. Estimates come from historical fare data and are not "
            f"live airline prices.</p>", unsafe_allow_html=True)
