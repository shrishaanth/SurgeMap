from __future__ import annotations

import streamlit as st

import data
from logic import capture_share, study_rows


def render() -> None:
    st.title("SurgeMap")
    st.subheader("Short-term taxi demand forecasting and fleet repositioning for New York City")
    st.write(
        "SurgeMap forecasts how many taxi pickups each of New York's taxi zones will see in the next 5 to 60 "
        "minutes, finds the busiest spots ahead of time, and tests whether using the forecast to move idle "
        "vehicles before demand appears gets riders served sooner. The forecaster is a spatio-temporal graph "
        "neural network (ST-GNN, a Graph WaveNet) trained on four months of NYC yellow-taxi trips.")

    metrics, d, result = data.forecast_metrics(), data.predictions(), data.evaluation()
    models, horizons = metrics["models"], metrics["horizons"]
    j = horizons.index(3) if 3 in horizons else 0
    best = data.FORECAST_MODEL if data.FORECAST_MODEL in models else "stgnn"
    c1, c2, c3 = st.columns(3)
    c1.metric("Zones forecast", f"{len(d['zone_ids'])}")
    c2.metric("Test forecasts", f"{metrics['n_anchors']:,}")
    if best in models and "persistence" in models:
        gain = 100 * (1 - models[best]["rmse"][j] / models["persistence"]["rmse"][j])
        c3.metric(f"Forecast error vs the naive baseline ({horizons[j] * 5} min)", f"-{gain:.0f}%")

    fleet, theta = result["config"]["sweep_fleet"], result["config"]["theta"]
    rows = study_rows(result["summary"], fleet, theta)
    base, model, top = rows["none"], rows["stgnn"], rows["oracle"]
    share = capture_share(base["mean_wait_penalized"], model["mean_wait_penalized"], top["mean_wait_penalized"])
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Requests left unserved ({fleet:,} vehicles)", f"{100 * model['unmet_rate']:.1f}%",
              f"{100 * (model['unmet_rate'] - base['unmet_rate']):+.1f} pts with repositioning", delta_color="inverse")
    r2.metric("Mean rider wait", f"{model['mean_wait_penalized']:.1f} min",
              f"{model['mean_wait_penalized'] - base['mean_wait_penalized']:+.1f} min with repositioning",
              delta_color="inverse")
    if share is not None:
        r3.metric("Share of the best possible gain", f"{100 * share:.0f}%")

    st.markdown("""
**Pages**
- **Demand forecast map**: predicted vs actual demand for every zone at any time in the test window, and how many
  of the real busiest spots were found.
- **Zone explorer**: actual demand against the forecast for one zone.
- **Forecast accuracy**: the only page that compares against other methods (gradient boosting, ridge regression,
  historical average and persistence).
- **Repositioning**: what the forecast is worth to a fleet, and an interactive simulator.
""")

    st.subheader("How to read the results")
    st.markdown("""
- The test window is the **last 15% of the month**; the model never saw it during fitting. Data comes from the
  official NYC TLC trip file.
- The model is trained on four months (October 2023 to January 2024). It is an average of three Graph WaveNets:
  temporal convolutions read each zone's history while graph convolutions let zones share what they find.
- Repositioning results come from a **simulation**: the trips are real but the fleet is synthetic, because
  public trip data has no vehicle positions. Riders are matched to the nearest idle vehicle within 15 minutes,
  and requests with none are lost. "Perfect knowledge" of future demand is an upper limit, not a real system.
  Compare scenarios with each other rather than reading absolute numbers as real-world performance.
- Predictions in this app are precomputed; nothing here retrains a model.
""")
    st.caption("Data: NYC Taxi & Limousine Commission trip records; zone shapes from NYC Open Data.")
