from __future__ import annotations

import streamlit as st

import data


def render() -> None:
    st.title("SurgeMap")
    st.subheader("Short-term taxi demand forecasting and fleet repositioning for New York City")
    st.write(
        "SurgeMap forecasts how many taxi pickups each of New York's taxi zones will see in the next 5 to 60 "
        "minutes, and tests whether using those forecasts to pre-position idle vehicles reduces how long riders "
        "wait. The forecaster is a directed-graph spatio-temporal neural network (ST-GNN) trained on four months of "
        "NYC yellow-taxi trips.")

    metrics, d = data.forecast_metrics(), data.predictions()
    models, horizons = metrics["models"], metrics["horizons"]
    j = horizons.index(3) if 3 in horizons else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Zones forecast", f"{len(d['zone_ids'])}")
    c2.metric("Test forecasts", f"{metrics['n_anchors']:,}")
    best = "stgnn_cal" if "stgnn_cal" in models else "stgnn"
    if best in models and "persistence" in models:
        gain = 100 * (1 - models[best]["rmse"][j] / models["persistence"]["rmse"][j])
        c3.metric(f"RMSE vs persistence ({horizons[j] * 5} min)", f"-{gain:.0f}%")

    st.markdown("""
**Pages**
- **Demand forecast map**: predicted vs actual demand for every zone at any time in the test window, with the
  predicted hotspots outlined.
- **Zone explorer**: actual demand against each model for one zone.
- **Forecast accuracy**: error by horizon for the ST-GNN and three baselines.
- **Repositioning**: the simulation study and an interactive simulator.
""")

    st.subheader("How to read the results")
    st.markdown("""
- The test window is the **last 15% of the month**; models never saw it during fitting. Data comes from the
  official NYC TLC trip file.
- Every fitted model here is trained on the same four months (October 2023 to January 2024). The ST-GNN, an
  average of three Graph WaveNets, is the most accurate at every horizon: 3 to 6% ahead of a gradient-boosted
  model with simple features (2.4 to 3.4% after calibration) and 4 to 7% ahead in surges. It is 23 to 35%
  better than persistence.
- That accuracy came from how the networks are trained (shuffled batches, a loss suited to count data, yesterday's
  and last week's demand as inputs, more history) and from the architecture: a Graph WaveNet, in which zones
  exchange information at every layer over a partly learned graph, is 1 to 2% better than the first design.
- For repositioning the choice of forecaster hardly matters once it is unbiased: the ST-GNN, gradient boosting and
  a plain four-month time-of-day average perform alike, within the seed-to-seed spread.
- Repositioning results come from a **simulation**: the trips are real but the fleet is synthetic, because
  public trip data has no vehicle positions. Riders are matched to the nearest idle vehicle within 15 minutes,
  and requests with none are lost. Compare policies with each other rather than reading absolute numbers as
  real-world performance.
- Predictions in this app are precomputed; nothing here retrains a model.
""")
    st.caption("Data: NYC Taxi & Limousine Commission trip records; zone shapes from NYC Open Data.")
