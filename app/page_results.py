from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import data


def render() -> None:
    st.title("Forecast accuracy")
    metrics, d = data.forecast_metrics(), data.predictions()
    horizons = metrics["horizons"]
    times = d["anchor_times"]
    st.write(f"Held-out test window: **{times[0]:%a %d %b %H:%M} to {times[-1]:%a %d %b %H:%M}** "
             f"({metrics['n_anchors']:,} forecast times, {len(d['zone_ids'])} zones). Models were fitted on "
             "earlier days only; errors below are in pickups per zone per 5-minute bin.")

    rows = []
    for model, v in metrics["models"].items():
        for j, h in enumerate(horizons):
            rows.append({"Model": data.LABELS[model], "Horizon (min)": h * 5, "RMSE": v["rmse"][j],
                         "MAE": v["mae"][j], "Top-3 hit rate": v["top3_hit_rate"][j],
                         "Top-5 overlap": v["top5_overlap"][j]})
    frame = pd.DataFrame(rows)

    chart = (alt.Chart(frame).mark_bar()
             .encode(x=alt.X("Horizon (min):O", title="Forecast horizon (minutes)"),
                     xOffset="Model:N", y=alt.Y("RMSE:Q", title="RMSE (pickups per zone per bin)"),
                     color=alt.Color("Model:N", legend=alt.Legend(orient="top")),
                     tooltip=["Model", "Horizon (min)", alt.Tooltip("RMSE:Q", format=".3f")])
             .properties(height=340))
    st.altair_chart(chart)

    pivot = frame.pivot(index="Model", columns="Horizon (min)", values="RMSE")
    st.subheader("RMSE by horizon")
    st.dataframe(pivot.round(3))

    lead = "ST-GNN, calibrated" if "ST-GNN, calibrated" in pivot.index else "ST-GNN"
    if {lead, "Persistence", "Ridge regression"} <= set(pivot.index):
        gain = pd.DataFrame({
            "vs persistence": 100 * (1 - pivot.loc[lead] / pivot.loc["Persistence"]),
            "vs ridge regression": 100 * (1 - pivot.loc[lead] / pivot.loc["Ridge regression"]),
            "vs historical average": 100 * (1 - pivot.loc[lead] / pivot.loc["Historical average"]),
            **({"vs gradient boosting, calibrated": 100 * (1 - pivot.loc[lead] / pivot.loc["Gradient boosting, calibrated"])}
               if "Gradient boosting, calibrated" in pivot.index else {}),
            **({"ST-GNN as trained vs gradient boosting as trained":
                100 * (1 - pivot.loc["ST-GNN"] / pivot.loc["Gradient boosting"])}
               if "Gradient boosting" in pivot.index else {}),
        }).T
        st.subheader(f"{lead}: RMSE reduction (%)")
        st.dataframe(gain.round(1))
        st.caption("Positive means the ST-GNN is better. Calibrated models get the same two corrections fitted on the "
                   "validation days: a per-zone bias correction and a blend with the time-of-day average. As "
                   "trained, the ST-GNN trails gradient boosting (last row); once both are calibrated it leads.")

    st.subheader("Hotspot ranking")
    st.dataframe(frame.pivot(index="Model", columns="Horizon (min)", values="Top-3 hit rate").round(3))
    st.caption("Top-3 hit rate: share of forecast times where at least one predicted top-3 zone is a real top-3 "
               "zone. Top-5 overlap (mean number of the 5 busiest zones that were also predicted):")
    st.dataframe(frame.pivot(index="Model", columns="Horizon (min)", values="Top-5 overlap").round(2))
