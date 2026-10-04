from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import data


def render() -> None:
    st.title("Zone explorer")
    d = data.predictions()
    names = data.zone_names()
    zone_ids = d["zone_ids"]
    horizons = [int(h) for h in d["horizons"]]

    busiest = np.argsort(-d["actual"][:, :, 0].sum(axis=0))
    labels = {int(zone_ids[n]): f"{names.get(int(zone_ids[n]), ['Zone ' + str(zone_ids[n])])[0]} "
                                f"({names.get(int(zone_ids[n]), ['', ''])[1]})" for n in busiest}
    repeated = {label for label in labels.values() if list(labels.values()).count(label) > 1}
    labels = {zone: f"{label} #{zone}" if label in repeated else label for zone, label in labels.items()}
    c1, c2, c3 = st.columns([2, 1, 2])
    zone_id = c1.selectbox("Zone (busiest first)", list(labels), format_func=labels.get)
    minutes = c2.selectbox("Horizon", [h * 5 for h in horizons], format_func=lambda m: f"{m} min ahead")
    shown = c3.multiselect("Models", data.MODELS, default=["stgnn", "gbm"], format_func=data.LABELS.get)

    n = int(np.nonzero(zone_ids == zone_id)[0][0])
    h = horizons.index(minutes // 5)
    when = d["anchor_times"] + pd.Timedelta(minutes=5 * (horizons[h] - 1))
    frame = pd.DataFrame({"time": when, "Actual": d["actual"][:, n, h]})
    for model in shown:
        frame[data.LABELS[model]] = d[model][:, n, h]
    long = frame.melt("time", var_name="Series", value_name="Pickups per 5 min")

    domain = ["Actual"] + [data.LABELS[m] for m in shown]
    chart = (alt.Chart(long).mark_line()
             .encode(x=alt.X("time:T", title=None), y="Pickups per 5 min:Q",
                     color=alt.Color("Series:N", scale=alt.Scale(domain=domain), legend=alt.Legend(orient="top")),
                     size=alt.condition(alt.datum.Series == "Actual", alt.value(2.2), alt.value(1.2)),
                     tooltip=["time:T", "Series:N", alt.Tooltip("Pickups per 5 min:Q", format=".1f")])
             .properties(height=380).interactive(bind_y=False))
    st.altair_chart(chart)

    rows = []
    for model in data.MODELS:
        err = d[model][:, n, h] - d["actual"][:, n, h]
        rows.append({"Model": data.LABELS[model], "RMSE": float(np.sqrt(np.mean(err ** 2))),
                     "MAE": float(np.abs(err).mean())})
    st.subheader("Error in this zone")
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
    st.caption(f"{names.get(int(zone_id), ['Zone'])[0]}: mean {d['actual'][:, n, h].mean():.1f} pickups per 5-minute "
               "bin over the test window. Errors are in pickup counts, not standardised units.")
