from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import data
from logic import anchor_index, diverging_colors, hotspot_table, sequential_colors

VIEWS = ("Predicted", "Actual", "Error (predicted minus actual)")


def render() -> None:
    st.title("Demand forecast map")
    st.caption(f"Forecast by the {data.LABELS[data.FORECAST_MODEL]} model. Baselines are compared on the Forecast accuracy page.")
    d = data.predictions()
    names, geo = data.zone_names(), data.zone_geojson()
    horizons = [int(h) for h in d["horizons"]]
    times = d["anchor_times"]

    model = data.FORECAST_MODEL
    c1, c2, c3 = st.columns(3)
    minutes = c1.selectbox("Forecast horizon", [h * 5 for h in horizons], format_func=lambda m: f"{m} min ahead")
    view = c2.radio("Show", VIEWS, horizontal=False)
    k = c3.selectbox("Top spots to find (top-k)", (3, 5, 10))

    when = st.slider("Forecast issued at", min_value=times[0].to_pydatetime(),
                     max_value=times[-1].to_pydatetime(), value=times[len(times) // 3].to_pydatetime(),
                     step=pd.Timedelta(minutes=5).to_pytimedelta(), format="ddd DD MMM, HH:mm")
    i = anchor_index(times[0], pd.Timestamp(when), len(times))
    h = horizons.index(minutes // 5)
    target_time = times[i] + pd.Timedelta(minutes=5 * (horizons[h] - 1))
    st.caption(f"Uses demand observed up to {times[i]:%a %d %b %H:%M}; predicts pickups in the 5-minute "
               f"bin starting {target_time:%a %d %b %H:%M}.")

    zone_ids = d["zone_ids"]
    pred, actual = d[model][i, :, h].astype(float), d["actual"][i, :, h].astype(float)
    shown = {VIEWS[0]: pred, VIEWS[1]: actual, VIEWS[2]: pred - actual}[view]
    if view == VIEWS[2]:
        vmax = max(float(np.abs(shown).max()), 1.0)
        colors = diverging_colors(shown, vmax)
    else:
        vmax = max(float(max(pred.max(), actual.max())), 1.0)
        colors = sequential_colors(shown, vmax)
    row_of = {int(z): n for n, z in enumerate(zone_ids)}
    top = {int(zone_ids[n]) for n in np.argsort(-pred, kind="stable")[:k]}

    features, outlines = [], []
    for feature in geo["features"]:
        zone_id = feature["properties"]["location_id"]
        n = row_of.get(zone_id)
        props = dict(feature["properties"])
        if n is None:
            props.update(fill=[200, 200, 200, 90], pred="-", actual="-")
        else:
            props.update(fill=colors[n].tolist(), pred=f"{pred[n]:.1f}", actual=int(actual[n]))
        item = {"type": "Feature", "properties": props, "geometry": feature["geometry"]}
        features.append(item)
        if zone_id in top:
            outlines.append(item)

    layers = [
        pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": features},
                  get_fill_color="properties.fill", get_line_color=[90, 90, 90],
                  line_width_min_pixels=0.5, pickable=True, stroked=True, filled=True),
        pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": outlines},
                  filled=False, stroked=True, get_line_color=[0, 0, 0], line_width_min_pixels=3),
    ]
    deck = pdk.Deck(layers=layers, map_style="light",
                    initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.96, zoom=9.6),
                    tooltip={"html": "<b>{zone}</b> ({borough})<br/>Predicted: {pred}<br/>Actual: {actual}"})

    left, right = st.columns([5, 4])
    with left:
        st.pydeck_chart(deck, height=560)
        scale = (f"Red = predicted above actual, blue = below (up to ±{vmax:.0f} requests)" if view == VIEWS[2]
                 else f"Colour: 0 to {vmax:.0f} requests per 5-minute bin (square-root scale). "
                      "Grey zones have no data. Black outline = predicted top hotspots.")
        st.caption(scale)
    with right:
        table, hits = hotspot_table(pred, actual, zone_ids, names, k)
        st.subheader(f"Predicted top {k} spots")
        st.metric("Real top spots found at this time", f"{hits} of {k}")
        compact = table.assign(Found=table["Hit"].map({True: "yes", False: "no"}))[["Zone", "Predicted", "Actual", "Actual rank", "Found"]].rename(columns={"Actual rank": "Rank"})
        st.dataframe(compact, hide_index=True)
        st.caption("Found: the zone really was among the k busiest zones in that 5-minute bin. Rank is its real rank.")
        overall = data.window_hotspot_rate(h, k)
        st.subheader("Over the whole test window")
        o1, o2 = st.columns(2)
        o1.metric(f"Real top-{k} found, on average", f"{overall['mean_hits']:.1f} of {k}")
        o2.metric("At least one found", f"{100 * overall['at_least_one']:.0f}%")
        st.caption(f"Across all {overall['n']:,} forecast times, at {minutes} minutes ahead.")
