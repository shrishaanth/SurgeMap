from __future__ import annotations

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import data
from logic import moves_to_arcs

POLICIES = ("stgnn_cal", "gbm_cal", "stgnn", "gbm", "persistence", "oracle", "histavg", "ridge")
WARMUP = 36


class Recorder:
    """Wraps a policy and keeps every set of moves it returns."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def __call__(self, t, sim):
        moves = self.inner(t, sim)
        self.log.append((t, moves, sim.idle.copy()))
        return moves


def study_section() -> None:
    result = data.evaluation()
    summary = pd.DataFrame(result["summary"])
    fleet = result["config"]["sweep_fleet"]
    st.subheader(f"Study: rider wait vs vehicle driving, {fleet:,} vehicles")
    sweep = summary[(summary["fleet"] == fleet) & summary["policy"].isin(["persistence", "stgnn", "stgnn_cal", "gbm", "gbm_cal", "oracle"])].copy()
    sweep["Policy"] = sweep["policy"].map(data.LABELS)
    sweep["kmin"] = sweep["empty_minutes"] / 1e3
    base = summary[(summary["fleet"] == fleet) & (summary["policy"] == "none")]
    line = (alt.Chart(sweep).mark_line(point=True)
            .encode(x=alt.X("kmin:Q", title="Vehicle-minutes driving empty (thousands)", scale=alt.Scale(zero=False)),
                    y=alt.Y("mean_wait_penalized:Q", title="Mean rider wait (min, unmet counted as 15)",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("Policy:N", legend=alt.Legend(orient="top")),
                    order="theta:Q",
                    tooltip=["Policy", alt.Tooltip("theta:Q", title="theta"),
                             alt.Tooltip("unmet_rate:Q", format=".3f"),
                             alt.Tooltip("mean_wait_penalized:Q", format=".2f"),
                             alt.Tooltip("kmin:Q", format=".0f")]))
    layers = [line]
    if len(base):
        layers.append(alt.Chart(pd.DataFrame({"y": [float(base.iloc[0]["mean_wait_penalized"])]}))
                      .mark_rule(strokeDash=[4, 4], color="grey").encode(y="y:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=360))
    st.caption("Each point is one value of theta (the price of driving a vehicle one minute against a rider "
               "waiting one minute); the grey line is dispatch alone. A curve further down and to the left is "
               "better: the same wait for less empty driving. Mean of the seeds shown in the tooltip data.")

    gain = result.get("frontier_gain", {})
    lines = []
    for model in ("stgnn_cal", "gbm_cal", "gbm", "stgnn"):
        if model in gain:
            line = f"{data.LABELS[model]}: **{gain[model]['gain']:.2f} min** less wait than persistence"
            if "share_of_oracle_gain" in gain[model]:
                line += f" ({100 * gain[model]['share_of_oracle_gain']:.0f}% of the gain from perfect demand knowledge)"
            lines.append(line)
    if lines:
        bullets = "\n".join(f"- {line}" for line in lines)
        st.info(f"At equal driving cost, averaged over the sweep:\n\n{bullets}")

    table = summary[(summary["theta"].isin([0.0, result["config"]["theta"]]))].copy()
    table["Policy"] = table["policy"].map(data.LABELS)
    table = table.rename(columns={"fleet": "Fleet", "unmet_rate": "Unmet", "mean_wait_penalized": "Wait (min)",
                                  "empty_minutes": "Empty vehicle-min"})
    st.subheader("Results by fleet size")
    st.dataframe(table[["Fleet", "Policy", "Unmet", "Wait (min)", "Empty vehicle-min"]]
                 .sort_values(["Fleet", "Wait (min)"]).round({"Unmet": 3, "Wait (min)": 2, "Empty vehicle-min": 0}),
                 hide_index=True)


def run_demo(policy: str, fleet: int, theta: float, t0: int, hours: int, seed: int) -> dict:
    from surgemap.simulation.policy import ArrayForecast, LPRepositioner
    from surgemap.simulation.simulator import run_episode

    world, d = data.world(), data.predictions()
    horizons = tuple(int(h) for h in d["horizons"])
    first = int(d["anchors"][0])
    end = t0 + hours * 12
    common = dict(world=world, n_vehicles=fleet, seed=seed, start=t0 - WARMUP, end=end, warmup=WARMUP)
    base = run_episode(**common, policy=None)
    source = d["actual"] if policy == "oracle" else d[policy]
    recorder = Recorder(LPRepositioner(world, ArrayForecast(source.astype(float), first), horizons, theta=theta))
    result = run_episode(**common, policy=recorder)
    return {"base": base, "result": result, "log": recorder.log, "policy": policy, "t0": t0}


def demo_section() -> None:
    st.subheader("Try it: run the simulator")
    d = data.predictions()
    times, anchors = d["anchor_times"], d["anchors"]
    c1, c2, c3 = st.columns(3)
    policy = c1.selectbox("Forecast driving the policy", POLICIES, format_func=data.LABELS.get)
    fleet = c2.select_slider("Fleet size", options=list(range(500, 4001, 250)), value=2000)
    theta = c3.select_slider("theta (cost of driving)", options=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1)
    c4, c5, c6 = st.columns(3)
    hours = c4.select_slider("Simulated duration (hours)", options=[1, 2, 3, 4, 6], value=3)
    latest = times[-1] - pd.Timedelta(hours=hours)
    start = c5.slider("Start", min_value=times[0].to_pydatetime(), max_value=latest.to_pydatetime(),
                      value=times[len(times) // 4].to_pydatetime(),
                      step=pd.Timedelta(minutes=5).to_pytimedelta(), format="ddd DD MMM, HH:mm")
    seed = c6.number_input("Random seed", 0, 99, 0)
    t0 = int(anchors[0]) + int(round((pd.Timestamp(start) - times[0]) / pd.Timedelta(minutes=5)))

    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating..."):
            st.session_state["demo"] = run_demo(policy, int(fleet), float(theta), t0, int(hours), int(seed))
    demo = st.session_state.get("demo")
    if not demo:
        st.caption("The simulator replays real pickups from the test window against a fleet of idle vehicles. "
                   "Dispatch alone sends the nearest idle vehicle within 15 minutes; the policy also moves idle "
                   "vehicles toward forecast demand every 15 minutes.")
        return

    base, res = demo["base"], demo["result"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unmet requests", f"{100 * res['unmet_rate']:.1f}%",
              f"{100 * (res['unmet_rate'] - base['unmet_rate']):+.1f} pts vs dispatch only", delta_color="inverse")
    m2.metric("Mean wait, unmet as 15 min", f"{res['mean_wait_penalized']:.2f} min",
              f"{res['mean_wait_penalized'] - base['mean_wait_penalized']:+.2f} min", delta_color="inverse")
    m3.metric("Empty vehicle-minutes", f"{res['empty_minutes']:,.0f}",
              f"{res['empty_minutes'] - base['empty_minutes']:+,.0f}", delta_color="inverse")
    m4.metric("Vehicles repositioned", f"{res['moved']:,}")
    st.caption(f"{res['demand']:,} requests simulated. Deltas compare with dispatch alone on the same seed.")

    log = [(t, m, idle) for t, m, idle in demo["log"] if m is not None]
    if not log:
        st.info("The policy did not reposition any vehicles in this run.")
        return
    step = st.slider("Show moves at decision", 1, len(log), 1)
    t, moves, idle = log[step - 1]
    d = data.predictions()
    zone_ids, names, centroids = d["zone_ids"], data.zone_names(), data.zone_centroids()
    arcs = moves_to_arcs(moves, zone_ids, centroids, names)
    clock = d["anchor_times"][0] + pd.Timedelta(minutes=5 * (t - int(d["anchors"][0])))
    st.write(f"**{clock:%a %d %b %H:%M}**: {int(moves.sum())} vehicles sent, {len(arcs)} routes")
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": f["geometry"]} for f in data.zone_geojson()["features"]]}
    deck = pdk.Deck(
        map_style="light", initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.96, zoom=9.6),
        layers=[pdk.Layer("GeoJsonLayer", geo, filled=True, stroked=True, get_fill_color=[235, 235, 235, 120],
                          get_line_color=[170, 170, 170], line_width_min_pixels=0.5),
                pdk.Layer("ArcLayer", arcs, get_source_position=["from_lon", "from_lat"],
                          get_target_position=["to_lon", "to_lat"], get_width="vehicles",
                          width_scale=1.2, width_min_pixels=1.5, get_source_color=[30, 120, 200, 200],
                          get_target_color=[220, 50, 40, 220], pickable=True)],
        tooltip={"html": "{from_zone} to {to_zone}<br/><b>{vehicles}</b> vehicles"})
    st.pydeck_chart(deck, height=520)
    st.caption("Blue end = where vehicles leave, red end = where they are sent.")
    if len(arcs):
        st.dataframe(arcs.sort_values("vehicles", ascending=False)[["from_zone", "to_zone", "vehicles"]].head(12),
                     hide_index=True)


def render() -> None:
    st.title("Repositioning")
    st.write("A forecast is only useful if it changes a decision. Every 15 minutes a small linear program "
             "decides which idle vehicles to send where, using the forecast demand, and the simulator measures "
             "what that does to rider waiting and to empty driving.")
    study_section()
    st.divider()
    demo_section()
