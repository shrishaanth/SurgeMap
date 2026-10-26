from __future__ import annotations

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import data
from logic import capture_share, moves_to_arcs, study_rows

WARMUP = 36
COLORS = {"none": "#8A94A6", "stgnn_cal": "#14213D", "oracle": "#F5B700"}


class Recorder:
    """Wraps a policy and keeps every set of moves it returns."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def __call__(self, t, sim):
        moves = self.inner(t, sim)
        self.log.append((t, moves, sim.idle.copy()))
        return moves


def _bar(frame: pd.DataFrame, value: str, title: str, fmt: str) -> alt.Chart:
    labels = [data.POLICY_LABELS[p] for p in ("none", "stgnn_cal", "oracle")]
    return (alt.Chart(frame).mark_bar()
            .encode(x=alt.X("Fleet:O", title="Fleet size (vehicles)", axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("Policy:N", sort=labels),
                    y=alt.Y(f"{value}:Q", title=title),
                    color=alt.Color("Policy:N", sort=labels, legend=alt.Legend(orient="top", title=None),
                                    scale=alt.Scale(domain=labels, range=[COLORS[p] for p in ("none", "stgnn_cal", "oracle")])),
                    tooltip=["Fleet", "Policy", alt.Tooltip(f"{value}:Q", format=fmt)])
            .properties(height=300))


def study_section() -> None:
    result = data.evaluation()
    summary, theta = result["summary"], result["config"]["theta"]
    fleets = sorted({r["fleet"] for r in summary})
    st.subheader("What is repositioning worth?")
    fleet = st.select_slider("Fleet size", options=fleets, value=result["config"]["sweep_fleet"],
                             format_func=lambda f: f"{f:,} vehicles")
    rows = study_rows(summary, fleet, theta)
    base, model, best = rows["none"], rows["stgnn_cal"], rows["oracle"]
    share = capture_share(base["mean_wait_penalized"], model["mean_wait_penalized"], best["mean_wait_penalized"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests left unserved", f"{100 * model['unmet_rate']:.1f}%",
              f"{100 * (model['unmet_rate'] - base['unmet_rate']):+.1f} pts vs no repositioning", delta_color="inverse")
    c2.metric("Mean rider wait", f"{model['mean_wait_penalized']:.2f} min",
              f"{model['mean_wait_penalized'] - base['mean_wait_penalized']:+.2f} min", delta_color="inverse")
    c3.metric("Share of the best possible gain", "n/a" if share is None else f"{100 * share:.0f}%",
              help="How much of the improvement that perfect knowledge of future demand would give is achieved "
                   "with the ST-GNN forecast.", delta=None)
    c4.metric("Empty driving", f"{100 * (model['empty_minutes'] / base['empty_minutes'] - 1):+.0f}%",
              "the cost of repositioning", delta_color="off")
    st.caption(f"ST-GNN forecast with {fleet:,} vehicles, averaged over {model['n_seeds']} simulation runs; rider wait "
               f"varies by about ±{model['mean_wait_penalized_std']:.2f} min between runs. Unserved requests count as "
               "a 15-minute wait.")

    frame = pd.DataFrame([{"Fleet": f"{f:,}", "Policy": data.POLICY_LABELS[p], "Wait (min)": r["mean_wait_penalized"],
                           "Unserved (%)": 100 * r["unmet_rate"]}
                          for f in fleets for p, r in study_rows(summary, f, theta).items()])
    left, right = st.columns(2)
    left.altair_chart(_bar(frame, "Wait (min)", "Mean rider wait (min)", ".2f"))
    right.altair_chart(_bar(frame, "Unserved (%)", "Requests left unserved (%)", ".1f"))
    st.caption("Lower is better. Repositioning helps most when the fleet is large enough to be moved around; with "
               "very few vehicles there is little to gain. Perfect knowledge of future demand is an upper limit, "
               "not something a real system has.")

    with st.expander("Cost versus benefit: the full trade-off curve"):
        sweep_fleet = result["config"]["sweep_fleet"]
        sweep = pd.DataFrame(summary)
        sweep = sweep[(sweep["fleet"] == sweep_fleet) & sweep["policy"].isin(["stgnn_cal", "oracle"])].copy()
        sweep["Policy"] = sweep["policy"].map(data.POLICY_LABELS)
        sweep["kmin"] = sweep["empty_minutes"] / 1e3
        line = (alt.Chart(sweep).mark_line(point=True)
                .encode(x=alt.X("kmin:Q", title="Vehicle-minutes driving empty (thousands)", scale=alt.Scale(zero=False)),
                        y=alt.Y("mean_wait_penalized:Q", title="Mean rider wait (min)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Policy:N", legend=alt.Legend(orient="top", title=None),
                                        scale=alt.Scale(domain=[data.POLICY_LABELS["stgnn_cal"], data.POLICY_LABELS["oracle"]],
                                                        range=[COLORS["stgnn_cal"], COLORS["oracle"]])),
                        order="theta:Q",
                        tooltip=["Policy", alt.Tooltip("theta:Q", title="theta"), alt.Tooltip("kmin:Q", format=".0f"),
                                 alt.Tooltip("mean_wait_penalized:Q", format=".2f")]))
        none = [r for r in summary if r["policy"] == "none" and r["fleet"] == sweep_fleet]
        layers = [line]
        if none:
            layers.append(alt.Chart(pd.DataFrame({"y": [float(none[0]["mean_wait_penalized"])]}))
                          .mark_rule(strokeDash=[4, 4], color=COLORS["none"]).encode(y="y:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=340))
        st.caption(f"{sweep_fleet:,} vehicles. Each point is one value of theta, the price of driving a vehicle one "
                   "minute against a rider waiting one minute; the grey line is no repositioning. Further down and "
                   "to the left is better.")

    with st.expander("Full results by fleet size"):
        table = pd.DataFrame([{"Fleet": f, "Policy": data.POLICY_LABELS[p], "Unserved": r["unmet_rate"],
                               "Wait (min)": r["mean_wait_penalized"], "Empty vehicle-min": r["empty_minutes"]}
                              for f in fleets for p, r in study_rows(summary, f, theta).items()])
        st.dataframe(table.round({"Unserved": 3, "Wait (min)": 2, "Empty vehicle-min": 0}), hide_index=True)


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
    policy = c1.selectbox("Forecast driving the policy", data.POLICIES, format_func=data.POLICY_LABELS.get)
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
                   "Without repositioning, the nearest idle vehicle within 15 minutes serves each request; with it, "
                   "idle vehicles are also moved toward forecast demand every 15 minutes.")
        return

    base, res = demo["base"], demo["result"]
    st.markdown(f"**Result for the {data.POLICY_LABELS[demo['policy']]}**, compared with no repositioning on the same "
                "requests:")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Requests left unserved", f"{100 * res['unmet_rate']:.1f}%",
              f"{100 * (res['unmet_rate'] - base['unmet_rate']):+.1f} pts", delta_color="inverse")
    m2.metric("Mean rider wait", f"{res['mean_wait_penalized']:.2f} min",
              f"{res['mean_wait_penalized'] - base['mean_wait_penalized']:+.2f} min", delta_color="inverse")
    m3.metric("Empty driving (vehicle-min)", f"{res['empty_minutes']:,.0f}",
              f"{res['empty_minutes'] - base['empty_minutes']:+,.0f}", delta_color="inverse")
    m4.metric("Vehicles repositioned", f"{res['moved']:,}")
    st.caption(f"{res['demand']:,} requests simulated. Unserved requests count as a 15-minute wait.")

    log = [(t, m, idle) for t, m, idle in demo["log"] if m is not None]
    if not log:
        st.info("The policy did not reposition any vehicles in this run.")
        return
    step = st.slider("Show moves at decision", 1, len(log), 1) if len(log) > 1 else 1
    t, moves, idle = log[step - 1]
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
    st.write("A forecast is only useful if it changes a decision. Every 15 minutes, a small optimisation decides "
             "which idle vehicles to send where, using the forecast demand. A simulator replays real pickups and "
             "measures what that does to rider waiting and to empty driving.")
    study_section()
    st.divider()
    demo_section()
