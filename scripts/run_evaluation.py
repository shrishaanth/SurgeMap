from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from reposition import ArrayForecast, LPRepositioner
from simulator import load_world, run_episode

METRICS = ("unmet_rate", "mean_wait_served", "mean_wait_penalized", "empty_minutes",
           "reposition_minutes", "moved")
FORECAST_POLICIES = ("persistence", "histavg", "ridge", "stgnn", "oracle")

_STATE = {}


def _init_worker(data_dir: str, predictions_path: str, lookahead: int) -> None:
    world = load_world(data_dir)
    data = np.load(predictions_path)
    anchors = data["anchors"]
    forecasts = {name: ArrayForecast(data[name].astype(np.float64), anchors[0])
                 for name in FORECAST_POLICIES if name in data and name != "oracle"}
    forecasts["oracle"] = ArrayForecast(data["actual"].astype(np.float64), anchors[0])
    _STATE.update(world=world, forecasts=forecasts, horizons=tuple(int(h) for h in data["horizons"]),
                  start=int(anchors[0]), end=int(anchors[-1]) + 1, lookahead=lookahead)


def _run(task: dict) -> dict:
    world, s = _STATE["world"], _STATE
    policy = None
    if task["policy"] != "none":
        policy = LPRepositioner(world, s["forecasts"][task["policy"]], s["horizons"],
                                theta=task["theta"], lookahead=s["lookahead"])
    result = run_episode(world, task["fleet"], task["seed"], s["start"], s["end"],
                         policy=policy, decision_interval=s["lookahead"])
    assert result["fleet_conserved"]
    return {**task, **{m: float(result[m]) for m in METRICS}, "demand": int(result["demand"])}


def summarize(runs: list[dict]) -> list[dict]:
    """Mean and std over seeds per (policy, fleet, theta), plus paired differences to persistence."""
    groups = defaultdict(dict)
    for run in runs:
        groups[(run["policy"], run["fleet"], run["theta"])][run["seed"]] = run
    rows = []
    for (policy, fleet, theta), by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        row = {"policy": policy, "fleet": fleet, "theta": theta, "n_seeds": len(seeds)}
        for metric in METRICS:
            values = np.array([by_seed[s][metric] for s in seeds])
            row[metric] = float(values.mean())
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        reference = groups.get(("persistence", fleet, theta))
        if reference and policy not in ("persistence", "none"):
            shared = [s for s in seeds if s in reference]
            for metric in ("mean_wait_penalized", "unmet_rate", "empty_minutes"):
                diff = np.array([by_seed[s][metric] - reference[s][metric] for s in shared])
                row[f"{metric}_vs_persistence"] = float(diff.mean())
                row[f"{metric}_vs_persistence_std"] = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        rows.append(row)
    return rows


def frontier_gain(rows: list[dict], fleet: int, reference: str = "persistence",
                  metric: str = "mean_wait_penalized", cost: str = "empty_minutes") -> dict:
    """Rider-metric improvement over `reference` at equal fleet driving cost.

    Each policy's theta sweep gives a (cost, metric) curve. Policies are compared with the
    reference at the same cost, averaged over the overlap of their cost ranges, so a policy
    that simply drives less is not mistaken for a worse one. Positive means better.
    """
    curves = {}
    for row in rows:
        if row["fleet"] == fleet and row["policy"] != "none":
            curves.setdefault(row["policy"], []).append((row[cost], row[metric]))
    curves = {p: np.array(sorted(pts)) for p, pts in curves.items() if len(pts) >= 3}
    if reference not in curves:
        return {}
    ref = curves[reference]
    out = {}
    for policy, pts in curves.items():
        if policy == reference:
            continue
        lo, hi = max(ref[0, 0], pts[0, 0]), min(ref[-1, 0], pts[-1, 0])
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 50)
        gain = np.interp(grid, ref[:, 0], ref[:, 1]) - np.interp(grid, pts[:, 0], pts[:, 1])
        out[policy] = {"gain": float(gain.mean()), "cost_range": [float(lo), float(hi)]}
    if "oracle" in out and out["oracle"]["gain"] > 1e-9:
        for policy, entry in out.items():
            entry["share_of_oracle_gain"] = entry["gain"] / out["oracle"]["gain"]
    return out


def build_tasks(args) -> list[dict]:
    tasks = []
    for fleet in args.fleet_sizes:
        for seed in args.seeds:
            tasks.append({"policy": "none", "fleet": fleet, "seed": seed, "theta": 0.0})
            for policy in args.policies:
                tasks.append({"policy": policy, "fleet": fleet, "seed": seed, "theta": args.theta})
    for theta in args.thetas:
        if theta == args.theta and args.sweep_fleet in args.fleet_sizes:
            continue
        for seed in args.seeds:
            for policy in args.sweep_policies:
                tasks.append({"policy": policy, "fleet": args.sweep_fleet, "seed": seed, "theta": theta})
    seen, unique = set(), []
    for task in tasks:
        key = task_key(task)
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


def task_key(task: dict) -> tuple:
    return (task["policy"], task["fleet"], task["seed"], task["theta"])


def load_cache(path: str) -> dict:
    """Completed episodes from an earlier, possibly interrupted, run."""
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    run = json.loads(line)
                    done[task_key(run)] = run
    return done


def parse_list(text: str, cast):
    return [cast(part) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--fleet-sizes", default="1000,1500,2000,3000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--policies", default=",".join(FORECAST_POLICIES))
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--thetas", default="0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--sweep-fleet", type=int, default=2000)
    parser.add_argument("--sweep-policies", default="persistence,stgnn,oracle")
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="results/evaluation.json")
    args = parser.parse_args()
    args.fleet_sizes = parse_list(args.fleet_sizes, int)
    args.seeds = parse_list(args.seeds, int)
    args.policies = parse_list(args.policies, str)
    args.thetas = parse_list(args.thetas, float)
    args.sweep_policies = parse_list(args.sweep_policies, str)

    tasks = build_tasks(args)
    cache_path = args.out + ".runs.jsonl"
    done = load_cache(cache_path)
    todo = [t for t in tasks if task_key(t) not in done]
    print(f"[evaluate] {len(tasks)} episodes, {len(tasks) - len(todo)} cached, "
          f"{len(todo)} to run on {args.workers} workers")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if todo:
        pool = ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                   initargs=(args.data_dir, args.predictions, args.lookahead))
        with pool, open(cache_path, "a", encoding="utf-8") as cache:
            for i, run in enumerate(pool.map(_run, todo), 1):
                done[task_key(run)] = run
                cache.write(json.dumps(run) + "\n")
                cache.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"[evaluate] {i}/{len(todo)} done")
    runs = [done[task_key(t)] for t in tasks]

    rows = summarize(runs)
    result = {"config": {k: v for k, v in vars(args).items()}, "runs": runs, "summary": rows,
              "frontier_gain": frontier_gain(rows, args.sweep_fleet)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("policy,fleet,theta,unmet_rate,mean_wait_penalized,empty_minutes")
    for r in rows:
        if r["theta"] in (0.0, args.theta):
            print(f"{r['policy']},{r['fleet']},{r['theta']},{r['unmet_rate']:.4f},"
                  f"{r['mean_wait_penalized']:.3f},{r['empty_minutes']:.0f}")
    print(f"[evaluate] wrote {args.out}")


if __name__ == "__main__":
    main()
