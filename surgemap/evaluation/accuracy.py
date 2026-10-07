"""Fit the extra baselines, bootstrap the ST-GNN's lead over every model, and merge them into the forecasts."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from surgemap import paths
from surgemap.evaluation.metrics import block_bootstrap_gap, rmse_per_horizon
from surgemap.models.baselines import gbm_predictions, load_weather, ridge_with_histavg


def metrics_entry(pred, actual, horizons) -> dict:
    from surgemap.evaluation.inference import ranking_metrics
    ranking = ranking_metrics(pred, actual, horizons, (3, 5))
    return {"rmse": rmse_per_horizon(pred, actual).tolist(),
            "mae": np.mean(np.abs(pred - actual), axis=(0, 1)).tolist(),
            "top3_hit_rate": [ranking[str(h)]["topk"]["3"]["hit_rate"] for h in horizons],
            "top5_overlap": [ranking[str(h)]["topk"]["5"]["overlap"] for h in horizons]}


def merge_into_artifacts(predictions_path, names, preds, actual, horizons) -> None:
    """Add the extra models to predictions.npz and forecast_metrics.json next to it."""
    with np.load(predictions_path) as z:
        arrays = {k: z[k] for k in z.files}
    for name in names:
        arrays[name] = preds[name].astype(np.float32)
    np.savez_compressed(predictions_path, **arrays)
    metrics_path = os.path.join(os.path.dirname(predictions_path), "forecast_metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    for name in names:
        metrics["models"][name] = metrics_entry(preds[name], actual, horizons)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=paths.JANUARY)
    parser.add_argument("--predictions", default=paths.PREDICTIONS)
    parser.add_argument("--out", default=paths.RESULTS_DIR + "/accuracy_checks.json")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--merge", action="store_true", help="add the extra models to predictions.npz and forecast_metrics.json")
    parser.add_argument("--extras", default="gbm", help="comma-separated: gbm, gbm_spatial, ridge_hist")
    args = parser.parse_args()

    p = np.load(args.predictions)
    anchors, horizons = p["anchors"], tuple(int(h) for h in p["horizons"])
    actual = p["actual"].astype(np.float64)
    demand = np.load(os.path.join(args.data_dir, "demand.npy")).astype(np.float64)
    times = np.load(os.path.join(args.data_dir, "times.npy"))
    features = np.load(os.path.join(args.data_dir, "features_clipped.npy"))
    weather = load_weather(args.data_dir, demand.shape[1])
    with open(os.path.join(args.data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    train_end = meta["split"]["train"][1]
    train_anchors = np.arange(max(args.window, 0), train_end - max(horizons) + 1)

    preds = {name: p[name].astype(np.float64) for name in ("stgnn", "ridge", "persistence", "histavg")}
    cache_path = args.out + ".preds.npz"
    cached = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
    wanted = [e for e in args.extras.split(",") if e]
    for extra in wanted:
        if extra in cached:
            print(f"[checks] {extra}: cached", flush=True)
            continue
        print(f"[checks] computing {extra}", flush=True)
        if extra == "ridge_hist":
            cached[extra] = ridge_with_histavg(features, demand, times, train_end, train_anchors, anchors,
                                               args.window, horizons)
        elif extra == "gbm_spatial":
            graph = (np.load(os.path.join(args.data_dir, "A_in.npy")).astype(np.float64),
                     np.load(os.path.join(args.data_dir, "A_out.npy")).astype(np.float64))
            cached[extra] = gbm_predictions(demand, times, weather, train_end, train_anchors, anchors,
                                            horizons, spatial=graph)
        elif extra == "gbm":
            cached[extra] = gbm_predictions(demand, times, weather, train_end, train_anchors, anchors, horizons)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, **cached)
    preds.update({k: cached[k].astype(np.float64) for k in wanted if k in cached})

    if args.merge:
        merge_into_artifacts(args.predictions, [k for k in wanted if k in preds], preds, actual, horizons)
        print(f"[checks] merged {', '.join(k for k in wanted if k in preds)} into {args.predictions}")
    report = {"horizons": list(horizons), "rmse": {k: rmse_per_horizon(v, actual).tolist() for k, v in preds.items()},
              "stgnn_vs": {}}
    for name, pred in preds.items():
        if name == "stgnn":
            continue
        gap, lo, hi = block_bootstrap_gap(preds["stgnn"], pred, actual)
        report["stgnn_vs"][name] = {"rmse_gap": gap.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nRMSE (pickups per zone per bin)    " + "  ".join(f"{h * 5:>5d}m" for h in horizons))
    for name, values in report["rmse"].items():
        print(f"{name:14s}                     " + "  ".join(f"{v:6.3f}" for v in values))
    print("\nST-GNN advantage: RMSE(other) - RMSE(ST-GNN), 95% block-bootstrap interval; positive = ST-GNN better")
    for name, r in report["stgnn_vs"].items():
        cells = "  ".join(f"{g:+.3f} [{lo:+.3f},{hi:+.3f}]" for g, lo, hi in zip(r["rmse_gap"], r["ci_low"], r["ci_high"]))
        print(f"{name:12s} {cells}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
