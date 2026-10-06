from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hotspot_eval import (collect_predictions, load_checkpoint, ranking_metrics, resolve_checkpoints,
                          split_bounds)
from train_multihorizon_torch import MultiHorizonDataset

BIN_MINUTES = 5
BINS_PER_DAY = 24 * 60 // BIN_MINUTES
MODELS = ("stgnn", "ridge", "persistence", "histavg")


def train_stats(demand: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-zone mean and std of log1p(demand) on the train split, as fitted in preprocess.py."""
    log_counts = np.log1p(demand[:, :train_end])
    return log_counts.mean(axis=1), np.maximum(log_counts.std(axis=1), 1e-6)


def to_counts(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Invert the per-zone log z-score. z has shape [N, Z, H]; the result is demand counts."""
    log_counts = z * sigma[None, :, None] + mu[None, :, None]
    return np.clip(np.expm1(log_counts), 0.0, None).astype(np.float32)


def gather_targets(series: np.ndarray, anchors: np.ndarray, horizons) -> np.ndarray:
    """Values of a [Z, T] series at bin anchor+h-1 for each horizon, shaped [N, Z, H]."""
    return np.stack([series[:, anchors + h - 1].T for h in horizons], axis=-1)


def anchors_for(features: np.ndarray, bounds, window: int, horizons) -> np.ndarray:
    dataset = MultiHorizonDataset(features, *bounds, window, horizons)
    return np.asarray(dataset.times, dtype=np.int64)


def persistence_counts(demand: np.ndarray, anchors: np.ndarray, n_horizons: int) -> np.ndarray:
    last = demand[:, anchors - 1].T
    return np.repeat(last[..., None], n_horizons, axis=-1).astype(np.float32)


def historical_average(demand: np.ndarray, times: np.ndarray, train_end: int,
                       anchors: np.ndarray, horizons, smooth: int = 3) -> np.ndarray:
    """Mean train-split demand for the target bin's (zone, weekday, time-of-day)."""
    index = pd.DatetimeIndex(times)
    dow = index.dayofweek.to_numpy()
    tod = (index.hour * 60 + index.minute).to_numpy() // BIN_MINUTES
    n_zones = demand.shape[0]
    sums = np.zeros((n_zones, 7, BINS_PER_DAY))
    counts = np.zeros((7, BINS_PER_DAY))
    np.add.at(sums, (slice(None), dow[:train_end], tod[:train_end]), demand[:, :train_end])
    np.add.at(counts, (dow[:train_end], tod[:train_end]), 1.0)
    smooth_sums, smooth_counts = np.zeros_like(sums), np.zeros_like(counts)
    for shift in range(-smooth, smooth + 1):
        smooth_sums += np.roll(sums, shift, axis=2)
        smooth_counts += np.roll(counts, shift, axis=1)
    zone_tod_mean = sums.sum(axis=1) / np.maximum(counts.sum(axis=0), 1.0)
    table = np.where(smooth_counts[None] > 0,
                     smooth_sums / np.maximum(smooth_counts[None], 1.0),
                     zone_tod_mean[:, None, :])
    out = []
    for h in horizons:
        target = anchors + h - 1
        out.append(table[:, dow[target], tod[target]].T)
    return np.stack(out, axis=-1).astype(np.float32)


def ridge_predictions(features: np.ndarray, train_anchors: np.ndarray, test_anchors: np.ndarray,
                      window: int, horizons, alpha: float = 1e-3) -> np.ndarray:
    """Per-zone ridge on the flattened input window; predictions are in z-score space."""
    n_zones = features.shape[0]
    pred = np.empty((len(test_anchors), n_zones, len(horizons)), dtype=np.float32)
    for zone in range(n_zones):
        windows = sliding_window_view(features[zone], window, axis=0)
        x_train = windows[train_anchors - window].reshape(len(train_anchors), -1).astype(np.float64)
        x_test = windows[test_anchors - window].reshape(len(test_anchors), -1).astype(np.float64)
        y_train = np.stack([features[zone, train_anchors + h - 1, 0] for h in horizons],
                           axis=1).astype(np.float64)
        design = np.hstack([x_train, np.ones((len(x_train), 1))])
        penalty = alpha * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
        pred[:, zone] = np.hstack([x_test, np.ones((len(x_test), 1))]) @ coef
    return pred


def rmse_by_horizon(pred: np.ndarray, target: np.ndarray) -> list[float]:
    return np.sqrt(np.mean((pred - target) ** 2, axis=(0, 1))).tolist()


def mae_by_horizon(pred: np.ndarray, target: np.ndarray) -> list[float]:
    return np.mean(np.abs(pred - target), axis=(0, 1)).tolist()


def load_reported(horizons) -> dict[str, list[float]]:
    """RMSE values (z-score space) already reported by the training and baseline scripts."""
    root = os.path.dirname(SCRIPT_DIR)
    reported = {}
    paths = {
        "ridge": ("multihorizon_baselines_265_clipped.json", lambda d: d["methods"]["ridge"]),
        "persistence": ("multihorizon_baselines_265_clipped.json", lambda d: d["methods"]["persistence"]),
    }
    for name, (filename, pick) in paths.items():
        path = os.path.join(root, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                section = pick(json.load(f))
            reported[name] = [section[str(h)]["rmse"] for h in horizons]
    return reported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--checkpoint", default="model",
                        help="a checkpoint file, a comma-separated list, or a directory of .pt files to average")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demand = np.load(os.path.join(args.data_dir, "demand.npy"))
    times = np.load(os.path.join(args.data_dir, "times.npy"))
    zone_ids = np.load(os.path.join(args.data_dir, "zone_ids.npy"))

    # Several checkpoints are averaged in pickup counts, giving one ensemble forecast.
    member_z, member_counts = [], []
    for path in resolve_checkpoints(args.checkpoint):
        model, features, a_out, a_in, horizons, meta = load_checkpoint(path, args.data_dir, device=device)
        bounds = split_bounds(meta, features.shape[1])
        train_end = bounds[0][1]
        mu, sigma = train_stats(demand, train_end)
        print(f"[export] running {path} on the test split")
        pred_z, target_z, test_anchors = collect_predictions(
            model, features, a_out, a_in, bounds, args.window, horizons,
            batch_size=args.batch_size, device=device)
        member_z.append(pred_z)
        member_counts.append(to_counts(pred_z, mu, sigma))
    stgnn_z = np.mean(member_z, axis=0)
    print(f"[export] ST-GNN forecast is the average of {len(member_counts)} checkpoint(s)")
    train_anchors = anchors_for(features, bounds[0], args.window, horizons)

    print("[export] fitting per-zone ridge baseline")
    ridge_z = ridge_predictions(features, train_anchors, test_anchors, args.window,
                                horizons, args.ridge)
    persistence_z = np.repeat(features[:, test_anchors - 1, 0].T[..., None], len(horizons), axis=-1)

    z_space = {"stgnn": stgnn_z, "ridge": ridge_z, "persistence": persistence_z}
    reported = load_reported(horizons)
    print("[export] RMSE in z-score space vs previously reported values")
    for name, pred in z_space.items():
        ours = rmse_by_horizon(pred, target_z)
        line = " ".join(f"{v:.4f}" for v in ours)
        if name in reported:
            diff = max(abs(a - b) for a, b in zip(ours, reported[name]))
            print(f"  {name:12s} {line}  max|diff|={diff:.5f}")
        else:
            print(f"  {name:12s} {line}")

    actual = gather_targets(demand, test_anchors, horizons)
    predictions = {
        "stgnn": np.mean(member_counts, axis=0).astype(np.float32),
        "ridge": to_counts(ridge_z, mu, sigma),
        "persistence": persistence_counts(demand, test_anchors, len(horizons)),
        "histavg": historical_average(demand, times, train_end, test_anchors, horizons),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "predictions.npz"),
        anchors=test_anchors, anchor_times=times[test_anchors], zone_ids=zone_ids,
        horizons=np.asarray(horizons), actual=actual, **predictions)

    metrics = {"horizons": list(horizons), "space": "demand counts per zone per 5-min bin",
               "n_anchors": int(len(test_anchors)), "models": {}}
    for name, pred in predictions.items():
        ranking = ranking_metrics(pred, actual, horizons, (3, 5))
        metrics["models"][name] = {
            "rmse": rmse_by_horizon(pred, actual),
            "mae": mae_by_horizon(pred, actual),
            "top3_hit_rate": [ranking[str(h)]["topk"]["3"]["hit_rate"] for h in horizons],
            "top5_overlap": [ranking[str(h)]["topk"]["5"]["overlap"] for h in horizons],
        }
    with open(os.path.join(args.out_dir, "forecast_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[export] count-space RMSE")
    for name, values in metrics["models"].items():
        print(f"  {name:12s} " + " ".join(f"{v:.3f}" for v in values["rmse"]))
    print(f"[export] wrote {args.out_dir}/predictions.npz and forecast_metrics.json")


if __name__ == "__main__":
    main()
