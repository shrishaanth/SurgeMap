"""Add validation-calibrated versions of the ST-GNN and the boosted model to the forecasts.

Both get the same two corrections, fitted on the validation split only: a per-zone,
per-horizon bias correction, then a per-horizon blend with the time-of-day average.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from surgemap import paths
from surgemap.evaluation.accuracy import merge_into_artifacts
from surgemap.evaluation.metrics import rmse_per_horizon
from surgemap.models.baselines import gbm_predictions, histavg_for_bins, load_weather
from surgemap.evaluation.export import gather_targets, to_counts, train_stats
from surgemap.data.windows import split_bounds
from surgemap.evaluation.inference import collect_predictions, load_checkpoint, resolve_checkpoints


def to_z(counts, mu, sigma):
    z = (np.log1p(np.clip(counts, 0, None)) - mu[None, :, None]) / sigma[None, :, None]
    return np.clip(z, -6.0, 6.0)


def correct(val_pred, val_actual, val_hist, test_pred, test_hist):
    """Two corrections fitted on validation only: a per-zone, per-horizon multiplicative bias
    correction on (1 + count), then a per-horizon linear blend with the time-of-day average."""
    smear = np.clip(np.mean(np.exp(np.log1p(val_actual) - np.log1p(val_pred)), axis=0), 0.5, 3.0)
    val_fixed = np.clip((1.0 + val_pred) * smear[None] - 1.0, 0.0, None)
    test_fixed = np.clip((1.0 + test_pred) * smear[None] - 1.0, 0.0, None)
    blended, weights = np.empty_like(test_fixed), []
    for j in range(test_pred.shape[-1]):
        design = np.column_stack([val_fixed[:, :, j].ravel(), val_hist[:, :, j].ravel()])
        coef, *_ = np.linalg.lstsq(design, val_actual[:, :, j].ravel(), rcond=None)
        weights.append(coef.tolist())
        blended[:, :, j] = np.clip(coef[0] * test_fixed[:, :, j] + coef[1] * test_hist[:, :, j], 0.0, None)
    return test_fixed, blended, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=paths.JANUARY)
    parser.add_argument("--checkpoint", default=paths.MODELS_DIR,
                        help="a checkpoint file, a comma-separated list, or a directory of .pt files to average")
    parser.add_argument("--predictions", default=paths.PREDICTIONS)
    parser.add_argument("--gbm-cache", default=paths.CACHE_DIR + "/gbm_val_test.npz")
    parser.add_argument("--window", type=int, default=48)
    args = parser.parse_args()

    demand = np.load(os.path.join(args.data_dir, "demand.npy")).astype(np.float64)
    times = np.load(os.path.join(args.data_dir, "times.npy"))
    p = np.load(args.predictions)
    actual, test_anchors = p["actual"].astype(np.float64), p["anchors"]
    test_hist = p["histavg"].astype(np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Several checkpoints are averaged in pickup counts, the same way export_predictions.py does.
    val_members = []
    for path in resolve_checkpoints(args.checkpoint):
        model, features, a_out, a_in, horizons, meta = load_checkpoint(path, args.data_dir, device=device)
        bounds = split_bounds(meta, features.shape[1])
        train_end = bounds[0][1]
        mu, sigma = train_stats(demand, train_end)
        print(f"[calibrate] {path} on the validation split")
        val_z, _, val_anchors = collect_predictions(model, features, a_out, a_in, (bounds[0], bounds[1], bounds[1]),
                                                    args.window, horizons, batch_size=64, device=device)
        val_members.append(to_counts(val_z, mu, sigma).astype(np.float64))
    val_stgnn = np.mean(val_members, axis=0)
    val_actual = gather_targets(demand, val_anchors, horizons)
    val_hist = np.stack([histavg_for_bins(demand, times, train_end, val_anchors + h - 1).T for h in horizons], axis=-1)

    if os.path.exists(args.gbm_cache):
        gbm_both = np.load(args.gbm_cache)["pred"].astype(np.float64)
    else:
        print("[calibrate] fitting the boosted model for validation and test")
        weather = load_weather(args.data_dir, demand.shape[1])
        train_anchors = np.arange(args.window, train_end - max(horizons) + 1)
        gbm_both = gbm_predictions(demand, times, weather, train_end, train_anchors,
                                   np.concatenate([val_anchors, test_anchors]), horizons).astype(np.float64)
        os.makedirs(os.path.dirname(os.path.abspath(args.gbm_cache)), exist_ok=True)
        np.savez_compressed(args.gbm_cache, pred=gbm_both)

    sources = {"stgnn_cal": (val_stgnn, p["stgnn"].astype(np.float64)),
               "gbm_cal": (gbm_both[:len(val_anchors)], gbm_both[len(val_anchors):])}
    calibrated = {}
    for name, (val_pred, test_pred) in sources.items():
        _, calibrated[name], weights = correct(val_pred, val_actual, val_hist, test_pred, test_hist)
        print(f"[calibrate] {name:10s} RMSE " + " ".join(f"{v:.3f}" for v in rmse_per_horizon(calibrated[name], actual))
              + "  blend (model/average) " + " ".join(f"{a:.2f}/{b:.2f}" for a, b in weights))
    merge_into_artifacts(args.predictions, list(calibrated), calibrated, actual, horizons)
    print(f"[calibrate] merged {', '.join(calibrated)} into {args.predictions}")


if __name__ == "__main__":
    main()
