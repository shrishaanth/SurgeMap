"""Add validation-calibrated versions of the ST-GNN and the boosted model to the forecasts.

Both get the same two corrections, fitted on the validation split only: a per-zone,
per-horizon bias correction, then a per-horizon blend with the time-of-day average.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch

from accuracy_checks import (gbm_predictions, histavg_for_bins, load_weather, merge_into_artifacts,
                             rmse_per_horizon)
from diagnose_stgnn import correct
from export_predictions import gather_targets, to_counts, train_stats
from hotspot_eval import collect_predictions, load_checkpoint, resolve_checkpoints, split_bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--checkpoint", default="model",
                        help="a checkpoint file, a comma-separated list, or a directory of .pt files to average")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--gbm-cache", default="results/gbm_val_test.npz")
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
