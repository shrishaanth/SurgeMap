"""Score ST-GNN checkpoints on the test split in pickup counts and compare them with the shipped models."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from accuracy_checks import block_bootstrap_gap, rmse_per_horizon
from export_predictions import to_counts, train_stats
from hotspot_eval import collect_predictions, load_checkpoint, split_bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--references", default="stgnn,stgnn_cal,gbm,gbm_cal")
    args = parser.parse_args()

    shipped = np.load(args.predictions)
    actual = shipped["actual"].astype(np.float64)
    demand = np.load(os.path.join(args.data_dir, "demand.npy")).astype(np.float64)
    rows, new = {}, {}
    for path in args.checkpoints:
        model, features, a_out, a_in, horizons, meta = load_checkpoint(path, args.data_dir)
        bounds = split_bounds(meta, features.shape[1])
        mu, sigma = train_stats(demand, bounds[0][1])
        pred_z, _, anchors = collect_predictions(model, features, a_out, a_in, bounds, args.window, horizons)
        if not np.array_equal(anchors, shipped["anchors"]):
            raise ValueError(f"{path}: test anchors differ from {args.predictions}")
        name = os.path.splitext(os.path.basename(path))[0]
        new[name] = to_counts(pred_z, mu, sigma).astype(np.float64)
        rows[name] = new[name]
    for ref in [r for r in args.references.split(",") if r in shipped]:
        rows[ref] = shipped[ref].astype(np.float64)

    labels = [f"{int(h) * 5}m" for h in shipped["horizons"]]
    print("count RMSE (pickups per zone per bin)   " + "  ".join(f"{l:>6s}" for l in labels) + "   predicted/actual volume")
    for name, pred in rows.items():
        ratio = pred.sum() / actual.sum()
        print(f"{name:38s}  " + "  ".join(f"{v:6.3f}" for v in rmse_per_horizon(pred, actual)) + f"   {ratio:.3f}")
    print("\nRMSE(reference) - RMSE(checkpoint), 95% block bootstrap; positive = checkpoint better")
    for name, pred in new.items():
        for ref in [r for r in args.references.split(",") if r in shipped]:
            gap, lo, hi = block_bootstrap_gap(pred, rows[ref], actual)
            print(f"{name} vs {ref:10s} " + "  ".join(f"{g:+.3f} [{l:+.3f},{h:+.3f}]" for g, l, h in zip(gap, lo, hi)))


if __name__ == "__main__":
    main()
