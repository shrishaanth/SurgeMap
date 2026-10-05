"""Score ST-GNN checkpoints on the test split in pickup counts and compare them with the shipped models.

With --calibrate each checkpoint also gets the validation-fitted bias correction and blend
with the time-of-day average that the shipped calibrated models have, so the comparison
with those is like for like.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from accuracy_checks import block_bootstrap_gap, histavg_for_bins, rmse_per_horizon
from diagnose_stgnn import correct
from export_predictions import gather_targets, to_counts, train_stats
from hotspot_eval import collect_predictions, load_checkpoint, split_bounds


def score_checkpoint(path, data_dir, demand, times, window, shipped, calibrate):
    """Test-split forecasts in pickup counts, as trained and (optionally) calibrated."""
    model, features, a_out, a_in, horizons, meta = load_checkpoint(path, data_dir)
    bounds = split_bounds(meta, features.shape[1])
    train_end = bounds[0][1]
    mu, sigma = train_stats(demand, train_end)
    pred_z, _, anchors = collect_predictions(model, features, a_out, a_in, bounds, window, horizons)
    if not np.array_equal(anchors, shipped["anchors"]):
        raise ValueError(f"{path}: test anchors differ from the shipped forecasts")
    raw = to_counts(pred_z, mu, sigma).astype(np.float64)
    if not calibrate:
        return raw, None
    val_z, _, val_anchors = collect_predictions(model, features, a_out, a_in, (bounds[0], bounds[1], bounds[1]),
                                                window, horizons)
    val_actual = gather_targets(demand, val_anchors, horizons)
    val_hist = np.stack([histavg_for_bins(demand, times, train_end, val_anchors + h - 1).T for h in horizons], axis=-1)
    _, calibrated, _ = correct(to_counts(val_z, mu, sigma).astype(np.float64), val_actual, val_hist, raw,
                               shipped["histavg"].astype(np.float64))
    return raw, calibrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--references", default="stgnn,stgnn_cal,gbm,gbm_cal")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--out", default=None, help="optional JSON file for the numbers")
    args = parser.parse_args()

    shipped = np.load(args.predictions)
    actual = shipped["actual"].astype(np.float64)
    demand = np.load(os.path.join(args.data_dir, "demand.npy")).astype(np.float64)
    times = np.load(os.path.join(args.data_dir, "times.npy"))
    references = [r for r in args.references.split(",") if r in shipped]
    rows, new = {}, {}
    for path in args.checkpoints:
        name = os.path.splitext(os.path.basename(path))[0]
        raw, calibrated = score_checkpoint(path, args.data_dir, demand, times, args.window, shipped, args.calibrate)
        rows[name] = new[name] = raw
        if calibrated is not None:
            rows[name + " (calibrated)"] = new[name + " (calibrated)"] = calibrated
    for ref in references:
        rows[ref] = shipped[ref].astype(np.float64)

    labels = [f"{int(h) * 5}m" for h in shipped["horizons"]]
    report = {"rmse": {}, "volume_ratio": {}, "gaps": {}}
    print("count RMSE (pickups per zone per bin)   " + "  ".join(f"{l:>6s}" for l in labels) + "   predicted/actual volume")
    for name, pred in rows.items():
        report["rmse"][name] = rmse_per_horizon(pred, actual).tolist()
        report["volume_ratio"][name] = float(pred.sum() / actual.sum())
        print(f"{name:38s}  " + "  ".join(f"{v:6.3f}" for v in report["rmse"][name])
              + f"   {report['volume_ratio'][name]:.3f}")
    print("\nRMSE(reference) - RMSE(checkpoint), 95% block bootstrap; positive = checkpoint better")
    for name, pred in new.items():
        for ref in references:
            gap, lo, hi = block_bootstrap_gap(pred, rows[ref], actual)
            report["gaps"][f"{name} vs {ref}"] = {"gap": gap.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist()}
            print(f"{name} vs {ref:10s} " + "  ".join(f"{g:+.3f} [{l:+.3f},{h:+.3f}]" for g, l, h in zip(gap, lo, hi)))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
