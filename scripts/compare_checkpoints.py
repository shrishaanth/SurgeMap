"""Score ST-GNN checkpoints on the test split in pickup counts and compare them with the shipped models.

With --calibrate each checkpoint also gets the validation-fitted bias correction and blend
with the time-of-day average that the shipped calibrated models have, so the comparison
with those is like for like. --ensemble NAME=a.pt,b.pt averages the forecasts of several
checkpoints (before calibration) and scores the average as one more model.
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


def predict_counts(path, data_dir, demand, times, window, shipped, with_validation):
    """Forecasts of one checkpoint in pickup counts: test split, and optionally validation
    together with the validation actuals and time-of-day averages needed for calibration."""
    model, features, a_out, a_in, horizons, meta = load_checkpoint(path, data_dir)
    bounds = split_bounds(meta, features.shape[1])
    train_end = bounds[0][1]
    mu, sigma = train_stats(demand, train_end)
    pred_z, _, anchors = collect_predictions(model, features, a_out, a_in, bounds, window, horizons)
    if not np.array_equal(anchors, shipped["anchors"]):
        raise ValueError(f"{path}: test anchors differ from the shipped forecasts")
    out = {"test": to_counts(pred_z, mu, sigma).astype(np.float64)}
    if with_validation:
        val_z, _, val_anchors = collect_predictions(model, features, a_out, a_in,
                                                    (bounds[0], bounds[1], bounds[1]), window, horizons)
        out["val"] = to_counts(val_z, mu, sigma).astype(np.float64)
        out["val_actual"] = gather_targets(demand, val_anchors, horizons)
        out["val_hist"] = np.stack([histavg_for_bins(demand, times, train_end, val_anchors + h - 1).T
                                    for h in horizons], axis=-1)
    return out


def calibrate(pred, test_hist):
    return correct(pred["val"], pred["val_actual"], pred["val_hist"], pred["test"], test_hist)[1]


def score_checkpoint(path, data_dir, demand, times, window, shipped, calibrate_too):
    """Test-split forecasts in pickup counts, as trained and (optionally) calibrated."""
    pred = predict_counts(path, data_dir, demand, times, window, shipped, calibrate_too)
    return pred["test"], (calibrate(pred, shipped["histavg"].astype(np.float64)) if calibrate_too else None)


def average(members):
    """Average the forecasts of several checkpoints; validation parts are averaged the same way."""
    out = {"test": np.mean([m["test"] for m in members], axis=0)}
    if "val" in members[0]:
        out["val"] = np.mean([m["val"] for m in members], axis=0)
        out["val_actual"], out["val_hist"] = members[0]["val_actual"], members[0]["val_hist"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*")
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--references", default="stgnn,stgnn_cal,gbm,gbm_cal")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--ensemble", action="append", default=[], metavar="NAME=a.pt,b.pt",
                        help="score the average of several checkpoints; may be repeated")
    parser.add_argument("--cache-dir", default=None,
                        help="keep each checkpoint's forecasts here so an interrupted or repeated run is fast")
    parser.add_argument("--save", default=None, help="optional .npz file for the scored forecasts")
    parser.add_argument("--out", default=None, help="optional JSON file for the numbers")
    args = parser.parse_args()

    shipped = np.load(args.predictions)
    actual = shipped["actual"].astype(np.float64)
    test_hist = shipped["histavg"].astype(np.float64)
    demand = np.load(os.path.join(args.data_dir, "demand.npy")).astype(np.float64)
    times = np.load(os.path.join(args.data_dir, "times.npy"))
    references = [r for r in args.references.split(",") if r in shipped]

    ensembles = {}
    for spec in args.ensemble:
        name, _, members = spec.partition("=")
        ensembles[name] = [m for m in members.split(",") if m]
    paths = list(dict.fromkeys(args.checkpoints + [m for members in ensembles.values() for m in members]))
    cache = {}
    for path in paths:
        stored = None
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            stored = os.path.join(args.cache_dir, os.path.splitext(os.path.basename(path))[0] + ".npz")
        if stored and os.path.exists(stored) and os.path.getmtime(stored) >= os.path.getmtime(path):
            with np.load(stored) as z:
                cache[path] = {k: z[k] for k in z.files}
        else:
            cache[path] = predict_counts(path, args.data_dir, demand, times, args.window, shipped, True)
            if stored:
                np.savez_compressed(stored, **cache[path])
        print(f"[compare] forecasts ready: {path}", flush=True)

    rows, new = {}, {}

    def add(name, pred):
        rows[name] = new[name] = pred["test"]
        if args.calibrate:
            rows[name + " (calibrated)"] = new[name + " (calibrated)"] = calibrate(pred, test_hist)

    for path in args.checkpoints:
        add(os.path.splitext(os.path.basename(path))[0], cache[path])
    for name, members in ensembles.items():
        add(name, average([cache[m] for m in members]))
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
    if args.save:
        np.savez_compressed(args.save, **{k.replace(" (calibrated)", "_cal"): v.astype(np.float32) for k, v in new.items()})
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
