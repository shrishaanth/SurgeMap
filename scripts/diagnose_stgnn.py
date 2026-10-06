"""Diagnostics for the trained ST-GNN: where its error comes from and what would fix it.

Everything here needs only inference with the existing checkpoint, no retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from accuracy_checks import block_bootstrap_gap, gbm_predictions, histavg_for_bins, load_weather
from export_predictions import gather_targets, to_counts, train_stats
from hotspot_eval import collect_predictions, load_checkpoint, split_bounds


def rmse(a, b, axis=None):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


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
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--checkpoint", default="experiments/r3_poisson_lag.pt")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--metrics", default="experiments/r3_poisson_lag_metrics.json")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--out", default="results/stgnn_diagnostics.json")
    args = parser.parse_args()
    out = {}

    model, features, a_out, a_in, horizons, meta = load_checkpoint(args.checkpoint, args.data_dir)
    bounds = split_bounds(meta, features.shape[1])
    demand = np.load(os.path.join(args.data_dir, "demand.npy")).astype(np.float64)
    times = np.load(os.path.join(args.data_dir, "times.npy"))
    train_end = bounds[0][1]
    mu, sigma = train_stats(demand, train_end)
    p = np.load(args.predictions)
    actual = p["actual"].astype(np.float64)
    test_anchors = p["anchors"]
    preds = {k: p[k].astype(np.float64) for k in ("stgnn", "gbm", "ridge_hist", "ridge", "persistence", "histavg")
             if k in p}
    H = len(horizons)
    labels = [f"{h * 5}m" for h in horizons]

    # 1. training curve ------------------------------------------------------------------
    with open(args.metrics, encoding="utf-8") as f:
        history = json.load(f)["history"]
    tr = [e["train_rmse"] for e in history]
    va = [e["val_rmse"] for e in history]
    out["training"] = {"epochs": len(history), "train_first": tr[0], "train_last": tr[-1],
                       "val_first": va[0], "val_best": min(va), "val_best_epoch": int(np.argmin(va)) + 1,
                       "train_minus_val_at_end": tr[-1] - va[-1],
                       "val_gain_last_10_epochs": va[-11] - min(va[-10:]) if len(va) > 11 else None}
    print("1. TRAINING CURVE (z-score RMSE)")
    print(f"   epoch 1: train {tr[0]:.4f} val {va[0]:.4f} | epoch {len(tr)}: train {tr[-1]:.4f} val {va[-1]:.4f}"
          f" | best val {min(va):.4f} at epoch {int(np.argmin(va)) + 1}")
    print(f"   train is {'BELOW' if tr[-1] < va[-1] else 'above'} val by {abs(tr[-1] - va[-1]):.4f}; "
          f"val improved {out['training']['val_gain_last_10_epochs']:.4f} over the last 10 epochs")

    # 2. the metric the model was trained on --------------------------------------------
    target_z = to_z(actual, mu, sigma)
    out["z_rmse"] = {k: rmse(to_z(v, mu, sigma), target_z, axis=(0, 1)).tolist() for k, v in preds.items()}
    out["count_rmse"] = {k: rmse(v, actual, axis=(0, 1)).tolist() for k, v in preds.items()}
    print("\n2. RMSE IN THE SPACE IT WAS TRAINED IN (per-zone standardised log) vs IN PICKUP COUNTS")
    print("   model         z-space: " + " ".join(f"{l:>6s}" for l in labels) + "   counts: " + " ".join(f"{l:>6s}" for l in labels))
    for k in preds:
        print(f"   {k:12s}          " + " ".join(f"{v:6.3f}" for v in out["z_rmse"][k]) + "           "
              + " ".join(f"{v:6.3f}" for v in out["count_rmse"][k]))

    # 3. bias ----------------------------------------------------------------------------
    out["volume_ratio"] = {k: (v.sum(axis=(0, 1)) / actual.sum(axis=(0, 1))).tolist() for k, v in preds.items()}
    print("\n3. BIAS: total predicted pickups / total actual pickups (1.00 = unbiased)")
    for k in preds:
        print(f"   {k:12s} " + " ".join(f"{v:6.3f}" for v in out["volume_ratio"][k]))

    # 4. where the error is --------------------------------------------------------------
    zone_mean = actual[:, :, 0].mean(axis=0)
    order = np.argsort(-zone_mean)
    groups = {"top 10 zones": order[:10], "zones 11-30": order[10:30], "zones 31-60": order[30:60],
              "other 198 zones": order[60:]}
    total_sse = ((preds["stgnn"] - actual) ** 2).sum()
    out["by_zone_group"] = {}
    print("\n4. WHERE THE ERROR IS (all horizons pooled)")
    print("   group             mean demand  share of ST-GNN error   RMSE: stgnn    gbm   | bias ratio: stgnn   gbm")
    for name, idx in groups.items():
        row = {"mean_demand": float(zone_mean[idx].mean()),
               "sse_share": float(((preds["stgnn"][:, idx] - actual[:, idx]) ** 2).sum() / total_sse),
               "rmse_stgnn": float(rmse(preds["stgnn"][:, idx], actual[:, idx])),
               "rmse_gbm": float(rmse(preds["gbm"][:, idx], actual[:, idx])),
               "ratio_stgnn": float(preds["stgnn"][:, idx].sum() / actual[:, idx].sum()),
               "ratio_gbm": float(preds["gbm"][:, idx].sum() / actual[:, idx].sum())}
        out["by_zone_group"][name] = row
        print(f"   {name:16s} {row['mean_demand']:10.2f} {100 * row['sse_share']:16.1f}%          "
              f"{row['rmse_stgnn']:6.3f} {row['rmse_gbm']:6.3f}   |        {row['ratio_stgnn']:6.3f} {row['ratio_gbm']:6.3f}")

    hours = pd.DatetimeIndex(p["anchor_times"]).hour.to_numpy()
    buckets = {"00-06": (0, 6), "06-10": (6, 10), "10-16": (10, 16), "16-20": (16, 20), "20-24": (20, 24)}
    out["by_hour_60min"] = {}
    print("   60-minute forecasts by time of day:   RMSE stgnn    gbm    histavg")
    for name, (lo, hi) in buckets.items():
        m = (hours >= lo) & (hours < hi)
        row = {k: float(rmse(preds[k][m][:, :, H - 1], actual[m][:, :, H - 1])) for k in ("stgnn", "gbm", "histavg")}
        out["by_hour_60min"][name] = row
        print(f"   {name}                                 {row['stgnn']:6.3f} {row['gbm']:6.3f} {row['histavg']:6.3f}")

    # 5. does it use the graph? ----------------------------------------------------------
    state = model.state_dict()
    norms = {k: float(state[f"spatial.{k}.weight"].norm()) for k in ("w_out", "w_in", "self_loop")}
    print("\n5. DOES IT USE THE GRAPH?")
    print(f"   weight norms: outgoing-neighbour {norms['w_out']:.2f}, incoming-neighbour {norms['w_in']:.2f}, "
          f"own-zone {norms['self_loop']:.2f}")
    zero = np.zeros_like(a_out)
    no_graph_z, target_feat, _ = collect_predictions(model, features, zero, zero, bounds, args.window, horizons)
    with_graph_z = to_z(preds["stgnn"], mu, sigma)
    no_graph = to_counts(no_graph_z, mu, sigma).astype(np.float64)
    out["graph"] = {"weight_norms": norms,
                    "count_rmse_graph_removed_at_inference": rmse(no_graph, actual, axis=(0, 1)).tolist(),
                    "count_rmse_with_graph": out["count_rmse"]["stgnn"]}
    print("   count RMSE with graph:                   " + " ".join(f"{v:6.3f}" for v in out["count_rmse"]["stgnn"]))
    print("   count RMSE, graph switched off (no retrain):" + " ".join(f"{v:6.3f}" for v in out["graph"]["count_rmse_graph_removed_at_inference"]))
    print("   (the network was trained with the graph, so switching it off can only show how much it leans on it)")

    # 6. fixes that need no retraining ---------------------------------------------------
    val_bounds = (bounds[0], bounds[1], bounds[1])
    val_z, _, val_anchors = collect_predictions(model, features, a_out, a_in, val_bounds, args.window, horizons)
    val_actual = gather_targets(demand, val_anchors, horizons)
    val_hist = np.stack([histavg_for_bins(demand, times, train_end, val_anchors + h - 1).T for h in horizons], axis=-1)
    val_stgnn = to_counts(val_z, mu, sigma).astype(np.float64)

    cache = args.out + ".gbm.npz"
    if os.path.exists(cache):
        gbm_both = np.load(cache)["pred"].astype(np.float64)
    else:
        weather = load_weather(args.data_dir, demand.shape[1])
        train_anchors = np.arange(args.window, train_end - max(horizons) + 1)
        gbm_both = gbm_predictions(demand, times, weather, train_end, train_anchors,
                                   np.concatenate([val_anchors, test_anchors]), horizons).astype(np.float64)
        np.savez_compressed(cache, pred=gbm_both)
    val_gbm, test_gbm = gbm_both[:len(val_anchors)], gbm_both[len(val_anchors):]

    fixes, corrected = {}, {}
    for name, val_pred, test_pred in (("stgnn", val_stgnn, preds["stgnn"]), ("gbm", val_gbm, test_gbm)):
        debiased, blended, weights = correct(val_pred, val_actual, val_hist, test_pred, preds["histavg"])
        corrected[name] = (debiased, blended)
        fixes[name] = {"plain": rmse(test_pred, actual, axis=(0, 1)).tolist(),
                       "bias_corrected": rmse(debiased, actual, axis=(0, 1)).tolist(),
                       "blended_with_histavg": rmse(blended, actual, axis=(0, 1)).tolist(),
                       "blend_weights_model_histavg": weights}
    out["no_retrain_fixes"] = fixes
    print("\n6. FIXES THAT NEED NO RETRAINING (fitted on the validation split, scored on test)")
    print("   the same two corrections are applied to both models")
    for name, label in (("stgnn", "ST-GNN"), ("gbm", "gradient boosting")):
        for key, text in (("plain", "as trained"), ("bias_corrected", "+ bias correction"),
                          ("blended_with_histavg", "+ blend with time-of-day average")):
            print(f"   {label:18s} {text:34s} " + " ".join(f"{v:6.3f}" for v in fixes[name][key]))
        print(f"   {'':18s} blend weights (model/average): "
              + "  ".join(f"{a:.2f}/{b:.2f}" for a, b in fixes[name]["blend_weights_model_histavg"]))

    out["corrected_gap"] = {}
    print("\n7. CORRECTED ST-GNN vs CORRECTED GRADIENT BOOSTING (RMSE gap, 95% block bootstrap; positive = ST-GNN better)")
    for i, text in ((0, "bias correction only"), (1, "bias correction + blend")):
        gap, lo, hi = block_bootstrap_gap(corrected["stgnn"][i], corrected["gbm"][i], actual)
        out["corrected_gap"][text] = {"gap": gap.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist()}
        print(f"   {text:26s} " + "  ".join(f"{g:+.3f} [{l:+.3f},{h:+.3f}]" for g, l, h in zip(gap, lo, hi)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
