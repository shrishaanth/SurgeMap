"""How much forecasting accuracy is left to gain, and which no-retraining methods capture some of it.

Everything is computed from the shipped test-split forecasts. Online methods are causal: at each
forecast time they use only outcomes that would already have been observed.
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

from accuracy_checks import block_bootstrap_gap, rmse_per_horizon


def online_bias_correction(pred, actual, horizons, half_life: float = 288.0, prior: float = 20.0):
    """Per-zone multiplicative correction from exponentially weighted past actual/predicted totals.

    A forecast issued at step t for horizon h is scored once bin t+h-1 has ended, so at step t
    only forecasts issued at or before t-h contribute. `prior` pickups of pseudo-data at ratio 1
    keep quiet zones stable.
    """
    n, z, hn = pred.shape
    decay = 0.5 ** (1.0 / half_life)
    out = np.empty_like(pred)
    for j, h in enumerate(horizons):
        sum_a, sum_p = np.zeros(z), np.zeros(z)
        for t in range(n):
            s = t - h
            if s >= 0:
                sum_a = decay * sum_a + actual[s, :, j]
                sum_p = decay * sum_p + pred[s, :, j]
            out[t, :, j] = pred[t, :, j] * (sum_a + prior) / (sum_p + prior)
    return out


def online_residual_correction(pred, actual, horizons, half_life: float = 288.0):
    """Add phi times the most recent observed residual of the same zone and horizon.

    phi is one number per horizon, fitted online by exponentially weighted least squares.
    """
    n, z, hn = pred.shape
    decay = 0.5 ** (1.0 / half_life)
    out = pred.copy()
    phis = []
    for j, h in enumerate(horizons):
        resid = actual[:, :, j] - pred[:, :, j]
        sxy = sxx = 0.0
        phi_track = []
        for t in range(n):
            last, prev = t - h, t - 2 * h
            if prev >= 0:                                   # pair (resid[prev] -> resid[last]) is now observed
                sxy = decay * sxy + float((resid[prev] * resid[last]).sum())
                sxx = decay * sxx + float((resid[prev] ** 2).sum())
            phi = sxy / sxx if sxx > 0 else 0.0
            phi_track.append(phi)
            if last >= 0:
                out[t, :, j] = np.clip(pred[t, :, j] + phi * resid[last], 0.0, None)
        phis.append(float(np.mean(phi_track[n // 4:])))
    return out, phis


def hedge(experts: dict, actual, horizons, half_life: float = 288.0):
    """Hedge (exponential weights) over forecasters, one weight vector per horizon.

    Losses are each expert's summed squared error on the newly observed forecasts, discounted
    with the given half-life and scaled by the best expert's loss so the learning rate is unitless.
    """
    names = list(experts)
    stack = np.stack([experts[k] for k in names])            # [K, N, Z, H]
    k, n, z, hn = stack.shape
    decay = 0.5 ** (1.0 / half_life)
    out = np.empty((n, z, hn))
    weights_mean = {}
    for j, h in enumerate(horizons):
        cum = np.zeros(k)
        track = []
        for t in range(n):
            s = t - h
            if s >= 0:
                cum = decay * cum + ((stack[:, s, :, j] - actual[s, :, j]) ** 2).sum(axis=1)
            scale = max(cum.min(), 1e-9)
            eta = 20.0                                       # weights shift by e-fold per 5% excess loss
            w = np.exp(-eta * (cum - cum.min()) / scale)
            w /= w.sum()
            track.append(w)
            out[t, :, j] = np.tensordot(w, stack[:, t, :, j], axes=1)
        weights_mean[f"{h * 5}min"] = dict(zip(names, np.mean(track[n // 4:], axis=0).round(3).tolist()))
    return out, weights_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--out", default="results/headroom_analysis.json")
    args = parser.parse_args()
    p = np.load(args.predictions)
    horizons = [int(h) for h in p["horizons"]]
    actual = p["actual"].astype(np.float64)
    m = {k: p[k].astype(np.float64) for k in ("stgnn", "stgnn_cal", "gbm", "gbm_cal", "ridge_hist", "histavg", "persistence")}
    labels = [f"{h * 5}m" for h in horizons]
    out = {"horizons": horizons}

    def show(name, pred, ref="stgnn_cal"):
        r = rmse_per_horizon(pred, actual)
        gap, lo, hi = block_bootstrap_gap(pred, m[ref], actual)
        out.setdefault("methods", {})[name] = {"rmse": r.tolist(), "gain_vs_stgnn_cal": gap.tolist(),
                                                "ci_low": lo.tolist(), "ci_high": hi.tolist()}
        print(f"   {name:44s} " + " ".join(f"{v:6.3f}" for v in r) + "   "
              + "  ".join(f"{g:+.3f}{'*' if l > 0 or u < 0 else ' '}" for g, l, u in zip(gap, lo, hi)))

    # 1. noise floor ---------------------------------------------------------------------
    best = m["stgnn_cal"]
    floor = float(np.sqrt(actual.mean()))
    out["poisson_floor"] = floor
    out["rmse_stgnn_cal"] = rmse_per_horizon(best, actual).tolist()
    print("1. NOISE FLOOR")
    print(f"   mean pickups per zone per bin: {actual.mean():.3f}; if arrivals were Poisson with a perfectly known rate,")
    print(f"   RMSE could not go below sqrt(mean) = {floor:.3f}")
    print("   calibrated ST-GNN RMSE:            " + " ".join(f"{v:6.3f}" for v in out["rmse_stgnn_cal"]))
    print("   excess over the Poisson floor:     " + " ".join(f"{100 * (v / floor - 1):5.1f}%" for v in out["rmse_stgnn_cal"]))
    pred1, act1 = best[:, :, 0].ravel(), actual[:, :, 0].ravel()
    edges = [0, 0.5, 2, 5, 10, 20, 1e9]
    out["dispersion_5min"] = []
    print("   5-minute forecasts by predicted rate:  share of cells   share of squared error   squared error / predicted rate")
    total = ((pred1 - act1) ** 2).sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pred1 >= lo) & (pred1 < hi)
        se = ((pred1[sel] - act1[sel]) ** 2)
        row = {"range": [lo, hi], "cells": float(sel.mean()), "sse_share": float(se.sum() / total),
               "dispersion": float(se.mean() / max(pred1[sel].mean(), 1e-9))}
        out["dispersion_5min"].append(row)
        print(f"     predicted {lo:>4g} to {hi if hi < 1e8 else 'inf':>4}                 {100 * row['cells']:5.1f}%            "
              f"{100 * row['sse_share']:5.1f}%                  {row['dispersion']:.2f}")
    print("   (a ratio of 1.00 is exactly Poisson noise; above 1 is extra variance, part of it reducible)")

    # 2. residual structure --------------------------------------------------------------
    busy = np.argsort(-actual[:, :, 0].mean(axis=0))[:30]
    out["residual_autocorr_top30"] = {}
    print("\n2. ARE THE ERRORS PREDICTABLE FROM RECENT ERRORS? (30 busiest zones, calibrated ST-GNN)")
    for j, h in enumerate(horizons):
        resid = actual[:, busy, j] - best[:, busy, j]
        corr = [float(np.mean([np.corrcoef(resid[lag:, c], resid[:-lag, c])[0, 1] for c in range(len(busy))]))
                for lag in (h, 2 * h)]
        out["residual_autocorr_top30"][labels[j]] = corr
        print(f"   {labels[j]:>4s} horizon: correlation with the latest observed error {corr[0]:+.3f}, one before {corr[1]:+.3f}")
    cross = np.corrcoef((actual[:, busy, 0] - best[:, busy, 0]).T)
    off = cross[~np.eye(len(busy), dtype=bool)]
    out["residual_cross_zone_corr_top30"] = float(off.mean())
    print(f"   mean correlation of 5-minute errors between different busy zones: {off.mean():+.3f}")

    # 3. no-retraining methods -----------------------------------------------------------
    print("\n3. METHODS THAT NEED NO RETRAINING   RMSE " + " ".join(f"{l:>6s}" for l in labels)
          + "   gain vs calibrated ST-GNN (* = interval excludes 0)")
    show("calibrated ST-GNN (shipped)", m["stgnn_cal"])
    show("calibrated gradient boosting", m["gbm_cal"])
    show("average of the two calibrated models", 0.5 * (m["stgnn_cal"] + m["gbm_cal"]))
    online_s = online_bias_correction(m["stgnn"], actual, horizons)
    online_g = online_bias_correction(m["gbm"], actual, horizons)
    show("ST-GNN + online bias correction", online_s)
    show("  ... on top of the calibrated ST-GNN", online_bias_correction(m["stgnn_cal"], actual, horizons))
    ar, phis = online_residual_correction(m["stgnn_cal"], actual, horizons)
    out["residual_phi"] = phis
    show("calibrated ST-GNN + last-error correction", ar)
    experts = {"stgnn_cal": m["stgnn_cal"], "gbm_cal": m["gbm_cal"], "ridge_hist": m["ridge_hist"],
               "histavg": m["histavg"], "persistence": m["persistence"]}
    hedged, weights = hedge(experts, actual, horizons)
    out["hedge_weights"] = weights
    show("Hedge over 5 forecasters", hedged)
    ens = 0.5 * (m["stgnn_cal"] + m["gbm_cal"])
    ens_ar, _ = online_residual_correction(online_bias_correction(ens, actual, horizons), actual, horizons)
    show("average + online bias + last-error correction", ens_ar)
    print("   last-error coefficients by horizon: " + " ".join(f"{v:.2f}" for v in phis))
    print("   Hedge mean weights: " + json.dumps(weights))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
