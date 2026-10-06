from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

BIN_MINUTES = 5
BINS_PER_DAY = 288
DAY, WEEK = BINS_PER_DAY, 7 * BINS_PER_DAY


def calendar_keys(times) -> tuple[np.ndarray, np.ndarray]:
    index = pd.DatetimeIndex(times)
    return index.dayofweek.to_numpy(), ((index.hour * 60 + index.minute) // BIN_MINUTES).to_numpy()


def histavg_for_bins(demand, times, train_end, bins, smooth: int = 3) -> np.ndarray:
    """Train-split mean demand for each bin's (zone, weekday, time of day), shaped [Z, len(bins)].

    A bin inside the train split would count itself, so its own contribution is removed.
    """
    dow, tod = calendar_keys(times)
    z = demand.shape[0]
    sums, counts = np.zeros((z, 7, BINS_PER_DAY)), np.zeros((7, BINS_PER_DAY))
    np.add.at(sums, (slice(None), dow[:train_end], tod[:train_end]), demand[:, :train_end])
    np.add.at(counts, (dow[:train_end], tod[:train_end]), 1.0)
    s_sum, s_cnt = np.zeros_like(sums), np.zeros_like(counts)
    for shift in range(-smooth, smooth + 1):
        s_sum += np.roll(sums, shift, axis=2)
        s_cnt += np.roll(counts, shift, axis=1)
    bins = np.asarray(bins)
    total = s_sum[:, dow[bins], tod[bins]]
    count = np.broadcast_to(s_cnt[dow[bins], tod[bins]], total.shape).copy()
    inside = bins < train_end
    total[:, inside] -= demand[:, bins[inside]]
    count[:, inside] -= 1.0
    return total / np.maximum(count, 1.0)


def load_weather(data_dir, n_bins: int) -> np.ndarray:
    """The [T, 2] weather channels of a dataset, or zeros when it has none (constant inputs
    carry no information, so the boosted model simply ignores them)."""
    path = os.path.join(data_dir, "weather.npy")
    if os.path.exists(path):
        return np.load(path)
    return np.zeros((n_bins, 2), dtype=np.float32)


def zscore_params(demand, train_end):
    log_counts = np.log1p(demand[:, :train_end])
    return log_counts.mean(axis=1), np.maximum(log_counts.std(axis=1), 1e-6)


def ridge_with_histavg(features, demand, times, train_end, train_anchors, test_anchors,
                       window, horizons, alpha: float = 1e-3) -> np.ndarray:
    """Per-zone ridge on the input window plus the time-of-day average of each target bin.

    Predictions are returned in demand counts, shaped [N, Z, H].
    """
    mu, sigma = zscore_params(demand, train_end)

    def hist_z(anchors):
        cols = []
        for h in horizons:
            hist = histavg_for_bins(demand, times, train_end, anchors + h - 1)
            cols.append((np.log1p(hist) - mu[:, None]) / sigma[:, None])
        return np.stack(cols, axis=-1)                        # [Z, N, H]

    hist_train, hist_test = hist_z(train_anchors), hist_z(test_anchors)
    n_zones = features.shape[0]
    out = np.empty((len(test_anchors), n_zones, len(horizons)), dtype=np.float32)
    for zone in range(n_zones):
        windows = sliding_window_view(features[zone], window, axis=0)
        x_tr = np.hstack([windows[train_anchors - window].reshape(len(train_anchors), -1), hist_train[zone]])
        x_te = np.hstack([windows[test_anchors - window].reshape(len(test_anchors), -1), hist_test[zone]])
        y_tr = np.stack([features[zone, train_anchors + h - 1, 0] for h in horizons], axis=1)
        design = np.hstack([x_tr, np.ones((len(x_tr), 1))]).astype(np.float64)
        penalty = alpha * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_tr.astype(np.float64))
        pred_z = np.hstack([x_te, np.ones((len(x_te), 1))]) @ coef
        out[:, zone] = np.clip(np.expm1(pred_z * sigma[zone] + mu[zone]), 0.0, None)
    return out


def gbm_features(demand, times, weather, train_end, anchors, h, zone_level, spatial=None) -> np.ndarray:
    """Feature matrix with one row per (anchor, zone), for target bin anchor + h - 1."""
    z, n = demand.shape[0], len(anchors)
    target = anchors + h - 1
    log_d = np.log1p(demand)
    cols = []

    def per_zone(values):                     # values [Z, N] -> flat in (anchor, zone) order
        return np.asarray(values, dtype=np.float32).T.reshape(-1)

    for lag in (1, 2, 3, 6):
        cols.append(per_zone(log_d[:, anchors - lag]))
    for span in (12, 48):
        cols.append(per_zone(np.stack([log_d[:, a - span:a].mean(axis=1) for a in anchors], axis=1)))
    for lag in (DAY, WEEK):
        idx = target - lag
        vals = np.where(idx[None, :] >= 0, log_d[:, np.maximum(idx, 0)], np.nan)
        cols.append(per_zone(vals))
    cols.append(per_zone(np.log1p(histavg_for_bins(demand, times, train_end, target))))
    dow, tod = calendar_keys(times)
    angle = 2 * np.pi * tod[target] / BINS_PER_DAY
    for series in (np.sin(angle), np.cos(angle), dow[target].astype(float), (dow[target] >= 5).astype(float)):
        cols.append(np.repeat(series.astype(np.float32), z))
    for channel in range(weather.shape[1]):
        cols.append(np.repeat(weather[anchors - 1, channel].astype(np.float32), z))
    cols.append(np.repeat(np.log1p(demand.sum(axis=0))[anchors - 1].astype(np.float32), z))
    cols.append(np.tile(zone_level.astype(np.float32), n))
    if spatial is not None:                   # flow-weighted demand of upstream and downstream zones
        for matrix in spatial:
            neighbour = matrix @ log_d
            for lag in (1, 2, 3):
                cols.append(per_zone(neighbour[:, anchors - lag]))
    return np.column_stack(cols)


def gbm_predictions(demand, times, weather, train_end, train_anchors, test_anchors, horizons,
                    stride: int = 3, max_iter: int = 200, spatial=None) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    z = demand.shape[0]
    zone_level = np.log1p(demand[:, :train_end]).mean(axis=1)
    fit_anchors = train_anchors[::stride]
    out = np.empty((len(test_anchors), z, len(horizons)), dtype=np.float32)
    for j, h in enumerate(horizons):
        x_tr = gbm_features(demand, times, weather, train_end, fit_anchors, h, zone_level, spatial)
        y_tr = demand[:, fit_anchors + h - 1].T.reshape(-1)
        # The L2 term matters for stability, not just accuracy. With a Poisson loss the score of
        # a zone that is almost always empty drifts very low, its curvature goes to zero, and a
        # single rare pickup then produces an unbounded leaf value. On four months of data that
        # made the unregularised model predict infinities for the quiet zones.
        model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.1, max_iter=max_iter,
                                              max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
                                              early_stopping=False, random_state=7)
        model.fit(x_tr, y_tr)
        x_te = gbm_features(demand, times, weather, train_end, test_anchors, h, zone_level, spatial)
        predicted = model.predict(x_te)
        if not np.isfinite(predicted).all() or predicted.max() > 100.0 * max(float(y_tr.max()), 1.0):
            raise FloatingPointError(f"boosted model produced unbounded predictions at horizon {h} "
                                     f"(max {np.nanmax(predicted):.3g}, largest training count {y_tr.max():.0f})")
        out[:, :, j] = predicted.reshape(len(test_anchors), z)
        print(f"[checks] gbm horizon {h * 5} min fitted on {len(y_tr):,} rows", flush=True)
    return out


def rmse_per_horizon(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2, axis=(0, 1)))


def block_bootstrap_gap(pred_a, pred_b, actual, block: int = 36, draws: int = 1000, seed: int = 0):
    """RMSE(b) - RMSE(a) per horizon with a block-bootstrap 95% interval; positive means a is better."""
    rng = np.random.default_rng(seed)
    sse_a = ((pred_a - actual) ** 2).sum(axis=1)               # [N, H]
    sse_b = ((pred_b - actual) ** 2).sum(axis=1)
    n_blocks = int(np.ceil(len(actual) / block))
    pad = n_blocks * block - len(actual)
    blocks_a = np.pad(sse_a, ((0, pad), (0, 0))).reshape(n_blocks, block, -1).sum(axis=1)
    blocks_b = np.pad(sse_b, ((0, pad), (0, 0))).reshape(n_blocks, block, -1).sum(axis=1)
    sizes = np.full(n_blocks, block, dtype=float)
    sizes[-1] = block - pad
    cells = actual.shape[1]
    gaps = []
    for _ in range(draws):
        pick = rng.integers(0, n_blocks, n_blocks)
        denom = sizes[pick].sum() * cells
        gaps.append(np.sqrt(blocks_b[pick].sum(axis=0) / denom) - np.sqrt(blocks_a[pick].sum(axis=0) / denom))
    gaps = np.array(gaps)
    point = np.sqrt(sse_b.sum(axis=0) / (len(actual) * cells)) - np.sqrt(sse_a.sum(axis=0) / (len(actual) * cells))
    return point, np.percentile(gaps, 2.5, axis=0), np.percentile(gaps, 97.5, axis=0)


def metrics_entry(pred, actual, horizons) -> dict:
    from hotspot_eval import ranking_metrics
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
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--out", default="results/accuracy_checks.json")
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
