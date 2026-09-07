from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Sequence

import numpy as np


DEFAULT_HORIZONS = (1, 3, 6, 12)


def parse_horizons(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        horizons = tuple(int(part) for part in parts)
    else:
        horizons = tuple(int(h) for h in value)
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must contain at least one positive integer")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    return horizons


def split_bounds(meta: dict, total: int) -> tuple[tuple[int, int], ...]:
    split = meta.get("split", {})
    train = split.get("train")
    validation = split.get("validation")
    test = split.get("test")
    train_end = int(split.get("train_end", train[1] if train else 0.70 * total))
    val_end = int(split.get("val_end", validation[1] if validation else 0.85 * total))
    train = train or [0, train_end]
    validation = validation or [train_end, val_end]
    test = test or [val_end, total]
    return tuple((int(bounds[0]), int(bounds[1])) for bounds in (train, validation, test))


def make_windows(features: np.ndarray, start: int, end: int, window: int,
                 horizons: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 3:
        raise ValueError("features must have shape [zones, time, features]")
    if window < 1:
        raise ValueError("window must be positive")
    horizons = parse_horizons(horizons)
    z, total, n_features = features.shape
    start, end = int(start), int(end)
    if not (0 <= start < end <= total):
        raise ValueError(f"invalid interval [{start}, {end}) for {total} bins")
    first = max(start, window)
    last = end - max(horizons)
    times = np.asarray(list(range(first, max(first, last + 1))), dtype=np.int64)
    times = times[times + max(horizons) - 1 < end]
    if len(times) == 0:
        raise ValueError("interval has no windows; reduce window or horizons")
    x = np.stack([features[:, t - window:t, :] for t in times], axis=0)
    y = np.stack([
        np.stack([features[:, t + h - 1, 0] for h in horizons], axis=-1)
        for t in times
    ], axis=0)
    assert x.shape == (len(times), z, window, n_features)
    assert y.shape == (len(times), z, len(horizons))
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), times


def evaluate_persistence(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim != 4 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
        raise ValueError("expected x [N,Z,W,F] and y [N,Z,H]")
    return np.repeat(x[:, :, -1:, 0], y.shape[-1], axis=-1)


def ridge_fit_multi(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    design = np.concatenate([X.astype(np.float64), np.ones((len(X), 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y.astype(np.float64))


def fit_per_zone_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    if x.ndim != 4 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
        raise ValueError("expected x [N,Z,W,F] and y [N,Z,H]")
    n, z, window, features = x.shape
    pred = np.empty_like(y, dtype=np.float32)
    for zone in range(z):
        coef = ridge_fit_multi(x[:, zone].reshape(n, window * features), y[:, zone], alpha)
        test_design = np.concatenate(
            [x[:, zone].reshape(n, window * features).astype(np.float64),
             np.ones((n, 1), dtype=np.float64)], axis=1,
        )
        pred[:, zone] = (test_design @ coef).astype(np.float32)
    return pred


def error_by_horizon(pred: np.ndarray, target: np.ndarray,
                     horizons: Sequence[int], zone_ids: Sequence[int] | None = None) -> dict:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("pred and target must both have shape [N,Z,H]")
    horizons = parse_horizons(horizons)
    if len(horizons) != pred.shape[-1]:
        raise ValueError("horizon count does not match prediction shape")
    if zone_ids is not None and len(zone_ids) != pred.shape[1]:
        raise ValueError("zone_ids count does not match zone dimension")
    result = {}
    for i, h in enumerate(horizons):
        diff = pred[..., i] - target[..., i]
        row = {
            "minutes": h * 5,
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff))),
        }
        if zone_ids is not None:
            row["per_zone"] = {
                str(int(zone_ids[z])): {
                    "rmse": float(np.sqrt(np.mean(diff[:, z] ** 2))),
                    "mae": float(np.mean(np.abs(diff[:, z]))),
                }
                for z in range(pred.shape[1])
            }
        result[str(h)] = row
    return result


def evaluate(features: np.ndarray, bounds: Sequence[tuple[int, int]], window: int,
             horizons: Sequence[int], ridge: float, zone_ids=None) -> dict:
    horizons = parse_horizons(horizons)
    train_x, train_y, train_times = make_windows(features, *bounds[0], window, horizons)
    test_x, test_y, test_times = make_windows(features, *bounds[2], window, horizons)
    persistence = evaluate_persistence(test_x, test_y)
    ridge_pred = fit_per_zone_ridge(train_x, train_y, ridge)
    n, z, w, f = train_x.shape
    ridge_test = np.empty_like(test_y, dtype=np.float32)
    for zone in range(z):
        coef = ridge_fit_multi(train_x[:, zone].reshape(n, w * f), train_y[:, zone], ridge)
        design = np.concatenate([test_x[:, zone].reshape(len(test_x), w * f).astype(np.float64),
                                 np.ones((len(test_x), 1))], axis=1)
        ridge_test[:, zone] = (design @ coef).astype(np.float32)
    return {
        "n_train": int(len(train_x)), "n_test": int(len(test_x)),
        "train_anchor_start": int(train_times[0]), "train_anchor_end": int(train_times[-1]),
        "test_anchor_start": int(test_times[0]), "test_anchor_end": int(test_times[-1]),
        "methods": {
            "persistence": error_by_horizon(persistence, test_y, horizons, zone_ids),
            "ridge": error_by_horizon(ridge_test, test_y, horizons, zone_ids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="real_processed_fixed")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    horizons = parse_horizons(args.horizons)
    features_path = os.path.join(args.data_dir, "features_clipped.npy")
    if not os.path.exists(features_path):
        features_path = os.path.join(args.data_dir, "features.npy")
    features = np.load(features_path)
    with open(os.path.join(args.data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    bounds = split_bounds(meta, features.shape[1])
    zone_path = os.path.join(args.data_dir, "zone_ids.npy")
    zone_ids = np.load(zone_path).tolist() if os.path.exists(zone_path) else meta.get("zone_ids")
    result = {"config": vars(args), "horizons": list(horizons),
              "split": {"train": list(bounds[0]), "validation": list(bounds[1]), "test": list(bounds[2])}}
    result.update(evaluate(features, bounds, args.window, horizons, args.ridge, zone_ids))
    print("method," + ",".join(f"{h * 5}min_rmse" for h in horizons))
    for method, values in result["methods"].items():
        print(method + "," + ",".join(f"{values[str(h)]['rmse']:.6f}" for h in horizons))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
