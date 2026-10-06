"""Re-index a forecast file onto another dataset's time grid.

Forecasts exported from a dataset covering several months carry bin indices into that
dataset. The simulator and the app index into real_processed_265, so the `anchors` are
rewritten to that grid by matching timestamps. Zones and actual counts must agree.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def align(predictions: dict, times: np.ndarray, zone_ids: np.ndarray, demand: np.ndarray) -> dict:
    stamps = predictions["anchor_times"].astype("datetime64[s]")
    grid = times.astype("datetime64[s]")
    anchors = np.searchsorted(grid, stamps)
    if (anchors >= len(grid)).any() or not np.array_equal(grid[anchors], stamps):
        raise ValueError("forecast times are not all present in the target dataset")
    if not np.array_equal(np.asarray(predictions["zone_ids"]), np.asarray(zone_ids)):
        raise ValueError("zones differ between the forecasts and the target dataset")
    horizons = [int(h) for h in predictions["horizons"]]
    actual = np.stack([demand[:, anchors + h - 1].T for h in horizons], axis=-1)
    if not np.allclose(actual, predictions["actual"]):
        raise ValueError("actual pickups differ between the forecasts and the target dataset")
    out = dict(predictions)
    out["anchors"] = anchors.astype(np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="real_processed_265")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with np.load(args.predictions) as z:
        predictions = {k: z[k] for k in z.files}
    aligned = align(predictions,
                    np.load(os.path.join(args.data_dir, "times.npy")),
                    np.load(os.path.join(args.data_dir, "zone_ids.npy")),
                    np.load(os.path.join(args.data_dir, "demand.npy")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **aligned)
    print(f"[align] {len(aligned['anchors'])} forecast times now index {args.data_dir} "
          f"(first bin {aligned['anchors'][0]}); wrote {args.out}")


if __name__ == "__main__":
    main()
