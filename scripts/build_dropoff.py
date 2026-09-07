from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from preprocess import PICKUP, DROPOFF, PULOC, DOLOC, STEP, row_validity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--data-dir", default="real_processed_265")
    p.add_argument("--chunksize", type=int, default=400_000)
    return p.parse_args()


def build_dropoff(raw_csv: str, data_dir: str, chunksize: int = 400_000) -> np.ndarray:
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    zone_ids = np.load(os.path.join(data_dir, "zone_ids.npy")).tolist()
    zone_set = set(zone_ids)
    zone_index = {z: i for i, z in enumerate(zone_ids)}
    Z = len(zone_ids)
    start = pd.Timestamp(meta["start"])
    end = pd.Timestamp(meta["end"])
    T = meta["n_bins"]
    grid = pd.date_range(start=start, periods=T, freq=STEP)
    grid_index = {t: i for i, t in enumerate(grid)}

    dropoff = np.zeros((Z, T), dtype=np.float32)
    cols = [PICKUP, DROPOFF, PULOC, DOLOC]
    for chunk in pd.read_csv(raw_csv, usecols=cols, chunksize=chunksize, parse_dates=[PICKUP, DROPOFF]):
        pu, do, pu_id, do_id, time_ok, loc_ok, dur_ok, _ = row_validity(chunk, start, end)
        ok = time_ok & loc_ok & dur_ok
        if not ok.any():
            continue
        bins = do[ok].dt.floor(STEP)
        ids = do_id[ok].astype(np.int64).to_numpy()
        keep = np.array([z in zone_set for z in ids], dtype=bool)
        if not keep.any():
            continue
        rows = np.fromiter((zone_index[z] for z in ids[keep]), dtype=np.int64)
        times = np.fromiter((grid_index[t] for t in bins[keep]), dtype=np.int64)
        np.add.at(dropoff, (rows, times), 1)
    return dropoff


def main():
    a = parse_args()
    dropoff = build_dropoff(a.input, a.data_dir, a.chunksize)
    out_path = os.path.join(a.data_dir, "dropoff.npy")
    np.save(out_path, dropoff)
    demand_path = os.path.join(a.data_dir, "demand.npy")
    if os.path.exists(demand_path):
        demand = np.load(demand_path)
        print(f"sanity check: total pickups={demand.sum():.0f}, total dropoffs={dropoff.sum():.0f}")
    print(f"saved {out_path} shape={dropoff.shape}")


if __name__ == "__main__":
    main()
