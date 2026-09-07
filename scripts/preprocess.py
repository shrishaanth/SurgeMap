from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

PICKUP = "tpep_pickup_datetime"
DROPOFF = "tpep_dropoff_datetime"
PULOC = "PULocationID"
DOLOC = "DOLocationID"
STEP = "5min"
CLIP_BOUND = 6.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="real_processed")
    p.add_argument("--month", default="2024-01")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--all-zones", action="store_true")
    p.add_argument("--chunksize", type=int, default=400_000)
    p.add_argument("--train-frac", type=float, default=0.70)
    return p.parse_args()


def row_validity(chunk, start, end):
    pu = pd.to_datetime(chunk[PICKUP], errors="coerce")
    do = pd.to_datetime(chunk[DROPOFF], errors="coerce")
    time_ok = (pu.notna() & do.notna() & (pu >= start) & (pu < end)
               & (do >= start) & (do < end))
    pu_id = pd.to_numeric(chunk[PULOC], errors="coerce")
    do_id = pd.to_numeric(chunk[DOLOC], errors="coerce")
    loc_ok = pu_id.notna() & do_id.notna() & (pu_id > 0) & (do_id > 0)
    duration = (do - pu) / pd.Timedelta("1min")
    dur_ok = duration.notna() & (duration >= 1.0) & (duration <= 360.0)
    return pu, do, pu_id, do_id, time_ok, loc_ok, dur_ok, duration


def row_norm(matrix):
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


def main():
    a = parse_args()
    all_zones = getattr(a, "all_zones", False)
    if not all_zones and (a.top_k < 1 or not 0 < a.train_frac < 1):
        raise ValueError("--top-k must be positive and --train-frac must be in (0, 1)")
    os.makedirs(a.out, exist_ok=True)
    start = pd.Timestamp(f"{a.month}-01")
    end = start + pd.offsets.MonthBegin(1)
    days = (end - start).days
    T = days * 24 * 60 // 5
    grid = pd.date_range(start=start, periods=T, freq=STEP)
    cols = [PICKUP, DROPOFF, PULOC, DOLOC]

    total_rows = invalid_time = invalid_location = invalid_duration = 0
    valid_rows = 0
    pickup_counts = Counter()
    flow_counts = Counter()
    duration_sum = Counter()
    duration_count = Counter()

    print(f"[preprocess] pass 1: {a.input}; strict range [{start}, {end})")
    for chunk in pd.read_csv(a.input, usecols=cols, chunksize=a.chunksize,
                             parse_dates=[PICKUP, DROPOFF]):
        total_rows += len(chunk)
        pu, do, pu_id, do_id, time_ok, loc_ok, dur_ok, duration = row_validity(chunk, start, end)
        invalid_time += int((~time_ok).sum())
        invalid_location += int((time_ok & ~loc_ok).sum())
        invalid_duration += int((time_ok & loc_ok & ~dur_ok).sum())
        ok = time_ok & loc_ok & dur_ok
        if not ok.any():
            continue
        p = pu_id[ok].astype(np.int64).to_numpy()
        d = do_id[ok].astype(np.int64).to_numpy()
        mins = np.rint(duration[ok].to_numpy()).astype(np.float32)
        valid_rows += len(p)
        pickup_counts.update(p.tolist())
        flow_counts.update(zip(p.tolist(), d.tolist()))
        for key, val in zip(zip(p.tolist(), d.tolist()), mins.tolist()):
            duration_sum[key] += float(val)
            duration_count[key] += 1

    if valid_rows == 0:
        raise RuntimeError(f"No valid records found in month {a.month}")
    if all_zones:
        zone_ids = sorted(pickup_counts.keys())
    else:
        zone_ids = [z for z, _ in pickup_counts.most_common(a.top_k)]
    zone_set = set(zone_ids)
    zone_index = {z: i for i, z in enumerate(zone_ids)}
    Z = len(zone_ids)
    print(f"[preprocess] rows total={total_rows:,}, valid={valid_rows:,}, "
          f"invalid_time={invalid_time:,}, invalid_location={invalid_location:,}, "
          f"invalid_duration={invalid_duration:,}")
    print(f"[preprocess] selected zones ({Z}): {zone_ids}")

    flow = np.zeros((Z, Z), dtype=np.float32)
    travel_time = np.full((Z, Z), np.inf, dtype=np.float32)
    for (p, d), count in flow_counts.items():
        if p in zone_set and d in zone_set:
            i, j = zone_index[p], zone_index[d]
            flow[i, j] = count
            if duration_count[(p, d)]:
                travel_time[i, j] = duration_sum[(p, d)] / duration_count[(p, d)]

    A_out = row_norm(flow)
    A_in = row_norm(flow.T)
    undirected = flow + flow.T
    np.fill_diagonal(undirected, 0)
    undirected += np.eye(Z, dtype=np.float32)
    degree = undirected.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    A_sym = (inv_sqrt[:, None] * undirected * inv_sqrt[None, :]).astype(np.float32)
    src, dst = np.nonzero(flow)
    edge_index = np.stack([src, dst]).astype(np.int64)

    demand = np.zeros((Z, T), dtype=np.float32)
    dropoff = np.zeros((Z, T), dtype=np.float32)
    grid_index = {t: i for i, t in enumerate(grid)}
    print(f"[preprocess] pass 2: aggregating complete grid ({T} bins)")
    for chunk in pd.read_csv(a.input, usecols=cols, chunksize=a.chunksize,
                             parse_dates=[PICKUP, DROPOFF]):
        pu, do, pu_id, do_id, time_ok, loc_ok, dur_ok, _ = row_validity(chunk, start, end)
        ok = time_ok & loc_ok & dur_ok
        if not ok.any():
            continue
        bins = pu[ok].dt.floor(STEP)
        ids = pu_id[ok].astype(np.int64).to_numpy()
        keep = np.array([z in zone_set for z in ids], dtype=bool)
        if keep.any():
            rows = np.fromiter((zone_index[z] for z in ids[keep]), dtype=np.int64)
            times = np.fromiter((grid_index[t] for t in bins[keep]), dtype=np.int64)
            np.add.at(demand, (rows, times), 1)
        do_bins = do[ok].dt.floor(STEP)
        do_ids = do_id[ok].astype(np.int64).to_numpy()
        do_keep = np.array([z in zone_set for z in do_ids], dtype=bool)
        if do_keep.any():
            do_rows = np.fromiter((zone_index[z] for z in do_ids[do_keep]), dtype=np.int64)
            do_times = np.fromiter((grid_index[t] for t in do_bins[do_keep]), dtype=np.int64)
            np.add.at(dropoff, (do_rows, do_times), 1)

    train_end = max(1, int(T * a.train_frac))

    def log_zscore(counts):
        log_counts = np.log1p(counts)
        mu = log_counts[:, :train_end].mean(axis=1, keepdims=True)
        sigma = log_counts[:, :train_end].std(axis=1, keepdims=True)
        return (log_counts - mu) / np.maximum(sigma, 1e-6)

    demand_z = log_zscore(demand)
    dropoff_z = log_zscore(dropoff)
    ts = pd.DatetimeIndex(grid)
    minute_of_day = ts.hour * 60 + ts.minute
    hour_angle = 2 * np.pi * minute_of_day / 1440.0
    week_angle = 2 * np.pi * ts.dayofweek / 7.0
    calendar = np.stack((np.sin(hour_angle), np.cos(hour_angle),
                         np.sin(week_angle), np.cos(week_angle),
                         (ts.dayofweek >= 5).astype(np.float32)), axis=1).astype(np.float32)
    features = np.empty((Z, T, 7), dtype=np.float32)
    features[:, :, 0] = demand_z
    features[:, :, 1:6] = calendar[None, :, :]
    features[:, :, 6] = dropoff_z

    features_clipped = features.copy()
    features_clipped[:, :, 0] = np.clip(features_clipped[:, :, 0], -CLIP_BOUND, CLIP_BOUND)
    features_clipped[:, :, 6] = np.clip(features_clipped[:, :, 6], -CLIP_BOUND, CLIP_BOUND)

    arrays = {
        "demand.npy": demand, "dropoff.npy": dropoff,
        "times.npy": grid.to_numpy(dtype="datetime64[ns]"),
        "features.npy": features, "features_clipped.npy": features_clipped,
        "A_out.npy": A_out, "A_in.npy": A_in,
        "A_sym.npy": A_sym, "adjacency.npy": A_sym, "edge_index.npy": edge_index,
        "travel_time.npy": travel_time,
        "zone_ids.npy": np.asarray(zone_ids, dtype=np.int32),
    }
    for name, value in arrays.items():
        np.save(os.path.join(a.out, name), value)

    train_end = int(train_end)
    val_end = train_end + int((T - train_end) / 2)
    metadata = {
        "input": os.path.abspath(a.input), "month": a.month,
        "start": start.isoformat(), "end": end.isoformat(), "step": "5min",
        "grid": {"start": start.isoformat(), "end_exclusive": end.isoformat(),
                 "periods": T, "frequency": "5min", "complete": True},
        "zone_ids": zone_ids, "n_zones": Z, "n_bins": T,
        "shapes": {name[:-4]: list(value.shape) for name, value in arrays.items()},
        "filter": {"total_rows": total_rows, "valid_rows": valid_rows,
                   "invalid_time": invalid_time, "invalid_location": invalid_location,
                   "invalid_duration": invalid_duration},
        "split": {"train": [0, train_end], "validation": [train_end, val_end],
                  "test": [val_end, T], "train_fraction": a.train_frac,
                  "scaler_fit_end": train_end},
        "feature_names": ["log_demand_zscore", "hour_sin", "hour_cos",
                          "weekday_sin", "weekday_cos", "is_weekend",
                          "log_dropoff_zscore"],
        "travel_time": "mean valid trip duration in minutes per selected directed edge; inf absent",
        "graph": "A_out and A_in preserve direction; A_sym is an ablation",
        "clipping": f"features_clipped.npy clips log_demand_zscore and log_dropoff_zscore to "
                    f"[-{CLIP_BOUND:g}, {CLIP_BOUND:g}]; calendar channels are untouched; "
                    f"training/eval scripts prefer this file when present.",
    }
    with open(os.path.join(a.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[preprocess] wrote {a.out}: demand={demand.shape}, times={grid.shape}, "
          f"features={features.shape}, edges={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
