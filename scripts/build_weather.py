from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURE_NAMES = ["precip_log_zscore", "temp_zscore"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weather-csv", required=True)
    p.add_argument("--data-dir", default="real_processed_265")
    return p.parse_args()


def _parse_precip(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"T": "0.005", "nan": np.nan})
    s = s.str.replace(r"[A-Za-z]+$", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def build_weather(weather_csv: str, data_dir: str) -> tuple[np.ndarray, list[str]]:
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    start = pd.Timestamp(meta["start"])
    T = meta["n_bins"]
    grid = pd.date_range(start=start, periods=T, freq="5min")

    w = pd.read_csv(weather_csv, low_memory=False)
    w["DATE"] = pd.to_datetime(w["DATE"])
    hourly = w[w["REPORT_TYPE"].astype(str).str.contains("FM-15", na=False)].sort_values("DATE")
    hourly = hourly.drop_duplicates(subset="DATE")

    precip = _parse_precip(hourly["HourlyPrecipitation"])
    temp = pd.to_numeric(hourly["HourlyDryBulbTemperature"], errors="coerce")

    obs_src = pd.DataFrame({"DATE": hourly["DATE"].to_numpy(),
                             "precip": precip.to_numpy(), "temp": temp.to_numpy()})
    grid_df = pd.DataFrame({"DATE": grid})
    obs = pd.merge_asof(grid_df, obs_src, on="DATE", direction="backward")
    obs[["precip", "temp"]] = obs[["precip", "temp"]].bfill()
    if obs[["precip", "temp"]].isna().any().any():
        raise ValueError("weather alignment left NaNs after merge_asof+bfill")

    precip_arr = obs["precip"].to_numpy(dtype=np.float32)
    temp_arr = obs["temp"].to_numpy(dtype=np.float32)

    train_end = meta["split"]["train"][1]
    log_precip = np.log1p(precip_arr)
    precip_z = (log_precip - log_precip[:train_end].mean()) / max(float(log_precip[:train_end].std()), 1e-6)
    temp_z = (temp_arr - temp_arr[:train_end].mean()) / max(float(temp_arr[:train_end].std()), 1e-6)

    weather = np.stack([precip_z, temp_z], axis=1).astype(np.float32)
    return weather, FEATURE_NAMES


def main():
    a = parse_args()
    weather, names = build_weather(a.weather_csv, a.data_dir)
    out_path = os.path.join(a.data_dir, "weather.npy")
    np.save(out_path, weather)
    print(f"saved {out_path} shape={weather.shape} features={names}")
    print(f"precip_log_zscore: min={weather[:,0].min():.2f} max={weather[:,0].max():.2f}")
    print(f"temp_zscore: min={weather[:,1].min():.2f} max={weather[:,1].max():.2f}")


if __name__ == "__main__":
    main()
