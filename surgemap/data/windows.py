"""Training windows: the dataset, the lagged inputs for the output heads and the split bounds."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from surgemap.models.baselines import histavg_for_bins, zscore_params

HORIZONS = (1, 3, 6, 12)
CLIP = 6.0


class MultiHorizonDataset(Dataset):
    """Windows of features with multi-horizon targets and an optional per-target prior.

    Each item is (x [Z, W, F], y [Z, H], prior [Z, H], lagged [Z, H, E]). The prior is zero
    and the lagged inputs have E = 0 when they are not given.
    """

    def __init__(self, features: np.ndarray, start: int, end: int,
                 window: int = 48, horizons=HORIZONS, prior: np.ndarray | None = None,
                 lagged: np.ndarray | None = None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.start, self.end = int(start), int(end)
        self.window = int(window)
        self.horizons = tuple(int(h) for h in horizons)
        if self.features.ndim != 3:
            raise ValueError("features must have shape [Z,T,F]")
        if self.window < 1 or not self.horizons or min(self.horizons) < 1:
            raise ValueError("window and horizons must be positive")
        if prior is not None and prior.shape != (features.shape[0], features.shape[1], len(self.horizons)):
            raise ValueError("prior must have shape [Z,T,H]")
        self.prior = None if prior is None else torch.as_tensor(prior, dtype=torch.float32)
        self._no_prior = torch.zeros(features.shape[0], len(self.horizons))
        if lagged is not None and lagged.shape[:3] != (features.shape[0], features.shape[1], len(self.horizons)):
            raise ValueError("lagged must have shape [Z,T,H,E]")
        self.lagged = None if lagged is None else torch.as_tensor(lagged, dtype=torch.float32)
        self._no_lagged = torch.zeros(features.shape[0], len(self.horizons), 0)
        first = max(self.start, self.window)
        last = self.end - max(self.horizons)
        self.times = list(range(first, max(first, last + 1)))
        self.times = [t for t in self.times if t + max(self.horizons) - 1 < self.end]

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        t = self.times[index]
        x = self.features[:, t - self.window:t, :]
        y = torch.stack([self.features[:, t + h - 1, 0] for h in self.horizons], dim=-1)
        prior = self._no_prior if self.prior is None else self.prior[:, t, :]
        lagged = self._no_lagged if self.lagged is None else self.lagged[:, t]
        return x, y, prior, lagged


LAG_BINS = (288, 2016)          # one day and one week of 5-minute bins


def lagged_inputs(features: np.ndarray, horizons) -> np.ndarray:
    """For each anchor bin and horizon: standardised demand of the target bin one day and one
    week earlier, plus a flag for each saying whether that bin exists. Shape [Z, T, H, 4].

    The target bin is anchor + h - 1, so a lag of at least h bins is already observed at the
    anchor; a day and a week both are for every horizon used here.
    """
    if max(horizons) > min(LAG_BINS):
        raise ValueError("horizons must not exceed the shortest lag")
    z, total = features.shape[0], features.shape[1]
    out = np.zeros((z, total, len(horizons), 2 * len(LAG_BINS)), dtype=np.float32)
    for j, h in enumerate(horizons):
        target = np.minimum(np.arange(total) + h - 1, total - 1)
        for k, lag in enumerate(LAG_BINS):
            source = target - lag
            ok = source >= 0
            out[:, ok, j, 2 * k] = features[:, source[ok], 0]
            out[:, ok, j, 2 * k + 1] = 1.0
    return out


def prepare_inputs(data_dir: str, features: np.ndarray, train_end: int, horizons,
                   use_weather: bool = False, prior_kind: str = "none", lag_features: bool = False):
    """Optional extra inputs: weather channels appended to the features, a prior [Z, T, H],
    and lagged head inputs [Z, T, H, E]. Returns (features, prior, lagged).

    The prior is a time-of-day average for each target bin, in the standardised log units the
    network predicts. "histavg_z" averages the standardised log demand itself, which is what
    the squared-error target calls for. "histavg" standardises the log of the average count,
    which sits above the average of the logs (most for quiet zones); it is kept only so that
    checkpoints trained with it still load. Bins inside the train split exclude their own
    value, so the prior does not leak the target.
    """
    prior = None
    lagged = lagged_inputs(features, horizons) if lag_features else None
    if use_weather:
        weather = np.load(os.path.join(data_dir, "weather.npy")).astype(np.float32)
        tiled = np.broadcast_to(weather[None], (features.shape[0],) + weather.shape)
        features = np.concatenate([features, tiled], axis=2)
    if prior_kind in ("histavg", "histavg_z"):
        demand = np.load(os.path.join(data_dir, "demand.npy")).astype(np.float64)
        times = np.load(os.path.join(data_dir, "times.npy"))
        total = demand.shape[1]
        mu, sigma = zscore_params(demand, train_end)
        if prior_kind == "histavg_z":
            z = np.clip((np.log1p(demand) - mu[:, None]) / sigma[:, None], -CLIP, CLIP)
            hist_z = histavg_for_bins(z, times, train_end, np.arange(total))
        else:
            hist = histavg_for_bins(demand, times, train_end, np.arange(total))
            hist_z = np.clip((np.log1p(hist) - mu[:, None]) / sigma[:, None], -CLIP, CLIP)
        prior = np.stack([hist_z[:, np.minimum(np.arange(total) + h - 1, total - 1)] for h in horizons],
                         axis=-1).astype(np.float32)
    elif prior_kind != "none":
        raise ValueError(f"unknown prior: {prior_kind}")
    return features, prior, lagged


def split_bounds(meta, total):
    split = meta.get("split", {})
    train_default = int(.70 * total)
    val_default = int(.85 * total)
    train = split.get("train")
    validation = split.get("validation")
    test = split.get("test")
    train_end = int(split.get("train_end", train[1] if train else train_default))
    val_end = int(split.get("val_end", validation[1] if validation else val_default))
    if train is None:
        train = [0, train_end]
    if validation is None:
        validation = [train_end, val_end]
    if test is None:
        test = [val_end, total]
    return ((int(train[0]), int(train[1])),
            (int(validation[0]), int(validation[1])),
            (int(test[0]), int(test[1])))
