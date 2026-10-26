"""Baseline forecasters: the time-of-day average, ridge regression and gradient boosting."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
BIN_MINUTES = 5


BINS_PER_DAY = 288


DAY, WEEK = BINS_PER_DAY, 7 * BINS_PER_DAY




'''
The below function is for converting timestamp into
Day of the week: Monday = 0, Tuesday = 1, … Sunday = 6.
Time-of-day bin: an integer from 0 to 287.
'''
def calendar_keys(times) -> tuple[np.ndarray, np.ndarray]:
    index = pd.DatetimeIndex(times)
    return index.dayofweek.to_numpy(), ((index.hour * 60 + index.minute) // BIN_MINUTES).to_numpy()


def histavg_for_bins(demand, times, train_end, bins, smooth: int = 3) -> np.ndarray:
    """Train-split mean demand for each bin's (zone, weekday, time of day), shaped [Z, len(bins)].

    A bin inside the train split would count itself, so its own contribution is removed.
    """
    dow, tod = calendar_keys(times)  # day of week , time of day
    z = demand.shape[0]   # no of zones
    sums, counts = np.zeros((z, 7, BINS_PER_DAY)), np.zeros((7, BINS_PER_DAY)) # zone → weekday → time of day
    '''
    counts records how many observations contributed to each weekday/time-of-day group. It is shared across zones because the timestamps are common to all zones.
    np.add.at(array, indices, values)
    add values into array at the specified indices.
    '''
    np.add.at(sums, (slice(None), dow[:train_end], tod[:train_end]), demand[:, :train_end])
    np.add.at(counts, (dow[:train_end], tod[:train_end]), 1.0)
    # take every zone's demand at time t and add it to that observation's weekday/time-bin group.  
    s_sum, s_cnt = np.zeros_like(sums), np.zeros_like(counts)
    # creates a new array filled with zeros that has the same shape as the given array.
    # this loop is for smoothening
    for shift in range(-smooth, smooth + 1):
        s_sum += np.roll(sums, shift, axis=2)
        s_cnt += np.roll(counts, shift, axis=1)
    '''
    Shift	Time bin	Time
    -3	105	8:45 AM
    -2	106	8:50 AM
    -1	107	8:55 AM
    0	108	9:00 AM
    +1	109	9:05 AM
    +2	110	9:10 AM
    +3	111	9:15 AM
    The loop adds the corresponding arrays for each shifts_sum contains the combined demand totals across these neighboring bins, and s_cnt contains the corresponding combined observation counts.
    '''
    bins = np.asarray(bins)
    # bins contains the time indices for which we want to calculate historical average demand.
    # coverting it into numpy array 
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
    log_counts = np.log1p(demand[:, :train_end]) # Log-transform the demand
    return log_counts.mean(axis=1), np.maximum(log_counts.std(axis=1), 1e-6)
    # two arrays -> mean and standard deviation for each zone

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
        windows = sliding_window_view(features[zone], window, axis=0) # for creating the window view
        # hstack for stacking horizontally
        # stack usually stacks , creating new dimensions
        x_tr = np.hstack([windows[train_anchors - window].reshape(len(train_anchors), -1), hist_train[zone]])
        x_te = np.hstack([windows[test_anchors - window].reshape(len(test_anchors), -1), hist_test[zone]])
        y_tr = np.stack([features[zone, train_anchors + h - 1, 0] for h in horizons], axis=1)
        design = np.hstack([x_tr, np.ones((len(x_tr), 1))]).astype(np.float64)  # adding ones for intercept
        penalty = alpha * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_tr.astype(np.float64))
        pred_z = np.hstack([x_te, np.ones((len(x_te), 1))]) @ coef
        out[:, zone] = np.clip(np.expm1(pred_z * sigma[zone] + mu[zone]), 0.0, None)
    return out


def gbm_features(demand, times, weather, train_end, anchors, h, zone_level, spatial=None,
                 use_histavg: bool = False) -> np.ndarray:
    """Feature matrix with one row per (anchor, zone), for target bin anchor + h - 1."""
    z, n = demand.shape[0], len(anchors)
    target = anchors + h - 1
    log_d = np.log1p(demand)
    cols = []

    def per_zone(values):                     # values [Z, N] -> flat in (anchor, zone) order
        return np.asarray(values, dtype=np.float32).T.reshape(-1)
    #-1 means flatten into one dimension, with NumPy determining the resulting length automatically.
    for lag in (1, 2, 3, 6):
        cols.append(per_zone(log_d[:, anchors - lag]))
    # takes the log-demand for every zone at those earlier timestamps and adds it as a feature column.
    for span in (12, 48):
        cols.append(per_zone(np.stack([log_d[:, a - span:a].mean(axis=1) for a in anchors], axis=1)))
    '''
    This calculates average log-demand over two historical windows:
span	Duration
12 bins	1 hour
48 bins	4 hours
For each anchor, it looks backward over that many bins and calculates the mean for each zone.

For example, when span = 12, it averages the 12 five-minute intervals before the anchor.

np.stack(..., axis=1) combines the averages for all anchors into a [Z, N] array. Then per_zone() flattens them into one feature column.
    '''
    for lag in (DAY, WEEK):
        idx = target - lag
        vals = np.where(idx[None, :] >= 0, log_d[:, np.maximum(idx, 0)], np.nan)
        cols.append(per_zone(vals))
    
    if use_histavg:                           # the time-of-day average of the target bin; off by default
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


'''
Parameter	Meaning
demand	Demand values for all zones over time, shaped [Z, T]
times	Timestamps corresponding to the time intervals
weather	Weather features for each time interval
train_end	End of the training period
anchors	Time indices for the prediction samples
h	Forecast horizon
zone_level	A feature describing each zone
spatial	Optional matrices used to calculate neighboring-zone demand features
use_histavg	Whether to include historical-average demand as an extra feature
'''

def gbm_predictions(demand, times, weather, train_end, train_anchors, test_anchors, horizons,
                    stride: int = 3, max_iter: int = 200, spatial=None,
                    use_histavg: bool = False) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    z = demand.shape[0]
    zone_level = np.log1p(demand[:, :train_end]).mean(axis=1)
    fit_anchors = train_anchors[::stride]
    out = np.empty((len(test_anchors), z, len(horizons)), dtype=np.float32)
    for j, h in enumerate(horizons):
        x_tr = gbm_features(demand, times, weather, train_end, fit_anchors, h, zone_level, spatial, use_histavg)
        y_tr = demand[:, fit_anchors + h - 1].T.reshape(-1)
        # The L2 term matters for stability, not just accuracy. With a Poisson loss the score of
        # a zone that is almost always empty drifts very low, its curvature goes to zero, and a
        # single rare pickup then produces an unbounded leaf value. On four months of data that
        # made the unregularised model predict infinities for the quiet zones.
        model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.1, max_iter=max_iter,
                                              max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
                                              early_stopping=False, random_state=7)
        model.fit(x_tr, y_tr)
        x_te = gbm_features(demand, times, weather, train_end, test_anchors, h, zone_level, spatial, use_histavg)
        predicted = model.predict(x_te)
        if not np.isfinite(predicted).all() or predicted.max() > 100.0 * max(float(y_tr.max()), 1.0):
            raise FloatingPointError(f"boosted model produced unbounded predictions at horizon {h} "
                                     f"(max {np.nanmax(predicted):.3g}, largest training count {y_tr.max():.0f})")
        out[:, :, j] = predicted.reshape(len(test_anchors), z)
        print(f"[checks] gbm horizon {h * 5} min fitted on {len(y_tr):,} rows", flush=True)
    return out
