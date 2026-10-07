import numpy as np
import pandas as pd

from surgemap.evaluation.export import (gather_targets, historical_average, persistence_counts, ridge_predictions,
                                        to_counts, train_stats)


def test_to_counts_inverts_the_preprocess_zscore():
    rng = np.random.default_rng(0)
    demand = rng.poisson(5.0, size=(4, 200)).astype(np.float32)
    train_end = 140
    mu, sigma = train_stats(demand, train_end)
    z = ((np.log1p(demand) - mu[:, None]) / sigma[:, None])[:, :, None]
    counts = to_counts(np.transpose(z, (1, 0, 2)), mu, sigma)
    np.testing.assert_allclose(counts[:, :, 0].T, demand, rtol=1e-3, atol=1e-3)


def test_to_counts_is_never_negative():
    mu, sigma = np.array([0.0]), np.array([1.0])
    assert to_counts(np.full((1, 1, 1), -50.0), mu, sigma).min() >= 0.0


def test_gather_targets_uses_anchor_plus_horizon_minus_one():
    series = np.arange(2 * 20).reshape(2, 20)
    out = gather_targets(series, np.array([5, 6]), (1, 3))
    assert out.shape == (2, 2, 2)
    np.testing.assert_array_equal(out[0, 0], [series[0, 5], series[0, 7]])
    np.testing.assert_array_equal(out[1, 1], [series[1, 6], series[1, 8]])


def test_persistence_repeats_the_last_observed_bin():
    demand = np.arange(3 * 10, dtype=np.float32).reshape(3, 10)
    out = persistence_counts(demand, np.array([4, 7]), 2)
    np.testing.assert_array_equal(out[0, :, 0], demand[:, 3])
    np.testing.assert_array_equal(out[1, :, 1], demand[:, 6])


def test_historical_average_recovers_a_weekly_pattern_without_smoothing():
    times = pd.date_range("2024-01-01", periods=14 * 288, freq="5min").to_numpy()
    index = pd.DatetimeIndex(times)
    tod = (index.hour * 60 + index.minute) // 5
    pattern = (index.dayofweek.to_numpy() * 10 + tod.to_numpy() % 7).astype(np.float32)
    demand = np.stack([pattern, 2 * pattern])
    train_end = 7 * 288
    anchors = np.array([train_end + 100, train_end + 500])
    out = historical_average(demand, times, train_end, anchors, (1, 3), smooth=0)
    for i, anchor in enumerate(anchors):
        for j, h in enumerate((1, 3)):
            np.testing.assert_allclose(out[i, :, j], demand[:, anchor + h - 1])


def test_ridge_recovers_a_linear_recurrence():
    t = np.arange(260)
    features = np.stack([np.stack([np.sin(0.3 * t + z), np.cos(0.1 * t)], axis=1)
                         for z in range(3)]).astype(np.float32)
    train = np.arange(20, 150)
    test = np.arange(160, 240)
    pred = ridge_predictions(features, train, test, window=4, horizons=(1,), alpha=1e-8)
    truth = features[:, test, 0].T[..., None]
    np.testing.assert_allclose(pred, truth, atol=1e-2)
