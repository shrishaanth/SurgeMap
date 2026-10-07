import numpy as np
import pandas as pd

from surgemap.evaluation.metrics import block_bootstrap_gap, rmse_per_horizon
from surgemap.models.baselines import BINS_PER_DAY, WEEK, gbm_features, histavg_for_bins, ridge_with_histavg


def weekly_demand(weeks=3, zones=2):
    times = pd.date_range("2024-01-01", periods=weeks * 7 * BINS_PER_DAY, freq="5min").to_numpy()
    index = pd.DatetimeIndex(times)
    pattern = (index.dayofweek.to_numpy() * 7 + (index.hour * 12 + index.minute // 5).to_numpy() % 5).astype(float)
    return times, np.stack([pattern + 1, 2 * pattern + 3])[:zones]


def test_histavg_removes_a_train_bin_from_its_own_average():
    times, demand = weekly_demand()
    train_end = 2 * 7 * BINS_PER_DAY
    demand = demand.copy()
    spike_bin = 3 * BINS_PER_DAY + 40
    demand[:, spike_bin] += 1000.0
    loo = histavg_for_bins(demand, times, train_end, np.array([spike_bin]), smooth=0)
    clean = weekly_demand()[1][:, spike_bin]
    np.testing.assert_allclose(loo[:, 0], clean)       # the spike must not leak into its own average


def test_histavg_for_a_later_bin_uses_all_train_weeks():
    times, demand = weekly_demand()
    train_end = 2 * 7 * BINS_PER_DAY
    bin_ = 2 * 7 * BINS_PER_DAY + 3 * BINS_PER_DAY + 50
    out = histavg_for_bins(demand, times, train_end, np.array([bin_]), smooth=0)
    np.testing.assert_allclose(out[:, 0], demand[:, bin_])


def test_block_bootstrap_gap_is_zero_for_identical_models_and_positive_for_a_worse_one():
    rng = np.random.default_rng(0)
    actual = rng.poisson(3, size=(200, 5, 2)).astype(float)
    good = actual + rng.normal(0, 0.5, actual.shape)
    bad = actual + rng.normal(0, 1.5, actual.shape)
    gap, lo, hi = block_bootstrap_gap(good, good, actual)
    np.testing.assert_allclose(gap, 0.0, atol=1e-12)
    gap, lo, hi = block_bootstrap_gap(good, bad, actual)
    assert (gap > 0).all() and (lo > 0).all() and (hi >= gap).all()


def test_rmse_per_horizon_matches_a_direct_computation():
    pred = np.zeros((4, 3, 2))
    actual = np.stack([np.full((4, 3), 2.0), np.full((4, 3), 4.0)], axis=-1)
    np.testing.assert_allclose(rmse_per_horizon(pred, actual), [2.0, 4.0])


def test_gbm_features_rows_follow_anchor_then_zone_order():
    times, demand = weekly_demand(weeks=3, zones=2)
    weather = np.zeros((demand.shape[1], 2))
    anchors = np.array([WEEK + 100, WEEK + 101])
    feats = gbm_features(demand, times, weather, 2 * WEEK, anchors, 3, np.array([0.5, 1.5]))
    assert feats.shape[0] == len(anchors) * demand.shape[0]
    log_d = np.log1p(demand)
    np.testing.assert_allclose(feats[0, 0], log_d[0, anchors[0] - 1], rtol=1e-5)    # lag 1, anchor 0, zone 0
    np.testing.assert_allclose(feats[1, 0], log_d[1, anchors[0] - 1], rtol=1e-5)    # zone 1
    np.testing.assert_allclose(feats[2, 0], log_d[0, anchors[1] - 1], rtol=1e-5)    # anchor 1, zone 0
    assert np.isfinite(feats[:, :6]).all()


def test_ridge_with_histavg_returns_nonnegative_counts_of_the_right_shape():
    times, demand = weekly_demand(weeks=3, zones=2)
    z = np.log1p(demand)
    features = np.stack([(z - z[:, :WEEK].mean(1, keepdims=True)) / z[:, :WEEK].std(1, keepdims=True)], axis=-1)
    train = np.arange(48, 2 * WEEK - 12)
    test = np.arange(2 * WEEK + 48, 2 * WEEK + 100)
    out = ridge_with_histavg(features, demand, times, 2 * WEEK, train, test, 48, (1, 3))
    assert out.shape == (len(test), 2, 2) and (out >= 0).all()


def test_gbm_spatial_features_are_flow_weighted_neighbour_demand():
    times, demand = weekly_demand(weeks=3, zones=2)
    weather = np.zeros((demand.shape[1], 2))
    anchors = np.array([WEEK + 100])
    graph = (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.5, 0.5], [0.5, 0.5]]))
    plain = gbm_features(demand, times, weather, 2 * WEEK, anchors, 1, np.array([0.5, 1.5]))
    spatial = gbm_features(demand, times, weather, 2 * WEEK, anchors, 1, np.array([0.5, 1.5]), graph)
    assert spatial.shape[1] == plain.shape[1] + 6
    log_d = np.log1p(demand)
    np.testing.assert_allclose(spatial[0, plain.shape[1]], log_d[1, anchors[0] - 1], rtol=1e-5)   # zone 0 <- zone 1
    np.testing.assert_allclose(spatial[1, plain.shape[1]], log_d[0, anchors[0] - 1], rtol=1e-5)   # zone 1 <- zone 0


def test_boosted_model_stays_bounded_with_a_nearly_empty_zone():
    from surgemap.models.baselines import gbm_predictions
    rng = np.random.default_rng(0)
    total = 16 * BINS_PER_DAY
    times = pd.date_range("2024-01-01", periods=total, freq="5min").to_numpy()
    demand = np.stack([rng.poisson(6.0, total), rng.poisson(1.0, total), np.zeros(total)]).astype(float)
    demand[2, rng.choice(total, 6, replace=False)] = 1.0          # a zone with a handful of pickups in all
    train_end = 14 * BINS_PER_DAY
    train = np.arange(WEEK + 48, train_end - 12)
    test = np.arange(train_end + 48, total - 12)
    pred = gbm_predictions(demand, times, np.zeros((total, 2)), train_end, train, test, (1, 12),
                           stride=2, max_iter=120)
    assert pred.shape == (len(test), 3, 2) and np.isfinite(pred).all()
    assert pred.max() < 100.0 and pred[:, 2].max() < 1.0
    assert abs(pred[:, 0].mean() - 6.0) < 0.5


def test_boosted_features_leave_out_the_time_of_day_average_unless_asked():
    times, demand = weekly_demand(weeks=3, zones=2)
    weather = np.zeros((demand.shape[1], 2))
    anchors = np.array([WEEK + 100, WEEK + 101])
    level = np.array([0.5, 1.5])
    plain = gbm_features(demand, times, weather, 2 * WEEK, anchors, 3, level)
    with_average = gbm_features(demand, times, weather, 2 * WEEK, anchors, 3, level, use_histavg=True)
    assert with_average.shape[1] == plain.shape[1] + 1
    expected = np.log1p(histavg_for_bins(demand, times, 2 * WEEK, anchors + 3 - 1)).T.reshape(-1)
    extra = [c for c in range(with_average.shape[1]) if np.allclose(with_average[:, c], expected, rtol=1e-5)]
    assert extra                                                   # the average is a column only when asked for
    assert not any(np.allclose(plain[:, c], expected, rtol=1e-5) for c in range(plain.shape[1]))
