import numpy as np

from surgemap.evaluation.headroom import hedge, online_bias_correction, online_residual_correction

H = (1, 3)


def data(n=600, z=4, seed=0):
    rng = np.random.default_rng(seed)
    rate = np.array([12.0, 6.0, 2.0, 0.3])[None, :z, None] * np.ones((n, 1, len(H)))
    return rng.poisson(rate).astype(float), rate


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_online_bias_correction_removes_a_constant_under_prediction():
    actual, rate = data()
    biased = 0.8 * rate
    fixed = online_bias_correction(biased, actual, H, half_life=100.0)
    assert rmse(fixed[200:], actual[200:]) < rmse(biased[200:], actual[200:])
    assert abs(fixed[200:, 0].mean() / rate[200:, 0].mean() - 1.0) < 0.05
    np.testing.assert_allclose(fixed[0], biased[0])          # nothing observed yet at the first step


def test_online_bias_correction_only_uses_outcomes_already_observed():
    actual, rate = data()
    pred = 0.8 * rate
    base = online_bias_correction(pred, actual, H)
    changed = actual.copy()
    changed[300:] += 50.0                                    # the future changes
    again = online_bias_correction(pred, changed, H)
    np.testing.assert_allclose(again[:301, :, 0], base[:301, :, 0])      # horizon 1: forecast 300 uses outcomes < 300
    np.testing.assert_allclose(again[:303, :, 1], base[:303, :, 1])      # horizon 3: outcomes up to step 299


def test_residual_correction_helps_when_errors_persist_and_not_otherwise():
    rng = np.random.default_rng(1)
    n, z = 800, 3
    drift = np.cumsum(rng.normal(0, 0.3, (n, z, 1)), axis=0) * np.ones((1, 1, len(H)))
    actual = 20.0 + drift + rng.normal(0, 0.2, (n, z, len(H)))
    pred = np.full_like(actual, 20.0)
    fixed, phis = online_residual_correction(pred, actual, H, half_life=200.0)
    assert rmse(fixed[100:], actual[100:]) < 0.5 * rmse(pred[100:], actual[100:])
    assert phis[0] > 0.8
    noise = 20.0 + rng.normal(0, 1.0, (n, z, len(H)))
    _, phis_noise = online_residual_correction(pred, noise, H, half_life=200.0)
    assert abs(phis_noise[0]) < 0.1


def test_hedge_moves_its_weight_to_the_better_forecaster():
    actual, rate = data()
    experts = {"good": rate, "bad": rate * 2.0}
    mixed, weights = hedge(experts, actual, H)
    assert weights["5min"]["good"] > 0.95
    assert rmse(mixed[100:], actual[100:]) < 1.02 * rmse(rate[100:], actual[100:])
