import numpy as np

from diagnose_stgnn import correct, to_z


def test_correct_removes_a_multiplicative_bias_learned_on_validation():
    rng = np.random.default_rng(0)
    val_actual = rng.poisson(6, size=(300, 4, 2)).astype(float)
    test_actual = rng.poisson(6, size=(200, 4, 2)).astype(float)
    val_pred = (1 + val_actual) / 2 - 1          # model reports half of (1 + count)
    test_pred = (1 + test_actual) / 2 - 1
    hist = np.full_like(val_actual, 6.0)
    debiased, blended, weights = correct(val_pred, val_actual, hist, test_pred, np.full_like(test_actual, 6.0))
    np.testing.assert_allclose(debiased, test_actual, atol=1e-9)
    np.testing.assert_allclose(blended, test_actual, atol=1e-6)
    for model_weight, hist_weight in weights:
        assert abs(model_weight - 1.0) < 1e-6 and abs(hist_weight) < 1e-6


def test_correct_leans_on_the_average_when_the_model_is_noise():
    rng = np.random.default_rng(1)
    val_actual = np.full((400, 3, 1), 5.0) + rng.normal(0, 0.1, (400, 3, 1))
    noise = rng.uniform(0, 10, (400, 3, 1))
    _, _, weights = correct(noise, val_actual, np.full_like(val_actual, 5.0), noise[:50], np.full((50, 3, 1), 5.0))
    assert weights[0][1] > 0.8 and abs(weights[0][0]) < 0.2


def test_to_z_standardises_log_counts_and_clips():
    mu, sigma = np.array([np.log1p(4.0)]), np.array([0.5])
    z = to_z(np.array([[[4.0]], [[1e9]]]), mu, sigma)
    assert z[0, 0, 0] == 0.0 and z[1, 0, 0] == 6.0
