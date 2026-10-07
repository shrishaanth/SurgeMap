import numpy as np

from surgemap.evaluation.regimes import PERIODS, ZONES, analyse, gap_interval


def test_gap_interval_is_positive_when_the_model_has_less_error():
    n = np.full(200, 10.0)
    worse, better = np.full(200, 40.0), np.full(200, 10.0)
    out = gap_interval((worse, n), (better, n))
    assert np.isclose(out["gap"], 1.0) and out["ci_low"] > 0 and out["ci_low"] <= out["gap"] <= out["ci_high"]


def test_analyse_splits_cover_every_cell_and_find_where_the_model_wins():
    rng = np.random.default_rng(0)
    n, z, h = 400, 20, 2
    level = np.linspace(0.2, 10.0, z)[None, :, None]
    surge = (1.0 + 0.5 * rng.normal(size=(n, 1, 1))).clip(0.2)
    histavg = np.broadcast_to(level, (n, z, h)).copy()
    actual = histavg * surge
    good = actual + 0.01 * rng.normal(size=actual.shape)
    out = analyse(actual, {"model": good, "other": histavg}, histavg, (1, 3), "model", "other")
    for kind, parts in (("periods", PERIODS), ("zones", ZONES)):
        assert list(out[kind]) == [name for name, _, _ in parts]
        for key in ("5min", "15min"):
            assert np.isclose(sum(out[kind][name][key]["share_of_cells"] for name, _, _ in parts), 1.0)
    extreme, normal = out["periods"][PERIODS[-1][0]]["5min"], out["periods"][PERIODS[2][0]]["5min"]
    assert extreme["citywide_pickups_vs_usual"] > normal["citywide_pickups_vs_usual"]
    assert extreme["model_vs_other"]["gap"] > normal["model_vs_other"]["gap"] > 0
    assert extreme["model_vs_other"]["ci_low"] > 0
