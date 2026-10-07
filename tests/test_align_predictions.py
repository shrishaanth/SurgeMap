import numpy as np
import pandas as pd
import pytest

from surgemap.models.baselines import load_weather
from surgemap.evaluation.align import align

H = np.array([1, 3])


def datasets():
    """A long dataset and a short one covering its last four days, with the same demand there."""
    rng = np.random.default_rng(0)
    long_times = pd.date_range("2023-12-28", periods=8 * 288, freq="5min").to_numpy()
    long_demand = rng.poisson(3, size=(2, 8 * 288)).astype(np.float32)
    offset = 4 * 288
    return long_times, long_demand, long_times[offset:], long_demand[:, offset:], offset


def forecasts(times, demand, anchors):
    return {"anchors": anchors, "anchor_times": times[anchors], "zone_ids": np.array([4, 7]), "horizons": H,
            "actual": np.stack([demand[:, anchors + h - 1].T for h in H], axis=-1),
            "stgnn": np.ones((len(anchors), 2, 2), dtype=np.float32)}


def test_anchors_are_moved_onto_the_short_dataset_by_timestamp():
    long_times, long_demand, short_times, short_demand, offset = datasets()
    anchors = np.arange(offset + 500, offset + 520)
    aligned = align(forecasts(long_times, long_demand, anchors), short_times, np.array([4, 7]), short_demand)
    np.testing.assert_array_equal(aligned["anchors"], anchors - offset)
    np.testing.assert_array_equal(short_times[aligned["anchors"]], long_times[anchors])
    assert aligned["stgnn"].shape == (20, 2, 2)


def test_mismatched_zones_actuals_or_times_are_rejected():
    long_times, long_demand, short_times, short_demand, offset = datasets()
    good = forecasts(long_times, long_demand, np.arange(offset + 500, offset + 520))
    with pytest.raises(ValueError, match="zones"):
        align(good, short_times, np.array([4, 8]), short_demand)
    with pytest.raises(ValueError, match="actual"):
        align(good, short_times, np.array([4, 7]), short_demand + 1)
    early = forecasts(long_times, long_demand, np.arange(100, 120))          # before the short dataset starts
    with pytest.raises(ValueError, match="times"):
        align(early, short_times, np.array([4, 7]), short_demand)


def test_load_weather_falls_back_to_zeros(tmp_path):
    assert load_weather(str(tmp_path), 50).shape == (50, 2) and not load_weather(str(tmp_path), 50).any()
    np.save(tmp_path / "weather.npy", np.ones((50, 2), dtype=np.float32))
    assert load_weather(str(tmp_path), 50).all()
