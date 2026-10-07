import numpy as np

from surgemap.simulation.policy import (ArrayForecast, LPRepositioner, oracle_forecast, persistence_forecast,
                                        round_rows, window_demand)
from surgemap.simulation.simulator import FleetSimulator, World, run_episode

HORIZONS = (1, 3, 6, 12)


def isolated_world(demand, move_bins=2):
    """Zones cannot borrow from each other, so only repositioning can bring vehicles."""
    z = demand.shape[0]
    dest = np.eye(z)
    minutes = np.full((z, z), 5.0 * move_bins)
    np.fill_diagonal(minutes, 0.0)
    return World(
        demand=np.asarray(demand, dtype=np.float64), dest_prob=dest,
        trip_bins=np.full((z, z), 3), move_minutes=minutes,
        move_bins=np.where(np.eye(z, dtype=bool), 0, move_bins),
        init_weights=np.full(z, 1.0 / z), zone_ids=np.arange(z),
        neighbors=[np.array([], dtype=np.int64) for _ in range(z)], max_wait_minutes=15.0)


def forecast_for(demand_row, h=len(HORIZONS)):
    return ArrayForecast(np.repeat(np.asarray(demand_row, dtype=float)[None, :, None], h, axis=-1), 0)


def test_window_demand_of_a_constant_forecast_scales_with_the_window():
    pred = np.full((2, 4), 2.0)
    np.testing.assert_allclose(window_demand(pred, HORIZONS, 3), [6.0, 6.0])


def test_window_demand_interpolates_between_horizons():
    pred = np.array([[0.0, 2.0, 5.0, 11.0]])      # value equals the bin offset 0, 2, 5, 11
    np.testing.assert_allclose(window_demand(pred, HORIZONS, 4), [0 + 1 + 2 + 3])
    np.testing.assert_allclose(window_demand(pred, HORIZONS, 14), [sum(range(12)) + 11 + 11])


def test_round_rows_keeps_row_sums_and_integrality():
    moves = np.array([[0.0, 1.4, 1.4, 1.2], [0.0, 0.0, 0.0, 0.0]])
    out = round_rows(moves)
    assert out.dtype.kind == "i" and out[0].sum() == 4 and out[1].sum() == 0
    assert (out >= 0).all()


def test_policy_sends_idle_vehicles_to_an_unserved_demand_zone():
    world = isolated_world(np.zeros((3, 10)))
    sim = FleetSimulator(world, n_vehicles=5, seed=0)
    sim.idle[:] = [0, 5, 0]
    policy = LPRepositioner(world, forecast_for([6.0, 0.0, 0.0]), HORIZONS, theta=0.01, lookahead=3)
    moves = policy(0, sim)
    assert moves is not None
    assert moves[1, 0] == 5 and moves.sum() == 5


def test_expensive_driving_discourages_moves():
    world = isolated_world(np.zeros((3, 10)))
    sim = FleetSimulator(world, n_vehicles=5, seed=0)
    sim.idle[:] = [0, 5, 0]
    # driving 10 minutes per vehicle at theta=5 costs 50, far above the 15-minute-per-request penalty
    policy = LPRepositioner(world, forecast_for([6.0, 0.0, 0.0]), HORIZONS, theta=5.0, lookahead=3)
    assert policy(0, sim) is None


def test_moves_never_exceed_the_idle_vehicles_in_a_zone():
    rng = np.random.default_rng(0)
    world = isolated_world(np.zeros((6, 10)))
    sim = FleetSimulator(world, n_vehicles=12, seed=0)
    sim.idle[:] = rng.integers(0, 4, size=6)
    policy = LPRepositioner(world, forecast_for(rng.uniform(0, 8, size=6)), HORIZONS, theta=0.01)
    moves = policy(0, sim)
    if moves is not None:
        assert (moves.sum(axis=1) <= sim.idle).all() and moves.diagonal().sum() == 0


def test_policy_returns_nothing_without_a_forecast_or_idle_vehicles():
    world = isolated_world(np.zeros((3, 10)))
    sim = FleetSimulator(world, n_vehicles=0, seed=0)
    assert LPRepositioner(world, forecast_for([5.0, 1.0, 1.0]), HORIZONS)(0, sim) is None
    sim.idle[:] = [0, 4, 0]
    assert LPRepositioner(world, ArrayForecast(np.zeros((0, 3, 4)), 0), HORIZONS)(0, sim) is None


def test_repositioning_with_a_good_forecast_reduces_unmet_demand_in_the_simulator():
    demand = np.zeros((3, 200))
    demand[0, :] = 3.0
    world = isolated_world(demand)
    world.init_weights = np.array([0.0, 0.5, 0.5])
    anchors = np.arange(1, 188)
    base = run_episode(world, 12, seed=0, start=0, end=190, policy=None, warmup=6)
    policy = LPRepositioner(world, oracle_forecast(demand, anchors, HORIZONS), HORIZONS,
                            theta=0.01, lookahead=3)
    moved = run_episode(world, 12, seed=0, start=0, end=190, policy=policy, warmup=6)
    assert moved["fleet_conserved"]
    assert moved["unmet_rate"] < base["unmet_rate"]
    assert moved["reposition_minutes"] > 0


def test_persistence_forecast_repeats_the_last_observed_bin():
    demand = np.arange(2 * 20, dtype=float).reshape(2, 20)
    fc = persistence_forecast(demand, np.arange(5, 15), HORIZONS)
    np.testing.assert_array_equal(fc(7)[:, 0], demand[:, 6])
    assert fc(7).shape == (2, 4) and fc(99) is None
