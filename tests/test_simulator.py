import numpy as np
import pytest

from simulator import FleetSimulator, World, build_world, repositioning_minutes, run_episode

INF = np.inf


def toy_world(demand, trip_to=2, trip_bins=3, move_bins=2, connected=True, max_wait=15.0):
    z = demand.shape[0]
    dest = np.zeros((z, z))
    dest[:, min(trip_to, z - 1)] = 1.0
    minutes = np.full((z, z), 5.0 * move_bins)
    np.fill_diagonal(minutes, 0.0)
    neighbors = [np.array([k for k in range(z) if k != i], dtype=np.int64) if connected
                 else np.array([], dtype=np.int64) for i in range(z)]
    return World(
        demand=np.asarray(demand, dtype=np.float64), dest_prob=dest,
        trip_bins=np.full((z, z), trip_bins), move_minutes=minutes,
        move_bins=np.where(np.eye(z, dtype=bool), 0, move_bins),
        init_weights=np.full(z, 1.0 / z), zone_ids=np.arange(z),
        neighbors=neighbors, max_wait_minutes=max_wait)


def test_repositioning_minutes_routes_through_intermediate_zones():
    tt = np.array([[0.0, 10.0, INF], [INF, 0.0, 20.0], [INF, INF, 0.0]])
    minutes = repositioning_minutes(tt)
    assert minutes[0, 2] == 30.0
    assert minutes[0, 1] == 10.0
    assert np.isinf(minutes[2, 0])
    assert np.all(np.diag(minutes) == 0)


def test_build_world_makes_distributions_lags_and_neighbors_valid():
    demand = np.random.default_rng(0).poisson(3, size=(3, 40))
    a_out = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tt = np.array([[1.0, 12.0, 6.0], [11.0, 1.0, INF], [INF, INF, 1.0]])
    world = build_world(demand, a_out, tt, [10, 20, 30], train_end=30, max_wait_minutes=12.0)
    np.testing.assert_allclose(world.dest_prob.sum(axis=1), 1.0)
    assert world.dest_prob[2, 2] == 1.0
    assert world.trip_bins[0, 1] == 3 and world.trip_bins[0, 2] == 2
    assert np.all(np.diag(world.move_bins) == 0)
    assert world.move_bins[0, 1] >= 1
    assert world.init_weights.sum() == pytest.approx(1.0)
    # zone 0 can be reached from zone 1 in 11 min (<= 12) and from nowhere else
    np.testing.assert_array_equal(world.neighbors[0], [1])
    assert len(world.neighbors[2]) == 1 and world.neighbors[2][0] == 0


def test_initial_placement_uses_the_whole_fleet():
    sim = FleetSimulator(toy_world(np.ones((4, 10))), n_vehicles=37, seed=1)
    assert sim.idle.sum() == 37 and sim.fleet_total() == 37


def test_local_vehicles_serve_with_no_wait_and_excess_borrows_from_neighbors():
    sim = FleetSimulator(toy_world(np.zeros((2, 5)), move_bins=2), n_vehicles=10, seed=0)
    sim.idle[:] = [4, 6]
    sim.begin(0)
    out = sim.finish(0, np.array([7, 2]))
    assert out["served"] == 9 and out["unmet"] == 0 and out["borrowed"] == 3
    assert out["wait_minutes"] == 3 * 10.0
    np.testing.assert_array_equal(sim.idle, [0, 1])


def test_requests_with_no_reachable_vehicle_are_lost():
    sim = FleetSimulator(toy_world(np.zeros((2, 5)), connected=False), n_vehicles=10, seed=0)
    sim.idle[:] = [4, 6]
    sim.begin(0)
    out = sim.finish(0, np.array([7, 2]))
    assert out["served"] == 4 + 2 and out["unmet"] == 3 and out["borrowed"] == 0
    np.testing.assert_array_equal(sim.idle, [0, 4])


def test_served_trip_returns_to_the_destination_after_its_duration():
    sim = FleetSimulator(toy_world(np.zeros((3, 10)), trip_to=2, trip_bins=3), n_vehicles=1, seed=0)
    sim.idle[:] = [1, 0, 0]
    sim.begin(0)
    sim.finish(0, np.array([1, 0, 0]))
    for t in (1, 2):
        sim.begin(t)
        assert sim.idle[2] == 0
    sim.begin(3)
    assert sim.idle[2] == 1 and sim.fleet_total() == 1


def test_borrowed_vehicle_returns_after_pickup_drive_plus_trip():
    sim = FleetSimulator(toy_world(np.zeros((3, 10)), trip_to=2, trip_bins=3, move_bins=2),
                         n_vehicles=1, seed=0)
    sim.idle[:] = [0, 1, 0]
    sim.begin(0)
    out = sim.finish(0, np.array([1, 0, 0]))
    assert out["borrowed"] == 1
    for t in range(1, 5):
        sim.begin(t)
        assert sim.idle[2] == 0
    sim.begin(5)
    assert sim.idle[2] == 1


def test_repositioned_vehicles_arrive_after_the_travel_lag():
    sim = FleetSimulator(toy_world(np.zeros((3, 10)), move_bins=2), n_vehicles=5, seed=0)
    sim.idle[:] = [5, 0, 0]
    sim.begin(0)
    moves = np.zeros((3, 3), dtype=int)
    moves[0, 1] = 4
    moved, minutes = sim.reposition(0, moves)
    assert moved == 4 and minutes == 4 * 10.0
    np.testing.assert_array_equal(sim.idle, [1, 0, 0])
    sim.begin(1)
    assert sim.idle[1] == 0
    sim.begin(2)
    assert sim.idle[1] == 4


def test_two_sources_to_one_destination_with_equal_lag_both_arrive():
    sim = FleetSimulator(toy_world(np.zeros((3, 10)), move_bins=2), n_vehicles=6, seed=0)
    sim.idle[:] = [3, 3, 0]
    sim.begin(0)
    moves = np.zeros((3, 3), dtype=int)
    moves[0, 2], moves[1, 2] = 2, 3
    sim.reposition(0, moves)
    sim.begin(2)
    assert sim.idle[2] == 5


def test_moving_more_than_idle_is_rejected():
    sim = FleetSimulator(toy_world(np.zeros((2, 5))), n_vehicles=4, seed=0)
    sim.idle[:] = [1, 3]
    moves = np.array([[0, 2], [0, 0]])
    with pytest.raises(ValueError):
        sim.reposition(0, moves)


def test_episode_conserves_the_fleet_and_counts_demand():
    rng = np.random.default_rng(3)
    world = toy_world(rng.poisson(4, size=(3, 120)).astype(float))

    def greedy(t, sim):
        moves = np.zeros((3, 3), dtype=int)
        src = int(np.argmax(sim.idle))
        if sim.idle[src] > 1:
            moves[src, (src + 1) % 3] = 1
        return moves

    result = run_episode(world, n_vehicles=20, seed=2, start=0, end=120, policy=greedy, warmup=10)
    assert result["fleet_conserved"]
    assert result["served"] + result["unmet"] == result["demand"]
    assert result["demand"] == int(world.demand[:, 10:120].sum())
    assert result["moved"] > 0 and result["reposition_minutes"] > 0
    assert result["empty_minutes"] == result["wait_minutes"] + result["reposition_minutes"]


def test_episode_without_policy_never_repositions_and_is_reproducible():
    world = toy_world(np.random.default_rng(1).poisson(3, size=(3, 80)).astype(float))
    a = run_episode(world, 15, seed=5, start=0, end=80)
    b = run_episode(world, 15, seed=5, start=0, end=80)
    assert a == b and a["moved"] == 0 and a["decisions"] == 0
