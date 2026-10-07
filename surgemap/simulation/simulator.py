from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

BIN_MINUTES = 5
BUFFER_BINS = 128


@dataclass
class World:
    """Replayed demand plus the flow and travel-time structure of the city."""
    demand: np.ndarray          # [Z, T] actual pickups per bin
    dest_prob: np.ndarray       # [Z, Z] row-stochastic destination distribution
    trip_bins: np.ndarray       # [Z, Z] int, trip duration in bins
    move_minutes: np.ndarray    # [Z, Z] float, repositioning time; inf if unreachable
    move_bins: np.ndarray       # [Z, Z] int, repositioning time in bins (0 on the diagonal)
    init_weights: np.ndarray    # [Z] initial fleet placement probabilities
    zone_ids: np.ndarray        # [Z]
    neighbors: list             # neighbors[i]: zones whose idle vehicles can reach i in time, nearest first
    max_wait_minutes: float = 15.0

    @property
    def n_zones(self) -> int:
        return self.demand.shape[0]


def repositioning_minutes(travel_time: np.ndarray) -> np.ndarray:
    """Zone-to-zone driving time: shortest paths over the observed mean trip durations.

    travel_time is inf for pairs with no observed trips, so the missing pairs are filled
    by routing through intermediate zones. Pairs that stay disconnected remain inf.
    """
    n = travel_time.shape[0]
    off_diagonal = np.isfinite(travel_time) & ~np.eye(n, dtype=bool)
    rows, cols = np.nonzero(off_diagonal)
    graph = csr_matrix((travel_time[rows, cols].astype(np.float64), (rows, cols)), shape=(n, n))
    minutes = shortest_path(graph, method="D", directed=True)
    np.fill_diagonal(minutes, 0.0)
    return minutes


def build_world(demand, a_out, travel_time, zone_ids, train_end: int,
                max_wait_minutes: float = 15.0) -> World:
    demand = np.asarray(demand, dtype=np.float64)
    n = demand.shape[0]
    flow = np.asarray(a_out, dtype=np.float64).copy()
    sums = flow.sum(axis=1, keepdims=True)
    empty = sums[:, 0] == 0
    flow[empty, :] = 0.0
    flow[empty, np.nonzero(empty)[0]] = 1.0
    dest_prob = flow / flow.sum(axis=1, keepdims=True)

    trip_minutes = np.where(np.isfinite(travel_time), travel_time, BIN_MINUTES)
    trip_bins = np.clip(np.ceil(trip_minutes / BIN_MINUTES), 1, BUFFER_BINS - 1).astype(np.int64)

    move_minutes = repositioning_minutes(np.asarray(travel_time, dtype=np.float64))
    move_bins = np.full((n, n), BUFFER_BINS - 1, dtype=np.int64)
    reachable = np.isfinite(move_minutes)
    move_bins[reachable] = np.clip(np.ceil(move_minutes[reachable] / BIN_MINUTES), 1, BUFFER_BINS - 1)
    np.fill_diagonal(move_bins, 0)

    if not 0 < max_wait_minutes <= 60:
        raise ValueError("max_wait_minutes must be in (0, 60]")
    neighbors = []
    for zone in range(n):
        minutes = move_minutes[:, zone]
        near = np.nonzero((minutes <= max_wait_minutes) & (np.arange(n) != zone))[0]
        neighbors.append(near[np.argsort(minutes[near], kind="stable")])

    weights = demand[:, :train_end].mean(axis=1)
    return World(demand=demand, dest_prob=dest_prob, trip_bins=trip_bins,
                 move_minutes=move_minutes, move_bins=move_bins,
                 init_weights=weights / weights.sum(), zone_ids=np.asarray(zone_ids),
                 neighbors=neighbors, max_wait_minutes=float(max_wait_minutes))


def load_world(data_dir: str, max_wait_minutes: float = 15.0) -> World:
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return build_world(
        np.load(os.path.join(data_dir, "demand.npy")),
        np.load(os.path.join(data_dir, "A_out.npy")),
        np.load(os.path.join(data_dir, "travel_time.npy")),
        np.load(os.path.join(data_dir, "zone_ids.npy")),
        meta["split"]["train"][1],
        max_wait_minutes,
    )


class FleetSimulator:
    """Idle-vehicle bookkeeping for one episode, stepped one 5-minute bin at a time.

    Per bin: begin() brings in vehicles that arrive now, the policy may then reposition
    idle vehicles, and finish() dispatches vehicles to that bin's requests. A request is
    served by an idle vehicle in its own zone (no wait) or else by the nearest idle
    vehicle within `max_wait_minutes`, whose drive time is the rider's wait. Requests with
    no such vehicle are lost. A served trip returns its vehicle to a destination zone
    sampled from the observed flows after the observed mean trip duration.
    """

    def __init__(self, world: World, n_vehicles: int, seed: int = 0):
        self.world = world
        self.n_vehicles = int(n_vehicles)
        init_rng, self.rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
        self.idle = init_rng.multinomial(self.n_vehicles, world.init_weights).astype(np.int64)
        self.pending = np.zeros((BUFFER_BINS, world.n_zones), dtype=np.int64)

    def fleet_total(self) -> int:
        return int(self.idle.sum() + self.pending.sum())

    def arrivals_within(self, t: int, k: int) -> np.ndarray:
        """Vehicles per zone that will become idle in bins t+1 .. t+k."""
        slots = [(t + lag) % BUFFER_BINS for lag in range(1, k + 1)]
        return self.pending[slots].sum(axis=0)

    def begin(self, t: int) -> None:
        slot = t % BUFFER_BINS
        self.idle += self.pending[slot]
        self.pending[slot] = 0

    def reposition(self, t: int, moves: np.ndarray) -> tuple[int, float]:
        """Send idle vehicles; moves[i, j] vehicles leave zone i for zone j."""
        moves = np.asarray(moves, dtype=np.int64).copy()
        np.fill_diagonal(moves, 0)
        if (moves < 0).any() or (moves.sum(axis=1) > self.idle).any():
            raise ValueError("moves must be non-negative and not exceed the idle vehicles in each zone")
        src, dst = np.nonzero(moves)
        if len(src) == 0:
            return 0, 0.0
        if not np.isfinite(self.world.move_minutes[src, dst]).all():
            raise ValueError("moves include unreachable zone pairs")
        count = moves[src, dst]
        self.idle -= moves.sum(axis=1)
        lags = self.world.move_bins[src, dst]
        np.add.at(self.pending, ((t + lags) % BUFFER_BINS, dst), count)
        return int(count.sum()), float((count * self.world.move_minutes[src, dst]).sum())

    def finish(self, t: int, demand_t: np.ndarray) -> dict:
        world = self.world
        demand_t = np.rint(demand_t).astype(np.int64)
        local = np.minimum(self.idle, demand_t)
        self.idle -= local
        remaining = demand_t - local
        taken = {}
        wait_minutes = 0.0
        for zone in self.rng.permutation(np.nonzero(remaining)[0]):
            near = world.neighbors[zone]
            if len(near) == 0:
                continue
            available = self.idle[near]
            before = np.cumsum(available) - available
            take = np.clip(remaining[zone] - before, 0, available)
            used = np.nonzero(take)[0]
            if len(used) == 0:
                continue
            self.idle[near[used]] -= take[used]
            remaining[zone] -= take[used].sum()
            taken[zone] = (near[used], take[used])
            wait_minutes += float((take[used] * world.move_minutes[near[used], zone]).sum())
        borrowed = 0
        for zone in np.nonzero(demand_t - remaining)[0]:
            lags = [np.zeros(local[zone], dtype=np.int64)]
            if zone in taken:
                origins, counts = taken[zone]
                lags.append(np.repeat(world.move_bins[origins, zone], counts))
                borrowed += int(counts.sum())
            lags = np.concatenate(lags)
            dest = np.repeat(np.arange(world.n_zones),
                             self.rng.multinomial(len(lags), world.dest_prob[zone]))
            dest = self.rng.permutation(dest)
            slots = np.minimum(lags + world.trip_bins[zone, dest], BUFFER_BINS - 1)
            np.add.at(self.pending, ((t + slots) % BUFFER_BINS, dest), 1)
        return {"served": int((demand_t - remaining).sum()), "borrowed": borrowed,
                "unmet": int(remaining.sum()), "wait_minutes": wait_minutes}


def run_episode(world: World, n_vehicles: int, seed: int, start: int, end: int,
                policy=None, warmup: int = 36, decision_interval: int = 3) -> dict:
    """Replay bins [start, end). The policy is switched on after `warmup` bins.

    policy(t, sim) returns an [Z, Z] integer move matrix (or None for no moves) and is
    called every `decision_interval` bins. Metrics cover the bins after the warm-up.
    """
    sim = FleetSimulator(world, n_vehicles, seed)
    totals = {"demand": 0, "served": 0, "unmet": 0, "borrowed": 0, "wait_minutes": 0.0,
              "moved": 0, "reposition_minutes": 0.0, "decisions": 0}
    active_from = start + warmup
    for t in range(start, end):
        sim.begin(t)
        if policy is not None and t >= active_from and (t - active_from) % decision_interval == 0:
            moves = policy(t, sim)
            totals["decisions"] += 1
            if moves is not None:
                moved, minutes = sim.reposition(t, moves)
                totals["moved"] += moved
                totals["reposition_minutes"] += minutes
        step = sim.finish(t, world.demand[:, t])
        if t >= active_from:
            totals["served"] += step["served"]
            totals["unmet"] += step["unmet"]
            totals["borrowed"] += step["borrowed"]
            totals["wait_minutes"] += step["wait_minutes"]
            totals["demand"] += step["served"] + step["unmet"]
    demand = max(totals["demand"], 1)
    totals["unmet_rate"] = totals["unmet"] / demand
    totals["mean_wait_served"] = totals["wait_minutes"] / max(totals["served"], 1)
    totals["mean_wait_penalized"] = (totals["wait_minutes"]
                                     + totals["unmet"] * world.max_wait_minutes) / demand
    totals["empty_minutes"] = totals["wait_minutes"] + totals["reposition_minutes"]
    totals["fleet_conserved"] = sim.fleet_total() == sim.n_vehicles
    return totals
