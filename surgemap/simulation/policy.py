from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

TRIP_BINS = 3


def window_demand(pred: np.ndarray, horizons, lookahead: int) -> np.ndarray:
    """Expected requests per zone over the next `lookahead` bins.

    pred is [Z, H], the forecast of bin t+h-1 for each horizon h. Bins between forecast
    horizons are linearly interpolated; bins past the last horizon repeat its value.
    """
    offsets = np.asarray(horizons, dtype=np.float64) - 1.0
    total = np.zeros(pred.shape[0])
    for b in range(lookahead):
        if b <= offsets[0]:
            total += pred[:, 0]
        elif b >= offsets[-1]:
            total += pred[:, -1]
        else:
            hi = int(np.searchsorted(offsets, b))
            lo = hi - 1
            w = (b - offsets[lo]) / (offsets[hi] - offsets[lo])
            total += (1.0 - w) * pred[:, lo] + w * pred[:, hi]
    return np.clip(total, 0.0, None)


class ArrayForecast:
    """Serves forecasts stored as [N, Z, H], one row per anchor bin first_anchor + n."""

    def __init__(self, pred: np.ndarray, first_anchor: int):
        self.pred, self.first_anchor = pred, int(first_anchor)

    def __call__(self, t: int):
        n = t - self.first_anchor
        return self.pred[n] if 0 <= n < len(self.pred) else None


def oracle_forecast(demand: np.ndarray, anchors: np.ndarray, horizons) -> ArrayForecast:
    pred = np.stack([demand[:, anchors + h - 1].T for h in horizons], axis=-1)
    return ArrayForecast(pred.astype(np.float64), anchors[0])


def persistence_forecast(demand: np.ndarray, anchors: np.ndarray, horizons) -> ArrayForecast:
    last = demand[:, anchors - 1].T
    return ArrayForecast(np.repeat(last[..., None], len(horizons), axis=-1).astype(np.float64), anchors[0])


def round_rows(moves: np.ndarray) -> np.ndarray:
    """Round a non-negative matrix to integers, preserving each row sum (largest remainder)."""
    floors = np.floor(moves + 1e-9)
    out = floors.astype(np.int64)
    target = np.rint(moves.sum(axis=1)).astype(np.int64)
    for row in np.nonzero(target - out.sum(axis=1) > 0)[0]:
        need = int(target[row] - out[row].sum())
        order = np.argsort(-(moves[row] - floors[row]))[:need]
        out[row, order] += 1
    return out


class LPRepositioner:
    """Forecast-driven repositioning as a small min-cost flow solved every decision step.

    Variables: m[i, j] idle vehicles sent from i to j, y[k, i] requests in zone i served by
    vehicles in k (rider wait = driving time k -> i), and u[i] requests left unserved.

        minimise  theta * sum(m * drive_minutes) + sum(y * wait_minutes) + penalty * sum(u)
        s.t.      y[:, i].sum() + u[i] = forecast_i
                  sum_i y[k, i] <= capacity * (idle_k + arriving_k - out_k + in_k)
                  out_k <= idle_k

    Moves are limited to trips that finish inside the look-ahead window, so they can be
    credited to the window's supply. theta trades fleet driving against rider wait.
    """

    def __init__(self, world, forecast, horizons, theta: float = 0.1, lookahead: int = 3,
                 move_radius_minutes: float | None = None, unmet_penalty: float | None = None,
                 min_demand: float = 0.05):
        self.world, self.forecast, self.horizons = world, forecast, tuple(horizons)
        self.theta, self.lookahead, self.min_demand = float(theta), int(lookahead), float(min_demand)
        self.capacity = max(1.0, self.lookahead / TRIP_BINS)
        self.penalty = float(unmet_penalty if unmet_penalty is not None else world.max_wait_minutes)
        radius = float(move_radius_minutes if move_radius_minutes is not None else 5 * self.lookahead)
        n = world.n_zones
        reachable = (world.move_minutes <= radius) & ~np.eye(n, dtype=bool)
        self.move_src, self.move_dst = np.nonzero(reachable)
        self.move_cost = world.move_minutes[self.move_src, self.move_dst]
        serve_k, serve_i = [], []
        for zone in range(n):
            ks = np.concatenate([[zone], world.neighbors[zone]]).astype(np.int64)
            serve_k.append(ks)
            serve_i.append(np.full(len(ks), zone, dtype=np.int64))
        self.serve_k, self.serve_i = np.concatenate(serve_k), np.concatenate(serve_i)
        self.serve_cost = np.where(self.serve_k == self.serve_i, 0.0,
                                   world.move_minutes[self.serve_k, self.serve_i])

    def __call__(self, t: int, sim):
        pred = self.forecast(t)
        if pred is None:
            return None
        n = self.world.n_zones
        demand = window_demand(pred, self.horizons, self.lookahead)
        idle = sim.idle
        supply = idle + sim.arrivals_within(t, self.lookahead - 1)

        keep_y = demand[self.serve_i] >= self.min_demand
        yk, yi, y_cost = self.serve_k[keep_y], self.serve_i[keep_y], self.serve_cost[keep_y]
        if len(yk) == 0 or idle.sum() == 0:
            return None
        useful = np.zeros(n, dtype=bool)
        useful[yk] = True
        keep_m = (idle[self.move_src] > 0) & useful[self.move_dst]
        ms, md, m_cost = self.move_src[keep_m], self.move_dst[keep_m], self.move_cost[keep_m]
        n_m, n_y = len(ms), len(yk)
        demand_zones = np.unique(yi)
        n_u = len(demand_zones)

        m_cols, y_cols = np.arange(n_m), n_m + np.arange(n_y)
        u_cols = n_m + n_y + np.arange(n_u)
        rows, cols, vals = [], [], []

        rows += [yk, ms, md]
        cols += [y_cols, m_cols, m_cols]
        vals += [np.ones(n_y), np.full(n_m, self.capacity), np.full(n_m, -self.capacity)]
        rows += [n + ms]
        cols += [m_cols]
        vals += [np.ones(n_m)]
        a_ub = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(2 * n, n_m + n_y + n_u)).tocsr()
        b_ub = np.concatenate([self.capacity * supply, idle]).astype(np.float64)

        eq_rows = np.concatenate([np.searchsorted(demand_zones, yi), np.arange(n_u)])
        eq_cols = np.concatenate([y_cols, u_cols])
        a_eq = coo_matrix((np.ones(n_y + n_u), (eq_rows, eq_cols)),
                          shape=(n_u, n_m + n_y + n_u)).tocsr()
        b_eq = demand[demand_zones]

        cost = np.concatenate([self.theta * m_cost, y_cost, np.full(n_u, self.penalty)])
        result = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                         bounds=(0, None), method="highs")
        if not result.success:
            return None
        moves = np.zeros((n, n))
        moves[ms, md] = result.x[:n_m]
        moves = round_rows(moves)
        return moves if moves.any() else None
