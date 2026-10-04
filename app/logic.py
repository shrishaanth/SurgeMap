"""Streamlit-free helpers for the SurgeMap app, kept separate so they can be unit tested."""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN = pd.Timedelta(minutes=5)

# Light-to-dark sequential gradient (ColorBrewer YlOrRd)
SEQUENTIAL = np.array([(255, 255, 204), (254, 217, 118), (253, 141, 60), (227, 26, 28), (128, 0, 38)], float)
DIVERGING_LOW = np.array((49, 130, 189), float)
DIVERGING_MID = np.array((247, 247, 247), float)
DIVERGING_HIGH = np.array((222, 45, 38), float)


def sequential_colors(values: np.ndarray, vmax: float, alpha: int = 200) -> np.ndarray:
    """RGBA per value; sqrt-compressed so quiet zones stay distinguishable from busy ones."""
    t = np.sqrt(np.clip(np.asarray(values, float) / max(vmax, 1e-9), 0.0, 1.0))
    pos = t * (len(SEQUENTIAL) - 1)
    lo = np.minimum(np.floor(pos).astype(int), len(SEQUENTIAL) - 2)
    frac = (pos - lo)[:, None]
    rgb = SEQUENTIAL[lo] * (1 - frac) + SEQUENTIAL[lo + 1] * frac
    return np.column_stack([rgb, np.full(len(rgb), alpha)]).astype(np.uint8)


def diverging_colors(values: np.ndarray, vmax: float, alpha: int = 200) -> np.ndarray:
    """Blue for negative, grey-white at zero, red for positive."""
    t = np.clip(np.asarray(values, float) / max(vmax, 1e-9), -1.0, 1.0)[:, None]
    pos = DIVERGING_MID * (1 - t) + DIVERGING_HIGH * t
    neg = DIVERGING_MID * (1 + t) + DIVERGING_LOW * (-t)
    rgb = np.where(t >= 0, pos, neg)
    return np.column_stack([rgb, np.full(len(rgb), alpha)]).astype(np.uint8)


def anchor_index(first: pd.Timestamp, when: pd.Timestamp, n: int) -> int:
    return int(np.clip(round((pd.Timestamp(when) - pd.Timestamp(first)) / BIN), 0, n - 1))


def ranks_desc(values: np.ndarray) -> np.ndarray:
    """1 for the largest value; ties keep their original order."""
    order = np.argsort(-np.asarray(values, float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def hotspot_table(pred: np.ndarray, actual: np.ndarray, zone_ids, names: dict, k: int):
    """Top-k predicted zones with their actual rank, plus the number that are real top-k zones."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    actual_rank = ranks_desc(actual)
    top = np.argsort(-pred, kind="stable")[:k]
    rows = []
    for z in top:
        zone, borough = names.get(int(zone_ids[z]), ["Zone %d" % zone_ids[z], ""])
        rows.append({"Zone": zone, "Borough": borough, "Predicted": round(float(pred[z]), 1),
                     "Actual": int(actual[z]), "Actual rank": int(actual_rank[z]),
                     "Hit": bool(actual_rank[z] <= k)})
    table = pd.DataFrame(rows)
    return table, int(table["Hit"].sum()) if len(table) else 0


def geometry_centroid(geometry: dict) -> tuple[float, float]:
    """Mean of the outer-ring vertices of the largest polygon (adequate for drawing moves)."""
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    outer = max((np.asarray(p[0], float) for p in polygons), key=len)
    return float(outer[:, 0].mean()), float(outer[:, 1].mean())


def moves_to_arcs(moves: np.ndarray, zone_ids, centroids: dict, names: dict) -> pd.DataFrame:
    """One row per (source, destination) pair with at least one vehicle and known map positions."""
    src, dst = np.nonzero(moves)
    rows = []
    for i, j in zip(src, dst):
        a, b = centroids.get(int(zone_ids[i])), centroids.get(int(zone_ids[j]))
        if a is None or b is None:
            continue
        rows.append({"from_zone": names.get(int(zone_ids[i]), ["?"])[0],
                     "to_zone": names.get(int(zone_ids[j]), ["?"])[0],
                     "vehicles": int(moves[i, j]), "from_lon": a[0], "from_lat": a[1],
                     "to_lon": b[0], "to_lat": b[1]})
    return pd.DataFrame(rows, columns=["from_zone", "to_zone", "vehicles", "from_lon", "from_lat",
                                       "to_lon", "to_lat"])
