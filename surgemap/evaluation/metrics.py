"""Error metrics in pickup counts and block-bootstrap intervals on the gap between two forecasts."""
from __future__ import annotations

import numpy as np

def rmse_per_horizon(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2, axis=(0, 1)))


def block_bootstrap_gap(pred_a, pred_b, actual, block: int = 36, draws: int = 1000, seed: int = 0):
    """RMSE(b) - RMSE(a) per horizon with a block-bootstrap 95% interval; positive means a is better."""
    rng = np.random.default_rng(seed)
    sse_a = ((pred_a - actual) ** 2).sum(axis=1)               # [N, H]
    sse_b = ((pred_b - actual) ** 2).sum(axis=1)
    n_blocks = int(np.ceil(len(actual) / block))
    pad = n_blocks * block - len(actual)
    blocks_a = np.pad(sse_a, ((0, pad), (0, 0))).reshape(n_blocks, block, -1).sum(axis=1)
    blocks_b = np.pad(sse_b, ((0, pad), (0, 0))).reshape(n_blocks, block, -1).sum(axis=1)
    sizes = np.full(n_blocks, block, dtype=float)
    sizes[-1] = block - pad
    cells = actual.shape[1]
    gaps = []
    for _ in range(draws):
        pick = rng.integers(0, n_blocks, n_blocks)
        denom = sizes[pick].sum() * cells
        gaps.append(np.sqrt(blocks_b[pick].sum(axis=0) / denom) - np.sqrt(blocks_a[pick].sum(axis=0) / denom))
    gaps = np.array(gaps)
    point = np.sqrt(sse_b.sum(axis=0) / (len(actual) * cells)) - np.sqrt(sse_a.sum(axis=0) / (len(actual) * cells))
    return point, np.percentile(gaps, 2.5, axis=0), np.percentile(gaps, 97.5, axis=0)
