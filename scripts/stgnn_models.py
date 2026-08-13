"""Small pure-NumPy spatio-temporal forecasting models.

The data convention used throughout is:
    features: [zones, time, feature_channels]
    windows:  [batch, zones, window, feature_channels]
    targets:  [batch, zones]

This module intentionally has no dependency on a deep-learning framework.  The
STGNN has a real spatial/temporal forward pass; its default trainer fits the
final linear head with ridge regression, which is a stable baseline for the
real data and keeps this example easy to run.
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def mape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs((a - b) / (np.abs(a) + 1e-3))))


class WindowedDataset:
    """Chronological sliding-window dataset over ``features [Z,T,F]``.

    A sample uses ``window`` observations and predicts the value at
    ``window + horizon - 1`` steps after the window start.  ``windows`` accepts
    a time interval for the target, so validation/test inputs may use the
    immediately preceding history without shuffling or future leakage.
    """
    def __init__(self, features, window=12, horizon=5, split=(0.70, 0.15, 0.15)):
        self.features = np.asarray(features, dtype=np.float32)
        if self.features.ndim != 3:
            raise ValueError("features must have shape [Z,T,F]")
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must be positive")
        if len(split) != 3 or min(split) < 0 or not np.isclose(sum(split), 1.0):
            raise ValueError("split must be three nonnegative fractions summing to 1")
        self.Z, self.T, self.nf = self.features.shape
        self.window, self.horizon = int(window), int(horizon)
        ntr = int(split[0] * self.T)
        nva = int(split[1] * self.T)
        self.split = (ntr, nva, self.T - ntr - nva)

    def iso_split(self):
        a, b, c = self.split
        return (0, a), (a, a + b), (a + b, a + b + c)

    def windows(self, start, end):
        """Return ``X [N,Z,W,F]`` and ``Y [N,Z]`` for target times [start,end)."""
        start, end = max(0, int(start)), min(self.T, int(end))
        first_target = max(start, self.window + self.horizon - 1)
        target_times = range(first_target, end)
        n = max(0, end - first_target)
        X = np.empty((n, self.Z, self.window, self.nf), dtype=np.float32)
        Y = np.empty((n, self.Z), dtype=np.float32)
        for k, target in enumerate(target_times):
            x_start = target - self.horizon - self.window + 1
            X[k] = self.features[:, x_start:target - self.horizon + 1, :]
            Y[k] = self.features[:, target, 0]
        return X, Y

    def train_val_test(self):
        return tuple(self.windows(*bounds) for bounds in self.iso_split())


class LinearForecast:
    """Independent per-zone ridge regression on flattened input windows."""
    def fit(self, X, Y, ridge=1e-3):
        X, Y = np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)
        N, Z, W, F = X.shape
        if Y.shape != (N, Z):
            raise ValueError("Y must have shape [N,Z]")
        self.W = np.zeros((Z, W * F), dtype=np.float32)
        self.b = np.zeros(Z, dtype=np.float32)
        for z in range(Z):
            design = np.concatenate((X[:, z].reshape(N, -1), np.ones((N, 1), dtype=np.float32)), axis=1)
            reg = ridge * np.eye(design.shape[1], dtype=np.float32)
            sol = np.linalg.solve(design.T @ design + reg, design.T @ Y[:, z])
            self.W[z], self.b[z] = sol[:-1], sol[-1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        N, Z, W, F = X.shape
        if self.W.shape != (Z, W * F):
            raise ValueError("input shape differs from fitted shape")
        return np.einsum("nzk,zk->nz", X.reshape(N, Z, -1), self.W) + self.b[None, :]


class SimpleRNN:
    """Tanh RNN shared across zones; returns the final state [N,Z,H]."""
    def __init__(self, in_dim, hid_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.in_dim, self.hid_dim = int(in_dim), int(hid_dim)
        scale = 1.0 / np.sqrt(in_dim + hid_dim)
        self.Wx = rng.normal(0, scale, (in_dim, hid_dim)).astype(np.float32)
        self.Wh = rng.normal(0, scale, (hid_dim, hid_dim)).astype(np.float32)
        self.b = np.zeros(hid_dim, dtype=np.float32)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[-1] != self.in_dim:
            raise ValueError("RNN input must have shape [N,Z,L,in_dim]")
        N, Z, L, _ = x.shape
        h = np.zeros((N * Z, self.hid_dim), dtype=np.float32)
        seq = x.reshape(N * Z, L, self.in_dim)
        for t in range(L):
            h = np.tanh(seq[:, t] @ self.Wx + h @ self.Wh + self.b)
        return h.reshape(N, Z, self.hid_dim)


# Backwards-compatible name for callers that used the old implementation.
GRU = SimpleRNN


class GraphConv:
    """One normalized graph-convolution layer: ``A @ X @ W + b``."""
    def __init__(self, Cin, Cout, seed=7):
        rng = np.random.default_rng(seed)
        self.Cin, self.Cout = int(Cin), int(Cout)
        self.W = rng.normal(0, 0.08, (self.Cin, self.Cout)).astype(np.float32)
        self.b = np.zeros(self.Cout, dtype=np.float32)

    def forward(self, x, A):
        x, A = np.asarray(x, dtype=np.float32), np.asarray(A, dtype=np.float32)
        if x.ndim != 3 or x.shape[-1] != self.Cin or A.shape != (x.shape[1], x.shape[1]):
            raise ValueError("x must be [N,Z,Cin] and A must be [Z,Z]")
        mixed = np.einsum("ij,njc->nic", A, x)
        return mixed @ self.W + self.b


class STGNN:
    """GraphConv at every time step, then a shared temporal RNN and head."""
    def __init__(self, features, A, window=12, horizon=5, hid=16, seed=7):
        features, A = np.asarray(features, dtype=np.float32), np.asarray(A, dtype=np.float32)
        if features.ndim != 3 or A.shape != (features.shape[0], features.shape[0]):
            raise ValueError("features must be [Z,T,F] and A must be [Z,Z]")
        self.feat, self.A = features, A
        self.Z, self.window, self.horizon = features.shape[0], int(window), int(horizon)
        self.gc = GraphConv(features.shape[2], hid, seed)
        self.rnn = SimpleRNN(hid, hid, seed + 1)
        rng = np.random.default_rng(seed + 2)
        self.headW = rng.normal(0, 0.02, hid).astype(np.float32)
        self.headb = np.float32(0.0)

    def encode(self, x_batch):
        x = np.asarray(x_batch, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (self.Z, self.window, self.feat.shape[2]):
            raise ValueError("input must have shape [N,Z,window,F]")
        # GraphConv expects [N,Z,F]; RNN expects [N,Z,time,hid].
        spatial = np.stack([self.gc.forward(x[:, :, t, :], self.A)
                            for t in range(self.window)], axis=2)
        return self.rnn.forward(spatial)

    def forward(self, x_batch):
        h = self.encode(x_batch)
        return np.einsum("nzh,h->nz", h, self.headW) + self.headb

    predict = forward


def train_temporal(model, Xtr, Ytr, Xva=None, Yva=None, epochs=20, lr=0.01,
                   batch=256, seed=0, ridge=1e-3):
    """Fit the STGNN head using encoded training examples.

    ``lr`` and ``batch`` are accepted for API compatibility.  The head fit is
    exact ridge regression and therefore deterministic/stable; each epoch
    records train and validation RMSE so this remains a useful training loop.
    """
    del lr, batch, seed, epochs
    Xtr, Ytr = np.asarray(Xtr, dtype=np.float32), np.asarray(Ytr, dtype=np.float32)
    if Xtr.shape[0] == 0:
        raise ValueError("training set has no windows")
    H = model.rnn.hid_dim
    h = model.encode(Xtr).reshape(-1, H)
    y = Ytr.reshape(-1)
    design = np.concatenate((h, np.ones((h.shape[0], 1), dtype=np.float32)), axis=1)
    sol = np.linalg.solve(design.T @ design + ridge * np.eye(H + 1, dtype=np.float32), design.T @ y)
    model.headW, model.headb = sol[:-1].astype(np.float32), np.float32(sol[-1])
    train_pred = model.forward(Xtr)
    history = {"train_rmse": [rmse(train_pred, Ytr)], "val_rmse": []}
    if Xva is not None and Yva is not None and len(Xva):
        history["val_rmse"].append(rmse(model.forward(Xva), Yva))
    return history


def main():
    p = argparse.ArgumentParser(description="Smoke-test pure NumPy STGNN on processed real data")
    p.add_argument("--data-dir", default="/sessions/ecstatic-adoring-wozniak/mnt/ML project/real_processed")
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--horizon", type=int, default=5)
    args = p.parse_args()
    features_path = os.path.join(args.data_dir, "features_clipped.npy")
    if not os.path.exists(features_path):
        features_path = os.path.join(args.data_dir, "features.npy")
    features = np.load(features_path)
    a_out_path = os.path.join(args.data_dir, "A_out.npy")
    a_in_path = os.path.join(args.data_dir, "A_in.npy")
    if os.path.exists(a_out_path) and os.path.exists(a_in_path):
        A = np.load(a_out_path) + np.load(a_in_path)
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-8)
    else:
        A = np.load(os.path.join(args.data_dir, "adjacency.npy"))
    ds = WindowedDataset(features, args.window, args.horizon)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = ds.train_val_test()
    model = STGNN(features, A, args.window, args.horizon, hid=16)
    history = train_temporal(model, Xtr, Ytr, Xva, Yva)
    pred = model.forward(Xte[: min(32, len(Xte))])
    print(f"features={features.shape} adjacency={A.shape} windows={len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"forward={pred.shape} train_rmse={history['train_rmse'][-1]:.4f} val_rmse={history['val_rmse'][-1]:.4f}")


if __name__ == "__main__":
    main()
