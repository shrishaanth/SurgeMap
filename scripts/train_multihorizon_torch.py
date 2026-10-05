from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from accuracy_checks import histavg_for_bins, zscore_params
from train_stgnn_torch import DirectedGraphConv

HORIZONS = (1, 3, 6, 12)
CLIP = 6.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiHorizonDataset(Dataset):
    """Windows of features with multi-horizon targets and an optional per-target prior.

    Each item is (x [Z, W, F], y [Z, H], prior [Z, H]); the prior is zero when none is given.
    """

    def __init__(self, features: np.ndarray, start: int, end: int,
                 window: int = 48, horizons=HORIZONS, prior: np.ndarray | None = None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.start, self.end = int(start), int(end)
        self.window = int(window)
        self.horizons = tuple(int(h) for h in horizons)
        if self.features.ndim != 3:
            raise ValueError("features must have shape [Z,T,F]")
        if self.window < 1 or not self.horizons or min(self.horizons) < 1:
            raise ValueError("window and horizons must be positive")
        if prior is not None and prior.shape != (features.shape[0], features.shape[1], len(self.horizons)):
            raise ValueError("prior must have shape [Z,T,H]")
        self.prior = None if prior is None else torch.as_tensor(prior, dtype=torch.float32)
        self._no_prior = torch.zeros(features.shape[0], len(self.horizons))
        first = max(self.start, self.window)
        last = self.end - max(self.horizons)
        self.times = list(range(first, max(first, last + 1)))
        self.times = [t for t in self.times if t + max(self.horizons) - 1 < self.end]

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        t = self.times[index]
        x = self.features[:, t - self.window:t, :]
        y = torch.stack([self.features[:, t + h - 1, 0] for h in self.horizons], dim=-1)
        prior = self._no_prior if self.prior is None else self.prior[:, t, :]
        return x, y, prior


class MultiHorizonSTGNN(nn.Module):
    """Graph convolution per time step, a GRU over time, and one linear head per horizon.

    With zone_dim > 0 each zone gets a learned embedding that is fed to the heads, so the
    network can tell zones apart. A prior passed to forward() is added to the output, so the
    network then learns the residual over it.
    """

    def __init__(self, n_features: int, hidden: int = 64, horizons=HORIZONS,
                 n_zones: int = 0, zone_dim: int = 0):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.spatial = DirectedGraphConv(n_features, hidden)
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.zone_embedding = nn.Embedding(n_zones, zone_dim) if zone_dim > 0 else None
        self.heads = nn.ModuleList(nn.Linear(hidden + zone_dim, 1) for _ in self.horizons)

    def encode(self, x, a_out, a_in):
        spatial = self.spatial(x, a_out, a_in)
        b, z, w, hidden = spatial.shape
        encoded, _ = self.temporal(spatial.reshape(b * z, w, hidden))
        return encoded[:, -1, :].reshape(b, z, hidden)

    def forward(self, x, a_out, a_in, prior=None):
        encoded = self.encode(x, a_out, a_in)
        if self.zone_embedding is not None:
            zones = self.zone_embedding.weight.unsqueeze(0).expand(encoded.shape[0], -1, -1)
            encoded = torch.cat([encoded, zones], dim=-1)
        out = torch.stack([head(encoded).squeeze(-1) for head in self.heads], dim=-1)
        return out if prior is None else out + prior


def prepare_inputs(data_dir: str, features: np.ndarray, train_end: int, horizons,
                   use_weather: bool = False, prior_kind: str = "none"):
    """Optional extra inputs: weather channels appended to the features, and a prior [Z, T, H].

    The prior is a time-of-day average for each target bin, in the standardised log units the
    network predicts. "histavg_z" averages the standardised log demand itself, which is what
    the squared-error target calls for. "histavg" standardises the log of the average count,
    which sits above the average of the logs (most for quiet zones); it is kept only so that
    checkpoints trained with it still load. Bins inside the train split exclude their own
    value, so the prior does not leak the target.
    """
    prior = None
    if use_weather:
        weather = np.load(os.path.join(data_dir, "weather.npy")).astype(np.float32)
        tiled = np.broadcast_to(weather[None], (features.shape[0],) + weather.shape)
        features = np.concatenate([features, tiled], axis=2)
    if prior_kind in ("histavg", "histavg_z"):
        demand = np.load(os.path.join(data_dir, "demand.npy")).astype(np.float64)
        times = np.load(os.path.join(data_dir, "times.npy"))
        total = demand.shape[1]
        mu, sigma = zscore_params(demand, train_end)
        if prior_kind == "histavg_z":
            z = np.clip((np.log1p(demand) - mu[:, None]) / sigma[:, None], -CLIP, CLIP)
            hist_z = histavg_for_bins(z, times, train_end, np.arange(total))
        else:
            hist = histavg_for_bins(demand, times, train_end, np.arange(total))
            hist_z = np.clip((np.log1p(hist) - mu[:, None]) / sigma[:, None], -CLIP, CLIP)
        prior = np.stack([hist_z[:, np.minimum(np.arange(total) + h - 1, total - 1)] for h in horizons],
                         axis=-1).astype(np.float32)
    elif prior_kind != "none":
        raise ValueError(f"unknown prior: {prior_kind}")
    return features, prior


def zone_weights(data_dir: str, train_end: int, power: float) -> np.ndarray:
    """Loss weight per zone, ((1 + mean pickups) * sigma) ** power, scaled to mean 1.

    A z-score error of dz moves the pickup count by about (1 + count) * sigma * dz, so
    power 2 approximates squared error in pickups and power 0 is the unweighted loss.
    """
    demand = np.load(os.path.join(data_dir, "demand.npy")).astype(np.float64)
    _, sigma = zscore_params(demand, train_end)
    weight = ((1.0 + demand[:, :train_end].mean(axis=1)) * sigma) ** power
    return (weight / weight.mean()).astype(np.float32)


def run_epoch(model, loader, a_out, a_in, device, optimizer=None, weight=None):
    training = optimizer is not None
    model.train(training)
    total, count, predictions, targets = 0.0, 0, [], []
    for x, y, prior in loader:
        x, y, prior = x.to(device), y.to(device), prior.to(device)
        pred = model(x, a_out, a_in, prior)
        squared = (pred - y) ** 2
        loss = squared.mean() if weight is None else (squared * weight[None, :, None]).mean()
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
        predictions.append(pred.detach().cpu())
        targets.append(y.detach().cpu())
    if not predictions:
        raise ValueError("dataset has no windows; reduce window or check metadata splits")
    return float(np.sqrt(total / max(count, 1))), torch.cat(predictions), torch.cat(targets)


def scores(pred, target):
    result = {}
    for i, horizon in enumerate(HORIZONS):
        p, y = pred[..., i], target[..., i]
        actual_top3 = torch.topk(y, k=min(3, y.shape[1]), dim=1).indices
        predicted_top3 = torch.topk(p, k=min(3, p.shape[1]), dim=1).indices
        actual_top5 = torch.topk(y, k=min(5, y.shape[1]), dim=1).indices
        predicted_top5 = torch.topk(p, k=min(5, p.shape[1]), dim=1).indices
        hit3 = (predicted_top3.unsqueeze(-1) == actual_top3.unsqueeze(1)).any(dim=(1, 2)).float().mean()
        overlap5 = (predicted_top5.unsqueeze(-1) == actual_top5.unsqueeze(1)).any(dim=1).float().sum(dim=1)
        result[str(horizon)] = {
            "minutes": horizon * 5,
            "rmse": float(torch.sqrt(torch.mean((p - y) ** 2))),
            "mae": float(torch.mean(torch.abs(p - y))),
            "hotspot_top3_hit_rate": float(hit3),
            "hotspot_top5_recall": float((overlap5 / actual_top5.shape[1]).mean()),
        }
    return result


def split_bounds(meta, total):
    split = meta.get("split", {})
    train_default = int(.70 * total)
    val_default = int(.85 * total)
    train = split.get("train")
    validation = split.get("validation")
    test = split.get("test")
    train_end = int(split.get("train_end", train[1] if train else train_default))
    val_end = int(split.get("val_end", validation[1] if validation else val_default))
    if train is None:
        train = [0, train_end]
    if validation is None:
        validation = [train_end, val_end]
    if test is None:
        test = [val_end, total]
    return ((int(train[0]), int(train[1])),
            (int(validation[0]), int(validation[1])),
            (int(test[0]), int(test[1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--out", default="multihorizon_stgnn_checkpoint.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle", action="store_true", help="shuffle training windows each epoch")
    parser.add_argument("--prior", choices=("none", "histavg_z", "histavg"), default="none",
                        help="predict the residual over the time-of-day average of each target bin")
    parser.add_argument("--zone-dim", type=int, default=0, help="size of a learned per-zone embedding (0 = off)")
    parser.add_argument("--weather", action="store_true", help="append the weather channels to the inputs")
    parser.add_argument("--loss-power", type=float, default=0.0,
                        help="weight zones by ((1 + mean pickups) * sigma) ** power; 2 approximates count error")
    parser.add_argument("--no-graph", action="store_true", help="ablation: remove the neighbour terms")
    args = parser.parse_args()
    seed_all(args.seed)

    data_dir = args.data_dir
    features_path = os.path.join(data_dir, "features_clipped.npy")
    if not os.path.exists(features_path):
        features_path = os.path.join(data_dir, "features.npy")
    features = np.load(features_path)
    a_out_np = np.load(os.path.join(data_dir, "A_out.npy"))
    a_in_np = np.load(os.path.join(data_dir, "A_in.npy"))
    if args.no_graph:
        a_out_np, a_in_np = np.zeros_like(a_out_np), np.zeros_like(a_in_np)
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    bounds = split_bounds(meta, features.shape[1])
    features, prior = prepare_inputs(data_dir, features, bounds[0][1], HORIZONS, args.weather, args.prior)
    datasets = [MultiHorizonDataset(features, *bound, args.window, prior=prior) for bound in bounds]
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(datasets[0], args.batch_size, shuffle=args.shuffle, generator=generator),
               DataLoader(datasets[1], args.batch_size, shuffle=False),
               DataLoader(datasets[2], args.batch_size, shuffle=False)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_out = torch.as_tensor(a_out_np, dtype=torch.float32, device=device)
    a_in = torch.as_tensor(a_in_np, dtype=torch.float32, device=device)
    weight = None
    if args.loss_power > 0:
        weight = torch.as_tensor(zone_weights(data_dir, bounds[0][1], args.loss_power), device=device)
    model = MultiHorizonSTGNN(features.shape[2], args.hidden, n_zones=features.shape[0],
                              zone_dim=args.zone_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_epoch, wait, history = float("inf"), 0, 0, []
    print(f"device={device} features={features.shape} split=" + "/".join(map(str, map(len, datasets))))
    for epoch in range(1, args.epochs + 1):
        train_rmse, _, _ = run_epoch(model, loaders[0], a_out, a_in, device, optimizer, weight)
        val_rmse, _, _ = run_epoch(model, loaders[1], a_out, a_in, device, weight=weight)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})
        print(f"epoch={epoch:03d} train_rmse={train_rmse:.5f} val_rmse={val_rmse:.5f}", flush=True)
        if val_rmse < best:
            best, best_epoch, wait = val_rmse, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "horizons": HORIZONS,
                        "best_val_rmse": best, "best_epoch": best_epoch}, args.out)
        else:
            wait += 1
            if wait >= args.patience:
                print("early stopping")
                break
    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, pred, target = run_epoch(model, loaders[2], a_out, a_in, device)
    metrics = {"config": vars(args), "horizons": HORIZONS, "best_val_rmse": best,
               "best_epoch": best_epoch, "test": scores(pred, target), "history": history}
    print("horizon,bins,minutes,rmse,mae,top3_hit_rate,top5_recall")
    for h, values in metrics["test"].items():
        print(f"{h},{h},{values['minutes']},{values['rmse']:.6f},{values['mae']:.6f},"
              f"{values['hotspot_top3_hit_rate']:.6f},{values['hotspot_top5_recall']:.6f}")
    metrics_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved_checkpoint={args.out}")
    print(f"saved_metrics={metrics_path}")


if __name__ == "__main__":
    main()
