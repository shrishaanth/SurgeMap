from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_stgnn_torch import DirectedGraphConv

HORIZONS = (1, 3, 6, 12)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiHorizonDataset(Dataset):
    def __init__(self, features: np.ndarray, start: int, end: int,
                 window: int = 48, horizons=HORIZONS):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.start, self.end = int(start), int(end)
        self.window = int(window)
        self.horizons = tuple(int(h) for h in horizons)
        if self.features.ndim != 3:
            raise ValueError("features must have shape [Z,T,F]")
        if self.window < 1 or not self.horizons or min(self.horizons) < 1:
            raise ValueError("window and horizons must be positive")
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
        return x, y


class MultiHorizonSTGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, horizons=HORIZONS):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.spatial = DirectedGraphConv(n_features, hidden)
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in self.horizons)

    def encode(self, x, a_out, a_in):
        spatial = self.spatial(x, a_out, a_in)
        b, z, w, hidden = spatial.shape
        encoded, _ = self.temporal(spatial.reshape(b * z, w, hidden))
        return encoded[:, -1, :].reshape(b, z, hidden)

    def forward(self, x, a_out, a_in):
        encoded = self.encode(x, a_out, a_in)
        return torch.stack([head(encoded).squeeze(-1) for head in self.heads], dim=-1)


def run_epoch(model, loader, a_out, a_in, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count, predictions, targets = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, a_out, a_in)
        loss = nn.functional.mse_loss(pred, y)
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
    args = parser.parse_args()
    seed_all(args.seed)

    data_dir = args.data_dir
    features_path = os.path.join(data_dir, "features_clipped.npy")
    if not os.path.exists(features_path):
        features_path = os.path.join(data_dir, "features.npy")
    features = np.load(features_path)
    a_out_np = np.load(os.path.join(data_dir, "A_out.npy"))
    a_in_np = np.load(os.path.join(data_dir, "A_in.npy"))
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    bounds = split_bounds(meta, features.shape[1])
    datasets = [MultiHorizonDataset(features, *bound, args.window) for bound in bounds]
    loaders = [DataLoader(ds, args.batch_size, shuffle=False) for ds in datasets]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_out = torch.as_tensor(a_out_np, dtype=torch.float32, device=device)
    a_in = torch.as_tensor(a_in_np, dtype=torch.float32, device=device)
    model = MultiHorizonSTGNN(features.shape[2], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_epoch, wait, history = float("inf"), 0, 0, []
    print(f"device={device} features={features.shape} split=" + "/".join(map(str, map(len, datasets))))
    for epoch in range(1, args.epochs + 1):
        train_rmse, _, _ = run_epoch(model, loaders[0], a_out, a_in, device, optimizer)
        val_rmse, _, _ = run_epoch(model, loaders[1], a_out, a_in, device)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})
        print(f"epoch={epoch:03d} train_rmse={train_rmse:.5f} val_rmse={val_rmse:.5f}")
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
