from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgemap.data.windows import HORIZONS, MultiHorizonDataset, prepare_inputs, split_bounds
from surgemap.models.baselines import zscore_params
from surgemap.models.build import build_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zone_weights(data_dir: str, train_end: int, power: float) -> np.ndarray:
    """Loss weight per zone, ((1 + mean pickups) * sigma) ** power, scaled to mean 1.

    A z-score error of dz moves the pickup count by about (1 + count) * sigma * dz, so
    power 2 approximates squared error in pickups and power 0 is the unweighted loss.
    """
    demand = np.load(os.path.join(data_dir, "demand.npy")).astype(np.float64)
    _, sigma = zscore_params(demand, train_end)
    weight = ((1.0 + demand[:, :train_end].mean(axis=1)) * sigma) ** power
    return (weight / weight.mean()).astype(np.float32)


def poisson_loss(pred, y, mu, sigma):
    """Poisson negative log-likelihood of the pickup counts, for a network that predicts
    standardised log demand. mu and sigma are the per-zone standardisation parameters."""
    log1p_rate = (pred * sigma[None, :, None] + mu[None, :, None]).clamp(max=9.0)
    rate = torch.expm1(log1p_rate).clamp(min=1e-3)
    counts = torch.expm1(y * sigma[None, :, None] + mu[None, :, None]).clamp(min=0.0)
    return (rate - counts * torch.log(rate)).mean()


def run_epoch(model, loader, a_out, a_in, device, optimizer=None, weight=None, count_stats=None,
              scaler=None):
    """One pass over the loader. Returns (objective, predictions, targets); the objective is
    the root of the (weighted) squared error, or the mean Poisson loss when count_stats is given.
    With a GradScaler the forward pass runs in mixed precision; the loss stays in float32."""
    training = optimizer is not None
    amp = scaler is not None
    model.train(training)
    total, count, predictions, targets = 0.0, 0, [], []
    for x, y, prior, lagged in loader:
        x, y, prior, lagged = x.to(device), y.to(device), prior.to(device), lagged.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(x, a_out, a_in, prior, lagged)
        pred = pred.float()
        if count_stats is not None:
            loss = poisson_loss(pred, y, *count_stats)
        else:
            squared = (pred - y) ** 2
            loss = squared.mean() if weight is None else (squared * weight[None, :, None]).mean()
        if training:
            optimizer.zero_grad()
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
        predictions.append(pred.detach().cpu())
        targets.append(y.detach().cpu())
    if not predictions:
        raise ValueError("dataset has no windows; reduce window or check metadata splits")
    mean = total / max(count, 1)
    objective = mean if count_stats is not None else float(np.sqrt(mean))
    return float(objective), torch.cat(predictions), torch.cat(targets)


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
    parser.add_argument("--layers", type=int, default=1, help="number of GRU layers")
    parser.add_argument("--lag-features", action="store_true",
                        help="give each head the target bin's demand a day and a week earlier")
    parser.add_argument("--loss", choices=("mse", "poisson"), default="mse",
                        help="poisson: likelihood of the pickup counts instead of squared error in log units")
    parser.add_argument("--arch", choices=("gru", "gwnet"), default="gru",
                        help="gru: graph conv then GRU; gwnet: dilated convolutions alternating with graph convs")
    parser.add_argument("--adaptive-dim", type=int, default=10,
                        help="gwnet: embedding size of the learned adjacency (0 = off)")
    parser.add_argument("--channels", type=int, default=32, help="gwnet: width of the residual layers")
    parser.add_argument("--end-channels", type=int, default=128, help="gwnet: width of the output layers")
    parser.add_argument("--dilations", default="1,2,4,8,16,1,2,4", help="gwnet: one dilation per layer")
    parser.add_argument("--gcn-order", type=int, default=2, help="gwnet: hops per graph convolution")
    parser.add_argument("--dropout", type=float, default=0.1, help="gwnet: dropout after each graph conv")
    parser.add_argument("--identity-dim", type=int, default=0,
                        help="gwnet: size of the learned zone, time-of-day and weekday embeddings (0 = off)")
    parser.add_argument("--amp", action="store_true", help="mixed precision on the GPU (faster, less memory)")
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
    features, prior, lagged = prepare_inputs(data_dir, features, bounds[0][1], HORIZONS, args.weather,
                                             args.prior, args.lag_features)
    datasets = [MultiHorizonDataset(features, *bound, args.window, prior=prior, lagged=lagged) for bound in bounds]
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
    count_stats = None
    if args.loss == "poisson":
        demand = np.load(os.path.join(data_dir, "demand.npy")).astype(np.float64)
        count_stats = tuple(torch.as_tensor(v, dtype=torch.float32, device=device)
                            for v in zscore_params(demand, bounds[0][1]))
    model = build_model(vars(args), features.shape[2], features.shape[0],
                        lag_dim=0 if lagged is None else lagged.shape[-1]).to(device)
    scaler = torch.amp.GradScaler() if args.amp and device.type == "cuda" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_epoch, wait, history = float("inf"), 0, 0, []
    print(f"device={device} features={features.shape} split=" + "/".join(map(str, map(len, datasets)))
          + f" arch={args.arch} parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        train_rmse, _, _ = run_epoch(model, loaders[0], a_out, a_in, device, optimizer, weight, count_stats,
                                     scaler)
        val_rmse, _, _ = run_epoch(model, loaders[1], a_out, a_in, device, weight=weight, count_stats=count_stats)
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
