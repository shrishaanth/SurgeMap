"""Trainable PyTorch ST-GNN for NYC taxi demand forecasting.

This is the actual model needed to replace the earlier random-encoder NumPy proof:
all graph, temporal, and head parameters receive gradients from the forecast loss.
It deliberately uses vanilla PyTorch (not PyG) so it is easier to install and run;
the graph operation is explicit A @ X @ W and supports directed A_out/A_in.

Input data is produced by preprocess.py:
  features.npy [Z,T,F], A_out.npy [Z,Z], A_in.npy [Z,Z]

Model:
  h_t = ReLU(A_out X_t W_out + A_in X_t W_in + b_g)
  q_t = GRU(h_t, q_{t-1})
  y_hat = q_last W_head + b_head

Run:
  python train_stgnn_torch.py --data-dir real_processed_fixed --epochs 50
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class WindowDataset(Dataset):
    """Chronological target indices; each input uses only historical bins."""
    def __init__(self, features, start, end, window=12, horizon=5):
        self.x = torch.as_tensor(features, dtype=torch.float32)
        self.start, self.end = int(start), int(end)
        self.window, self.horizon = int(window), int(horizon)
        self.first = max(self.start, self.window + self.horizon - 1)
        self.targets = list(range(self.first, self.end))
    def __len__(self): return len(self.targets)
    def __getitem__(self, k):
        target = self.targets[k]
        x_end = target - self.horizon + 1
        x = self.x[:, x_end - self.window:x_end, :]  # [Z,W,F]
        y = self.x[:, target, 0]                      # [Z]
        return x, y


class DirectedGraphConv(nn.Module):
    """Trainable directed message-passing layer, batched over the W time axis.

    A_out and A_in are fixed topology/weights from data; W_out and W_in are learned.
    There is no random frozen encoder: every parameter below is in autograd's graph.

    Input  x: [B, Z, W, F]   (W time slices)
    Output  : [B, Z, W, H]
    The neighbour mix is done for ALL W slices in a single einsum, avoiding the
    per-timestep Python loop that dominated CPU runtime at long windows.
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.w_out = nn.Linear(in_channels, hidden_channels, bias=False)
        self.w_in = nn.Linear(in_channels, hidden_channels, bias=False)
        self.self_loop = nn.Linear(in_channels, hidden_channels)

    def forward(self, x, a_out, a_in):
        # x: [B,Z,W,F]; out: [B,Z,W,H]
        # Mix the ZONE dimension while preserving time and feature channels.
        # A[i,j] transports messages from source zone j to destination zone i.
        out = torch.einsum("ij,bjwf->biwf", a_out, x)
        inc = torch.einsum("ij,bjwf->biwf", a_in, x)
        h = self.w_out(out) + self.w_in(inc) + self.self_loop(x)
        return torch.relu(h)


class STGNN(nn.Module):
    def __init__(self, n_features, hidden=64, horizon=5):
        super().__init__()
        self.spatial = DirectedGraphConv(n_features, hidden)
        self.temporal = nn.GRU(input_size=hidden, hidden_size=hidden,
                               num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.horizon = horizon

    def forward(self, x, a_out, a_in):
        # x [B,Z,W,F] -> spatially mixed [B,Z,W,H] (all steps at once).
        spatial = self.spatial(x, a_out, a_in)      # [B,Z,W,H]
        B, Z, W, H = spatial.shape
        seq = spatial.reshape(B * Z, W, H)
        encoded, _ = self.temporal(seq)
        last = encoded[:, -1, :].reshape(B, Z, H)
        return self.head(last).squeeze(-1)          # [B,Z]


def batches(model, loader, a_out, a_in, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    preds, ys = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, a_out, a_in)
        loss = nn.functional.mse_loss(pred, y)
        if training:
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * len(x); count += len(x)
        preds.append(pred.detach().cpu()); ys.append(y.detach().cpu())
    pred = torch.cat(preds); y = torch.cat(ys)
    return float(np.sqrt(total / max(count, 1))), pred, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="stgnn_checkpoint.pt")
    args = p.parse_args(); seed_all(args.seed)
    d = args.data_dir
    features_path = os.path.join(d, "features_clipped.npy")
    if not os.path.exists(features_path):
        features_path = os.path.join(d, "features.npy")
    features = np.load(features_path)
    a_out_np = np.load(os.path.join(d, "A_out.npy"))
    a_in_np = np.load(os.path.join(d, "A_in.npy"))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    T = features.shape[1]
    train_end = int(meta.get("split", {}).get("train_end", 0)) or int(.70*T)
    val_end = int(meta.get("split", {}).get("val_end", 0)) or int(.85*T)
    train = WindowDataset(features, 0, train_end, args.window, args.horizon)
    val = WindowDataset(features, train_end, val_end, args.window, args.horizon)
    test = WindowDataset(features, val_end, T, args.window, args.horizon)
    loaders = [DataLoader(z, args.batch_size, shuffle=False) for z in (train, val, test)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGNN(features.shape[2], args.hidden, args.horizon).to(device)
    a_out = torch.as_tensor(a_out_np, dtype=torch.float32, device=device)
    a_in = torch.as_tensor(a_in_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, wait, history = float("inf"), 0, []
    print(f"device={device} features={features.shape} split={len(train)}/{len(val)}/{len(test)}")
    print("parameter_count=", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(1, args.epochs + 1):
        tr, _, _ = batches(model, loaders[0], a_out, a_in, device, optimizer)
        va, _, _ = batches(model, loaders[1], a_out, a_in, device)
        history.append({"epoch": epoch, "train_rmse": tr, "val_rmse": va})
        print(f"epoch={epoch:03d} train_rmse={tr:.5f} val_rmse={va:.5f}")
        if va < best:
            best, wait = va, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "best_val_rmse": best}, args.out)
        else:
            wait += 1
            if wait >= args.patience:
                print("early stopping")
                break
    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    te, pred, y = batches(model, loaders[2], a_out, a_in, device)
    mae = torch.mean(torch.abs(pred-y)).item()
    print(f"BEST val_rmse={best:.5f} TEST rmse={te:.5f} mae={mae:.5f}")
    with open(os.path.splitext(args.out)[0] + "_history.json", "w") as f: json.dump(history, f, indent=2)

if __name__ == "__main__": main()
