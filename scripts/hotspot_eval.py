from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_multihorizon_torch import MultiHorizonDataset, MultiHorizonSTGNN
from train_multihorizon_torch import split_bounds as _split_bounds

DEFAULT_HORIZONS = (1, 3, 6, 12)
_DEFAULT_KS = (3, 5)


def parse_horizons(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_HORIZONS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        horizons = tuple(int(part) for part in parts)
    else:
        horizons = tuple(int(h) for h in value)
    if not horizons or any(h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be a non-empty sequence of unique positive integers")
    return horizons


def parse_topk(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return _DEFAULT_KS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        ks = tuple(int(part) for part in parts)
    else:
        ks = tuple(int(k) for k in value)
    if not ks or any(k < 1 for k in ks):
        raise ValueError("top-k must be a non-empty sequence of positive integers")
    return ks


def split_bounds(meta: dict, total: int) -> tuple[tuple[int, int], ...]:
    return _split_bounds(meta, total)


def resolve_data_dir(explicit: str | None, checkpoint_args: dict | None) -> str:
    if explicit:
        return explicit
    saved = checkpoint_args.get("data_dir") if checkpoint_args else None
    if saved:
        return str(saved)
    raise ValueError(
        "No data directory could be resolved. Pass --data-dir or train with it saved in the checkpoint."
    )


def normalize_horizons(from_checkpoint, explicit, head_count: int) -> tuple[int, ...]:
    if explicit:
        horizons = parse_horizons(explicit)
        if len(horizons) != head_count:
            raise ValueError(
                f"--horizons has {len(horizons)} entries but the checkpoint model has "
                f"{head_count} heads ({horizons}); pass the checkpoint's horizons."
            )
        return horizons
    from_checkpoint = parse_horizons(from_checkpoint)
    if len(from_checkpoint) != head_count:
        raise ValueError(f"checkpoint horizons {from_checkpoint} do not match {head_count} model heads")
    return from_checkpoint


def load_checkpoint(checkpoint_path: str, data_dir: str, explicit_horizons=None,
                    explicit_hidden=None, device=torch.device("cpu")) -> tuple:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    for required in ("A_out.npy", "A_in.npy", "metadata.json"):
        if not os.path.exists(os.path.join(data_dir, required)):
            raise FileNotFoundError(f"required input missing in {data_dir}: {required}")
    features_path = os.path.join(data_dir, "features_clipped.npy")
    if not os.path.exists(features_path):
        features_path = os.path.join(data_dir, "features.npy")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"required input missing in {data_dir}: features_clipped.npy or features.npy"
        )
    features = np.load(features_path)
    a_out = np.load(os.path.join(data_dir, "A_out.npy"))
    a_in = np.load(os.path.join(data_dir, "A_in.npy"))
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a torch serialized dictionary")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict) or not any(key.startswith("spatial.") for key in state):
        raise ValueError("checkpoint contains no 'model' state dict with spatial parameters")
    saved_args = checkpoint.get("args") or checkpoint.get("config") or {}
    if not isinstance(saved_args, dict):
        saved_args = {}
    saved_horizons = checkpoint.get("horizons") or saved_args.get("horizons") or DEFAULT_HORIZONS
    head_count = sum(1 for key in state if key.startswith("heads.") and key.endswith(".weight"))
    if head_count == 0:
        raise ValueError("checkpoint has no 'heads.*' layers; is it a multi-horizon checkpoint?")
    horizons = normalize_horizons(saved_horizons, explicit_horizons, head_count)
    hidden = explicit_hidden
    if hidden is None:
        if saved_args.get("hidden"):
            hidden = int(saved_args["hidden"])
        elif "spatial.w_out.weight" in state:
            hidden = state["spatial.w_out.weight"].shape[0]
        else:
            raise ValueError("cannot infer hidden size from checkpoint; pass --hidden")
    model = MultiHorizonSTGNN(features.shape[2], hidden=hidden, horizons=horizons).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            f"state dict is incompatible with MultiHorizonSTGNN (hidden={hidden}, "
            f"horizons={horizons}, n_features={features.shape[2]}): {exc}"
        ) from exc
    model.eval()
    return model, features, a_out, a_in, horizons, meta


def collect_predictions(model, features, a_out, a_in, bounds, window: int, horizons,
                        batch_size: int = 128, device=torch.device("cpu")) -> tuple:
    dataset = MultiHorizonDataset(features, *bounds[2], window, horizons)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, targets, anchors = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, torch.as_tensor(a_out, dtype=torch.float32, device=device),
                        torch.as_tensor(a_in, dtype=torch.float32, device=device))
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    if not preds:
        raise ValueError(f"test split has no windows; window={window} horizons={horizons}")
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    target = np.concatenate(targets, axis=0).astype(np.float32)
    anchors = np.asarray(dataset.times, dtype=np.int64)
    return pred, target, anchors


def ranking_metrics(pred: np.ndarray, target: np.ndarray, horizons: Sequence[int],
                    ks: Sequence[int], zone_ids: Sequence[int] | None = None) -> dict:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("pred and target must both have shape [N,Z,H]")
    horizons = parse_horizons(horizons)
    ks = parse_topk(ks)
    if len(horizons) != pred.shape[-1]:
        raise ValueError("horizon count does not match prediction shape")
    if zone_ids is not None and len(zone_ids) != pred.shape[1]:
        raise ValueError("zone_ids length does not match zone dimension")
    result = {}
    for i, h in enumerate(horizons):
        p, y = pred[..., i], target[..., i]
        row = {
            "minutes": h * 5,
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y))),
            "topk": {},
        }
        result[str(h)] = row
        for k in ks:
            n, z = p.shape
            kk = min(int(k), z)
            pred_top = np.argsort(p, axis=1)[:, -kk:][:, ::-1]
            act_top = np.argsort(y, axis=1)[:, -kk:][:, ::-1]
            intersect = np.array([
                len(set(pred_top[i].tolist()) & set(act_top[i].tolist())) for i in range(n)
            ], dtype=np.float32)
            result[str(h)]["topk"][str(int(k))] = {
                "precision": float(np.mean(intersect / kk)),
                "recall": float(np.mean(intersect / kk)),
                "hit_rate": float(np.mean(intersect > 0)),
                "overlap": float(np.mean(intersect)),
            }
    return result


@torch.no_grad()
def block_predict(model, x, y, a_out, a_in, device):
    return model(x.to(device), a_out, a_in)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="multihorizon_stgnn_checkpoint.pt")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--topk", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe_args = checkpoint_probe.get("args") or checkpoint_probe.get("config") or {}
    data_dir = resolve_data_dir(args.data_dir, probe_args)
    if not os.path.isabs(data_dir):
        candidate = os.path.join(os.path.dirname(checkpoint_path), data_dir)
        data_dir = candidate if os.path.isdir(candidate) else os.path.abspath(data_dir)
    model, features, a_out, a_in, horizons, meta = load_checkpoint(
        checkpoint_path, data_dir,
        explicit_horizons=args.horizons, explicit_hidden=args.hidden, device=device,
    )
    bounds = split_bounds(meta, features.shape[1])
    pred, target, anchors = collect_predictions(
        model, features, a_out, a_in, bounds, args.window, horizons,
        batch_size=args.batch_size, device=device,
    )
    ks = parse_topk(args.topk)
    zone_path = os.path.join(data_dir, "zone_ids.npy")
    zone_ids = np.load(zone_path).tolist() if os.path.exists(zone_path) else meta.get("zone_ids")
    metrics = {
        "config": vars(args), "dataset": {"n_test": int(len(anchors))},
        "horizons": list(horizons), "window": args.window,
    }
    metrics.update(ranking_metrics(pred, target, horizons, ks, zone_ids))
    print("horizon,minutes,rmse,mae,topk,precision,recall,hit_rate,overlap")
    for h_text in (str(h) for h in horizons):
        row = metrics[h_text]
        h = int(h_text)
        for k_text, krow in row["topk"].items():
            print(f"{h},{row['minutes']},{row['rmse']:.6f},{row['mae']:.6f},{k_text},"
                  f"{krow['precision']:.6f},{krow['recall']:.6f},{krow['hit_rate']:.6f},{krow['overlap']:.6f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved={args.out}")
    else:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = f"{stem}_hotspot_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved={out_path}")


if __name__ == "__main__":
    main()
