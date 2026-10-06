"""Where the ST-GNN's lead over gradient boosting comes from.

Splits the test window two ways and reports RMSE for each model in each part:

* by how unusual the period is: citywide pickups at the target bin relative to the time-of-day
  average, from far below normal to far above (surges);
* by how busy the zone is: zones ranked by their mean pickups over the test window.

The gap to gradient boosting gets a block-bootstrap interval over anchors in every part.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

PERIODS = (("far below normal (lowest 5%)", 0.0, 0.05), ("below normal (5-25%)", 0.05, 0.25),
           ("normal (25-75%)", 0.25, 0.75), ("above normal (75-95%)", 0.75, 0.95),
           ("far above normal (top 5%)", 0.95, 1.0))
ZONES = (("busiest 10% of zones", 0.9, 1.0), ("next 20%", 0.7, 0.9), ("quietest 70%", 0.0, 0.7))


def rmse(squared: np.ndarray) -> float:
    return float(np.sqrt(squared.mean()))


def gap_interval(se_other, se_model, block: int = 36, draws: int = 1000, seed: int = 0) -> dict:
    """RMSE(other) - RMSE(model) with a 95% interval from resampling blocks of consecutive anchors.

    Both inputs are per-anchor sums of squared error with a matching cell count in `n`.
    """
    (sum_other, n), (sum_model, _) = se_other, se_model
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(n), block)
    gaps = []
    for _ in range(draws):
        pick = np.concatenate([np.arange(s, min(s + block, len(n))) for s in rng.choice(starts, len(starts))])
        cells = n[pick].sum()
        if cells == 0:
            continue
        gaps.append(np.sqrt(sum_other[pick].sum() / cells) - np.sqrt(sum_model[pick].sum() / cells))
    total = n.sum()
    return {"gap": float(np.sqrt(sum_other.sum() / total) - np.sqrt(sum_model.sum() / total)),
            "ci_low": float(np.quantile(gaps, 0.025)), "ci_high": float(np.quantile(gaps, 0.975))}


def per_anchor(squared: np.ndarray, mask: np.ndarray) -> tuple:
    """Per-anchor sum of squared error and cell count over the masked [N, Z] cells."""
    return (squared * mask).sum(axis=1), mask.sum(axis=1).astype(np.float64)


def analyse(actual, forecasts: dict, histavg, horizons, model: str, other: str) -> dict:
    result = {"periods": {}, "zones": {}}
    zone_rank = np.argsort(np.argsort(actual.mean(axis=(0, 2)))) / (actual.shape[1] - 1)
    for j, horizon in enumerate(horizons):
        key = f"{int(horizon) * 5}min"
        squared = {name: (pred[:, :, j] - actual[:, :, j]) ** 2 for name, pred in forecasts.items()}
        ratio = actual[:, :, j].sum(axis=1) / np.maximum(histavg[:, :, j].sum(axis=1), 1e-9)
        rank = np.argsort(np.argsort(ratio)) / (len(ratio) - 1)
        splits = [("periods", name, ((rank >= lo) & (rank <= hi) if hi == 1.0 else (rank >= lo) & (rank < hi))[:, None]
                   * np.ones(actual.shape[1], dtype=bool)[None, :], float(np.median(ratio[(rank >= lo) & (rank <= hi)])))
                  for name, lo, hi in PERIODS]
        splits += [("zones", name, np.ones(len(ratio), dtype=bool)[:, None]
                    * ((zone_rank >= lo) & (zone_rank <= hi) if hi == 1.0 else (zone_rank >= lo) & (zone_rank < hi))[None, :],
                    None) for name, lo, hi in ZONES]
        for kind, name, mask, typical in splits:
            entry = {"rmse": {m: rmse(sq[mask]) for m, sq in squared.items()},
                     "share_of_cells": float(mask.mean()),
                     "share_of_squared_error": float(squared[model][mask].sum() / squared[model].sum()),
                     f"{model}_vs_{other}": gap_interval(per_anchor(squared[other], mask),
                                                         per_anchor(squared[model], mask))}
            entry[f"{model}_vs_{other}"]["percent"] = 100 * entry[f"{model}_vs_{other}"]["gap"] / entry["rmse"][other]
            if typical is not None:
                entry["citywide_pickups_vs_usual"] = typical
            result[kind].setdefault(name, {})[key] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/predictions.npz")
    parser.add_argument("--model", default="stgnn_cal")
    parser.add_argument("--other", default="gbm_cal")
    parser.add_argument("--extra", default="stgnn,gbm,histavg,persistence")
    parser.add_argument("--out", default="results/regime_analysis.json")
    args = parser.parse_args()

    p = np.load(args.predictions)
    names = [args.model, args.other] + [n for n in args.extra.split(",") if n and n in p]
    forecasts = {name: p[name].astype(np.float64) for name in dict.fromkeys(names)}
    result = analyse(p["actual"].astype(np.float64), forecasts, p["histavg"].astype(np.float64), p["horizons"],
                     args.model, args.other)
    result = {"model": args.model, "other": args.other, **result}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    pair = f"{args.model}_vs_{args.other}"
    for kind in ("periods", "zones"):
        print(f"{kind}: RMSE {args.model} / {args.other}, and how much lower the first is (95% interval)")
        for name, by_horizon in result[kind].items():
            cells = []
            for key, e in by_horizon.items():
                g = e[pair]
                cells.append(f"{key} {e['rmse'][args.model]:.3f}/{e['rmse'][args.other]:.3f} "
                             f"{g['percent']:+.1f}% [{g['gap']:+.3f} {g['ci_low']:+.3f},{g['ci_high']:+.3f}]")
            print(f"  {name:30s} " + " | ".join(cells))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
