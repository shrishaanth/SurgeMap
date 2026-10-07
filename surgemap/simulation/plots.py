from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surgemap import paths

LABELS = {"none": "dispatch only", "persistence": "persistence", "histavg": "historical average",
          "ridge": "ridge",
          "gbm": "gradient boosting", "gbm_cal": "gradient boosting, calibrated",
          "stgnn": "ST-GNN", "stgnn_cal": "ST-GNN, calibrated", "oracle": "oracle (true demand)"}
COLORS = {"none": "#888888", "persistence": "#1f77b4", "histavg": "#9467bd", "ridge": "#2ca02c", "gbm": "#ff7f0e", "gbm_cal": "#bc6c00", "stgnn_cal": "#7f0000",
          "stgnn": "#d62728", "oracle": "#000000"}


def tradeoff(summary, fleet, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in ("persistence", "stgnn", "stgnn_cal", "gbm", "gbm_cal", "oracle"):
        rows = sorted((r for r in summary if r["policy"] == policy and r["fleet"] == fleet),
                      key=lambda r: r["theta"])
        if len(rows) < 2:
            continue
        ax.errorbar([r["empty_minutes"] / 1e3 for r in rows], [r["mean_wait_penalized"] for r in rows],
                    yerr=[r["mean_wait_penalized_std"] for r in rows], marker="o", ms=4, capsize=2,
                    color=COLORS[policy], label=LABELS[policy])
    base = [r for r in summary if r["policy"] == "none" and r["fleet"] == fleet]
    if base:
        ax.axhline(base[0]["mean_wait_penalized"], color=COLORS["none"], ls="--", lw=1,
                   label=LABELS["none"])
    ax.set_xlabel("vehicle-minutes driving empty (thousands)")
    ax.set_ylabel("mean rider wait, unmet counted as 15 min")
    ax.set_title(f"Wait vs fleet driving, {fleet} vehicles (each point is one theta)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def by_fleet(summary, theta, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in ("none", "persistence", "ridge", "stgnn", "gbm", "gbm_cal", "stgnn_cal", "oracle"):
        rows = sorted((r for r in summary if r["policy"] == policy and r["theta"] in (0.0, theta)),
                      key=lambda r: r["fleet"])
        if rows:
            ax.errorbar([r["fleet"] for r in rows], [100 * r["unmet_rate"] for r in rows],
                        yerr=[100 * r["unmet_rate_std"] for r in rows], marker="o", ms=4, capsize=2,
                        color=COLORS[policy], label=LABELS[policy])
    ax.set_xlabel("fleet size")
    ax.set_ylabel("unmet requests (%)")
    ax.set_title(f"Unmet demand by fleet size (theta = {theta})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=paths.RESULTS_DIR + "/evaluation.json")
    parser.add_argument("--out-dir", default=paths.FIGURES_DIR)
    args = parser.parse_args()
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    tradeoff(data["summary"], data["config"]["sweep_fleet"], os.path.join(args.out_dir, "tradeoff.png"))
    by_fleet(data["summary"], data["config"]["theta"], os.path.join(args.out_dir, "fleet_size.png"))
    print(f"wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
