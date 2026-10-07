"""Command line: python -m surgemap <command> [options]. Each command is the main() of one module."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "download": ("surgemap.data.download", "download the monthly trip files from NYC TLC"),
    "preprocess": ("surgemap.data.preprocess", "build a processed dataset from TLC trip files"),
    "zones": ("surgemap.data.zones", "prepare the taxi-zone polygons and names for the app"),
    "weather": ("surgemap.data.weather", "add the weather channels to a processed dataset"),
    "dropoff": ("surgemap.data.dropoff", "add the dropoff counts to a processed dataset"),
    "train": ("surgemap.training.train", "train one ST-GNN"),
    "export": ("surgemap.evaluation.export", "forecasts of the shipped networks and the simple baselines"),
    "baselines": ("surgemap.evaluation.accuracy", "fit gradient boosting, bootstrap the gaps, merge"),
    "calibrate": ("surgemap.evaluation.calibration", "add the calibrated forecasts"),
    "align": ("surgemap.evaluation.align", "re-index forecasts onto the January dataset"),
    "compare": ("surgemap.evaluation.compare", "score checkpoints against the shipped models"),
    "hotspots": ("surgemap.evaluation.inference", "top-k hotspot metrics of one checkpoint"),
    "diagnose": ("surgemap.evaluation.diagnose", "where a checkpoint's error comes from"),
    "headroom": ("surgemap.evaluation.headroom", "the noise floor and after-the-fact corrections"),
    "regimes": ("surgemap.evaluation.regimes", "accuracy by how unusual the period and how busy the zone"),
    "simulate": ("surgemap.simulation.study", "the repositioning study"),
    "plots": ("surgemap.simulation.plots", "figures of the repositioning study"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in COMMANDS:
        print("usage: python -m surgemap <command> [options]" + chr(10))
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:11s} {help_text}")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f"python -m surgemap {command}"
    importlib.import_module(COMMANDS[command][0]).main()


if __name__ == "__main__":
    main()
