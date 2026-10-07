"""Where things live in the repository. Every command-line default comes from here."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rel(*parts: str) -> str:
    """A path under the repository, written relative to it so that messages and saved arguments stay short."""
    return "/".join(parts)


RAW_DIR = _rel("data", "raw")
PROCESSED_DIR = _rel("data", "processed")
JANUARY = _rel("data", "processed", "january_2024")          # one month; the simulator and the app use it
FOUR_MONTHS = _rel("data", "processed", "four_months")       # the shipped models are trained on this
MODELS_DIR = "models"
FORECASTS_DIR = _rel("outputs", "forecasts")
PREDICTIONS = _rel("outputs", "forecasts", "predictions.npz")
RESULTS_DIR = _rel("outputs", "results")
FIGURES_DIR = _rel("outputs", "figures")
EXPERIMENTS_DIR = _rel("outputs", "experiments")
SCORES_DIR = _rel("outputs", "experiments", "scores")
CACHE_DIR = _rel("outputs", "cache")

# Checkpoints record the folder they were trained on. These are the names those folders had
# before the repository was reorganised, so that older checkpoints still load.
DATASET_ALIASES = {"real_processed_265": "january_2024", "real_processed_4mo": "four_months",
                   "real_processed_2mo": "two_months"}


def dataset_name(path: str) -> str:
    """The name that identifies a processed dataset, whatever folder it is kept in."""
    name = os.path.basename(os.path.normpath(os.path.abspath(str(path))))
    return DATASET_ALIASES.get(name, name)



def locate_dataset(saved: str | None, default: str) -> str:
    """The folder holding the dataset a checkpoint names, or `default` when it names none.

    Looks at the saved path itself, then at `default` if it is the same dataset, then under
    data/processed. Raises if the dataset is not on this machine."""
    if not saved:
        return default
    if os.path.isdir(saved):
        return saved
    name = dataset_name(saved)
    if name == dataset_name(default):
        return default
    candidate = _rel(PROCESSED_DIR, name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"the dataset '{name}' is not present; build it with "
                            "'python -m surgemap preprocess' (see the Makefile)")
