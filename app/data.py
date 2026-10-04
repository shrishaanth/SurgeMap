"""Cached loaders for the files the app reads. Nothing here trains or runs a neural network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from logic import geometry_centroid

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "scripts"))

PREDICTIONS = ROOT / "artifacts" / "predictions.npz"
FORECAST_METRICS = ROOT / "artifacts" / "forecast_metrics.json"
EVALUATION = ROOT / "results" / "evaluation.json"
DATA_DIR = ROOT / "real_processed_265"

MODELS = ("stgnn", "gbm", "ridge_hist", "ridge", "persistence", "histavg")
LABELS = {"stgnn": "ST-GNN", "gbm": "Gradient boosting", "ridge_hist": "Ridge + time-of-day average",
          "ridge": "Ridge regression", "persistence": "Persistence",
          "histavg": "Historical average", "oracle": "Oracle (true demand)", "none": "Dispatch only"}


def require(path: Path, hint: str) -> None:
    """Stop the page with a clear message when a generated file is missing."""
    if not path.exists():
        st.error(f"`{path.relative_to(ROOT)}` is missing. {hint}")
        st.stop()


@st.cache_data(show_spinner=False)
def predictions() -> dict:
    require(PREDICTIONS, "Generate it with `make export`.")
    with np.load(PREDICTIONS) as z:
        out = {k: z[k] for k in z.files}
    out["anchor_times"] = pd.to_datetime(out["anchor_times"])
    return out


@st.cache_data(show_spinner=False)
def forecast_metrics() -> dict:
    require(FORECAST_METRICS, "Generate it with `make export`.")
    return json.loads(FORECAST_METRICS.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def evaluation() -> dict:
    require(EVALUATION, "Generate it with `make evaluate`.")
    return json.loads(EVALUATION.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def zone_names() -> dict:
    raw = json.loads((APP_DIR / "assets" / "zone_names.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


@st.cache_data(show_spinner=False)
def zone_geojson() -> dict:
    return json.loads((APP_DIR / "assets" / "taxi_zones.geojson").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def zone_centroids() -> dict:
    return {f["properties"]["location_id"]: geometry_centroid(f["geometry"])
            for f in zone_geojson()["features"]}


@st.cache_resource(show_spinner=False)
def world():
    from simulator import load_world
    require(DATA_DIR / "demand.npy", "Generate it with `make preprocess-265`.")
    return load_world(str(DATA_DIR))
