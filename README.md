# SurgeMap

**NYC Taxi Demand Forecasting & Fleet Repositioning using Spatio-Temporal Graph Neural Networks**

SurgeMap predicts taxi demand across 253 NYC taxi zones at 5-minute intervals using a multi-horizon Spatio-Temporal Graph Neural Network (ST-GNN). By forecasting demand surges 5, 15, 30, and 60 minutes ahead, fleet operators can proactively reposition vehicles to anticipated hotspots, reducing passenger wait times and increasing revenue.

Part of the [FleetMg](https://github.com/shris-xyz/FleetMg) (Fleet Management) ecosystem.

---

## Key Features

- **Multi-Horizon Forecasting** — Predicts demand at 5, 15, 30, and 60-minute horizons simultaneously
- **Directed Graph Convolutions** — Explicitly models inbound (`A_in`) and outbound (`A_out`) taxi flows as separate adjacency matrices
- **Hotspot Ranking Metrics** — Evaluates top-k precision, recall, and hit rate for fleet repositioning relevance
- **Memory-Efficient Preprocessing** — Chunked ingestion of NYC TLC yellow taxi CSV data
- **Reproducible Pipeline** — Centralized config (`config.yaml`), fixed seeds, and serialized checkpoints
- **Dual Implementation** — Lightweight NumPy baselines alongside PyTorch training with autograd

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Deep Learning | PyTorch >= 2.2 |
| Data Processing | NumPy, Pandas |
| Scientific Computing | SciPy |
| Configuration | PyYAML |
| Testing | pytest |
| Build Automation | Make |

---

## Architecture

```
Raw NYC Taxi CSV
       │
       ▼
  preprocess.py          Memory-efficient chunked ingestion → graph + features
       │
       ▼
  train_multihorizon_torch.py   Multi-horizon ST-GNN (shared spatial-temporal encoder)
       │
       ▼
  hotspot_eval.py        Top-k ranking evaluation on held-out test split
       │
       ▼
  Baselines / Simulator  Persistence, Ridge Regression, Fleet repositioning policies
```

### Model Architecture

The multi-horizon ST-GNN consists of:

1. **Spatial Encoder** — `DirectedGraphConv` layers that separately mix information via `A_out` and `A_in` adjacency matrices, plus learned self-loop projections
2. **Temporal Encoder** — 1-layer GRU that captures temporal dependencies across a 4-hour observation window (48 bins)
3. **Multi-Horizon Heads** — Separate linear prediction heads for each forecast horizon (1, 3, 6, 12 bins ahead), sharing the spatial-temporal encoder

### Features per Zone per Timestep

| Feature | Description |
|---------|-------------|
| `log_demand_zscore` | Log(1 + pickups), standardized on training split |
| `hour_sin` / `hour_cos` | Cyclic time-of-day encoding |
| `weekday_sin` / `weekday_cos` | Cyclic day-of-week encoding |
| `is_weekend` | Binary weekend flag |

---

## Dataset

- **Source**: [NYC TLC Yellow Taxi Trip Records — January 2024](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Zones**: 253 active NYC taxi zones (out of 265; 12 zones had no valid data)
- **Timesteps**: 8,928 five-minute bins covering the full month
- **Split**: Chronological 70% train / 15% validation / 15% test
- **Graph**: Directed adjacency matrices derived from actual pickup/dropoff flows

---

## Installation

```powershell
# Clone the repository
git clone https://github.com/shris-xyz/FleetMg.git
cd FleetMg/SurgeMap

# Create virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Usage

### Quick Start (265-Zone Pipeline)

```powershell
# 1. Preprocess all 253 NYC zones from raw CSV
python scripts\preprocess.py --input data\yellow_tripdata_2024-01.csv --out real_processed_265 --month 2024-01 --all-zones

# 2. Train multi-horizon ST-GNN
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 50 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out multihorizon_stgnn_checkpoint_265.pt

# 3. Evaluate baselines
python scripts\multihorizon_baseline.py --data-dir real_processed_265 --window 48 --horizons 1,3,6,12 --out multihorizon_baselines_265_clipped.json

# 4. Evaluate trained model hotspot performance
python scripts\hotspot_eval.py --checkpoint multihorizon_265_clipped.pt --data-dir real_processed_265 --horizons 1,3,6,12 --topk 3,5 --out hotspot_265_clipped.json
```

### Makefile Targets

```powershell
make install           # Install dependencies
make preprocess-265    # Preprocess 265 zones + build travel matrix
make train-multi-265   # Train multi-horizon ST-GNN
make test              # Run test suite
make smoke             # Quick dependency check
```

---

## Project Structure

```
SurgeMap/
├── README.md                        # Project documentation
├── README_RUN.md                     # Detailed run guide
├── config.yaml                       # Central configuration (seed, data, forecast, simulator)
├── requirements.txt                  # Python dependencies
├── Makefile                          # Build automation
├── real_processed_265/               # Preprocessed data for 253 NYC zones
│   ├── metadata.json                 # Dataset metadata (zones, splits, features, shapes)
│   ├── demand.npy                    # Raw pickup counts [253 zones × 8928 bins]
│   ├── features_clipped.npy          # Outlier-clipped features (used by training/eval)
│   ├── A_out.npy                     # Outgoing adjacency (directed) [253 × 253]
│   ├── A_in.npy                      # Incoming adjacency (directed) [253 × 253]
│   └── zone_ids.npy                  # NYC zone IDs (253 zones)
│   # A_sym/adjacency/edge_index/travel_time/times/features(unclipped) are regenerable via
│   # preprocess.py but not committed — unused by the current training/eval scripts.
├── scripts/
│   ├── preprocess.py                 # Raw CSV → tensors + graph
│   ├── stgnn_models.py               # Pure NumPy ST-GNN (baseline)
│   ├── train_stgnn_torch.py          # Single-horizon PyTorch ST-GNN trainer
│   ├── train_multihorizon_torch.py   # Multi-horizon PyTorch ST-GNN trainer
│   ├── multihorizon_baseline.py      # Persistence + ridge regression baselines
│   └── hotspot_eval.py               # Top-k hotspot ranking evaluation
├── multihorizon_265_clipped.pt       # Trained multi-horizon ST-GNN checkpoint
├── multihorizon_265_clipped_metrics.json
├── multihorizon_baselines_265_clipped.json
└── hotspot_265_clipped.json
```

---

## Model Performance

Trained checkpoint (`multihorizon_265_clipped.pt`): 50 epochs, hidden=64, batch=128 (seed=7, deterministic)

Hotspot numbers below are from `scripts/hotspot_eval.py`, the standalone evaluation script — treat these as
authoritative over any hit-rate figure the training script prints inline, which has a known discrepancy at
the 60-minute horizon (see `scripts/train_multihorizon_torch.py`'s `scores()`; it over-reports at h=12).

| Horizon | Test RMSE | Hotspot Top-3 Hit Rate |
|---------|-----------|-------------------------|
| 5 min   | 0.534     | 33.4%                   |
| 15 min  | 0.543     | 36.0%                   |
| 30 min  | 0.544     | 18.9%                   |
| 60 min  | 0.575     | 15.3%                   |

Compared against baselines (`scripts/multihorizon_baseline.py`) on the same test split:

| Horizon | Persistence RMSE | Ridge RMSE | ST-GNN RMSE | ST-GNN vs. Ridge |
|---------|-------------------|------------|-------------|------------------|
| 5 min   | 0.724             | 0.549      | 0.534       | 0.4% better      |
| 15 min  | 0.727             | 0.562      | 0.543       | 2.1% better      |
| 30 min  | 0.729             | 0.584      | 0.544       | 5.8% better      |
| 60 min  | 0.738             | 0.638      | 0.575       | 9.4% better      |

The ST-GNN beats the naive persistence baseline by ~22-25% RMSE at every horizon. Its edge over plain ridge
regression is thin at short horizons (5 min) and grows at longer horizons (30-60 min), where the graph's
cross-zone mixing and the GRU's sequential memory add more value than a flattened-window linear model can capture.

---

## Reproducibility

- Fixed random seed (`7`) across Python, NumPy, and PyTorch
- Configuration centralized in `config.yaml` and serialized into checkpoints
- Chronological data splits prevent future leakage
- All preprocessing artifacts (graph, features, scaler) versioned alongside checkpoints

---


---

## Acknowledgements

- NYC Taxi & Limousine Commission for the [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Built as part of the FleetMg fleet management research project
