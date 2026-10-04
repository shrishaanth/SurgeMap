# SurgeMap

**NYC taxi demand forecasting and fleet repositioning with a spatio-temporal graph neural network**

SurgeMap forecasts how many taxi pickups each of New York's taxi zones will see over the next 5, 15, 30 and 60
minutes, then tests in a simulator whether using those forecasts to pre-position idle vehicles shortens rider
waits. It is part of the [FleetMg](https://github.com/shris-xyz/FleetMg) fleet management project.

The repository contains:

- **forecasters**: a directed-graph ST-GNN trained on January 2024 NYC yellow-taxi trips, compared with
  persistence, historical-average, ridge and gradient-boosted baselines
- a **fleet simulator** that replays real pickups against a simulated fleet with nearest-vehicle dispatch
- a **repositioning policy**: a small min-cost-flow linear program driven by any forecast
- an **evaluation** of policies across fleet sizes, seeds and a cost trade-off sweep
- a **Streamlit app** to explore forecasts and run the simulator

---

## Data

- **Source**: [NYC TLC Yellow Taxi Trip Records, January 2024](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  (the official parquet file, 2,964,624 rows; 2,927,173 pass the validity filters)
- **Zones**: 258 taxi zones with at least one pickup; five of them have fewer than five trips in the month
- **Time grid**: 8,928 five-minute bins covering the whole month, none empty
- **Split**: chronological 70% train (to 22 Jan 16:45), 15% validation, 15% test (27 Jan 08:20 to 31 Jan 23:55)
- **Graph**: directed row-normalised flow matrices `A_out` and `A_in` from pickup-to-dropoff counts

**Correction note.** Earlier versions of this project were built from a CSV that stopped at 29 Jan 21:38 and
held about 5% fewer trips than the official file, so the last two days of the test split were empty. Every
metric produced from that extract was inflated and has been discarded. The pipeline now reads the official
parquet file (`data/yellow_tripdata_2024-01.parquet`; `data/` is not tracked).

### Features per zone and time step

| Channel | Description |
|---------|-------------|
| `log_demand_zscore` | log(1 + pickups), standardised per zone on the training split, clipped to ±6 |
| `hour_sin`, `hour_cos` | cyclic time of day |
| `weekday_sin`, `weekday_cos` | cyclic day of week |
| `is_weekend` | weekend flag |
| `log_dropoff_zscore` | log(1 + dropoffs), standardised and clipped like demand |

---

## Model

1. **Spatial encoder**: `DirectedGraphConv` mixes neighbour features through `A_out` and `A_in` separately, plus
   a learned self-loop projection.
2. **Temporal encoder**: a one-layer GRU over a 48-bin (4 hour) window.
3. **Multi-horizon heads**: one linear head per horizon (1, 3, 6, 12 bins ahead) on the shared encoder.

Trained with AdamW, MSE loss, gradient clipping and early stopping on validation RMSE (seed 7).

### Baselines

| Baseline | Description |
|----------|-------------|
| Persistence | the last observed bin |
| Historical average | mean training demand for the same zone, weekday and time of day |
| Ridge regression | one ridge model per zone on the flattened 48-bin window |
| Ridge + time-of-day average | the same ridge model with the historical average of each target bin as an extra input |
| Gradient boosting | one scikit-learn `HistGradientBoostingRegressor` per horizon across all zones, Poisson loss; lags, same time yesterday and last week, historical average, calendar, weather, citywide demand |

---

## Results

Held-out test window: 27 Jan 08:20 to 31 Jan 23:55 (1,329 forecast times, 258 zones). No model saw it during
fitting. The ST-GNN trained for 50 epochs on the corrected data (best validation epoch 47).

### Forecast accuracy

RMSE in pickups per zone per 5-minute bin (lower is better), from `artifacts/forecast_metrics.json`:

| Model | 5 min | 15 min | 30 min | 60 min |
|-------|-------|--------|--------|--------|
| **ST-GNN, calibrated** | **1.402** | **1.468** | **1.506** | **1.561** |
| Gradient boosting, calibrated | 1.422 | 1.506 | 1.539 | 1.577 |
| Gradient boosting | 1.429 | 1.523 | 1.558 | 1.610 |
| ST-GNN | 1.449 | 1.546 | 1.618 | 1.726 |
| Ridge + time-of-day average | 1.476 | 1.575 | 1.602 | 1.661 |
| Ridge regression | 1.467 | 1.580 | 1.681 | 1.844 |
| Persistence | 1.716 | 1.889 | 2.016 | 2.189 |
| Historical average | 1.701 | 1.702 | 1.703 | 1.706 |

"Calibrated" means two corrections fitted on the validation split only and applied identically to both models
(`scripts/calibrate_forecasts.py`): a per-zone, per-horizon bias correction, then a per-horizon blend with the
time-of-day average.

**As trained, the ST-GNN is not the best model.** It beats persistence by 16 to 21%, but a gradient-boosted model
with simple features (recent lags, the same time yesterday and last week, calendar, weather, citywide demand)
matches or beats it at every horizon. 95% block-bootstrap intervals on RMSE(other) minus RMSE(ST-GNN), positive
meaning the ST-GNN is better (`scripts/accuracy_checks.py`):

| Other model | 5 min | 15 min | 30 min | 60 min |
|-------------|-------|--------|--------|--------|
| Ridge regression | +0.017 [-0.012, +0.050] | +0.034 [-0.001, +0.074] | +0.062 [+0.018, +0.112] | +0.119 [+0.065, +0.187] |
| Historical average | +0.251 [+0.172, +0.332] | +0.156 [+0.088, +0.227] | +0.085 [+0.021, +0.154] | -0.020 [-0.084, +0.048] |
| Ridge + time-of-day average | +0.027 [+0.003, +0.053] | +0.029 [-0.004, +0.062] | -0.016 [-0.049, +0.017] | -0.065 [-0.097, -0.035] |
| Gradient boosting | -0.020 [-0.050, +0.005] | -0.023 [-0.048, +0.004] | -0.060 [-0.093, -0.025] | -0.116 [-0.167, -0.062] |

**After the same calibration, the ST-GNN is ahead of gradient boosting**: +0.020 [+0.000, +0.039] at 5 minutes,
+0.038 [+0.018, +0.061] at 15, +0.033 [+0.013, +0.054] at 30 and +0.016 [-0.005, +0.038] at 60. The lead is
significant at 15 and 30 minutes, borderline at 5, and not significant at 60.

### Why the ST-GNN trails as trained

`scripts/diagnose_stgnn.py` examines the trained network using inference only (`results/stgnn_diagnostics.json`):

- **It under-predicts total pickups by 8.5 to 12%** (gradient boosting: 2 to 4%). It is trained on log-scale
  demand, which converts back to something closer to a median than a mean, and the test days are about 9% busier
  than the training average for the same weekday and time.
- **It has no notion of which zone it is forecasting.** All zones share one set of weights and it sees only the
  last 4 hours, so it cannot learn a zone's own daily profile. Its gap to gradient boosting at 60 minutes is at
  night and in the evening, where it is also worse than a plain time-of-day average.
- **Its loss does not match the evaluation.** On the metric it was trained on (per-zone standardised log demand) it
  is the best model: 0.715 against 0.721 for ridge and about 0.90 for gradient boosting. But that loss weights all
  258 zones equally, while 86% of the squared error in pickups sits in the 30 busiest zones.
- **It underfits.** Training error (0.722) is close to validation error (0.739) and validation was still improving
  at epoch 47. The training loader also did not shuffle its batches.
- **It does rely on the graph**: switching the graph off at inference raises RMSE by 7 to 19%. That shows it leans
  on neighbouring zones. A later ablation confirmed the graph helps: the same network trained without it is 2 to 4%
  worse (see [Outcome of the retrain](#outcome-of-the-retrain)). Giving the gradient-boosted model flow-weighted
  neighbour demand, by contrast, left its error unchanged (`results/spatial_check.json`).

The calibration patches the first two points from outside the network. The trainer has options to address all
four inside it (see [Retraining the ST-GNN](#retraining-the-st-gnn)); that retrain did not beat the calibrated model.

Hotspot ranking, as the mean number of the 5 busiest zones also among the 5 predicted, across 5 to 60 minutes:
calibrated ST-GNN 3.19 to 3.08, calibrated gradient boosting 3.18 to 3.06, persistence 2.93 to 2.55. The top-3 hit
rate is 0.86 to 0.94 for every model, because the busiest zones barely change, so it does not separate them.

### Repositioning

Rider outcomes over the test window with `theta = 0.1`, by fleet size. Cells show unmet requests (%) and mean
rider wait (minutes, unmet requests counted as 15). Mean over 5 seeds; the standard deviation across seeds is
about 0.1 minute (0.16 at most).

| Policy driven by | 1000 vehicles | 1500 | 2000 | 3000 |
|------------------|---------------|------|------|------|
| Dispatch only | 61.5 / 9.98 | 49.7 / 8.49 | 42.8 / 7.63 | 33.7 / 6.50 |
| Persistence | 57.0 / 9.25 | 40.2 / 7.07 | 27.3 / 5.40 | 12.6 / 3.41 |
| Historical average | 57.7 / 9.32 | 40.6 / 7.08 | 27.9 / 5.44 | 15.3 / 3.79 |
| Ridge | 57.9 / 9.34 | 41.5 / 7.20 | 28.9 / 5.57 | 15.1 / 3.76 |
| Ridge + time-of-day average | 57.6 / 9.30 | 41.2 / 7.16 | 29.1 / 5.60 | 15.0 / 3.76 |
| ST-GNN | 57.7 / 9.33 | 41.3 / 7.18 | 28.4 / 5.51 | 14.6 / 3.70 |
| Gradient boosting | 57.8 / 9.34 | 40.6 / 7.09 | 27.2 / 5.34 | 12.9 / 3.40 |
| Gradient boosting, calibrated | 57.3 / 9.27 | 40.5 / 7.08 | 26.7 / 5.28 | 12.9 / 3.39 |
| **ST-GNN, calibrated** | 57.1 / 9.25 | 40.0 / 7.01 | 26.7 / 5.28 | 12.8 / 3.36 |
| Oracle (true demand) | 56.8 / 9.17 | 40.3 / 6.98 | 25.7 / 5.05 | 12.2 / 3.18 |

![Rider wait vs empty driving](results/tradeoff.png)

- Forecast-driven repositioning clearly helps once the fleet is large enough: at 2,000 vehicles unmet requests fall
  from 42.8% to 26 to 29% and mean wait from 7.6 to about 5.3 minutes. At 1,000 vehicles there is little to gain.
- A better forecast gives a better policy. Comparing at equal driving cost along the `theta` sweep at 2,000
  vehicles, the wait saved relative to a persistence-driven policy is 0.30 minutes for the calibrated ST-GNN (84%
  of what perfect demand knowledge would give), 0.30 for calibrated gradient boosting (83%), 0.28 for gradient
  boosting (77%) and 0.17 for the ST-GNN as trained (47%). Comparing at equal `theta` is misleading because
  policies drive different amounts.
- The two calibrated models are indistinguishable as repositioning policies, even though the ST-GNN forecasts
  slightly better. Both reach persistence's rider outcomes while driving about 13% less empty at 2,000 vehicles.
- Perfect foresight beats persistence by only 0.2 to 0.35 minutes of wait at 2,000 to 3,000 vehicles, which bounds
  how much any forecaster can add at a 15-minute decision horizon.

### What this means

Forecasts improve repositioning, and forecast quality carries through to the policy. The ST-GNN as first trained
was beaten by a simpler model, for reasons that turned out to be about its training setup rather than about graphs:
bias from the log scale, no zone-specific daily profile, a loss that does not match the evaluation, and too little
training. Correcting the first two from outside makes it the most accurate forecaster here, by a small margin that
is significant at 15 and 30 minutes. A no-graph ablation shows the graph contributes 2 to 4% of accuracy to the
network. A retrain that built the fixes into the network did not beat the calibrated original.

---

## Repositioning study

**Simulator** (`scripts/simulator.py`). Real pickups of the test window are replayed against a fleet of idle
vehicles placed in proportion to demand. A request is served by an idle vehicle in its own zone with no wait, or
by the nearest idle vehicle within 15 minutes, whose drive time is the rider's wait. Otherwise it is lost. A
served trip returns its vehicle to a destination drawn from the observed flows after the observed mean trip
duration. Zone-to-zone driving times are shortest paths over the observed mean trip durations, since most zone
pairs have no direct trips.

**Policy** (`scripts/reposition.py`). Every 15 minutes a linear program decides how many idle vehicles to send
between zones, given the forecast demand over the next 15 minutes. It minimises
`theta × vehicle driving minutes + rider wait minutes + 15 × unserved requests`. Any forecast can drive it;
`theta` sets the trade-off between fleet driving and rider waiting.

**Evaluation** (`scripts/run_evaluation.py`). Policies driven by persistence, historical average, ridge, ridge
with the time-of-day average, gradient boosting, the ST-GNN (both as trained and calibrated) and the true demand ("oracle", an upper bound) are compared with dispatch alone, over several fleet sizes
and seeds, with a `theta` sweep to trace each policy's wait vs driving curve. Policies are also compared with
persistence at equal driving cost, so a policy that simply moves fewer vehicles is not mistaken for a worse one.

The fleet is synthetic, because public trip data has no vehicle positions. Read the study as a comparison between
policies, not as a prediction of real-world performance.

---

## Usage

```powershell
# Install (Python 3.11+; a virtual environment is recommended)
python -m pip install -r requirements.txt

# 1. Preprocess the official parquet file into real_processed_265/
python scripts\preprocess.py --input data\yellow_tripdata_2024-01.parquet --out real_processed_265 --month 2024-01 --all-zones

# 2. Train the multi-horizon ST-GNN (about 3-4 hours on a CPU, a few minutes on a GPU)
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 50 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out multihorizon_265_clipped.pt

# 3. Export count-space forecasts and baselines for the test split
python scripts\export_predictions.py --checkpoint multihorizon_265_clipped.pt

# 4. Add the extra baselines (ridge + time-of-day average, gradient boosting) and the bootstrap intervals
python scripts\accuracy_checks.py --merge

# 5. Add the validation-calibrated ST-GNN and boosted model
python scripts\calibrate_forecasts.py

# 6. Run the repositioning study and draw the plots (a few minutes on 10 cores)
python scripts\run_evaluation.py --workers 10
python scripts\plot_results.py

# 7. Explore everything in the app
python -m streamlit run app\streamlit_app.py
```

The same steps are available as `make preprocess-265`, `make train-multi-265`, `make export`, `make evaluate`,
`make plots`, `make app` and `make test`.

---

## Retraining the ST-GNN

`train_multihorizon_torch.py` has options that target each diagnosed weakness. They are off by default, so the
original command still reproduces the shipped model.

| Option | What it does | Weakness it targets |
|--------|--------------|---------------------|
| `--shuffle` | shuffle training windows each epoch | underfitting (batches were consecutive windows) |
| `--prior histavg` | predict the residual over the time-of-day average of each target bin | no daily or weekly pattern |
| `--zone-dim 8` | learned embedding per zone, fed to the output heads | no notion of which zone it is |
| `--weather` | append the precipitation and temperature channels | unused weather data |
| `--loss-power 1` | weight zones by `((1 + mean pickups) x sigma) ** power` in the loss (2 approximates count error) | loss weights all zones equally |
| `--no-graph` | remove the neighbour terms (ablation) | tests whether the graph helps |

```powershell
# Recipe as run (batch size 32 fits a 15 GB GPU; early stopping ended both runs before epoch 30)
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --prior histavg --zone-dim 8 --weather --loss-power 1 --out multihorizon_v2.pt

# The same recipe without the graph, to measure what the graph contributes
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --prior histavg --zone-dim 8 --weather --loss-power 1 --no-graph --out multihorizon_v2_nograph.pt

# Score both on the test split against the shipped models, with bootstrap intervals
python scripts\compare_checkpoints.py multihorizon_v2.pt multihorizon_v2_nograph.pt
```

Or `make train-v2`, `make train-v2-nograph` and `make compare-v2`. With a weighted loss the `train_rmse` and
`val_rmse` printed during training are the weighted objective and are not comparable with the shipped model's
0.72; use `compare_checkpoints.py` for a like-for-like score in pickup counts.

These options do not remove the log-scale bias or the drift in demand level, so the validation calibration still
applies to a retrained model. On a 15 GB GPU use `--batch-size 32`; 128 runs out of memory.

### Outcome of the retrain

Both recipes were run once on a Kaggle T4 GPU with batch size 32 (`results/v2_retrain.json`). RMSE in pickups per
zone per 5-minute bin:

| Model | 5 min | 15 min | 30 min | 60 min |
|-------|-------|--------|--------|--------|
| Shipped ST-GNN, calibrated | 1.402 | 1.468 | 1.506 | 1.561 |
| Retrained, calibrated | 1.524 | 1.535 | 1.545 | 1.558 |
| Retrained without graph, calibrated | 1.532 | 1.557 | 1.572 | 1.592 |
| Shipped ST-GNN, as trained | 1.450 | 1.546 | 1.619 | 1.726 |
| Retrained, as trained | 1.574 | 1.580 | 1.597 | 1.633 |
| Retrained without graph, as trained | 1.602 | 1.629 | 1.660 | 1.686 |

- **The retrain did not improve on the shipped model.** As trained it is better at 60 minutes (1.633 vs 1.726) but
  worse at 5 and 15. After calibration it is worse at 5, 15 and 30 minutes and level at 60, so the shipped
  checkpoint stays. The retrained model's error is almost flat across horizons, which suggests it leans on the
  time-of-day prior and reacts less to the most recent demand. The recipe changed several things at once, so which
  change caused that is not known.
- **The graph helps.** With everything else equal, the network with the graph beats the one without at every
  horizon as trained: +0.028 [+0.003, +0.056], +0.049 [+0.015, +0.087], +0.063 [+0.016, +0.116] and
  +0.054 [+0.002, +0.114], about 2 to 4%. After calibration the gap narrows to 0.01 to 0.03 and is significant at 15,
  30 and 60 minutes. This is one training run per model, so seed-to-seed variation is not measured.

---

## Project structure

```
SurgeMap/
├── app/                         Streamlit app (pages, helpers, trimmed zone polygons)
├── scripts/
│   ├── preprocess.py            Raw trips (CSV or parquet) -> tensors and graph
│   ├── train_multihorizon_torch.py   Multi-horizon ST-GNN trainer
│   ├── train_stgnn_torch.py     Single-horizon trainer; defines the graph convolution
│   ├── multihorizon_baseline.py Persistence and ridge baselines (z-score space)
│   ├── hotspot_eval.py          Top-k hotspot evaluation of a checkpoint
│   ├── export_predictions.py    Count-space forecasts for the simulator and app
│   ├── accuracy_checks.py       Extra baselines, bootstrap intervals, merge into the forecasts
│   ├── calibrate_forecasts.py   Validation-fitted bias correction and blend for both models
│   ├── diagnose_stgnn.py        Where the ST-GNN's error comes from
│   ├── compare_checkpoints.py   Scores retrained checkpoints against the shipped models
│   ├── simulator.py             Fleet simulator with dispatch matching
│   ├── reposition.py            Forecast-driven LP repositioning policy
│   ├── run_evaluation.py        Policy comparison and trade-off sweep
│   ├── plot_results.py          Plots of the study
│   ├── prepare_zones.py         Trims the NYC taxi zone polygons for the app
│   ├── build_dropoff.py, build_weather.py, stgnn_models.py   Auxiliary tools
├── tests/                       pytest suite
├── real_processed_265/          Preprocessed arrays (demand, features, graph, metadata)
├── notebooks/                   Feature experiments
└── config.yaml                  Reference settings (not read at runtime)
```

---

## Limitations

- One month of data, so seasonality and holidays are not covered.
- The flow graph is built from the whole month, including the test period; this is a mild form of leakage.
- Observed pickups are what taxis actually served, which understates true demand where supply was short.
- The repositioning study uses a synthetic fleet, a simple dispatch rule and lost (not queued) requests.
- Only yellow taxis are included.

## Acknowledgements

- NYC Taxi & Limousine Commission for the trip record data
- NYC Open Data for the taxi zone boundaries
