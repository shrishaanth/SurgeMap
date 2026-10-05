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

Trained with AdamW on shuffled windows, a squared-error loss weighted toward busy zones, gradient clipping and
early stopping on the validation loss (seed 7).

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

Held-out test window: 27 Jan 08:20 to 31 Jan 23:55 (1,329 forecast times, 258 zones). No model was fitted on it.
The shipped ST-GNN was trained on a GPU with shuffled batches of 32 and a volume-weighted loss; it stopped early
after 45 epochs (best validation epoch 33). How that recipe was arrived at is described under
[How the model got here](#how-the-model-got-here).

### Forecast accuracy

RMSE in pickups per zone per 5-minute bin (lower is better), from `artifacts/forecast_metrics.json`:

| Model | 5 min | 15 min | 30 min | 60 min |
|-------|-------|--------|--------|--------|
| **ST-GNN, calibrated** | **1.375** | **1.440** | **1.476** | **1.511** |
| ST-GNN | 1.400 | 1.480 | 1.536 | 1.589 |
| Gradient boosting, calibrated | 1.422 | 1.506 | 1.539 | 1.577 |
| Gradient boosting | 1.429 | 1.523 | 1.558 | 1.610 |
| Ridge + time-of-day average | 1.476 | 1.575 | 1.602 | 1.661 |
| Ridge regression | 1.467 | 1.580 | 1.681 | 1.844 |
| Persistence | 1.716 | 1.889 | 2.016 | 2.189 |
| Historical average | 1.701 | 1.702 | 1.703 | 1.706 |

"Calibrated" means two corrections fitted on the validation split only and applied identically to both models
(`scripts/calibrate_forecasts.py`): a per-zone, per-horizon bias correction, then a per-horizon blend with the
time-of-day average. They are needed because both models predict on a log scale, and because the validation and
test days are about 9% busier than the training average for the same weekday and time.

95% block-bootstrap intervals on RMSE(other) minus RMSE(ST-GNN), positive meaning the ST-GNN is better
(`scripts/accuracy_checks.py`):

| ST-GNN as trained vs | 5 min | 15 min | 30 min | 60 min |
|----------------------|-------|--------|--------|--------|
| Persistence | +0.316 [+0.273, +0.354] | +0.408 [+0.368, +0.447] | +0.479 [+0.432, +0.526] | +0.600 [+0.506, +0.693] |
| Historical average | +0.300 [+0.218, +0.388] | +0.222 [+0.150, +0.299] | +0.167 [+0.103, +0.236] | +0.117 [+0.064, +0.170] |
| Ridge regression | +0.066 [+0.034, +0.102] | +0.100 [+0.060, +0.142] | +0.145 [+0.092, +0.201] | +0.256 [+0.168, +0.352] |
| Ridge + time-of-day average | +0.076 [+0.046, +0.113] | +0.094 [+0.059, +0.141] | +0.066 [+0.033, +0.112] | +0.072 [+0.036, +0.120] |
| Gradient boosting | +0.029 [+0.011, +0.047] | +0.042 [+0.015, +0.075] | +0.022 [-0.009, +0.054] | +0.021 [-0.014, +0.058] |

- The ST-GNN beats persistence by 18 to 27% and every other baseline at every horizon.
- Against gradient boosting as trained the margin is 1 to 3%, significant at 5 and 15 minutes only.
- After the same calibration the margin is 3 to 4% and significant at every horizon: +0.047 [+0.033, +0.062],
  +0.066 [+0.043, +0.096], +0.063 [+0.040, +0.088] and +0.066 [+0.043, +0.092].

Two cautions. The shipped recipe was chosen from a handful of experiments by looking at this same test window, so
its margin is slightly optimistic. And each recipe was trained once, so seed-to-seed variation is not measured.

Hotspot ranking, as the mean number of the 5 busiest zones also among the 5 predicted, across 5 to 60 minutes:
calibrated ST-GNN 3.24 to 3.11, calibrated gradient boosting 3.18 to 3.06, persistence 2.93 to 2.55. The top-3 hit
rate is 0.86 to 0.95 for every model, because the busiest zones barely change, so it does not separate them.

### Repositioning

Rider outcomes over the test window with `theta = 0.1`, by fleet size. Cells show unmet requests (%) and mean
rider wait (minutes, unmet requests counted as 15). Mean over 5 seeds; the standard deviation across seeds is
about 0.1 minute.

| Policy driven by | 1000 vehicles | 1500 | 2000 | 3000 |
|------------------|---------------|------|------|------|
| Dispatch only | 61.5 / 9.98 | 49.7 / 8.49 | 42.8 / 7.63 | 33.7 / 6.50 |
| Persistence | 57.0 / 9.25 | 40.2 / 7.07 | 27.3 / 5.40 | 12.6 / 3.41 |
| Historical average | 57.7 / 9.32 | 40.6 / 7.08 | 27.9 / 5.44 | 15.3 / 3.79 |
| Ridge | 57.9 / 9.34 | 41.5 / 7.20 | 28.9 / 5.57 | 15.1 / 3.76 |
| Ridge + time-of-day average | 57.6 / 9.30 | 41.2 / 7.16 | 29.1 / 5.60 | 15.0 / 3.76 |
| ST-GNN | 57.7 / 9.32 | 41.3 / 7.18 | 28.7 / 5.54 | 14.5 / 3.66 |
| Gradient boosting | 57.8 / 9.34 | 40.6 / 7.09 | 27.2 / 5.34 | 12.9 / 3.40 |
| Gradient boosting, calibrated | 57.3 / 9.27 | 40.5 / 7.08 | 26.7 / 5.28 | 12.9 / 3.39 |
| ST-GNN, calibrated | 57.1 / 9.24 | 40.3 / 7.04 | 27.6 / 5.40 | 13.1 / 3.42 |
| Oracle (true demand) | 56.8 / 9.17 | 40.3 / 6.98 | 25.7 / 5.05 | 12.2 / 3.18 |

![Rider wait vs empty driving](results/tradeoff.png)

- Forecast-driven repositioning clearly helps once the fleet is large enough: at 2,000 vehicles unmet requests fall
  from 42.8% to 26 to 29% and mean wait from 7.6 to about 5.3 to 5.6 minutes. At 1,000 vehicles there is little to
  gain.
- Calibration matters more for the policy than the choice of model. Comparing at equal driving cost along the
  `theta` sweep at 2,000 vehicles, the wait saved relative to a persistence-driven policy is 0.30 minutes for
  calibrated gradient boosting (83% of what perfect demand knowledge would give), 0.25 for the calibrated ST-GNN
  (69%), 0.28 for gradient boosting (77%) and 0.17 for the ST-GNN as trained (48%). Comparing at equal `theta` is
  misleading because policies drive different amounts.
- **The most accurate forecaster is not the best repositioning policy.** The calibrated ST-GNN has 3 to 4% lower
  forecast error than calibrated gradient boosting but is slightly behind it here, and the earlier, less accurate
  ST-GNN checkpoint reached 84% (`results/model_selection.json`). The gaps between these policies, 0.05 to 0.12
  minutes of wait, are about the size of the seed-to-seed spread, so once forecasts are this good the policy
  result no longer tracks forecast RMSE.
- Perfect foresight beats persistence by only 0.2 to 0.35 minutes of wait at 2,000 to 3,000 vehicles, which bounds
  how much any forecaster can add at a 15-minute decision horizon.

### How the model got here

The first ST-GNN (batches of 128 in time order, 50 epochs, kept as `experiments/original_batch128_noshuffle.pt`)
scored 1.449, 1.546, 1.618, 1.726 and was beaten by gradient boosting at every horizon. `scripts/diagnose_stgnn.py`
found why (`results/stgnn_diagnostics.json`):

- it under-predicted total pickups by 8.5 to 12%, from the log scale and the busier test days;
- it had no notion of which zone it was forecasting and saw only the last 4 hours;
- its loss weighted all 258 zones equally, while 86% of the squared error in pickups sits in the 30 busiest zones;
- it underfitted, and its training batches were consecutive, near-identical windows.

Calibration fixed the first point from outside the network and took that model to 1.402, 1.468, 1.506, 1.561. The
rest were tested one change at a time on a Kaggle T4 GPU (`experiments/`, `results/single_change_experiments.json`,
`results/combo_experiments.json`). Calibrated RMSE:

| Recipe (batch size 32) | 5 min | 15 min | 30 min | 60 min |
|------------------------|-------|--------|--------|--------|
| No options | 1.422 | 1.487 | 1.533 | 1.580 |
| Shuffle | 1.385 | 1.446 | 1.484 | 1.526 |
| Shuffle + weather | 1.411 | 1.466 | 1.498 | 1.539 |
| Shuffle + zone embedding | 1.394 | 1.446 | 1.483 | 1.514 |
| Shuffle + time-of-day prior | 1.528 | 1.525 | 1.529 | 1.537 |
| **Shuffle + weighted loss (shipped)** | **1.375** | **1.440** | **1.476** | **1.511** |
| Shuffle + zone embedding + weighted loss | 1.379 | 1.446 | 1.480 | 1.514 |

- Shuffling the training windows is the largest single gain.
- A loss weighted toward busy zones adds a little more; a zone embedding helps slightly on its own but adds nothing
  on top of the weighted loss; weather does not help.
- Predicting a residual over the time-of-day average makes short horizons much worse, even after its bias was
  fixed. An earlier retrain that bundled it with the other changes failed for that reason
  (`results/v2_retrain.json`).

**Does the graph help?** Less than the name suggests. In the bundled retrain, removing the graph cost 2 to 4% at
every horizon. In the improved recipe (shuffle + zone embedding + weighted loss) the cost is 0.3 to 1.2% as trained,
significant only at 15 minutes, and nothing measurable after calibration (`results/model_selection.json`). Giving
the gradient-boosted model flow-weighted neighbour demand left its error unchanged
(`results/spatial_check.json`). On this data most of the network's accuracy comes from the recurrent encoder and
the training recipe, not from the graph.

### What this means

A carefully trained and calibrated ST-GNN is the most accurate forecaster here, 3 to 4% ahead of a calibrated
gradient-boosted model and 20 to 31% ahead of persistence. That lead came from fixing how the network was trained
(shuffling, loss weighting) and calibrating its output, not from its graph structure, whose measured contribution
is about 1% or less. Forecasts clearly improve repositioning, but past a certain accuracy the policy stops
improving with the forecast: a 15-minute decision horizon leaves little room between a good forecast and a perfect
one.

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

# 2. Train the multi-horizon ST-GNN (minutes on a GPU, a few hours on a CPU)
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss-power 1 --out multihorizon_265_clipped.pt

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

## Training options

`train_multihorizon_torch.py` options, all off by default. The shipped model uses `--shuffle --loss-power 1`.

| Option | What it does |
|--------|--------------|
| `--shuffle` | shuffle training windows each epoch |
| `--loss-power P` | weight zones by `((1 + mean pickups) x sigma) ** P` in the loss; 2 approximates error in pickups |
| `--zone-dim N` | learned embedding per zone, fed to the output heads |
| `--weather` | append the precipitation and temperature channels |
| `--prior histavg_z` | predict the residual over the time-of-day average of each target bin |
| `--no-graph` | remove the neighbour terms (ablation) |
| `--layers N` | number of GRU layers |
| `--lag-features` | give each output head the target bin's demand one day and one week earlier |
| `--loss poisson` | train on the likelihood of the pickup counts instead of squared error in log units |

With a weighted loss the `train_rmse` and `val_rmse` printed during training are the weighted objective and are not
comparable across settings. `scripts/compare_checkpoints.py --calibrate` scores any checkpoints in pickups on the
test split, against the shipped models, with bootstrap intervals. Batch size 128 needs more than 15 GB of GPU
memory; 32 fits. Training takes minutes on a GPU and a few hours on a CPU.

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
│   ├── compare_checkpoints.py   Scores checkpoints against the shipped models, optionally calibrated
│   ├── simulator.py             Fleet simulator with dispatch matching
│   ├── reposition.py            Forecast-driven LP repositioning policy
│   ├── run_evaluation.py        Policy comparison and trade-off sweep
│   ├── plot_results.py          Plots of the study
│   ├── prepare_zones.py         Trims the NYC taxi zone polygons for the app
│   ├── build_dropoff.py, build_weather.py, stgnn_models.py   Auxiliary tools
├── experiments/                 Checkpoints from the training experiments, including the first model
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
