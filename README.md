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

Each output head also receives the demand of its own target time one day and one week earlier. Trained with AdamW
on shuffled windows, a Poisson loss on the pickup counts, gradient clipping and early stopping on the validation
loss. The shipped forecast is the average of five such networks trained with different seeds.

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
The shipped ST-GNN is the average of five networks (`model/`) trained with different random seeds on a GPU, each
with shuffled batches of 32, a Poisson loss on the pickup counts, and yesterday's and last week's demand for the
target time as extra inputs. How that recipe was arrived at is described under
[How the model got here](#how-the-model-got-here).

### Forecast accuracy

RMSE in pickups per zone per 5-minute bin (lower is better), from `artifacts/forecast_metrics.json`:

| Model | 5 min | 15 min | 30 min | 60 min |
|-------|-------|--------|--------|--------|
| **ST-GNN, calibrated** | **1.356** | **1.422** | **1.451** | **1.480** |
| ST-GNN | 1.360 | 1.430 | 1.465 | 1.501 |
| Gradient boosting, calibrated | 1.422 | 1.506 | 1.539 | 1.577 |
| Gradient boosting | 1.429 | 1.523 | 1.558 | 1.610 |
| Ridge + time-of-day average | 1.476 | 1.575 | 1.602 | 1.661 |
| Ridge regression | 1.467 | 1.580 | 1.681 | 1.844 |
| Persistence | 1.716 | 1.889 | 2.016 | 2.189 |
| Historical average | 1.701 | 1.702 | 1.703 | 1.706 |

"Calibrated" means two corrections fitted on the validation split only and applied identically to both models
(`scripts/calibrate_forecasts.py`): a per-zone, per-horizon bias correction, then a per-horizon blend with the
time-of-day average. With the Poisson loss the ST-GNN no longer under-predicts volume (predicted/actual 0.99), so
calibration now adds little to it.

95% block-bootstrap intervals on RMSE(other) minus RMSE(ST-GNN), positive meaning the ST-GNN is better
(`scripts/accuracy_checks.py`):

| ST-GNN as trained vs | 5 min | 15 min | 30 min | 60 min |
|----------------------|-------|--------|--------|--------|
| Persistence | +0.357 [+0.318, +0.392] | +0.459 [+0.414, +0.500] | +0.551 [+0.503, +0.601] | +0.688 [+0.603, +0.782] |
| Historical average | +0.341 [+0.254, +0.436] | +0.272 [+0.198, +0.353] | +0.239 [+0.167, +0.313] | +0.205 [+0.145, +0.268] |
| Ridge regression | +0.107 [+0.069, +0.149] | +0.150 [+0.102, +0.200] | +0.216 [+0.149, +0.283] | +0.343 [+0.243, +0.452] |
| Ridge + time-of-day average | +0.116 [+0.082, +0.160] | +0.144 [+0.102, +0.197] | +0.137 [+0.098, +0.189] | +0.160 [+0.108, +0.217] |
| Gradient boosting | +0.070 [+0.053, +0.088] | +0.093 [+0.062, +0.130] | +0.093 [+0.063, +0.126] | +0.109 [+0.082, +0.137] |

- The ST-GNN beats persistence by 21 to 32% and every other baseline at every horizon, with every interval
  excluding zero.
- Against gradient boosting the margin is 5 to 7% as trained and 5 to 6% after the same calibration:
  +0.066 [+0.050, +0.083], +0.085 [+0.060, +0.116], +0.088 [+0.062, +0.115] and +0.097 [+0.077, +0.120].
- Pickups arrive at random, so even a perfectly known demand rate leaves an RMSE of about 1.16 here. The
  calibrated ST-GNN is 17% above that floor at 5 minutes and 27% at 60 (`scripts/headroom_analysis.py`).

Two cautions. The recipe was chosen over several rounds of experiments by looking at this same test window, so
its margin is somewhat optimistic. Seed-to-seed variation is measured (about 0.002 RMSE at 5 minutes and 0.006 at
60 for one network) and is far smaller than the gaps to the baselines.

Hotspot ranking, as the mean number of the 5 busiest zones also among the 5 predicted, across 5 to 60 minutes:
calibrated ST-GNN 3.26 to 3.13, calibrated gradient boosting 3.18 to 3.06, persistence 2.93 to 2.55. The top-3 hit
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
| Gradient boosting | 57.8 / 9.34 | 40.6 / 7.09 | 27.2 / 5.34 | 12.9 / 3.40 |
| Gradient boosting, calibrated | 57.3 / 9.27 | 40.5 / 7.08 | 26.7 / 5.28 | 12.9 / 3.39 |
| ST-GNN | 57.1 / 9.25 | 40.4 / 7.07 | 26.7 / 5.28 | 12.8 / 3.37 |
| **ST-GNN, calibrated** | 57.5 / 9.30 | 40.2 / 7.03 | 26.7 / 5.27 | 12.6 / 3.32 |
| Oracle (true demand) | 56.8 / 9.17 | 40.3 / 6.98 | 25.7 / 5.05 | 12.2 / 3.18 |

![Rider wait vs empty driving](results/tradeoff.png)

- Forecast-driven repositioning clearly helps once the fleet is large enough: at 2,000 vehicles unmet requests fall
  from 42.8% to 26 to 29% and mean wait from 7.6 to about 5.3 to 5.6 minutes. At 1,000 vehicles there is little to
  gain.
- Comparing at equal driving cost along the `theta` sweep at 2,000 vehicles, the wait saved relative to a
  persistence-driven policy is 0.30 minutes for calibrated gradient boosting (83% of what perfect demand knowledge
  would give), 0.29 for the calibrated ST-GNN (81%), 0.28 for gradient boosting (77%) and 0.27 for the ST-GNN as
  trained (75%). Comparing at equal `theta` is misleading because policies drive different amounts.
- **Forecast accuracy and policy quality are only loosely linked.** The ST-GNN's forecasts are 5 to 6% more accurate
  than gradient boosting's, yet as repositioning policies the two are indistinguishable: the gaps between them,
  0.01 to 0.07 minutes of wait, are below the seed-to-seed spread. An earlier ST-GNN that was more accurate than
  its predecessor scored worse as a policy (`results/model_selection.json`). Removing bias matters more to the
  policy than the last few percent of RMSE.
- Perfect foresight beats persistence by only 0.2 to 0.35 minutes of wait at 2,000 to 3,000 vehicles, which bounds
  how much any forecaster can add at a 15-minute decision horizon.

### How the model got here

The first ST-GNN (batches of 128 in time order, squared error on log demand, kept as
`experiments/original_batch128_noshuffle.pt`) scored 1.449, 1.546, 1.618, 1.726 and was beaten by gradient boosting
at every horizon. `scripts/diagnose_stgnn.py` found why (`results/stgnn_diagnostics.json`):

- it under-predicted total pickups by 8.5 to 12%, from the log scale and the busier test days;
- it had no notion of which zone it was forecasting and saw only the last 4 hours;
- its loss weighted all 258 zones equally, while 86% of the squared error in pickups sits in the 30 busiest zones;
- it underfitted, and its training batches were consecutive, near-identical windows.

Each idea was then tested on a Kaggle T4 GPU, one change at a time (`experiments/`, and
`results/single_change_experiments.json`, `combo_experiments.json`, `round2_experiments.json`,
`round3_experiments.json`, `round4_experiments.json`). Calibrated RMSE, one network unless stated:

| Recipe (batch size 32) | 5 min | 15 min | 30 min | 60 min |
|------------------------|-------|--------|--------|--------|
| First model (batch 128, no shuffle) | 1.402 | 1.468 | 1.506 | 1.561 |
| No options | 1.422 | 1.487 | 1.533 | 1.580 |
| Shuffle | 1.385 | 1.446 | 1.484 | 1.526 |
| Shuffle + weather | 1.411 | 1.466 | 1.498 | 1.539 |
| Shuffle + zone embedding | 1.394 | 1.446 | 1.483 | 1.514 |
| Shuffle + time-of-day prior | 1.528 | 1.525 | 1.529 | 1.537 |
| Shuffle + weighted loss | 1.375 | 1.440 | 1.476 | 1.511 |
| ... range over five seeds | 1.372 to 1.381 | 1.434 to 1.444 | 1.467 to 1.478 | 1.496 to 1.513 |
| ... average of five seeds | 1.369 | 1.432 | 1.465 | 1.494 |
| Shuffle + weighted loss, 128 hidden units | 1.369 | 1.437 | 1.477 | 1.513 |
| Shuffle + weighted loss, two GRU layers | 1.373 | 1.439 | 1.477 | 1.509 |
| Shuffle + weighted loss + lagged inputs | 1.371 | 1.436 | 1.470 | 1.496 |
| Shuffle + Poisson loss | 1.373 | 1.438 | 1.467 | 1.497 |
| Shuffle + Poisson loss + lagged inputs | 1.364 | 1.429 | 1.459 | 1.490 |
| ... range over five seeds | 1.360 to 1.366 | 1.425 to 1.431 | 1.452 to 1.465 | 1.481 to 1.497 |
| **... average of five seeds (shipped)** | **1.356** | **1.422** | **1.451** | **1.480** |

- Shuffling the training windows is the largest single gain.
- A loss aimed at pickup counts helps: first by weighting busy zones, then, better, as a Poisson likelihood, which
  also removes the volume bias.
- Giving the output heads the target time's demand a day and a week earlier helps a little alone and more with
  the Poisson loss. Forcing the same information in as a fixed offset (the time-of-day prior) made short horizons
  much worse.
- More capacity does not help: 128 hidden units and a second GRU layer both land inside the seed-to-seed spread.
  Neither do weather inputs or a zone embedding on top of the better loss.
- Averaging five seeds gives a further 0.5 to 1%.
- Nothing applied after the fact adds much to the final model: averaging with gradient boosting, online bias
  correction, a last-error correction and Hedge weighting over five forecasters each gained 0.6% or less on the
  previous model (`results/headroom_analysis.json`).

**Does the graph help?** Less than the name suggests. In an early bundled retrain, removing the graph cost 2 to 4%
at every horizon. In a better-trained recipe (shuffle + zone embedding + weighted loss) the cost was 0.3 to 1.2% as
trained, significant only at 15 minutes, and nothing measurable after calibration
(`results/model_selection.json`). Giving the gradient-boosted model flow-weighted neighbour demand left its error
unchanged (`results/spatial_check.json`). On this data most of the network's accuracy comes from the recurrent
encoder and the training recipe, not from the graph. The ablation has not been repeated on the shipped recipe.

### What this means

A carefully trained ST-GNN is the most accurate forecaster here, 5 to 6% ahead of a calibrated gradient-boosted
model and 21 to 32% ahead of persistence. That lead came from fixing how the network is trained (shuffling, a loss
that matches count data, the right extra inputs, averaging seeds), not from its graph structure, whose measured
contribution is about 1% or less, nor from making it bigger. Forecasts clearly improve repositioning, but the
policy result depends on the forecast being unbiased far more than on the last few percent of accuracy: a
15-minute decision horizon leaves little room between a good forecast and a perfect one.

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

# 2. Train five seeds of the ST-GNN (about 10 minutes each on a GPU, hours each on a CPU)
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss poisson --lag-features --seed 7 --out model\poisson_lag_seed7.pt
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss poisson --lag-features --seed 1 --out model\poisson_lag_seed1.pt
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss poisson --lag-features --seed 2 --out model\poisson_lag_seed2.pt
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss poisson --lag-features --seed 3 --out model\poisson_lag_seed3.pt
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss poisson --lag-features --seed 4 --out model\poisson_lag_seed4.pt

# 3. Export count-space forecasts (the average of the checkpoints in model/) and baselines
python scripts\export_predictions.py --checkpoint model

# 4. Add the extra baselines (ridge + time-of-day average, gradient boosting) and the bootstrap intervals
python scripts\accuracy_checks.py --merge

# 5. Add the validation-calibrated ST-GNN and boosted model
python scripts\calibrate_forecasts.py --checkpoint model

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

`train_multihorizon_torch.py` options, all off by default. The shipped networks use
`--shuffle --loss poisson --lag-features`.

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

### Training on more months

`preprocess.py` accepts several trip files and a date range, can reuse the zones of an existing dataset, and takes
explicit split times. This builds a longer training history while keeping the same zones, validation days and test
window, so `compare_checkpoints.py` can score the result against the shipped forecasts:

```powershell
python scripts\preprocess.py --input data\yellow_tripdata_2023-12.parquet,data\yellow_tripdata_2024-01.parquet --out real_processed_2mo --start 2023-12-01 --end 2024-02-01 --zone-ids-from real_processed_265 --train-end "2024-01-22 16:45" --val-end "2024-01-27 08:20"
```

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
├── model/                       The five shipped checkpoints; their forecasts are averaged
├── experiments/                 Checkpoints from the training experiments, including earlier models
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
