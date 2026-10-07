# SurgeMap

**NYC taxi demand forecasting and fleet repositioning with a spatio-temporal graph neural network**

SurgeMap forecasts how many taxi pickups each of New York's taxi zones will see over the next 5, 15, 30 and 60
minutes, then tests in a simulator whether using those forecasts to pre-position idle vehicles shortens rider
waits. It is part of the [FleetMg](https://github.com/shris-xyz/FleetMg) fleet management project.

The repository contains:

- **forecasters**: a directed-graph ST-GNN (Graph WaveNet) trained on four months of NYC yellow-taxi trips, compared with
  persistence, historical-average, ridge and gradient-boosted baselines fitted on the same data
- a **fleet simulator** that replays real pickups against a simulated fleet with nearest-vehicle dispatch
- a **repositioning policy**: a small min-cost-flow linear program driven by any forecast
- an **evaluation** of policies across fleet sizes, seeds and a cost trade-off sweep
- a **Streamlit app** to explore forecasts and run the simulator

---

## Data

- **Source**: [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  (official parquet files; January 2024 has 2,964,624 rows, of which 2,927,173 pass the validity filters)
- **Zones**: 258 taxi zones with at least one pickup; five of them have fewer than five trips in the month
- **Time grid**: 8,928 five-minute bins covering the whole month, none empty
- **Split**: chronological 70% train (to 22 Jan 16:45), 15% validation, 15% test (27 Jan 08:20 to 31 Jan 23:55)
- **Graph**: directed row-normalised flow matrices `A_out` and `A_in` from pickup-to-dropoff counts
- **Training history**: the shipped models are fitted on October 2023 to January 2024 (35,424 bins, same
  zones), with the same validation days and test window as above. That dataset is built by
  `preprocess.py` from the four monthly files and is not tracked; `real_processed_265` (January) is what
  the simulator and the app run on.

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

The shipped network is a Graph WaveNet (Wu et al., 2019; `scripts/stgnn_arch.py`):

1. **Temporal layers**: eight gated, dilated convolutions (dilations 1, 2, 4, 8, 16, 1, 2, 4) read the 48-bin
   (4 hour) window of each zone.
2. **Spatial layers**: after every temporal layer a graph convolution mixes each zone's state with its
   neighbours', two hops at a time, over three graphs: the taxi-flow graph in each direction (`A_out`, `A_in`)
   and an adjacency the network learns itself from two sets of zone embeddings.
3. **Multi-horizon heads**: every layer contributes to a skip connection, and one linear head per horizon
   (1, 3, 6, 12 bins ahead) reads the result.

Each output head also receives the demand of its own target time one day and one week earlier. Trained with AdamW
on shuffled windows, a Poisson loss on the pickup counts, gradient clipping and early stopping on the validation
loss. One network has 147,644 parameters; the shipped forecast is the average of three trained with different
seeds.

The first design, kept in the code as `--arch gru`, applied one graph convolution to the raw inputs and then
encoded each zone separately with a GRU (26,644 parameters). [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
shows what the change bought.

### Baselines

| Baseline | Description |
|----------|-------------|
| Persistence | the last observed bin |
| Historical average | mean training demand for the same zone, weekday and time of day |
| Ridge regression | one ridge model per zone on the flattened 48-bin window |
| Gradient boosting | one scikit-learn `HistGradientBoostingRegressor` per horizon across all zones, Poisson loss with L2 regularisation; recent lags, same time yesterday and last week, calendar, weather, citywide demand |

---

## Results

Held-out test window: 27 Jan 08:20 to 31 Jan 23:55 2024 (1,329 forecast times, 258 zones). No model was fitted
on it. Every fitted model below, the ST-GNN and the baselines alike, was trained on the same four months
(1 October 2023 to 22 January 2024). The shipped ST-GNN is the average of three Graph WaveNets (`model/`) trained
with different random seeds. How the architecture and the recipe were arrived at is described in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### Forecast accuracy

RMSE in pickups per zone per 5-minute bin (lower is better), from `artifacts/forecast_metrics.json`:

| Model | 5 min | 15 min | 30 min | 60 min |
|-------|-------|--------|--------|--------|
| **ST-GNN, calibrated** | **1.326** | **1.386** | **1.406** | **1.421** |
| ST-GNN | 1.326 | 1.390 | 1.411 | 1.425 |
| Gradient boosting, calibrated | 1.358 | 1.423 | 1.447 | 1.471 |
| Gradient boosting | 1.371 | 1.447 | 1.480 | 1.514 |
| Ridge regression | 1.428 | 1.534 | 1.625 | 1.769 |
| Historical average | 1.543 | 1.545 | 1.547 | 1.552 |
| Persistence | 1.716 | 1.889 | 2.016 | 2.189 |

"Calibrated" means two corrections fitted on the validation split only and applied identically to both models
(`scripts/calibrate_forecasts.py`): a per-zone, per-horizon bias correction, then a per-horizon blend with the
time-of-day average. Both models are trained on pickup counts with a Poisson loss and are nearly unbiased already.
Calibration adds little to the ST-GNN and more to gradient boosting, which is not given the time-of-day average
as a feature and so gains from the blend.

95% block-bootstrap intervals on RMSE(other) minus RMSE(ST-GNN), positive meaning the ST-GNN is better
(`scripts/accuracy_checks.py`):

| ST-GNN as trained vs | 5 min | 15 min | 30 min | 60 min |
|----------------------|-------|--------|--------|--------|
| Persistence | +0.391 [+0.351, +0.426] | +0.499 [+0.452, +0.544] | +0.605 [+0.551, +0.661] | +0.764 [+0.667, +0.866] |
| Historical average | +0.218 [+0.168, +0.271] | +0.155 [+0.123, +0.192] | +0.136 [+0.106, +0.169] | +0.128 [+0.098, +0.158] |
| Ridge regression | +0.103 [+0.073, +0.135] | +0.145 [+0.109, +0.184] | +0.215 [+0.161, +0.269] | +0.344 [+0.254, +0.440] |
| Gradient boosting | +0.045 [+0.039, +0.052] | +0.058 [+0.048, +0.068] | +0.068 [+0.058, +0.080] | +0.089 [+0.073, +0.105] |

- The ST-GNN is the most accurate model at every horizon, and every interval excludes zero.
- **It leads gradient boosting by 3.3 to 5.9% as trained and 2.4 to 3.4% after calibration** (+0.033 [+0.026,
  +0.040], +0.037 [+0.028, +0.046], +0.041 [+0.030, +0.053], +0.050 [+0.037, +0.063]). The lead grows with the
  horizon.
- It beats the historical average by 8 to 14%, ridge regression by 7 to 19%, and persistence by 23 to 35%.
- Pickups arrive at random, so even a perfectly known demand rate leaves an RMSE of about 1.16 here. The
  calibrated ST-GNN is 14% above that floor at 5 minutes and 22% at 60 (`scripts/headroom_analysis.py`).

**Where the lead comes from** (`scripts/regime_analysis.py`, `results/regime_analysis.json`). Splitting the test
window by how unusual citywide demand is, and by how busy the zone is, the calibrated ST-GNN's RMSE is lower than
calibrated gradient boosting's by:

| Part of the test window | 5 min | 15 min | 30 min | 60 min |
|-------------------------|-------|--------|--------|--------|
| Demand far below usual (lowest 5% of periods) | 4.7% | 7.8% | 10.6%* | 10.1% |
| Ordinary periods (middle 50%) | 1.9% | 2.3% | 2.3% | 2.4% |
| Demand far above usual (top 5% of periods) | 3.7% | 4.7% | 5.7% | 7.4% |
| Busiest 10% of zones | 2.7% | 2.9% | 3.1% | 3.8% |
| Quietest 70% of zones | 0.2% | 0.2% | 0.2% | 0.2% |

\* the 95% interval touches zero; each 5% slice holds only about 66 forecast times.

The ST-GNN is ahead in every slice. Its lead is two to four times larger when demand departs from the usual
pattern than in ordinary periods: in surges it is 4 to 7% more accurate, which is when a forecast matters most.
Nearly all of the lead is in the busy zones; in the quietest 70% of zones, which see almost no pickups, the two
models are within 0.2%.

Two cautions. The architecture and the recipe were chosen over several rounds of experiments by looking at this
same test window, so the ST-GNN's margin is somewhat optimistic; the baselines were not tuned that way. Seed-to-seed
variation for one Graph WaveNet, measured on January, is about 0.005 RMSE at 5 minutes and 0.011 at 60, well below
the gap to gradient boosting.

Hotspot ranking, as the mean number of the 5 busiest zones also among the 5 predicted, across 5 to 60 minutes:
calibrated ST-GNN 3.29 to 3.18, calibrated gradient boosting 3.27 to 3.12, persistence 2.93 to 2.55. The top-3 hit
rate is 0.86 to 0.97 for every model, because the busiest zones barely change, so it does not separate them.

### Repositioning

Rider outcomes over the test window with `theta = 0.1`, by fleet size. Cells show unmet requests (%) and mean
rider wait (minutes, unmet requests counted as 15). Mean over 5 seeds; the standard deviation across seeds is
about 0.1 minute.

| Policy driven by | 1000 vehicles | 1500 | 2000 | 3000 |
|------------------|---------------|------|------|------|
| Dispatch only | 61.5 / 9.98 | 49.7 / 8.49 | 42.8 / 7.63 | 33.7 / 6.50 |
| Persistence | 57.0 / 9.25 | 40.2 / 7.07 | 27.3 / 5.40 | 12.6 / 3.41 |
| Ridge | 58.1 / 9.36 | 40.9 / 7.12 | 28.4 / 5.52 | 14.1 / 3.60 |
| Historical average | 57.5 / 9.30 | 39.7 / 6.96 | 26.5 / 5.23 | 12.2 / 3.24 |
| Gradient boosting | 57.9 / 9.35 | 40.6 / 7.10 | 26.3 / 5.23 | 12.5 / 3.32 |
| Gradient boosting, calibrated | 57.3 / 9.28 | 40.3 / 7.06 | 26.7 / 5.28 | 12.6 / 3.33 |
| ST-GNN | 57.7 / 9.32 | 40.5 / 7.07 | 26.9 / 5.30 | 12.8 / 3.37 |
| **ST-GNN, calibrated** | 57.2 / 9.26 | 39.9 / 7.00 | 26.5 / 5.25 | 12.6 / 3.33 |
| Oracle (true demand) | 56.8 / 9.17 | 40.3 / 6.98 | 25.7 / 5.05 | 12.2 / 3.18 |

![Rider wait vs empty driving](results/tradeoff.png)

- Forecast-driven repositioning clearly helps once the fleet is large enough: at 2,000 vehicles unmet requests fall
  from 42.8% to 26 to 29% and mean wait from 7.6 to about 5.2 to 5.5 minutes. At 1,000 vehicles there is little to
  gain.
- Comparing at equal driving cost along the `theta` sweep at 2,000 vehicles, the wait saved relative to a
  persistence-driven policy is 0.32 minutes for calibrated gradient boosting (90% of what perfect demand knowledge
  would give), 0.31 for the calibrated ST-GNN (87%), 0.29 for gradient boosting (80%) and 0.28 for the ST-GNN as
  trained (79%). Comparing at equal `theta` is misleading because policies drive different amounts.
- **A good forecast is enough; the best forecast adds nothing measurable.** The ST-GNN and gradient boosting are
  indistinguishable as policies even though the ST-GNN forecasts 2 to 6% better, and a plain four-month
  time-of-day average does as well as either at `theta = 0.1` (5.23 and 3.24 minutes at 2,000 and 3,000
  vehicles). The differences between these policies are within the seed-to-seed spread.
- Perfect foresight beats persistence by only 0.2 to 0.35 minutes of wait at 2,000 to 3,000 vehicles, which bounds
  how much any forecaster can add at a 15-minute decision horizon.

### What made the difference

The first ST-GNN here lost to gradient boosting at every horizon. What turned that round, in the order found:

- **Shuffling the training windows**, the largest single gain from the training recipe.
- **A Poisson loss on pickup counts** in place of squared error on log demand, which also removed an 8 to 12%
  under-prediction of total volume.
- **Yesterday's and last week's demand** for each target time as extra inputs.
- **Four months of history** in place of one, and averaging seeds.
- **The architecture**: a Graph WaveNet is 1 to 2% more accurate than the first design, which mixed zones once
  and then encoded each with a GRU.

A wider or deeper first design, weather inputs, zone embeddings, a time-of-day prior and several after-the-fact
corrections did not help. The measurements behind each statement are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### What this means

On equal data, the ST-GNN is the most accurate forecaster here: 3 to 6% ahead of a gradient-boosted model with
simple features, 2.4 to 3.4% once both are calibrated, and 4 to 7% ahead in surges. Two things produced that. The
first was how the network is trained: shuffling, a loss that matches count data, yesterday's and last week's
demand as inputs, more history and averaging seeds. The second was the architecture: a network that lets zones
exchange what they have encoded at every layer, over a graph it partly learns, beat one that mixed them once.
Making the first design bigger did nothing. The lead is real but modest, because most of the remaining error is
the randomness of individual pickups. For the repositioning policy the choice of forecaster does not matter once
it is unbiased and knows the daily pattern: a four-month time-of-day average performs as well as either model.

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

**Evaluation** (`scripts/run_evaluation.py`). Policies driven by persistence, historical average, ridge,
gradient boosting, the ST-GNN (both as trained and calibrated) and the true demand ("oracle", an upper bound) are compared with dispatch alone, over several fleet sizes
and seeds, with a `theta` sweep to trace each policy's wait vs driving curve. Policies are also compared with
persistence at equal driving cost, so a policy that simply moves fewer vehicles is not mistaken for a worse one.

The fleet is synthetic, because public trip data has no vehicle positions. Read the study as a comparison between
policies, not as a prediction of real-world performance.

---

## Usage

```powershell
# Install (Python 3.11+; a virtual environment is recommended)
python -m pip install -r requirements.txt

# 1. Preprocess January 2024 into real_processed_265/ (the dataset the simulator and app use)
python scripts\preprocess.py --input data\yellow_tripdata_2024-01.parquet --out real_processed_265 --month 2024-01 --all-zones

# 2. Build the four-month training dataset (same zones, validation days and test window)
python scripts\preprocess.py --input data\yellow_tripdata_2023-10.parquet,data\yellow_tripdata_2023-11.parquet,data\yellow_tripdata_2023-12.parquet,data\yellow_tripdata_2024-01.parquet --out real_processed_4mo --start 2023-10-01 --end 2024-02-01 --zone-ids-from real_processed_265 --train-end "2024-01-22 16:45" --val-end "2024-01-27 08:20"

# 3. Train three seeds of the ST-GNN on it (about two hours each on a GPU; impractical on a CPU)
python scripts\train_multihorizon_torch.py --data-dir real_processed_4mo --window 48 --epochs 45 --batch-size 32 --lr 0.001 --patience 6 --shuffle --loss poisson --lag-features --arch gwnet --amp --seed 7 --out model\graph_wavenet_seed7.pt
python scripts\train_multihorizon_torch.py --data-dir real_processed_4mo --window 48 --epochs 45 --batch-size 32 --lr 0.001 --patience 6 --shuffle --loss poisson --lag-features --arch gwnet --amp --seed 1 --out model\graph_wavenet_seed1.pt
python scripts\train_multihorizon_torch.py --data-dir real_processed_4mo --window 48 --epochs 45 --batch-size 32 --lr 0.001 --patience 6 --shuffle --loss poisson --lag-features --arch gwnet --amp --seed 2 --out model\graph_wavenet_seed2.pt

# 4. Forecasts (the average of the networks in model/), baselines fitted on the same data, calibration
python scripts\export_predictions.py --data-dir real_processed_4mo --checkpoint model --out-dir artifacts_4mo
python scripts\accuracy_checks.py --data-dir real_processed_4mo --predictions artifacts_4mo\predictions.npz --merge
python scripts\calibrate_forecasts.py --data-dir real_processed_4mo --checkpoint model --predictions artifacts_4mo\predictions.npz

# 5. Re-index the forecasts onto the January dataset used by the simulator and the app
python scripts\align_predictions.py --predictions artifacts_4mo\predictions.npz --out artifacts\predictions.npz
copy artifacts_4mo\forecast_metrics.json artifacts\forecast_metrics.json

# 6. Run the repositioning study and draw the plots (a few minutes on 10 cores)
python scripts\run_evaluation.py --workers 10
python scripts\plot_results.py

# 7. Explore everything in the app
python -m streamlit run app\streamlit_app.py
```

The same steps are available as `make preprocess-265`, `make preprocess-4mo`, `make train`, `make export`, `make evaluate`,
`make plots`, `make app` and `make test`.

---

## Project structure

```
SurgeMap/
├── app/                         Streamlit app (pages, helpers, trimmed zone polygons)
├── scripts/
│   ├── preprocess.py            Raw trips (CSV or parquet) -> tensors and graph
│   ├── train_multihorizon_torch.py   Multi-horizon ST-GNN trainer
│   ├── stgnn_arch.py            Graph WaveNet and its building blocks
│   ├── train_stgnn_torch.py     Single-horizon trainer; defines the graph convolution
│   ├── multihorizon_baseline.py Persistence and ridge baselines (z-score space)
│   ├── hotspot_eval.py          Top-k hotspot evaluation of a checkpoint
│   ├── export_predictions.py    Count-space forecasts for the simulator and app
│   ├── accuracy_checks.py       Extra baselines, bootstrap intervals, merge into the forecasts
│   ├── regime_analysis.py       Accuracy by how unusual the period and how busy the zone is
│   ├── calibrate_forecasts.py   Validation-fitted bias correction and blend for both models
│   ├── diagnose_stgnn.py        Where the ST-GNN's error comes from
│   ├── compare_checkpoints.py   Scores checkpoints against the shipped models, optionally calibrated
│   ├── simulator.py             Fleet simulator with dispatch matching
│   ├── reposition.py            Forecast-driven LP repositioning policy
│   ├── run_evaluation.py        Policy comparison and trade-off sweep
│   ├── plot_results.py          Plots of the study
│   ├── prepare_zones.py         Trims the NYC taxi zone polygons for the app
│   ├── build_dropoff.py, build_weather.py, stgnn_models.py   Auxiliary tools
├── model/                       The three shipped Graph WaveNet checkpoints (trained on four months); forecasts are averaged
├── experiments/                 Checkpoints from the training experiments, including earlier models
├── tests/                       pytest suite
├── real_processed_265/          Preprocessed arrays (demand, features, graph, metadata)
├── docs/                        EXPERIMENTS.md: what was tried and what each change was worth
├── notebooks/                   Feature experiments
└── config.yaml                  Reference settings (not read at runtime)
```

---

## Limitations

- Four months of training data and one five-day test window, so seasonality is barely covered and the test
  window is short.
- The flow graph is built from the whole month, including the test period; this is a mild form of leakage.
- Observed pickups are what taxis actually served, which understates true demand where supply was short.
- The repositioning study uses a synthetic fleet, a simple dispatch rule and lost (not queued) requests.
- Only yellow taxis are included.

## Acknowledgements

- NYC Taxi & Limousine Commission for the trip record data
- NYC Open Data for the taxi zone boundaries
