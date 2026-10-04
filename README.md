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
| **Gradient boosting** | **1.429** | **1.523** | **1.558** | **1.610** |
| ST-GNN | 1.449 | 1.546 | 1.618 | 1.726 |
| Ridge + time-of-day average | 1.476 | 1.575 | 1.602 | 1.661 |
| Ridge regression | 1.467 | 1.580 | 1.681 | 1.844 |
| Persistence | 1.716 | 1.889 | 2.016 | 2.189 |
| Historical average | 1.701 | 1.702 | 1.703 | 1.706 |

The ST-GNN is clearly better than persistence (16 to 21%), but **it is not the best model**. A gradient-boosted
model with simple features (recent lags, the same time yesterday and last week, calendar, weather, citywide
demand) matches or beats it at every horizon. `scripts/accuracy_checks.py` puts 95% block-bootstrap intervals on
the gap, RMSE(other) minus RMSE(ST-GNN), where positive means the ST-GNN is better:

| Other model | 5 min | 15 min | 30 min | 60 min |
|-------------|-------|--------|--------|--------|
| Ridge regression | +0.017 [-0.012, +0.050] | +0.034 [-0.001, +0.074] | +0.062 [+0.018, +0.112] | +0.119 [+0.065, +0.187] |
| Historical average | +0.251 [+0.172, +0.332] | +0.156 [+0.088, +0.227] | +0.085 [+0.021, +0.154] | -0.020 [-0.084, +0.048] |
| Ridge + time-of-day average | +0.027 [+0.003, +0.053] | +0.029 [-0.004, +0.062] | -0.016 [-0.049, +0.017] | -0.065 [-0.097, -0.035] |
| Gradient boosting | -0.020 [-0.050, +0.005] | -0.023 [-0.048, +0.004] | -0.060 [-0.093, -0.025] | -0.116 [-0.167, -0.062] |

Reading it honestly:

- The ST-GNN's lead over a per-zone ridge regression is statistically indistinguishable from zero at 5 and 15
  minutes, and clear only at 30 and 60 minutes.
- At 60 minutes it ties a plain time-of-day average, and a linear model that is simply given that average as an
  input beats it. The ST-GNN only sees the last 4 hours, so it misses the daily and weekly pattern.
- Gradient boosting is better at every horizon, significantly so from 30 minutes on (6.7% lower RMSE at 60 minutes).
  On this data the graph structure has not been shown to add accuracy beyond what simple features provide. A
  controlled test (the same network with the graph removed) has not been run.
- Spatial information is not the missing piece here. Giving the gradient-boosted model flow-weighted upstream
  and downstream neighbour demand (the same flows that define the ST-GNN's graph) leaves its error unchanged:
  RMSE 1.428, 1.536, 1.563, 1.608 against 1.429, 1.523, 1.558, 1.610 (`results/spatial_check.json`). A zone's own
  history, the time of day and citywide demand already carry the useful signal at 5 to 60 minutes; most zones
  are quiet, and the busy ones in Manhattan move together with the city as a whole. This is evidence about one
  kind of spatial signal (trip-flow neighbours, one hop), not a proof that no graph could help.

Hotspot ranking, as the mean number of the 5 busiest zones also among the 5 predicted: gradient boosting 3.18 to
3.04 and the ST-GNN 3.16 to 2.92 across 5 to 60 minutes, against 2.93 to 2.55 for persistence. The top-3 hit rate
is 0.86 to 0.94 for every model, because the busiest zones barely change, so it does not separate them. The
training script's inline hotspot numbers rank zones by standardised demand rather than raw counts and are not
comparable.

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
| Oracle (true demand) | 56.8 / 9.17 | 40.3 / 6.98 | 25.7 / 5.05 | 12.2 / 3.18 |

![Rider wait vs empty driving](results/tradeoff.png)

- Forecast-driven repositioning clearly helps once the fleet is large enough: at 2,000 vehicles unmet requests fall
  from 42.8% to 26 to 29% and mean wait from 7.6 to about 5.4 minutes. At 1,000 vehicles there is little to gain.
- A better forecast gives a better policy. Gradient boosting is the best forecast-driven policy at 2,000 and 3,000
  vehicles (5.34 and 3.40 minutes), level with persistence on wait while driving about 13% less empty at 2,000
  vehicles (1.13M vs 1.29M vehicle-minutes). The ST-GNN is behind both (5.51 and 3.70).
- Comparing at equal driving cost along the `theta` sweep at 2,000 vehicles, waiting less than persistence by:
  gradient boosting **0.28 minutes (77% of the improvement perfect demand knowledge would give)**, the ST-GNN
  0.17 minutes (47%). Comparing at equal `theta` is misleading because policies drive different amounts.
- Perfect foresight beats persistence by only 0.2 to 0.35 minutes of wait at 2,000 to 3,000 vehicles, which bounds
  how much any forecaster can add at a 15-minute decision horizon.

### What this means

The project's useful findings are that forecasts do improve repositioning, and that the choice of forecaster
matters, but that the ST-GNN is not the right tool here: a much simpler, faster model is more accurate and gives
a better policy. The most likely route to a better ST-GNN is to give it the daily and weekly pattern (as a
residual over the time-of-day average), the weather channels, and more months of training data; none of that has
been run.

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
with the time-of-day average, gradient boosting, the ST-GNN and the true demand ("oracle", an upper bound) are compared with dispatch alone, over several fleet sizes
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

# 5. Run the repositioning study and draw the plots (a few minutes on 10 cores)
python scripts\run_evaluation.py --workers 10
python scripts\plot_results.py

# 6. Explore everything in the app
python -m streamlit run app\streamlit_app.py
```

The same steps are available as `make preprocess-265`, `make train-multi-265`, `make export`, `make evaluate`,
`make plots`, `make app` and `make test`.

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
