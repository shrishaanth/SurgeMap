# How the model got here

The experiment record behind the results in the [README](../README.md): what was tried, in what order, and what
each change was worth. All numbers are RMSE in pickups per zone per 5-minute bin on the same test window
(27 Jan 08:20 to 31 Jan 23:55 2024), at 5, 15, 30 and 60 minutes. File paths are relative to the repository root.


The first ST-GNN (January only, batches of 128 in time order, squared error on log demand, kept as
`experiments/original_batch128_noshuffle.pt`) scored 1.449, 1.546, 1.618, 1.726 and was beaten by gradient boosting
at every horizon. `scripts/diagnose_stgnn.py` found why (`results/stgnn_diagnostics.json`):

- it under-predicted total pickups by 8.5 to 12%, from the log scale and the busier test days;
- it had no notion of which zone it was forecasting and saw only the last 4 hours;
- its loss weighted all 258 zones equally, while 86% of the squared error in pickups sits in the 30 busiest zones;
- it underfitted, and its training batches were consecutive, near-identical windows.

Each idea was then tested on a Kaggle T4 GPU, one change at a time (`experiments/`, and the
`results/*_experiments.json`, `round5_multi_month.json` and `round6_scores.json` files). Calibrated RMSE, one
network trained on January unless stated:

| Recipe (batch size 32) | 5 min | 15 min | 30 min | 60 min |
|------------------------|-------|--------|--------|--------|
| First model (batch 128, no shuffle) | 1.402 | 1.468 | 1.506 | 1.561 |
| No options | 1.422 | 1.487 | 1.533 | 1.580 |
| Shuffle | 1.385 | 1.446 | 1.484 | 1.526 |
| Shuffle + weather | 1.411 | 1.466 | 1.498 | 1.539 |
| Shuffle + zone embedding | 1.394 | 1.446 | 1.483 | 1.514 |
| Shuffle + time-of-day prior | 1.528 | 1.525 | 1.529 | 1.537 |
| Shuffle + weighted loss | 1.375 | 1.440 | 1.476 | 1.511 |
| ... average of five seeds | 1.369 | 1.432 | 1.465 | 1.494 |
| Shuffle + weighted loss, 128 hidden units | 1.369 | 1.437 | 1.477 | 1.513 |
| Shuffle + weighted loss, two GRU layers | 1.373 | 1.439 | 1.477 | 1.509 |
| Shuffle + weighted loss + lagged inputs | 1.371 | 1.436 | 1.470 | 1.496 |
| Shuffle + Poisson loss | 1.373 | 1.438 | 1.467 | 1.497 |
| Shuffle + Poisson loss + lagged inputs | 1.364 | 1.429 | 1.459 | 1.490 |
| ... average of five seeds | 1.356 | 1.422 | 1.451 | 1.480 |
| ... trained on two months | 1.360 | 1.420 | 1.445 | 1.468 |
| ... trained on four months, range over five seeds | 1.346 to 1.351 | 1.404 to 1.408 | 1.428 to 1.431 | 1.447 to 1.453 |
| ... four months, average of five seeds | 1.340 | 1.399 | 1.421 | 1.440 |
| Graph WaveNet, same recipe, January, one seed | 1.351 | 1.414 | 1.444 | 1.468 |
| ... with zone, time-of-day and weekday embeddings, range over two seeds | 1.349 to 1.355 | 1.410 to 1.419 | 1.439 to 1.450 | 1.460 to 1.471 |
| ... with embeddings, January, average of two seeds | 1.342 | 1.405 | 1.434 | 1.451 |
| **Graph WaveNet, four months, average of three seeds (shipped)** | **1.326** | **1.386** | **1.406** | **1.421** |

- Shuffling the training windows is the largest single gain from the training recipe.
- A loss aimed at pickup counts helps: first by weighting busy zones, then, better, as a Poisson likelihood, which
  also removes the volume bias.
- Giving the output heads the target time's demand a day and a week earlier helps a little alone and more with
  the Poisson loss. Forcing the same information in as a fixed offset (the time-of-day prior) made short horizons
  much worse.
- **The architecture mattered after all.** Replacing the graph-convolution-then-GRU design with a Graph WaveNet
  improved the calibrated RMSE by 1.3 to 2.1% on January (two seeds against two seeds, +0.018 [+0.009, +0.026] at
  5 minutes and +0.032 [+0.014, +0.050] at 60) and by 0.9 to 1.3% on four months, where three Graph WaveNets beat
  five of the earlier networks at every horizon with intervals excluding zero. Two Graph WaveNets trained on
  January alone nearly match five of the earlier networks trained on four months. Learned zone, time-of-day and
  weekday embeddings, suggested by the literature, made no measurable difference on top (`results/arch_january_scores.json`),
  so the shipped network does not use them.
- More capacity in the earlier design did not help: 128 hidden units and a second GRU layer both land inside the seed-to-seed spread.
  Neither do weather inputs or a zone embedding on top of the better loss.
- More training data helps steadily, most at long horizons. Averaging five seeds adds a further 0.4 to 0.6%.
- Nothing applied after the fact adds much: averaging the ST-GNN with gradient boosting, online bias correction, a
  last-error correction and Hedge weighting over five forecasters each gain 0.5% or less
  (`results/headroom_analysis.json`).

**More data helped the baselines more than the ST-GNN.** Calibrated or as-trained RMSE on the same test window:

| Model | Trained on | 5 min | 15 min | 30 min | 60 min |
|-------|------------|-------|--------|--------|--------|
| ST-GNN (first design), calibrated, five seeds | January | 1.356 | 1.422 | 1.451 | 1.480 |
| | four months | 1.340 | 1.399 | 1.421 | 1.440 |
| ST-GNN (Graph WaveNet), calibrated, two seeds | January | 1.342 | 1.405 | 1.434 | 1.451 |
| | four months, three seeds | 1.326 | 1.386 | 1.406 | 1.421 |
| Gradient boosting, calibrated | January | 1.384 | 1.460 | 1.500 | 1.525 |
| | four months | 1.358 | 1.423 | 1.447 | 1.471 |
| Historical average | January | 1.701 | 1.702 | 1.703 | 1.706 |
| | four months | 1.543 | 1.545 | 1.547 | 1.552 |

Both ST-GNN designs improved by 1 to 3% and gradient boosting by 2 to 4%. The Graph WaveNet's calibrated lead over
gradient boosting was 3 to 5% on January and is 2.4 to 3.4% on four months, so its advantage is somewhat larger
when history is short. The historical average gained most, 9%, because three weeks of history are too few to
estimate a weekly pattern. (The gradient-boosted model is fitted with an L2 term; without it the Poisson model
predicted infinities for near-empty zones on four months.)

**Does the graph help?** In the first design, little. In an early bundled retrain, removing the graph cost 2 to 4%
at every horizon; in a better-trained recipe (shuffle + zone embedding + weighted loss, January) the cost was 0.3 to
1.2% as trained, significant only at 15 minutes, and nothing measurable after calibration
(`results/model_selection.json`). That design used the graph once, on the raw inputs. The Graph WaveNet uses it
after every layer and adds a learned adjacency, and it is 1 to 2% more accurate, but the change replaced the GRU
at the same time, so this is not a clean measurement of the graph's share: the shipped network has not been
retrained without its graph. Giving the gradient-boosted model flow-weighted neighbour demand left its error
unchanged (`results/spatial_check.json`).

## Training options

`train_multihorizon_torch.py` options, all off by default. The shipped networks use
`--shuffle --loss poisson --lag-features` on the four-month dataset.

| Option | What it does |
|--------|--------------|
| `--shuffle` | shuffle training windows each epoch |
| `--loss-power P` | weight zones by `((1 + mean pickups) x sigma) ** P` in the loss; 2 approximates error in pickups |
| `--zone-dim N` | learned embedding per zone, fed to the output heads |
| `--weather` | append the precipitation and temperature channels |
| `--prior histavg_z` | predict the residual over the time-of-day average of each target bin |
| `--no-graph` | remove the neighbour terms (ablation) |
| `--arch gwnet` | Graph WaveNet (the shipped network); the default `gru` is the first design |
| `--identity-dim N` | Graph WaveNet: learned zone, time-of-day and weekday embeddings |
| `--amp` | mixed precision on a GPU |
| `--layers N` | first design: number of GRU layers |
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
