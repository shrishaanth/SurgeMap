import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hotspot_eval import collect_predictions, load_checkpoint, split_bounds
from train_multihorizon_torch import (HORIZONS, LAG_BINS, MultiHorizonDataset, MultiHorizonSTGNN, lagged_inputs,
                                      poisson_loss, prepare_inputs, run_epoch)

Z, T, F = 3, 9 * 288, 7


def make_data_dir(path):
    rng = np.random.default_rng(0)
    demand = rng.poisson(np.array([8.0, 2.0, 0.3])[:, None], size=(Z, T)).astype(np.float32)
    np.save(path / "demand.npy", demand)
    np.save(path / "times.npy", pd.date_range("2024-01-01", periods=T, freq="5min").to_numpy())
    np.save(path / "weather.npy", rng.normal(size=(T, 2)).astype(np.float32))
    np.save(path / "features_clipped.npy", rng.normal(size=(Z, T, F)).astype(np.float32))
    flow = np.full((Z, Z), 1.0 / Z, dtype=np.float32)
    np.save(path / "A_out.npy", flow)
    np.save(path / "A_in.npy", flow)
    train_end, val_end = 8 * 288, 8 * 288 + 144
    (path / "metadata.json").write_text(json.dumps(
        {"split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, T]}}))
    return demand


def test_lagged_inputs_hold_the_target_bin_a_day_and_a_week_earlier():
    features = np.random.default_rng(0).normal(size=(Z, T, F)).astype(np.float32)
    lagged = lagged_inputs(features, HORIZONS)
    assert lagged.shape == (Z, T, len(HORIZONS), 4)
    anchor = 2500
    for j, h in enumerate(HORIZONS):
        target = anchor + h - 1
        np.testing.assert_allclose(lagged[:, anchor, j, 0], features[:, target - LAG_BINS[0], 0])
        np.testing.assert_allclose(lagged[:, anchor, j, 2], features[:, target - LAG_BINS[1], 0])
        assert (lagged[:, anchor, j, 1] == 1).all() and (lagged[:, anchor, j, 3] == 1).all()
        assert target - LAG_BINS[0] < anchor          # the lagged bin is already observed at the anchor


def test_lagged_inputs_are_zero_and_flagged_where_the_history_does_not_reach():
    features = np.ones((Z, T, F), dtype=np.float32)
    lagged = lagged_inputs(features, HORIZONS)
    assert (lagged[:, 100, 0] == 0).all()                            # first day: neither lag exists
    np.testing.assert_array_equal(lagged[0, 400, 0], [1, 1, 0, 0])   # day lag exists, week lag does not


def test_deeper_model_with_lagged_inputs_has_the_expected_shapes():
    model = MultiHorizonSTGNN(F, hidden=8, n_zones=Z, zone_dim=2, n_layers=2, lag_dim=4)
    assert model.temporal.num_layers == 2 and model.heads[0].in_features == 8 + 2 + 4
    x = torch.randn(2, Z, 12, F)
    a = torch.full((Z, Z), 1.0 / Z)
    lagged = torch.randn(2, Z, 4, 4)
    out = model(x, a, a, None, lagged)
    assert out.shape == (2, Z, 4)
    changed = model(x, a, a, None, lagged + 1.0)
    assert not torch.allclose(out, changed)                          # the heads really use the lagged inputs


def test_poisson_loss_is_lowest_at_the_true_rate():
    mu, sigma = torch.tensor([1.0, 0.5]), torch.tensor([0.8, 0.6])
    counts = torch.tensor([[[6.0], [2.0]]])
    y = (torch.log1p(counts) - mu[None, :, None]) / sigma[None, :, None]
    at_truth = poisson_loss(y, y, mu, sigma)
    assert at_truth < poisson_loss(y + 0.5, y, mu, sigma)
    assert at_truth < poisson_loss(y - 0.5, y, mu, sigma)
    wild = torch.full_like(y, 50.0, requires_grad=True)
    loss = poisson_loss(wild, y, mu, sigma)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(wild.grad).all()


def test_training_with_the_poisson_loss_runs_and_improves():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    counts = rng.poisson(np.array([8.0, 2.0, 0.5])[:, None], size=(Z, 400)).astype(np.float64)
    log_counts = np.log1p(counts)
    mu, sigma = log_counts.mean(axis=1), log_counts.std(axis=1)
    features = np.zeros((Z, 400, F), dtype=np.float32)
    features[:, :, 0] = (log_counts - mu[:, None]) / sigma[:, None]
    loader = DataLoader(MultiHorizonDataset(features, 0, 400, window=12), batch_size=32, shuffle=True)
    model = MultiHorizonSTGNN(F, hidden=8)
    a = torch.full((Z, Z), 1.0 / Z)
    stats = (torch.tensor(mu, dtype=torch.float32), torch.tensor(sigma, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first, _, _ = run_epoch(model, loader, a, a, torch.device("cpu"), optimizer, count_stats=stats)
    for _ in range(4):
        last, _, _ = run_epoch(model, loader, a, a, torch.device("cpu"), optimizer, count_stats=stats)
    assert np.isfinite(last) and last < first


def test_checkpoint_with_lagged_inputs_and_two_layers_reloads_and_predicts(tmp_path):
    make_data_dir(tmp_path)
    args = {"hidden": 8, "lag_features": True, "layers": 2}
    model = MultiHorizonSTGNN(F, hidden=8, n_layers=2, lag_dim=4)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "args": args, "horizons": HORIZONS}, ckpt)
    loaded, features, a_out, a_in, horizons, meta = load_checkpoint(str(ckpt), str(tmp_path))
    assert loaded.temporal.num_layers == 2 and loaded.lag_dim == 4
    assert loaded.lagged_array.shape == (Z, T, 4, 4)
    bounds = split_bounds(meta, features.shape[1])
    pred, target, anchors = collect_predictions(loaded, features, a_out, a_in, bounds, 12, horizons)
    assert pred.shape == (len(anchors), Z, 4) and np.isfinite(pred).all()


def test_prepare_inputs_returns_lagged_only_when_asked(tmp_path):
    make_data_dir(tmp_path)
    features = np.load(tmp_path / "features_clipped.npy")
    _, _, lagged = prepare_inputs(str(tmp_path), features, 8 * 288, HORIZONS, lag_features=True)
    assert lagged.shape == (Z, T, 4, 4)
    assert prepare_inputs(str(tmp_path), features, 8 * 288, HORIZONS)[2] is None
