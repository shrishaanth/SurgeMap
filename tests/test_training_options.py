import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hotspot_eval import collect_predictions, load_checkpoint, split_bounds
from train_multihorizon_torch import (HORIZONS, MultiHorizonDataset, MultiHorizonSTGNN, prepare_inputs,
                                      run_epoch, zone_weights)

Z, T, F = 4, 3 * 288, 7


def make_data_dir(path):
    rng = np.random.default_rng(0)
    demand = rng.poisson(np.array([8.0, 3.0, 0.5, 0.05])[:, None], size=(Z, T)).astype(np.float32)
    np.save(path / "demand.npy", demand)
    np.save(path / "times.npy", pd.date_range("2024-01-01", periods=T, freq="5min").to_numpy())
    np.save(path / "weather.npy", rng.normal(size=(T, 2)).astype(np.float32))
    np.save(path / "features_clipped.npy", rng.normal(size=(Z, T, F)).astype(np.float32))
    flow = np.full((Z, Z), 1.0 / Z, dtype=np.float32)
    np.save(path / "A_out.npy", flow)
    np.save(path / "A_in.npy", flow)
    train_end, val_end = 2 * 288, 2 * 288 + 144
    (path / "metadata.json").write_text(json.dumps(
        {"split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, T]}}))
    return demand, train_end


def test_dataset_returns_a_zero_prior_unless_one_is_given():
    features = np.random.default_rng(0).normal(size=(Z, 200, F)).astype(np.float32)
    plain = MultiHorizonDataset(features, 0, 200, window=12)
    x, y, prior, lagged = plain[0]
    assert x.shape == (Z, 12, F) and y.shape == (Z, 4) and torch.count_nonzero(prior) == 0
    assert lagged.shape == (Z, 4, 0)
    given = np.random.default_rng(1).normal(size=(Z, 200, 4)).astype(np.float32)
    ds = MultiHorizonDataset(features, 0, 200, window=12, prior=given)
    _, _, prior, _ = ds[5]
    np.testing.assert_allclose(prior.numpy(), given[:, ds.times[5], :])


def test_model_adds_the_prior_and_uses_zone_embeddings():
    model = MultiHorizonSTGNN(F, hidden=8, n_zones=Z, zone_dim=3)
    x = torch.randn(2, Z, 12, F)
    a = torch.full((Z, Z), 1.0 / Z)
    prior = torch.randn(2, Z, 4)
    base = model(x, a, a)
    assert base.shape == (2, Z, 4)
    torch.testing.assert_close(model(x, a, a, prior), base + prior)
    assert model.heads[0].in_features == 8 + 3 and model.zone_embedding.weight.shape == (Z, 3)


def test_prepare_inputs_appends_weather_and_builds_a_leak_free_prior(tmp_path):
    demand, train_end = make_data_dir(tmp_path)
    features = np.load(tmp_path / "features_clipped.npy")
    out, prior, _ = prepare_inputs(str(tmp_path), features, train_end, HORIZONS, use_weather=True, prior_kind="histavg")
    assert out.shape == (Z, T, F + 2) and prior.shape == (Z, T, 4)
    np.testing.assert_allclose(out[0, :, F:], np.load(tmp_path / "weather.npy"))
    assert np.isfinite(prior).all() and np.abs(prior).max() <= 6.0

    spiked = demand.copy()
    spiked[0, 300] += 5000.0                                # a train bin: must not move its own prior
    np.save(tmp_path / "demand.npy", spiked)
    _, prior_spiked, _ = prepare_inputs(str(tmp_path), features, train_end, HORIZONS, prior_kind="histavg")
    np.save(tmp_path / "demand.npy", demand)
    _, prior_clean, _ = prepare_inputs(str(tmp_path), features, train_end, HORIZONS, prior_kind="histavg")
    # horizon index 0 targets bin t itself; the standardisation changes slightly, the average must not jump
    assert abs(prior_spiked[0, 300, 0] - prior_clean[0, 300, 0]) < 1.0

    none_features, none_prior, _ = prepare_inputs(str(tmp_path), features, train_end, HORIZONS)
    assert none_prior is None and none_features.shape == features.shape


def test_zone_weights_have_mean_one_and_favour_busy_zones(tmp_path):
    _, train_end = make_data_dir(tmp_path)
    flat = zone_weights(str(tmp_path), train_end, 0.0)
    np.testing.assert_allclose(flat, 1.0)
    weights = zone_weights(str(tmp_path), train_end, 2.0)
    assert abs(weights.mean() - 1.0) < 1e-5 and weights[0] > weights[2] > weights[3]


def test_weighted_training_step_runs_and_reduces_the_loss():
    torch.manual_seed(0)
    features = np.random.default_rng(0).normal(size=(Z, 300, F)).astype(np.float32)
    loader = DataLoader(MultiHorizonDataset(features, 0, 300, window=12), batch_size=32, shuffle=True)
    model = MultiHorizonSTGNN(F, hidden=8, n_zones=Z, zone_dim=2)
    a = torch.full((Z, Z), 1.0 / Z)
    weight = torch.tensor([2.0, 1.0, 0.5, 0.5])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first, _, _ = run_epoch(model, loader, a, a, torch.device("cpu"), optimizer, weight)
    for _ in range(4):
        last, _, _ = run_epoch(model, loader, a, a, torch.device("cpu"), optimizer, weight)
    assert last < first


def test_checkpoint_with_every_option_reloads_and_predicts(tmp_path):
    _, train_end = make_data_dir(tmp_path)
    args = {"hidden": 8, "weather": True, "prior": "histavg", "zone_dim": 3, "no_graph": True}
    model = MultiHorizonSTGNN(F + 2, hidden=8, n_zones=Z, zone_dim=3)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "args": args, "horizons": HORIZONS}, ckpt)
    loaded, features, a_out, a_in, horizons, meta = load_checkpoint(str(ckpt), str(tmp_path))
    assert features.shape[2] == F + 2 and loaded.prior_array.shape == (Z, T, 4)
    assert not a_out.any() and not a_in.any()
    bounds = split_bounds(meta, features.shape[1])
    pred, target, anchors = collect_predictions(loaded, features, a_out, a_in, bounds, 12, horizons)
    assert pred.shape == target.shape == (len(anchors), Z, 4) and np.isfinite(pred).all()


def test_old_style_checkpoint_still_loads_without_extras(tmp_path):
    make_data_dir(tmp_path)
    model = MultiHorizonSTGNN(F, hidden=8)
    ckpt = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "args": {"hidden": 8}, "horizons": HORIZONS}, ckpt)
    loaded, features, a_out, *_ = load_checkpoint(str(ckpt), str(tmp_path))
    assert features.shape[2] == F and loaded.prior_array is None and a_out.any()
    assert loaded.zone_embedding is None


def test_log_space_prior_is_unbiased_where_the_count_space_prior_is_not(tmp_path):
    demand, train_end = make_data_dir(tmp_path)
    features = np.load(tmp_path / "features_clipped.npy")
    _, prior_z, _ = prepare_inputs(str(tmp_path), features, train_end, HORIZONS, prior_kind="histavg_z")
    _, prior_c, _ = prepare_inputs(str(tmp_path), features, train_end, HORIZONS, prior_kind="histavg")
    log_d = np.log1p(demand[:, :train_end].astype(np.float64))
    z = (log_d - log_d.mean(axis=1, keepdims=True)) / log_d.std(axis=1, keepdims=True)
    for zone in (0, 1, 2):
        # the mean residual target - prior over the train split: ~0 for the log-space prior
        assert abs((z[zone] - prior_z[zone, :train_end, 0]).mean()) < 0.05
    quiet = 2                                   # mean 0.5 pickups: log of the mean exceeds the mean of the logs
    assert (z[quiet] - prior_c[quiet, :train_end, 0]).mean() < -0.1
    assert prior_z.shape == prior_c.shape == (Z, T, 4)
