import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from surgemap.data.windows import split_bounds
from surgemap.evaluation.inference import collect_predictions, load_checkpoint
from surgemap.models.graph_wavenet import GraphWaveNet, adaptive_adjacency, calendar_slots
from test_capacity_options import F, Z, make_data_dir
from surgemap.data.windows import HORIZONS, MultiHorizonDataset, prepare_inputs
from surgemap.models.build import build_model, parse_dilations
from surgemap.models.gru import MultiHorizonSTGNN
from surgemap.training.train import run_epoch

WINDOW = 12
SMALL = {"arch": "gwnet", "channels": 8, "end_channels": 16, "dilations": "1,2,4", "gcn_order": 2,
         "adaptive_dim": 4, "dropout": 0.1}


def flow(n):
    return torch.full((n, n), 1.0 / n)


def test_adaptive_adjacency_rows_sum_to_one():
    adjacency = adaptive_adjacency(torch.randn(5, 3), torch.randn(5, 3))
    assert adjacency.shape == (5, 5)
    assert (adjacency >= 0).all()
    torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(5))


def test_build_model_picks_the_architecture_named_in_the_arguments():
    assert isinstance(build_model({"hidden": 8}, F, Z), MultiHorizonSTGNN)
    assert isinstance(build_model(SMALL, F, Z), GraphWaveNet)


@pytest.mark.parametrize("lag_dim", [0, 4])
def test_graph_wavenet_output_shape_and_gradients(lag_dim):
    torch.manual_seed(0)
    model = build_model(SMALL, F, Z, lag_dim=lag_dim)
    assert isinstance(model, GraphWaveNet) and model.receptive_field == 8
    x = torch.randn(2, Z, WINDOW, F)
    lagged = torch.randn(2, Z, len(HORIZONS), lag_dim) if lag_dim else None
    out = model(x, flow(Z), flow(Z), lagged=lagged)
    assert out.shape == (2, Z, len(HORIZONS))
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.node_a.grad.abs().sum() > 0


def test_graph_wavenet_reads_the_whole_receptive_field_and_no_further():
    torch.manual_seed(0)
    model = build_model({**SMALL, "dropout": 0.0}, F, Z).eval()
    x = torch.randn(1, Z, WINDOW, F)
    base = model(x, flow(Z), flow(Z))
    inside, outside = x.clone(), x.clone()
    inside[:, :, WINDOW - model.receptive_field] += 1.0
    outside[:, :, WINDOW - model.receptive_field - 1] += 1.0
    assert not torch.allclose(model(inside, flow(Z), flow(Z)), base)
    torch.testing.assert_close(model(outside, flow(Z), flow(Z)), base)


def test_graph_wavenet_rejects_a_window_shorter_than_its_receptive_field():
    model = build_model(SMALL, F, Z)
    with pytest.raises(ValueError, match="receptive field"):
        model(torch.randn(1, Z, 5, F), flow(Z), flow(Z))


def test_parse_dilations():
    assert parse_dilations("1,2, 4") == (1, 2, 4)
    assert parse_dilations([1, 2]) == (1, 2)
    with pytest.raises(ValueError):
        parse_dilations("0,1")
    with pytest.raises(ValueError):
        build_model({"arch": "transformer"}, F, Z)


@pytest.mark.parametrize("config", [SMALL, {**SMALL, "identity_dim": 4}, {"arch": "gru", "hidden": 8}])
def test_new_architectures_train_and_reload_through_the_checkpoint_loader(tmp_path, config):
    make_data_dir(tmp_path)
    features = np.load(tmp_path / "features_clipped.npy")
    bounds = split_bounds({"split": {"train": [0, 8 * 288], "validation": [8 * 288, 8 * 288 + 144],
                                     "test": [8 * 288 + 144, features.shape[1]]}}, features.shape[1])
    features, prior, lagged = prepare_inputs(str(tmp_path), features, bounds[0][1], HORIZONS, False, "none", True)
    torch.manual_seed(0)
    model = build_model(config, F, Z, lag_dim=lagged.shape[-1])
    dataset = MultiHorizonDataset(features, 8 * 288 - 60, 8 * 288, WINDOW, prior=prior, lagged=lagged)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    a = flow(Z)
    losses = [run_epoch(model, DataLoader(dataset, 8), a, a, torch.device("cpu"), optimizer)[0] for _ in range(3)]
    assert np.isfinite(losses).all() and losses[-1] < losses[0]

    saved_args = {**config, "data_dir": str(tmp_path), "lag_features": True, "window": WINDOW}
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "args": saved_args, "horizons": HORIZONS}, path)
    loaded, feats, a_out, a_in, horizons, meta = load_checkpoint(str(path), str(tmp_path))
    assert type(loaded) is type(model) and not loaded.training
    pred, target, anchors = collect_predictions(loaded, feats, a_out, a_in, split_bounds(meta, feats.shape[1]),
                                                WINDOW, horizons, batch_size=16)
    model.eval()
    x, _, _, lag = MultiHorizonDataset(feats, *split_bounds(meta, feats.shape[1])[2], WINDOW, horizons,
                                       lagged=loaded.lagged_array)[0]
    with torch.no_grad():
        direct = model(x[None], a, a, lagged=lag[None])[0].numpy()
    np.testing.assert_allclose(pred[0], direct, rtol=1e-4, atol=1e-5)
    assert pred.shape == target.shape and len(anchors) == len(pred)


def test_calendar_slots_recover_time_of_day_and_weekday_from_the_features():
    times = pd.date_range('2024-01-05 22:00', periods=600, freq='5min')
    minute = (times.hour * 60 + times.minute).to_numpy()
    day = times.dayofweek.to_numpy()
    x = torch.zeros(1, 2, 600, F)
    x[0, :, :, 1] = torch.tensor(np.sin(2 * np.pi * minute / 1440), dtype=torch.float32)
    x[0, :, :, 2] = torch.tensor(np.cos(2 * np.pi * minute / 1440), dtype=torch.float32)
    x[0, :, :, 3] = torch.tensor(np.sin(2 * np.pi * day / 7), dtype=torch.float32)
    x[0, :, :, 4] = torch.tensor(np.cos(2 * np.pi * day / 7), dtype=torch.float32)
    slot, weekday = calendar_slots(x)
    np.testing.assert_array_equal(slot[0].numpy(), minute // 5)
    np.testing.assert_array_equal(weekday[0].numpy(), day)


def test_identity_embeddings_make_identical_windows_in_different_zones_differ():
    torch.manual_seed(0)
    x = torch.randn(1, 1, WINDOW, F).expand(1, Z, WINDOW, F).contiguous()
    plain = build_model({**SMALL, 'adaptive_dim': 0, 'dropout': 0.0}, F, Z).eval()
    out = plain(x, flow(Z), flow(Z))
    torch.testing.assert_close(out[:, 0], out[:, 1])
    with_identity = build_model({**SMALL, 'adaptive_dim': 0, 'dropout': 0.0, 'identity_dim': 4}, F, Z).eval()
    out = with_identity(x, flow(Z), flow(Z))
    assert not torch.allclose(out[:, 0], out[:, 1])
