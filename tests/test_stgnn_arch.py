import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hotspot_eval import collect_predictions, load_checkpoint, split_bounds
from stgnn_arch import GraphWaveNet, HiddenGraphMix, adaptive_adjacency
from test_capacity_options import F, Z, make_data_dir
from train_multihorizon_torch import (HORIZONS, MultiHorizonDataset, MultiHorizonSTGNN, build_model,
                                      parse_dilations, prepare_inputs, run_epoch)

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


def test_hidden_mix_lets_one_zone_change_anothers_state():
    torch.manual_seed(0)
    mix = HiddenGraphMix(6, 4, adaptive_dim=3)
    encoded = torch.randn(2, 4, 6)
    changed = encoded.clone()
    changed[:, 0] += 1.0
    before, after = mix(encoded, flow(4), flow(4)), mix(changed, flow(4), flow(4))
    assert before.shape == encoded.shape
    assert not torch.allclose(before[:, 1:], after[:, 1:])
    isolated = torch.zeros(4, 4)
    mix_fixed = HiddenGraphMix(6, 4)
    torch.testing.assert_close(mix_fixed(encoded, isolated, isolated)[:, 1:],
                               mix_fixed(changed, isolated, isolated)[:, 1:])


def test_gru_without_mix_keeps_the_original_parameters():
    plain = build_model({"hidden": 64, "adaptive_dim": 10}, F, Z, lag_dim=4)
    assert isinstance(plain, MultiHorizonSTGNN) and plain.mix is None
    assert not any(name.startswith("mix.") for name in plain.state_dict())
    mixed = build_model({"hidden": 64, "mix_hidden": True, "adaptive_dim": 10}, F, Z, lag_dim=4)
    assert any(name == "mix.node_a" for name in mixed.state_dict())


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


@pytest.mark.parametrize("config", [SMALL, {"arch": "gru", "hidden": 8, "mix_hidden": True, "adaptive_dim": 4},
                                    {"arch": "gru", "hidden": 8, "mix_hidden": True, "adaptive_dim": 0}])
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
