import json

import numpy as np
import torch

from hotspot_eval import load_checkpoint, ranking_metrics
from train_multihorizon_torch import HORIZONS, MultiHorizonSTGNN


def _write_data_dir(path, n_channels, zones=5, bins=300):
    rng = np.random.default_rng(0)
    np.save(path / "features_clipped.npy", rng.normal(size=(zones, bins, n_channels)).astype(np.float32))
    np.save(path / "A_out.npy", np.eye(zones, dtype=np.float32))
    np.save(path / "A_in.npy", np.eye(zones, dtype=np.float32))
    (path / "metadata.json").write_text(json.dumps({"split": {}}))


def test_checkpoint_trained_on_fewer_channels_uses_leading_channels(tmp_path):
    _write_data_dir(tmp_path, n_channels=7)
    model = MultiHorizonSTGNN(6, hidden=8)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "args": {"hidden": 8}, "horizons": HORIZONS}, ckpt)
    loaded, features, *_ = load_checkpoint(str(ckpt), str(tmp_path))
    assert features.shape[2] == 6
    np.testing.assert_array_equal(features, np.load(tmp_path / "features_clipped.npy")[:, :, :6])
    assert isinstance(loaded, MultiHorizonSTGNN)


def test_checkpoint_needing_more_channels_than_available_is_rejected(tmp_path):
    _write_data_dir(tmp_path, n_channels=5)
    model = MultiHorizonSTGNN(6, hidden=8)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "args": {"hidden": 8}, "horizons": HORIZONS}, ckpt)
    try:
        load_checkpoint(str(ckpt), str(tmp_path))
    except ValueError as exc:
        assert "feature channels" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ranking_metrics_on_a_hand_computed_case():
    target = np.array([[[9.0], [8.0], [1.0], [0.0]]])
    pred = np.array([[[9.0], [1.0], [8.0], [0.0]]])
    row = ranking_metrics(pred, target, (1,), (2,))["1"]["topk"]["2"]
    assert row["overlap"] == 1.0
    assert row["hit_rate"] == 1.0
    assert row["precision"] == 0.5


def test_resolve_checkpoints_accepts_a_file_a_list_and_a_directory(tmp_path):
    import pytest
    from hotspot_eval import resolve_checkpoints
    for name in ("b.pt", "a.pt", "notes.json"):
        (tmp_path / name).write_text("x")
    found = resolve_checkpoints(str(tmp_path))
    assert [p.replace("\\", "/").split("/")[-1] for p in found] == ["a.pt", "b.pt"]
    assert resolve_checkpoints("one.pt") == ["one.pt"]
    assert resolve_checkpoints("one.pt, two.pt") == ["one.pt", "two.pt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoints(str(empty))
