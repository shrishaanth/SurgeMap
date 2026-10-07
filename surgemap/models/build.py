"""Builds the network a set of training arguments describes; shared by the trainer and the loader."""
from __future__ import annotations

from torch import nn

from surgemap.data.windows import HORIZONS
from surgemap.models.graph_wavenet import GraphWaveNet
from surgemap.models.gru import MultiHorizonSTGNN

def parse_dilations(value) -> tuple:
    """"1,2,4" or a sequence of ints -> (1, 2, 4)."""
    parts = value.split(',') if isinstance(value, str) else value
    dilations = tuple(int(p) for p in parts if str(p).strip())
    if not dilations or min(dilations) < 1:
        raise ValueError(f"dilations must be positive integers, got {value!r}")
    return dilations


def build_model(config: dict, n_features: int, n_zones: int, horizons=HORIZONS, lag_dim: int = 0) -> nn.Module:
    """The network a set of training arguments describes; shared by the trainer and the loader."""
    arch = config.get("arch") or "gru"
    adaptive_dim = int(config.get("adaptive_dim") or 0)
    if arch == "gwnet":
        return GraphWaveNet(n_features, n_zones, horizons,
                            channels=int(config.get("channels") or 32),
                            dilations=parse_dilations(config.get("dilations") or "1,2,4,8,16,1,2,4"),
                            gcn_order=int(config.get("gcn_order") or 2), adaptive_dim=adaptive_dim,
                            dropout=float(config.get("dropout") or 0.0), lag_dim=lag_dim,
                            end_channels=int(config.get("end_channels") or 128),
                            identity_dim=int(config.get("identity_dim") or 0))
    if arch != "gru":
        raise ValueError(f"unknown architecture {arch!r}")
    return MultiHorizonSTGNN(n_features, int(config.get("hidden") or 64), horizons, n_zones=n_zones,
                             zone_dim=int(config.get("zone_dim") or 0),
                             n_layers=int(config.get("layers") or 1), lag_dim=lag_dim)
