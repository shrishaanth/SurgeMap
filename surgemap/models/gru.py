"""The first ST-GNN design: one graph convolution on the inputs, then a GRU per zone."""
from __future__ import annotations

import torch
from torch import nn

from surgemap.data.windows import HORIZONS

class DirectedGraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.w_out = nn.Linear(in_channels, hidden_channels, bias=False)
        self.w_in = nn.Linear(in_channels, hidden_channels, bias=False)
        self.self_loop = nn.Linear(in_channels, hidden_channels)

    def forward(self, x, a_out, a_in):
        out = torch.einsum("ij,bjwf->biwf", a_out, x)
        inc = torch.einsum("ij,bjwf->biwf", a_in, x)
        h = self.w_out(out) + self.w_in(inc) + self.self_loop(x)
        return torch.relu(h)


class MultiHorizonSTGNN(nn.Module):
    """Graph convolution per time step, a GRU over time, and one linear head per horizon.

    With zone_dim > 0 each zone gets a learned embedding that is fed to the heads, so the
    network can tell zones apart. A prior passed to forward() is added to the output, so the
    network then learns the residual over it. With lag_dim > 0 each head also receives
    lag_dim extra inputs for its own target bin (demand a day and a week earlier).
    """

    def __init__(self, n_features: int, hidden: int = 64, horizons=HORIZONS,
                 n_zones: int = 0, zone_dim: int = 0, n_layers: int = 1, lag_dim: int = 0):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.spatial = DirectedGraphConv(n_features, hidden)
        self.temporal = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        self.lag_dim = int(lag_dim)
        self.zone_embedding = nn.Embedding(n_zones, zone_dim) if zone_dim > 0 else None
        self.heads = nn.ModuleList(nn.Linear(hidden + zone_dim + self.lag_dim, 1) for _ in self.horizons)

    def encode(self, x, a_out, a_in):
        spatial = self.spatial(x, a_out, a_in)
        b, z, w, hidden = spatial.shape
        encoded, _ = self.temporal(spatial.reshape(b * z, w, hidden))
        return encoded[:, -1, :].reshape(b, z, hidden)

    def forward(self, x, a_out, a_in, prior=None, lagged=None):
        encoded = self.encode(x, a_out, a_in)
        if self.zone_embedding is not None:
            zones = self.zone_embedding.weight.unsqueeze(0).expand(encoded.shape[0], -1, -1)
            encoded = torch.cat([encoded, zones], dim=-1)
        if self.lag_dim:
            out = torch.stack([head(torch.cat([encoded, lagged[:, :, j, :]], dim=-1)).squeeze(-1)
                               for j, head in enumerate(self.heads)], dim=-1)
        else:
            out = torch.stack([head(encoded).squeeze(-1) for head in self.heads], dim=-1)
        return out if prior is None else out + prior
