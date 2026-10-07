"""Graph WaveNet: gated dilated temporal convolutions alternating with diffusion graph convolutions.

After Wu et al. (2019). Zones exchange what they have encoded after every layer, over several hops
of the fixed flow graph and of an adjacency the network learns itself. It takes the same inputs and
returns the same output as the first design (models/gru.py), so the trainer, the checkpoint loader
and everything downstream work with either.
"""
from __future__ import annotations

import math

import torch
from torch import nn

SLOTS_PER_DAY = 288      # 5-minute bins


def adaptive_adjacency(node_a: torch.Tensor, node_b: torch.Tensor) -> torch.Tensor:
    """A learned, row-normalised zone-to-zone weight matrix from two sets of node embeddings."""
    return torch.softmax(torch.relu(node_a @ node_b.T), dim=1)


def calendar_slots(x: torch.Tensor) -> tuple:
    """Time-of-day slot (0..287) and weekday (0..6) of every step of a window, read back from the
    calendar channels 1-4 of the features (hour sin/cos, weekday sin/cos). x is [B, Z, W, F];
    the calendar is the same for every zone, so zone 0 is used. Returns two [B, W] tensors."""
    cal = x[:, 0, :, 1:5]
    turn = 2 * math.pi
    slot = torch.round(torch.atan2(cal[..., 0], cal[..., 1]) / turn * SLOTS_PER_DAY).long() % SLOTS_PER_DAY
    weekday = torch.round(torch.atan2(cal[..., 2], cal[..., 3]) / turn * 7).long() % 7
    return slot, weekday


class DiffusionGraphConv(nn.Module):
    """Graph convolution on [B, C, Z, T]: for every support matrix, `order` successive hops."""

    def __init__(self, channels: int, n_supports: int, order: int, dropout: float):
        super().__init__()
        self.order = int(order)
        self.project = nn.Conv2d(channels * (1 + n_supports * self.order), channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, supports):
        terms = [x]
        for support in supports:
            hop = x
            for _ in range(self.order):
                hop = torch.einsum("vw,bcwt->bcvt", support, hop)
                terms.append(hop)
        return self.dropout(self.project(torch.cat(terms, dim=1)))


class GraphWaveNet(nn.Module):
    """Gated dilated temporal convolutions alternating with diffusion graph convolutions.

    Each layer shortens the time axis by its dilation, so the window must be longer than the
    sum of the dilations. Every layer contributes its most recent time step to a skip sum,
    which is projected to a per-zone vector and read by one linear head per horizon. As in
    MultiHorizonSTGNN, a prior is added to the output and lagged inputs go to the heads.

    With identity_dim > 0 the input of every zone at every step is extended with three learned
    embeddings: one per zone, one per time-of-day slot and one per weekday (the spatial and
    temporal identities of Shao et al., 2022), so the network can tell apart windows that look
    the same but belong to different places or times.
    """

    def __init__(self, n_features: int, n_zones: int, horizons, channels: int = 32,
                 dilations=(1, 2, 4, 8, 16, 1, 2, 4), gcn_order: int = 2, adaptive_dim: int = 10,
                 dropout: float = 0.1, lag_dim: int = 0, end_channels: int = 128,
                 identity_dim: int = 0):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.dilations = tuple(int(d) for d in dilations)
        self.lag_dim = int(lag_dim)
        self.adaptive = adaptive_dim > 0
        if self.adaptive:
            self.node_a = nn.Parameter(torch.randn(n_zones, adaptive_dim) * 0.1)
            self.node_b = nn.Parameter(torch.randn(n_zones, adaptive_dim) * 0.1)
        n_supports = 2 + int(self.adaptive)
        self.identity_dim = int(identity_dim)
        if self.identity_dim:
            self.zone_identity = nn.Embedding(n_zones, self.identity_dim)
            self.slot_identity = nn.Embedding(SLOTS_PER_DAY, self.identity_dim)
            self.weekday_identity = nn.Embedding(7, self.identity_dim)
        self.input_proj = nn.Conv2d(n_features + 3 * self.identity_dim, channels, kernel_size=1)
        conv = lambda d: nn.Conv2d(channels, channels, kernel_size=(1, 2), dilation=(1, d))
        self.filters = nn.ModuleList(conv(d) for d in self.dilations)
        self.gates = nn.ModuleList(conv(d) for d in self.dilations)
        self.graph_convs = nn.ModuleList(DiffusionGraphConv(channels, n_supports, gcn_order, dropout)
                                         for _ in self.dilations)
        self.norms = nn.ModuleList(nn.BatchNorm2d(channels) for _ in self.dilations)
        self.skips = nn.ModuleList(nn.Conv2d(channels, end_channels, kernel_size=1) for _ in self.dilations)
        self.end = nn.Conv2d(end_channels, end_channels, kernel_size=1)
        self.heads = nn.ModuleList(nn.Linear(end_channels + self.lag_dim, 1) for _ in self.horizons)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(self.dilations)

    def forward(self, x, a_out, a_in, prior=None, lagged=None):
        if x.shape[2] < self.receptive_field:
            raise ValueError(f"window of {x.shape[2]} bins is shorter than the receptive field "
                             f"({self.receptive_field}); use a longer window or fewer dilations")
        supports = [a_out, a_in]
        if self.adaptive:
            supports.append(adaptive_adjacency(self.node_a, self.node_b))
        if self.identity_dim:
            b, z, w, _ = x.shape
            slot, weekday = calendar_slots(x)
            x = torch.cat([x, self.zone_identity.weight[None, :, None, :].expand(b, z, w, -1),
                           self.slot_identity(slot)[:, None].expand(b, z, w, -1),
                           self.weekday_identity(weekday)[:, None].expand(b, z, w, -1)], dim=-1)
        h = self.input_proj(x.permute(0, 3, 1, 2))                 # [B, C, Z, W]
        skip = 0.0
        for i in range(len(self.dilations)):
            residual = h
            h = torch.tanh(self.filters[i](h)) * torch.sigmoid(self.gates[i](h))
            h = self.graph_convs[i](h, supports) + residual[..., -h.shape[-1]:]
            h = self.norms[i](h)
            skip = skip + self.skips[i](h[..., -1:])               # latest time step of this layer
        encoded = torch.relu(self.end(torch.relu(skip))).squeeze(-1).permute(0, 2, 1)     # [B, Z, E]
        if self.lag_dim:
            out = torch.stack([head(torch.cat([encoded, lagged[:, :, j, :]], dim=-1)).squeeze(-1)
                               for j, head in enumerate(self.heads)], dim=-1)
        else:
            out = torch.stack([head(encoded).squeeze(-1) for head in self.heads], dim=-1)
        return out if prior is None else out + prior
