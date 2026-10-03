"""Fresh capacity-only GCNs; inputs are caller-normalized public features."""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.models.gcn import GCNEncoder
from src.rl.capacity_adaptation_interface import project_effort_tensor
from src.rl.networks import nn, require_torch, torch


@dataclass(frozen=True)
class CapacityModelConfig:
    node_input_dim: int
    graph_widths: tuple[int, int]
    actor_head: tuple[int, int]
    site_hour_caps: tuple[float, ...]
    shared_hour_budget: float
    raw_hour_offset: float
    hours_divisor: float

    def __post_init__(self):
        for name in ("graph_widths", "actor_head", "site_hour_caps"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.node_input_dim) is not int or self.node_input_dim < 1:
            raise ValueError("positive supplied node_input_dim required")
        if self.graph_widths != (32, 32) or self.actor_head != (32, 1):
            raise ValueError("capacity pilot requires two 32-wide GCN layers and a 32/1 head")
        if len(self.site_hour_caps) < 3:
            raise ValueError("ring requires at least three sites")
        values = (*self.site_hour_caps, self.shared_hour_budget, self.hours_divisor)
        if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("finite positive capacity limits/divisor required")
        if not math.isfinite(self.raw_hour_offset) or self.raw_hour_offset < 0:
            raise ValueError("finite nonnegative raw-hour offset required")

    @property
    def num_nodes(self):
        return len(self.site_hour_caps)


def validate_features(features, config):
    require_torch()
    if (not isinstance(features, torch.Tensor) or features.device.type != "cpu"
            or features.dtype != torch.float32 or features.ndim != 3
            or tuple(features.shape[1:]) != (config.num_nodes, config.node_input_dim)
            or features.shape[0] < 1 or not torch.isfinite(features).all().item()):
        raise ValueError("finite CPU float32 [batch, sites, node_input_dim] required")


def validate_hours(hours, config, *, batch_size, legal=True):
    if (not isinstance(hours, torch.Tensor) or hours.device.type != "cpu"
            or hours.dtype != torch.float32 or tuple(hours.shape) != (batch_size, config.num_nodes)
            or not torch.isfinite(hours).all().item()):
        raise ValueError("finite CPU float32 [batch, sites] hours required")
    if legal:
        # Only allow float32 roundoff, not an alternative action projection.
        tolerance = torch.finfo(torch.float32).eps * max(config.shared_hour_budget, 1.0) * 4
        if ((hours < 0).any().item()
                or (hours > hours.new_tensor(config.site_hour_caps) + tolerance).any().item()
                or (hours.sum(-1) > config.shared_hour_budget + tolerance).any().item()):
            raise ValueError("executed hours violate site/shared capacity")


def _bind_adjacency(encoder, adjacency):
    if adjacency is None:
        return
    adjacency = torch.as_tensor(adjacency, device="cpu")
    if (adjacency.dtype != torch.float32 or adjacency.shape != encoder.adjacency.shape
            or not torch.isfinite(adjacency).all().item()
            or not torch.allclose(adjacency, encoder.adjacency, atol=1e-7, rtol=1e-6)):
        raise ValueError("supplied adjacency must be the float32 normalized self-loop ring")
    encoder.adjacency.copy_(adjacency.detach())


if torch is not None:

    class CapacityActor(nn.Module):
        def __init__(self, config: CapacityModelConfig, *, adjacency=None):
            super().__init__()
            self.config = config
            edges = tuple((i, (i + 1) % config.num_nodes) for i in range(config.num_nodes))
            self.encoder = GCNEncoder(config.node_input_dim, config.graph_widths, config.num_nodes, edges)
            self.head = nn.Sequential(nn.Linear(config.graph_widths[-1], config.actor_head[0]),
                                      nn.ReLU(), nn.Linear(*config.actor_head))
            self.to(device="cpu", dtype=torch.float32)
            _bind_adjacency(self.encoder, adjacency)

        def forward(self, features):
            """One module call returns clipped raw requests, before radial projection."""
            validate_features(features, self.config)
            z = self.head(self.encoder(features)).squeeze(-1)
            return torch.minimum(torch.clamp_min(self.config.raw_hour_offset + z, 0),
                                 z.new_tensor(self.config.site_hour_caps))

        def project(self, raw_hours):
            return project_effort_tensor(raw_hours, self.config)


    class CapacityCritic(nn.Module):
        def __init__(self, config: CapacityModelConfig, *, adjacency=None):
            super().__init__()
            self.config = config
            edges = tuple((i, (i + 1) % config.num_nodes) for i in range(config.num_nodes))
            self.encoder = GCNEncoder(config.node_input_dim + 1, config.graph_widths, config.num_nodes, edges)
            self.head = nn.Sequential(nn.Linear(config.graph_widths[-1], config.actor_head[0]),
                                      nn.ReLU(), nn.Linear(*config.actor_head))
            self.to(device="cpu", dtype=torch.float32)
            _bind_adjacency(self.encoder, adjacency)

        def forward(self, features, executed_hours):
            validate_features(features, self.config)
            validate_hours(executed_hours, self.config, batch_size=features.shape[0])
            nodes = torch.cat((features, executed_hours.unsqueeze(-1) / self.config.hours_divisor), dim=-1)
            return self.head(self.encoder(nodes).mean(dim=1))

else:
    class CapacityActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()

    class CapacityCritic(CapacityActor):  # pragma: no cover
        pass
