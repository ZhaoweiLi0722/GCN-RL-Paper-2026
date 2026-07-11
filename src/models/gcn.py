"""Small dense-adjacency GCN modules for graph-aware policies.

The 20-clinic manuscript setting has only 21 graph nodes once the central
capacity hub is included, so a dense normalized adjacency matrix is simpler
and more transparent than a sparse dependency.
"""

from __future__ import annotations

from typing import Sequence

from src.graph.edges import Edge
from src.rl.networks import nn, require_torch, torch

# Squashed-Gaussian constants — kept identical to the flat SAC/PPO backbones so a
# GCN encoder ∘ backbone composition reproduces the flat agent's action math.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


def build_normalized_adjacency(
    num_nodes: int,
    edges: Sequence[Edge],
    *,
    device=None,
):
    """Return symmetric normalized adjacency with self-loops."""

    require_torch()
    adjacency = torch.eye(num_nodes, dtype=torch.float32, device=device)
    for i, j in edges:
        source = int(i)
        target = int(j)
        if source == target:
            continue
        if source < 0 or target < 0 or source >= num_nodes or target >= num_nodes:
            raise ValueError(f"Edge {(source, target)} is outside the {num_nodes}-node graph")
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    degree = adjacency.sum(dim=1).clamp_min(1.0)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


if torch is not None:

    class GraphConvolution(nn.Module):
        """First-order GCN layer using a pre-normalized dense adjacency."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, node_features, adjacency):
            support = self.linear(node_features)
            return torch.matmul(adjacency, support)


    class GCNEncoder(nn.Module):
        """Stack of graph convolution layers."""

        def __init__(
            self,
            node_feature_dim: int,
            hidden_sizes: Sequence[int],
            num_nodes: int,
            edges: Sequence[Edge],
        ):
            super().__init__()
            if not hidden_sizes:
                raise ValueError("hidden_sizes must contain at least one GCN layer")
            self.register_buffer("adjacency", build_normalized_adjacency(num_nodes, edges))
            layers = []
            previous_dim = int(node_feature_dim)
            for hidden_dim in hidden_sizes:
                layers.append(GraphConvolution(previous_dim, int(hidden_dim)))
                previous_dim = int(hidden_dim)
            self.layers = nn.ModuleList(layers)
            self.output_dim = previous_dim

        def forward(self, node_features):
            x = node_features
            for layer in self.layers:
                x = torch.relu(layer(x, self.adjacency))
            return x


    def graph_readout(encoded, num_facilities, include_global_context):
        """Flatten facility node embeddings, optionally with a mean global context."""

        facility_encoded = encoded[:, :num_facilities, :]
        readout = facility_encoded.flatten(start_dim=1)
        if include_global_context:
            graph_context = encoded.mean(dim=1)
            readout = torch.cat((readout, graph_context), dim=-1)
        return readout


    def graph_readout_dim(num_facilities, encoder_output_dim, include_global_context):
        dim = int(num_facilities) * int(encoder_output_dim)
        if include_global_context:
            dim += int(encoder_output_dim)
        return dim


    class GraphFeatureExtractor(nn.Module):
        """GCN encoder + global_flat readout, shared by the stochastic heads."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
        ):
            super().__init__()
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.encoder = GCNEncoder(node_feature_dim, gcn_hidden_sizes, num_nodes, edges)
            self.output_dim = graph_readout_dim(
                self.num_facilities, self.encoder.output_dim, self.include_global_context
            )

        def forward(self, node_features):
            encoded = self.encoder(node_features)
            return graph_readout(encoded, self.num_facilities, self.include_global_context)


    class GCNActor(nn.Module):
        """GCN encoder followed by a deterministic DDPG actor head."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            action_dim: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            head_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
            readout_mode: str = "global_flat",
        ):
            super().__init__()
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.readout_mode = str(readout_mode)
            self.encoder = GCNEncoder(node_feature_dim, gcn_hidden_sizes, num_nodes, edges)
            if self.readout_mode == "global_flat":
                readout_dim = self.num_facilities * self.encoder.output_dim
                if self.include_global_context:
                    readout_dim += self.encoder.output_dim
                self.head = _build_mlp(readout_dim, head_hidden_sizes, action_dim, output_tanh=True)
            elif self.readout_mode == "facility_action":
                if int(action_dim) % self.num_facilities != 0:
                    raise ValueError(
                        "facility_action readout requires action_dim divisible by num_facilities"
                    )
                self.facility_action_dim = int(action_dim) // self.num_facilities
                facility_input_dim = self.encoder.output_dim
                if self.include_global_context:
                    facility_input_dim += self.encoder.output_dim
                self.head = _build_mlp(
                    facility_input_dim,
                    head_hidden_sizes,
                    self.facility_action_dim,
                    output_tanh=True,
                )
            else:
                raise ValueError(f"Unsupported GCN actor readout_mode: {self.readout_mode}")

        def forward(self, node_features):
            encoded = self.encoder(node_features)
            facility_encoded = encoded[:, : self.num_facilities, :]
            if self.readout_mode == "facility_action":
                if self.include_global_context:
                    graph_context = encoded.mean(dim=1, keepdim=True).expand(
                        -1, self.num_facilities, -1
                    )
                    facility_encoded = torch.cat((facility_encoded, graph_context), dim=-1)
                facility_actions = self.head(facility_encoded)
                return facility_actions.transpose(1, 2).flatten(start_dim=1)

            readout = graph_readout(encoded, self.num_facilities, self.include_global_context)
            return self.head(readout)


    class GCNCritic(nn.Module):
        """GCN encoder followed by a state-action Q-value head."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            action_dim: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            head_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
        ):
            super().__init__()
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.encoder = GCNEncoder(node_feature_dim, gcn_hidden_sizes, num_nodes, edges)
            readout_dim = self.num_facilities * self.encoder.output_dim
            if self.include_global_context:
                readout_dim += self.encoder.output_dim
            self.head = _build_mlp(
                readout_dim + int(action_dim),
                head_hidden_sizes,
                1,
                output_tanh=False,
            )

        def forward(self, node_features, action):
            encoded = self.encoder(node_features)
            readout = graph_readout(encoded, self.num_facilities, self.include_global_context)
            return self.head(torch.cat((readout, action), dim=-1))


    class GCNSquashedGaussianActor(nn.Module):
        """GCN encoder + SAC squashed-Gaussian head (mirrors flat SAC's actor)."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            action_dim: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            head_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim, num_facilities, num_nodes, edges, gcn_hidden_sizes,
                include_global_context=include_global_context,
            )
            self.backbone = _build_hidden_mlp(self.extractor.output_dim, head_hidden_sizes, "relu")
            last_dim = int(head_hidden_sizes[-1]) if head_hidden_sizes else self.extractor.output_dim
            self.mean = nn.Linear(last_dim, int(action_dim))
            self.log_std = nn.Linear(last_dim, int(action_dim))

        def forward(self, node_features):
            features = self.backbone(self.extractor(node_features))
            mean = self.mean(features)
            log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std

        def sample(self, node_features):
            mean, log_std = self.forward(node_features)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            pre_tanh = normal.rsample()
            action = torch.tanh(pre_tanh)
            log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPSILON)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob, torch.tanh(mean)

        def deterministic(self, node_features):
            mean, _ = self.forward(node_features)
            return torch.tanh(mean)


    class GCNGaussianActor(nn.Module):
        """GCN encoder + PPO Gaussian head with a state-independent log-std."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            action_dim: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            head_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim, num_facilities, num_nodes, edges, gcn_hidden_sizes,
                include_global_context=include_global_context,
            )
            self.backbone = _build_hidden_mlp(self.extractor.output_dim, head_hidden_sizes, "tanh")
            last_dim = int(head_hidden_sizes[-1]) if head_hidden_sizes else self.extractor.output_dim
            self.mean = nn.Linear(last_dim, int(action_dim))
            self.log_std = nn.Parameter(torch.zeros(int(action_dim), dtype=torch.float32))

        def forward(self, node_features):
            features = self.backbone(self.extractor(node_features))
            mean = self.mean(features)
            log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
            return mean, log_std

        def sample(self, node_features):
            mean, log_std = self.forward(node_features)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            pre_tanh = normal.rsample()
            action = torch.tanh(pre_tanh)
            log_prob = _squashed_log_prob(normal, pre_tanh, action)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
            return action, log_prob, entropy

        def deterministic(self, node_features):
            mean, _ = self.forward(node_features)
            return torch.tanh(mean)

        def log_prob(self, node_features, action):
            mean, log_std = self.forward(node_features)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            clipped_action = action.clamp(-1.0 + EPSILON, 1.0 - EPSILON)
            pre_tanh = torch.atanh(clipped_action)
            return _squashed_log_prob(normal, pre_tanh, clipped_action)

        def entropy(self, node_features):
            mean, log_std = self.forward(node_features)
            del mean
            std = log_std.exp()
            normal = torch.distributions.Normal(torch.zeros_like(std), std)
            return normal.entropy().sum(dim=-1, keepdim=True)


    class GCNValue(nn.Module):
        """GCN encoder + scalar state-value head (PPO baseline)."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            head_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim, num_facilities, num_nodes, edges, gcn_hidden_sizes,
                include_global_context=include_global_context,
            )
            self.head = _build_mlp(
                self.extractor.output_dim, head_hidden_sizes, 1, output_tanh=False
            )

        def forward(self, node_features):
            return self.head(self.extractor(node_features))


    def _squashed_log_prob(normal, pre_tanh, action):
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPSILON)
        return log_prob.sum(dim=-1, keepdim=True)


    def _build_hidden_mlp(input_dim: int, hidden_sizes: Sequence[int], activation: str):
        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(act())
            previous_dim = int(hidden_dim)
        return nn.Sequential(*layers)


    def _build_mlp(
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        *,
        output_tanh: bool,
    ):
        layers = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, int(output_dim)))
        if output_tanh:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

else:

    class GraphConvolution:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GCNEncoder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GCNActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GCNCritic:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GraphFeatureExtractor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GCNSquashedGaussianActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GCNGaussianActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class GCNValue:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
