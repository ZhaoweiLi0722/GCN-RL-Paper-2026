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
    edge_weights: Sequence[float] | None = None,
    device=None,
):
    """Return symmetric normalized adjacency with self-loops."""

    require_torch()
    weights = [1.0] * len(edges) if edge_weights is None else [float(value) for value in edge_weights]
    if len(weights) != len(edges):
        raise ValueError("edge_weights must have the same length as edges")
    adjacency = torch.eye(num_nodes, dtype=torch.float32, device=device)
    for (i, j), weight in zip(edges, weights):
        source = int(i)
        target = int(j)
        if source == target:
            continue
        if source < 0 or target < 0 or source >= num_nodes or target >= num_nodes:
            raise ValueError(f"Edge {(source, target)} is outside the {num_nodes}-node graph")
        adjacency[source, target] = max(float(weight), 0.0)
        adjacency[target, source] = max(float(weight), 0.0)

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
            edge_weights: Sequence[float] | None = None,
        ):
            super().__init__()
            if not hidden_sizes:
                raise ValueError("hidden_sizes must contain at least one GCN layer")
            self.register_buffer(
                "adjacency",
                build_normalized_adjacency(num_nodes, edges, edge_weights=edge_weights),
            )
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


    class TemporalDemandNodeEncoder(nn.Module):
        """Compress a fixed causal demand/error sequence for every graph node."""

        def __init__(
            self,
            node_feature_dim: int,
            sequence_start: int,
            sequence_length: int,
            hidden_size: int,
        ):
            super().__init__()
            self.node_feature_dim = int(node_feature_dim)
            self.sequence_start = int(sequence_start)
            self.sequence_length = int(sequence_length)
            self.hidden_size = int(hidden_size)
            if self.sequence_length < 1:
                raise ValueError("temporal sequence_length must be positive")
            if self.hidden_size < 1:
                raise ValueError("temporal hidden_size must be positive")
            sequence_width = 3 * self.sequence_length
            if (
                self.sequence_start < 0
                or self.sequence_start + sequence_width
                > self.node_feature_dim
            ):
                raise ValueError(
                    "Temporal demand sequence is outside the node feature layout"
                )
            self.sequence_width = sequence_width
            self.output_dim = (
                self.node_feature_dim - sequence_width + self.hidden_size
            )
            self.gru = nn.GRU(
                input_size=3,
                hidden_size=self.hidden_size,
                batch_first=True,
            )

        def forward(self, node_features):
            start = self.sequence_start
            stop = start + self.sequence_width
            sequence_block = node_features[:, :, start:stop]
            demand, error, mask = torch.split(
                sequence_block,
                self.sequence_length,
                dim=-1,
            )
            sequence = torch.stack((demand, error, mask), dim=-1)
            batch_size, num_nodes = sequence.shape[:2]
            sequence = sequence.reshape(
                batch_size * num_nodes,
                self.sequence_length,
                3,
            )
            _output, hidden = self.gru(sequence)
            temporal = hidden[-1].reshape(
                batch_size,
                num_nodes,
                self.hidden_size,
            )
            has_history = (
                mask.sum(dim=-1, keepdim=True) > 0.0
            ).to(dtype=temporal.dtype)
            temporal = temporal * has_history
            return torch.cat(
                (
                    node_features[:, :, :start],
                    temporal,
                    node_features[:, :, stop:],
                ),
                dim=-1,
            )


    class HistoryAwareGCNEncoder(nn.Module):
        """Shared per-clinic GRU followed by geographic message passing."""

        def __init__(
            self,
            node_feature_dim: int,
            hidden_sizes: Sequence[int],
            num_nodes: int,
            edges: Sequence[Edge],
            *,
            sequence_start: int,
            sequence_length: int,
            temporal_hidden_size: int,
            edge_weights: Sequence[float] | None = None,
        ):
            super().__init__()
            self.temporal = TemporalDemandNodeEncoder(
                node_feature_dim,
                sequence_start,
                sequence_length,
                temporal_hidden_size,
            )
            self.gcn = GCNEncoder(
                self.temporal.output_dim,
                hidden_sizes,
                num_nodes,
                edges,
                edge_weights=edge_weights,
            )
            self.output_dim = self.gcn.output_dim

        def forward(self, node_features):
            return self.gcn(self.temporal(node_features))


    def _make_gcn_encoder(
        node_feature_dim: int,
        hidden_sizes: Sequence[int],
        num_nodes: int,
        edges: Sequence[Edge],
        *,
        edge_weights: Sequence[float] | None,
        temporal_sequence_start: int | None,
        temporal_sequence_length: int,
        temporal_hidden_size: int,
    ):
        if temporal_hidden_size <= 0:
            return GCNEncoder(
                node_feature_dim,
                hidden_sizes,
                num_nodes,
                edges,
                edge_weights=edge_weights,
            )
        if temporal_sequence_start is None or temporal_sequence_start < 0:
            raise ValueError(
                "Temporal GCN encoder requires demand-sequence node features"
            )
        return HistoryAwareGCNEncoder(
            node_feature_dim,
            hidden_sizes,
            num_nodes,
            edges,
            sequence_start=temporal_sequence_start,
            sequence_length=temporal_sequence_length,
            temporal_hidden_size=temporal_hidden_size,
            edge_weights=edge_weights,
        )


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
            edge_weights: Sequence[float] | None = None,
            temporal_sequence_start: int | None = None,
            temporal_sequence_length: int = 0,
            temporal_hidden_size: int = 0,
        ):
            super().__init__()
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.encoder = _make_gcn_encoder(
                node_feature_dim,
                gcn_hidden_sizes,
                num_nodes,
                edges,
                edge_weights=edge_weights,
                temporal_sequence_start=temporal_sequence_start,
                temporal_sequence_length=temporal_sequence_length,
                temporal_hidden_size=temporal_hidden_size,
            )
            self.output_dim = graph_readout_dim(
                self.num_facilities, self.encoder.output_dim, self.include_global_context
            )

        def forward(self, node_features):
            encoded = self.encoder(node_features)
            return graph_readout(encoded, self.num_facilities, self.include_global_context)

    def _edge_pair_tensor(edges: Sequence[Edge]):
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.as_tensor(tuple(edges), dtype=torch.long).transpose(0, 1)


    def _edge_feature_tensor(
        features: Sequence[Sequence[float]],
        feature_dim: int,
    ):
        if not features:
            return torch.zeros((0, int(feature_dim)), dtype=torch.float32)
        if int(feature_dim) == 0 or len(tuple(features[0])) == 0:
            return torch.zeros((len(features), int(feature_dim)), dtype=torch.float32)
        return torch.as_tensor(tuple(features), dtype=torch.float32)


    def _validate_network_edge_metadata(
        specimen_edges: Sequence[Edge],
        resource_edges: Sequence[Edge],
        capacity_edges: Sequence[Edge],
        specimen_features: Sequence[Sequence[float]],
        resource_features: Sequence[Sequence[float]],
        capacity_features: Sequence[Sequence[float]],
    ) -> int:
        if len(specimen_edges) != len(specimen_features):
            raise ValueError(
                "network_residual requires one feature row per specimen edge"
            )
        if len(resource_edges) != len(resource_features):
            raise ValueError(
                "network_residual requires one feature row per resource edge"
            )
        if len(capacity_edges) != len(capacity_features):
            raise ValueError(
                "network_residual requires one feature row per capacity edge"
            )
        dimensions = {
            len(tuple(row))
            for row in (
                *specimen_features,
                *resource_features,
                *capacity_features,
            )
        }
        if len(dimensions) > 1:
            raise ValueError(
                "network_residual edge feature rows must have a common width"
            )
        return next(iter(dimensions), 0)


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
            edge_weights: Sequence[float] | None = None,
            specimen_routing_enabled: bool = False,
            specimen_edges: Sequence[Edge] | None = None,
            resource_edges: Sequence[Edge] | None = None,
            capacity_edges: Sequence[Edge] | None = None,
            specimen_edge_features: Sequence[Sequence[float]] | None = None,
            resource_edge_features: Sequence[Sequence[float]] | None = None,
            capacity_edge_features: Sequence[Sequence[float]] | None = None,
            edge_selector_enabled: bool = False,
            edge_selector_top_k: int = 2,
            edge_selector_temperature: float = 1.0,
            edge_selector_training_gate: str = "soft",
            temporal_sequence_start: int | None = None,
            temporal_sequence_length: int = 0,
            temporal_hidden_size: int = 0,
        ):
            super().__init__()
            self.node_feature_dim = int(node_feature_dim)
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.readout_mode = str(readout_mode)
            self.specimen_routing_enabled = bool(specimen_routing_enabled)
            self.last_edge_flows = {}
            self.last_edge_selector_logits = {}
            self.edge_selector_enabled = bool(edge_selector_enabled)
            self.edge_selector_top_k = max(int(edge_selector_top_k), 1)
            self.edge_selector_temperature = max(
                float(edge_selector_temperature),
                1e-6,
            )
            self.edge_selector_training_gate = str(
                edge_selector_training_gate
            )
            if self.edge_selector_training_gate not in (
                "none",
                "soft",
                "straight_through",
            ):
                raise ValueError(
                    "edge_selector_training_gate must be none, soft, "
                    "or straight_through"
                )
            self.encoder = _make_gcn_encoder(
                node_feature_dim,
                gcn_hidden_sizes,
                num_nodes,
                edges,
                edge_weights=edge_weights,
                temporal_sequence_start=temporal_sequence_start,
                temporal_sequence_length=temporal_sequence_length,
                temporal_hidden_size=temporal_hidden_size,
            )
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
            elif self.readout_mode == "pressure_intensity":
                if int(action_dim) != 4 * self.num_facilities:
                    raise ValueError(
                        "pressure_intensity readout requires a facility-net action layout"
                    )
                self.head = _build_mlp(
                    self.encoder.output_dim,
                    head_hidden_sizes,
                    3,
                    output_tanh=True,
                )
                if self.specimen_routing_enabled:
                    specimen_input_dim = self.encoder.output_dim
                    if self.include_global_context:
                        specimen_input_dim += self.encoder.output_dim
                    self.specimen_pressure_head = _build_mlp(
                        specimen_input_dim,
                        head_hidden_sizes,
                        1,
                        output_tanh=True,
                    )
            elif self.readout_mode == "network_residual":
                if int(action_dim) != 4 * self.num_facilities:
                    raise ValueError(
                        "network_residual readout requires a facility-net action layout"
                    )
                resource_edges = tuple(resource_edges or ())
                capacity_edges = tuple(capacity_edges or ())
                if self.specimen_routing_enabled:
                    specimen_edges = tuple(specimen_edges or ())
                    specimen_edge_features = tuple(
                        specimen_edge_features
                        if specimen_edge_features is not None
                        else (() for _edge in specimen_edges)
                    )
                else:
                    specimen_edges = ()
                    specimen_edge_features = ()
                resource_edge_features = tuple(
                    resource_edge_features
                    if resource_edge_features is not None
                    else (() for _edge in resource_edges)
                )
                capacity_edge_features = tuple(
                    capacity_edge_features
                    if capacity_edge_features is not None
                    else (() for _edge in capacity_edges)
                )
                edge_feature_dim = _validate_network_edge_metadata(
                    specimen_edges,
                    resource_edges,
                    capacity_edges,
                    specimen_edge_features,
                    resource_edge_features,
                    capacity_edge_features,
                )
                if self.specimen_routing_enabled:
                    self.register_buffer(
                        "specimen_edge_pairs",
                        _edge_pair_tensor(specimen_edges),
                    )
                    self.register_buffer(
                        "specimen_static_edge_features",
                        _edge_feature_tensor(
                            specimen_edge_features,
                            edge_feature_dim,
                        ),
                    )
                self.register_buffer(
                    "resource_edge_pairs",
                    _edge_pair_tensor(resource_edges),
                )
                self.register_buffer(
                    "capacity_edge_pairs",
                    _edge_pair_tensor(capacity_edges),
                )
                self.register_buffer(
                    "resource_static_edge_features",
                    _edge_feature_tensor(
                        resource_edge_features,
                        edge_feature_dim,
                    ),
                )
                self.register_buffer(
                    "capacity_static_edge_features",
                    _edge_feature_tensor(
                        capacity_edge_features,
                        edge_feature_dim,
                    ),
                )
                context_dim = (
                    self.encoder.output_dim if self.include_global_context else 0
                )
                node_input_dim = (
                    self.encoder.output_dim
                    + self.node_feature_dim
                    + context_dim
                )
                edge_input_dim = (
                    3 * self.encoder.output_dim
                    + self.node_feature_dim
                    + context_dim
                    + edge_feature_dim
                )
                self.replenishment_head = _build_mlp(
                    node_input_dim,
                    head_hidden_sizes,
                    1,
                    output_tanh=True,
                )
                self.reagent_edge_head = _build_mlp(
                    edge_input_dim,
                    head_hidden_sizes,
                    1,
                    output_tanh=True,
                )
                self.capacity_edge_head = _build_mlp(
                    edge_input_dim,
                    head_hidden_sizes,
                    1,
                    output_tanh=True,
                )
                if self.specimen_routing_enabled:
                    self.specimen_edge_head = _build_mlp(
                        edge_input_dim,
                        head_hidden_sizes,
                        1,
                        output_tanh=True,
                    )
                if self.edge_selector_enabled:
                    self.reagent_edge_selector_head = _build_mlp(
                        edge_input_dim,
                        head_hidden_sizes,
                        1,
                        output_tanh=False,
                    )
                    self.capacity_edge_selector_head = _build_mlp(
                        edge_input_dim,
                        head_hidden_sizes,
                        1,
                        output_tanh=False,
                    )
                    if self.specimen_routing_enabled:
                        self.specimen_edge_selector_head = _build_mlp(
                            edge_input_dim,
                            head_hidden_sizes,
                            1,
                            output_tanh=False,
                        )
                else:
                    self.reagent_edge_selector_head = None
                    self.capacity_edge_selector_head = None
                    if self.specimen_routing_enabled:
                        self.specimen_edge_selector_head = None
            else:
                raise ValueError(f"Unsupported GCN actor readout_mode: {self.readout_mode}")

        def forward(self, node_features):
            encoded = self.encoder(node_features)
            facility_encoded = encoded[:, : self.num_facilities, :]
            if self.readout_mode == "network_residual":
                facility_features = node_features[:, : self.num_facilities, :]
                graph_context = (
                    encoded.mean(dim=1)
                    if self.include_global_context
                    else None
                )
                node_inputs = [facility_encoded, facility_features]
                if graph_context is not None:
                    node_inputs.append(
                        graph_context.unsqueeze(1).expand(
                            -1,
                            self.num_facilities,
                            -1,
                        )
                    )
                replenishment = self.replenishment_head(
                    torch.cat(node_inputs, dim=-1)
                ).squeeze(-1)
                (
                    reagent_net,
                    reagent_edge_flow,
                    reagent_selector_logits,
                ) = self._edge_net_actions(
                    facility_encoded,
                    facility_features,
                    graph_context,
                    self.resource_edge_pairs,
                    self.resource_static_edge_features,
                    self.reagent_edge_head,
                    self.reagent_edge_selector_head,
                )
                (
                    capacity_net,
                    capacity_edge_flow,
                    capacity_selector_logits,
                ) = self._edge_net_actions(
                    facility_encoded,
                    facility_features,
                    graph_context,
                    self.capacity_edge_pairs,
                    self.capacity_static_edge_features,
                    self.capacity_edge_head,
                    self.capacity_edge_selector_head,
                )
                if self.specimen_routing_enabled:
                    (
                        specimen_net,
                        specimen_edge_flow,
                        specimen_selector_logits,
                    ) = self._edge_net_actions(
                        facility_encoded,
                        facility_features,
                        graph_context,
                        self.specimen_edge_pairs,
                        self.specimen_static_edge_features,
                        self.specimen_edge_head,
                        self.specimen_edge_selector_head,
                    )
                else:
                    specimen_net = torch.zeros_like(reagent_net)
                self.last_edge_flows = {
                    "reagent_transfer": reagent_edge_flow,
                    "capacity_transfer": capacity_edge_flow,
                }
                self.last_edge_selector_logits = {
                    "reagent_transfer": reagent_selector_logits,
                    "capacity_transfer": capacity_selector_logits,
                }
                if self.specimen_routing_enabled:
                    self.last_edge_flows["specimen_transfer"] = specimen_edge_flow
                    self.last_edge_selector_logits["specimen_transfer"] = (
                        specimen_selector_logits
                    )
                return torch.cat(
                    (
                        specimen_net,
                        reagent_net,
                        capacity_net,
                        replenishment,
                    ),
                    dim=1,
                )
            if self.readout_mode == "pressure_intensity":
                intensities = self.head(encoded.mean(dim=1))
                if self.specimen_routing_enabled:
                    specimen_inputs = [facility_encoded]
                    if self.include_global_context:
                        specimen_inputs.append(
                            encoded.mean(dim=1, keepdim=True).expand(
                                -1,
                                self.num_facilities,
                                -1,
                            )
                        )
                    specimen_pressure = self.specimen_pressure_head(
                        torch.cat(specimen_inputs, dim=-1)
                    ).squeeze(-1)
                    specimen_net = 0.5 * (
                        specimen_pressure
                        - specimen_pressure.mean(dim=1, keepdim=True)
                    )
                else:
                    specimen_net = torch.zeros(
                        encoded.shape[0],
                        self.num_facilities,
                        dtype=encoded.dtype,
                        device=encoded.device,
                    )
                reagent_intensity = intensities[:, 0:1].expand(
                    -1,
                    self.num_facilities,
                )
                capacity_intensity = intensities[:, 1:2].expand(
                    -1,
                    self.num_facilities,
                )
                replenishment_intensity = intensities[:, 2:3].expand(
                    -1,
                    self.num_facilities,
                )
                return torch.cat(
                    (
                        specimen_net,
                        reagent_intensity,
                        capacity_intensity,
                        replenishment_intensity,
                    ),
                    dim=1,
                )
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

        def _edge_net_actions(
            self,
            facility_encoded,
            facility_features,
            graph_context,
            edge_pairs,
            static_edge_features,
            edge_head,
            selector_head,
        ):
            batch_size = facility_encoded.shape[0]
            if edge_pairs.shape[1] == 0:
                net = torch.zeros(
                    batch_size,
                    self.num_facilities,
                    dtype=facility_encoded.dtype,
                    device=facility_encoded.device,
                )
                edge_flow = torch.zeros(
                    batch_size,
                    0,
                    dtype=facility_encoded.dtype,
                    device=facility_encoded.device,
                )
                return net, edge_flow, edge_flow
            source = edge_pairs[0]
            target = edge_pairs[1]
            source_encoded = facility_encoded[:, source, :]
            target_encoded = facility_encoded[:, target, :]
            edge_inputs = [
                source_encoded,
                target_encoded,
                target_encoded - source_encoded,
                facility_features[:, target, :] - facility_features[:, source, :],
            ]
            if graph_context is not None:
                edge_inputs.append(
                    graph_context.unsqueeze(1).expand(
                        -1,
                        source.shape[0],
                        -1,
                    )
                )
            if static_edge_features.shape[1] > 0:
                edge_inputs.append(
                    static_edge_features.unsqueeze(0).expand(
                        batch_size,
                        -1,
                        -1,
                    )
                )
            edge_inputs = torch.cat(edge_inputs, dim=-1)
            edge_flow = edge_head(edge_inputs).squeeze(-1)
            if selector_head is None:
                selector_logits = torch.zeros_like(edge_flow)
            else:
                selector_logits = selector_head(edge_inputs).squeeze(-1)
                edge_flow = edge_flow * self._edge_selector_gate(
                    selector_logits
                )
            net = torch.zeros(
                batch_size,
                self.num_facilities,
                dtype=edge_flow.dtype,
                device=edge_flow.device,
            )
            source_index = source.unsqueeze(0).expand(batch_size, -1)
            target_index = target.unsqueeze(0).expand(batch_size, -1)
            net.scatter_add_(1, source_index, -edge_flow)
            net.scatter_add_(1, target_index, edge_flow)
            normalizer = net.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
            return net / normalizer, edge_flow, selector_logits

        def _edge_selector_gate(self, selector_logits):
            edge_count = int(selector_logits.shape[1])
            if edge_count == 0:
                return selector_logits
            top_k = min(self.edge_selector_top_k, edge_count)
            if self.training:
                if self.edge_selector_training_gate == "none":
                    return torch.ones_like(selector_logits)
                probabilities = torch.softmax(
                    selector_logits / self.edge_selector_temperature,
                    dim=1,
                )
                soft_gate = torch.clamp(
                    probabilities * float(top_k),
                    max=1.0,
                )
                if self.edge_selector_training_gate == "soft":
                    return soft_gate
                selected = torch.topk(
                    selector_logits,
                    k=top_k,
                    dim=1,
                ).indices
                hard_gate = torch.zeros_like(selector_logits).scatter(
                    1,
                    selected,
                    1.0,
                )
                return hard_gate + soft_gate - soft_gate.detach()
            selected = torch.topk(
                selector_logits,
                k=top_k,
                dim=1,
            ).indices
            mask = torch.zeros_like(selector_logits)
            return mask.scatter(1, selected, 1.0)

        def zero_initialize_output_heads(self) -> None:
            """Initialize every actor output head to the zero residual policy."""

            if self.readout_mode == "network_residual":
                heads = [
                    self.replenishment_head,
                    self.reagent_edge_head,
                    self.capacity_edge_head,
                ]
                if self.specimen_routing_enabled:
                    heads.append(self.specimen_edge_head)
                if self.edge_selector_enabled:
                    heads.extend(
                        (
                            self.reagent_edge_selector_head,
                            self.capacity_edge_selector_head,
                        )
                    )
                    if self.specimen_routing_enabled:
                        heads.append(self.specimen_edge_selector_head)
            elif (
                self.readout_mode == "pressure_intensity"
                and self.specimen_routing_enabled
            ):
                heads = [self.head, self.specimen_pressure_head]
            else:
                heads = [self.head]
            for head in heads:
                output_layer = next(
                    module
                    for module in reversed(tuple(head.modules()))
                    if isinstance(module, nn.Linear)
                )
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)


    class GCNCorrectionGate(nn.Module):
        """Graph classifier deciding whether a residual correction is admissible."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            edges: Sequence[Edge],
            gcn_hidden_sizes: Sequence[int],
            head_hidden_sizes: Sequence[int],
            include_global_context: bool = True,
            edge_weights: Sequence[float] | None = None,
            output_dim: int = 1,
            temporal_sequence_start: int | None = None,
            temporal_sequence_length: int = 0,
            temporal_hidden_size: int = 0,
        ):
            super().__init__()
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.encoder = _make_gcn_encoder(
                node_feature_dim,
                gcn_hidden_sizes,
                num_nodes,
                edges,
                edge_weights=edge_weights,
                temporal_sequence_start=temporal_sequence_start,
                temporal_sequence_length=temporal_sequence_length,
                temporal_hidden_size=temporal_hidden_size,
            )
            readout_dim = graph_readout_dim(
                self.num_facilities,
                self.encoder.output_dim,
                self.include_global_context,
            )
            self.head = _build_mlp(
                readout_dim,
                head_hidden_sizes,
                int(output_dim),
                output_tanh=False,
            )
            output_layer = next(
                module
                for module in reversed(tuple(self.head.modules()))
                if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

        def forward(self, node_features):
            encoded = self.encoder(node_features)
            readout = graph_readout(
                encoded,
                self.num_facilities,
                self.include_global_context,
            )
            output = self.head(readout)
            return output.squeeze(-1) if output.shape[-1] == 1 else output


    def transfer_matching_parameters(source_state_dict, target_module):
        """Copy every source parameter/buffer whose name and shape match the target.

        Enables curriculum warm-start across graph sizes. A size-invariant
        ``facility_action`` actor trained on a small network transfers all of its
        learnable weights---the node-independent GCN convolution weights and the
        shared per-facility head---into a larger-network actor, because none of those
        shapes depend on the number of facilities. Only the non-learnable
        ``encoder.adjacency`` buffer differs in shape across graph sizes; it is skipped,
        and the target keeps its own adjacency rebuilt for its graph. A ``global_flat``
        actor instead transfers only the encoder, since its head width scales with the
        number of facilities.

        Returns a summary dict listing transferred and skipped keys.
        """

        target_state = target_module.state_dict()
        filtered = {}
        transferred, skipped = [], []
        for key, tensor in source_state_dict.items():
            if key in target_state and target_state[key].shape == tensor.shape:
                filtered[key] = tensor
                transferred.append(key)
            else:
                skipped.append(key)
        target_module.load_state_dict(filtered, strict=False)
        return {"transferred": transferred, "skipped": skipped}


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
            edge_weights: Sequence[float] | None = None,
            temporal_sequence_start: int | None = None,
            temporal_sequence_length: int = 0,
            temporal_hidden_size: int = 0,
        ):
            super().__init__()
            self.num_facilities = int(num_facilities)
            self.include_global_context = bool(include_global_context)
            self.encoder = _make_gcn_encoder(
                node_feature_dim,
                gcn_hidden_sizes,
                num_nodes,
                edges,
                edge_weights=edge_weights,
                temporal_sequence_start=temporal_sequence_start,
                temporal_sequence_length=temporal_sequence_length,
                temporal_hidden_size=temporal_hidden_size,
            )
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
            edge_weights: Sequence[float] | None = None,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim, num_facilities, num_nodes, edges, gcn_hidden_sizes,
                include_global_context=include_global_context,
                edge_weights=edge_weights,
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
            edge_weights: Sequence[float] | None = None,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim, num_facilities, num_nodes, edges, gcn_hidden_sizes,
                include_global_context=include_global_context,
                edge_weights=edge_weights,
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
            edge_weights: Sequence[float] | None = None,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim, num_facilities, num_nodes, edges, gcn_hidden_sizes,
                include_global_context=include_global_context,
                edge_weights=edge_weights,
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


    class GCNCorrectionGate:  # pragma: no cover
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
