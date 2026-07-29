"""History-aware flat modules used for matched graph ablations."""

from __future__ import annotations

from typing import Sequence

from src.models.gcn import TemporalDemandNodeEncoder
from src.models.graph_features import (
    GraphStateSpec,
    flat_state_to_node_features,
)
from src.rl.networks import nn, require_torch, torch


if torch is not None:

    class TemporalFacilityFeatureExtractor(nn.Module):
        """Shared per-facility GRU without graph message passing."""

        def __init__(
            self,
            graph_spec: GraphStateSpec,
            hidden_size: int,
            include_global_context: bool = True,
        ):
            super().__init__()
            if not graph_spec.include_demand_sequence_state:
                raise ValueError(
                    "Temporal facility encoder requires demand-sequence state"
                )
            self.graph_spec = graph_spec
            self.num_facilities = graph_spec.num_facilities
            self.temporal = TemporalDemandNodeEncoder(
                graph_spec.node_feature_dim,
                graph_spec.demand_sequence_feature_start,
                graph_spec.demand_sequence_length,
                hidden_size,
            )
            self.include_global_context = bool(include_global_context)
            self.output_dim = (
                self.num_facilities * self.temporal.output_dim
                + (
                    self.temporal.output_dim
                    if self.include_global_context
                    else 0
                )
            )

        def node_embeddings(self, flat_state):
            node_features = flat_state_to_node_features(
                flat_state,
                self.graph_spec,
            )
            return self.temporal(node_features)

        def forward(self, flat_state):
            encoded = self.node_embeddings(flat_state)
            readout = encoded[:, : self.num_facilities, :].flatten(
                start_dim=1
            )
            if self.include_global_context:
                readout = torch.cat(
                    (readout, encoded.mean(dim=1)),
                    dim=-1,
                )
            return readout


    class TemporalFlatActor(nn.Module):
        """Deterministic flat actor with shared clinic-level temporal encoding."""

        def __init__(
            self,
            graph_spec: GraphStateSpec,
            action_dim: int,
            temporal_hidden_size: int,
            hidden_sizes: Sequence[int],
            include_global_context: bool = True,
            readout_mode: str = "global_flat",
        ):
            super().__init__()
            self.extractor = TemporalFacilityFeatureExtractor(
                graph_spec,
                temporal_hidden_size,
                include_global_context=include_global_context,
            )
            self.num_facilities = int(graph_spec.num_facilities)
            self.action_dim = int(action_dim)
            self.readout_mode = str(readout_mode)
            if self.readout_mode in ("global_flat", "network_residual"):
                input_dim = self.extractor.output_dim
                output_dim = self.action_dim
            elif self.readout_mode == "pressure_intensity":
                if self.action_dim != 4 * self.num_facilities:
                    raise ValueError(
                        "pressure_intensity readout requires a facility-net "
                        "action layout"
                    )
                input_dim = self.extractor.temporal.output_dim
                output_dim = 3
            else:
                raise ValueError(
                    f"Unsupported temporal flat actor readout_mode: "
                    f"{self.readout_mode}"
                )
            self.head = _build_mlp(
                input_dim,
                hidden_sizes,
                output_dim,
                output_tanh=True,
            )

        def forward(self, flat_state):
            if self.readout_mode in ("global_flat", "network_residual"):
                return self.head(self.extractor(flat_state))
            encoded = self.extractor.node_embeddings(flat_state)
            intensities = self.head(encoded.mean(dim=1))
            specimen_net = torch.zeros(
                encoded.shape[0],
                self.num_facilities,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            return torch.cat(
                (
                    specimen_net,
                    intensities[:, 0:1].expand(-1, self.num_facilities),
                    intensities[:, 1:2].expand(-1, self.num_facilities),
                    intensities[:, 2:3].expand(-1, self.num_facilities),
                ),
                dim=1,
            )


    class TemporalFlatCritic(nn.Module):
        """State-action critic matched to :class:`TemporalFlatActor`."""

        def __init__(
            self,
            graph_spec: GraphStateSpec,
            action_dim: int,
            temporal_hidden_size: int,
            hidden_sizes: Sequence[int],
            include_global_context: bool = True,
        ):
            super().__init__()
            self.extractor = TemporalFacilityFeatureExtractor(
                graph_spec,
                temporal_hidden_size,
                include_global_context=include_global_context,
            )
            self.head = _build_mlp(
                self.extractor.output_dim + int(action_dim),
                hidden_sizes,
                1,
                output_tanh=False,
            )

        def forward(self, flat_state, action):
            features = self.extractor(flat_state)
            return self.head(torch.cat((features, action), dim=-1))


    class TemporalFlatCorrectionGate(nn.Module):
        """Grouped deployment gate with the same temporal flat encoder."""

        def __init__(
            self,
            graph_spec: GraphStateSpec,
            temporal_hidden_size: int,
            hidden_sizes: Sequence[int],
            output_dim: int,
            include_global_context: bool = True,
        ):
            super().__init__()
            self.extractor = TemporalFacilityFeatureExtractor(
                graph_spec,
                temporal_hidden_size,
                include_global_context=include_global_context,
            )
            self.head = _build_mlp(
                self.extractor.output_dim,
                hidden_sizes,
                int(output_dim),
                output_tanh=False,
            )
            output = next(
                module
                for module in reversed(tuple(self.head.modules()))
                if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

        def forward(self, flat_state):
            output = self.head(self.extractor(flat_state))
            return (
                output.squeeze(-1)
                if output.shape[-1] == 1
                else output
            )

    class TemporalFlatOptionQNetwork(nn.Module):
        """Dueling option network with temporal clinic encoding but no graph."""

        def __init__(
            self,
            graph_spec: GraphStateSpec,
            temporal_hidden_size: int,
            hidden_sizes: Sequence[int],
            num_options: int,
            include_global_context: bool = True,
        ):
            super().__init__()
            self.extractor = TemporalFacilityFeatureExtractor(
                graph_spec,
                temporal_hidden_size,
                include_global_context=include_global_context,
            )
            layers: list[nn.Module] = []
            feature_dim = self.extractor.output_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(feature_dim, int(hidden_size)))
                layers.append(nn.ReLU())
                feature_dim = int(hidden_size)
            self.encoder = nn.Sequential(*layers)
            self.value_head = nn.Linear(feature_dim, 1)
            self.advantage_head = nn.Linear(feature_dim, int(num_options))
            self.correction_gate_head = nn.Linear(
                feature_dim,
                int(num_options) - 1,
            )

        def forward(self, flat_state):
            features = self.encoder(self.extractor(flat_state))
            return self._option_values(features)

        def forward_with_gate(self, flat_state):
            features = self.encoder(self.extractor(flat_state))
            return (
                self._option_values(features),
                self.correction_gate_head(features),
            )

        def _option_values(self, features):
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage[:, 0:1]

        def correction_gate_logits(self, flat_state):
            features = self.encoder(self.extractor(flat_state))
            return self.correction_gate_head(features)


    def _build_mlp(
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        *,
        output_tanh: bool,
    ):
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous, int(hidden_size)))
            layers.append(nn.ReLU())
            previous = int(hidden_size)
        layers.append(nn.Linear(previous, int(output_dim)))
        if output_tanh:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)


else:

    class TemporalFacilityFeatureExtractor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class TemporalFlatActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class TemporalFlatCritic:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class TemporalFlatCorrectionGate:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class TemporalFlatOptionQNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
