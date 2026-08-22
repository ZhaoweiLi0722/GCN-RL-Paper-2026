"""Matched TD3 for the complete AFR-GCN residual policy surface."""

from __future__ import annotations

from typing import Any

from src.models.gcn_ddpg import GCNDDPGAgent, flat_state_to_node_features
from src.rl.matched_residual_td3 import MatchedResidualTD3Mixin


class ConservativeGCNResidualTD3Agent(
    MatchedResidualTD3Mixin,
    GCNDDPGAgent,
):
    """TD3 critic backbone with the routing-primary GCN-DDPG contract."""

    algorithm = "gcn_residual_mdl2_network_td3_bc"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ):
        super().__init__(state_dim, action_dim, config)
        self._initialize_matched_td3(config)

    def _td3_state_views(self, raw_states):
        node_features = flat_state_to_node_features(
            raw_states,
            self.graph_spec,
        )
        return node_features, node_features
