"""Matched flat-state residual-option Double-DQN baseline."""

from __future__ import annotations

from src.models.gcn_option_dqn import GCNResidualOptionDQNAgent


class FlatResidualOptionDQNAgent(GCNResidualOptionDQNAgent):
    """Use the same residual options and RL updates with an MLP state encoder."""

    algorithm = "flat_residual_mdl2_option_dqn_afd"
