"""Fail-closed deployment loading; no partial gate transfer or learner restore."""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.models.gcn_ddpg import GCNDDPGAgent
from src.utils.research_archive import sha256_file


POLICY_MODULES = ("actor", "correction_gate", "correction_safety_gate")
POLICY_METADATA = (
    "algorithm", "state_dim", "action_dim", "correction_gate_mode",
    "correction_gate_threshold", "correction_gate_include_proposed_residual_features",
    "correction_gate_proposed_residual_feature_mode", "correction_gate_group_thresholds",
    "correction_safety_gate_threshold", "correction_safety_gate_group_thresholds",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_module(module, saved, name):
    if module is None:
        if saved is not None:
            raise ValueError(f"Unexpected policy module: {name}")
        return
    expected = module.state_dict()
    if not isinstance(saved, dict) or saved.keys() != expected.keys():
        raise ValueError(f"Policy tensor keys differ: {name}")
    for key, tensor in saved.items():
        reference = expected[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid policy tensor: {name}.{key}")


def load_policy_payload_strict(agent, payload):
    """Validate all modules and semantics before mutating any live parameter."""
    for key in POLICY_METADATA:
        if key not in payload or canonical(payload[key]) != canonical(getattr(agent, key)):
            raise ValueError(f"Policy metadata mismatch: {key}")
    if canonical(payload.get("graph_spec")) != canonical(asdict(agent.graph_spec)):
        raise ValueError("Policy graph/observation contract mismatch")
    for name in POLICY_MODULES:
        if name not in payload:
            raise ValueError(f"Missing policy module declaration: {name}")
        validate_module(getattr(agent, name), payload[name], name)
    for name in POLICY_MODULES:
        module = getattr(agent, name)
        if module is not None:
            module.load_state_dict(payload[name], strict=True)
            module.eval()
            module.requires_grad_(False)


class StrictFrozenPolicy:
    """Inference-only facade over the existing deployment action implementation.

    Construction uses the legacy agent, with an empty one-entry replay allocation.
    No checkpoint critic, optimizer, replay, exploration or RNG state is restored.
    """

    def __init__(self, checkpoint: Path, effective_config: Path, *,
                 checkpoint_sha256: str, config_sha256: str, device="cpu"):
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint hash mismatch")
        if sha256_file(effective_config) != config_sha256:
            raise ValueError("Effective config hash mismatch")
        config = json.loads(effective_config.read_text())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        runtime = copy.deepcopy(config)
        runtime.update(device=device, replay_buffer_size=1)
        self._agent = GCNDDPGAgent(payload["state_dim"], payload["action_dim"], runtime)
        if self._agent.residual_temporal_guard.enabled:
            raise ValueError("Stateful temporal guard needs a separate continuation contract")
        load_policy_payload_strict(self._agent, payload)
        # The constructor's optimizers/critics are unused; the facade exposes no update.
        self.state_dim = self._agent.state_dim
        self.action_dim = self._agent.action_dim
        self.checkpoint_sha256 = checkpoint_sha256
        self.config_sha256 = config_sha256

    def act(self, observation, *, env=None):
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid frozen-policy observation")
        result = self._agent.select_action(state, explore=False, env=env)
        self._agent.actor.eval()
        if result.shape != (self.action_dim,) or not np.isfinite(result).all():
            raise ValueError("Invalid frozen-policy request")
        return result.copy()
