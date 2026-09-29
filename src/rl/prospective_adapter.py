"""Isolated N1/N2 acceptance adapter, not a production collector or learner.

Only explicitly declared records are accepted. Historical arrays are not
migrated, source assertions are not authenticated, and no environment is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.models.matched_inputs import InputSchema, ObservationBatch, MatchedInputs, build_matched_inputs
from src.rl.networks import require_torch, torch
from src.rl.validated_returns import ReplaySemantics, OneStepRecord, ReturnSample, make_return


@dataclass(frozen=True)
class ReplayInputContract:
    inputs: InputSchema
    replay: ReplaySemantics

    def __post_init__(self):
        if not isinstance(self.inputs, InputSchema) or not isinstance(self.replay, ReplaySemantics):
            raise TypeError("explicit input and replay contracts required")
        expected_state = self.inputs.definition_id + "/actor-flat"
        expected_action = self.inputs.definition_id + "/action"
        if (self.replay.state_schema_id != expected_state
                or self.replay.action_schema_id != expected_action):
            raise ValueError("replay schema IDs must bind to the declared input definition")
        n = len(self.inputs.node_ids)
        width = n * len(self.inputs.node_feature_names) + len(self.inputs.global_feature_names)
        width += n * n + len(self.inputs.action_names)
        if self.replay.state_dim != width or self.replay.action_dim != len(self.inputs.action_names):
            raise ValueError("replay dimensions do not match canonical input layout")


def pack_actor_state(observation, anchor_actions, contract: ReplayInputContract):
    """Return the canonical tensor layout; caller must persist the schema too."""
    if not isinstance(contract, ReplayInputContract):
        raise TypeError("explicit ReplayInputContract required")
    return build_matched_inputs(observation, contract.inputs, anchor_actions,
                                role="actor", message_mode="self_only").flat


def completed_segment_windows(
    records: Sequence[OneStepRecord], semantics: ReplaySemantics, *, max_steps: int,
) -> tuple[tuple[OneStepRecord, ...], ...]:
    """One window per start, including every short tail of a closed segment.

    The caller must supply one genuine terminated/truncated trajectory segment.
    This is a batch acceptance helper, not a live queue, reset or resume manager.
    Independent counterfactuals bypass it and remain separate one-step windows.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    records = tuple(records)
    if not records:
        raise ValueError("nonempty completed segment required")
    for record in records:
        make_return((record,), semantics)
        if record.origin != "trajectory":
            raise ValueError("segment requires trajectory records, not independent counterfactuals")
    # Validate lineage even when max_steps=1, which otherwise hides discontinuities.
    for pair in zip(records, records[1:]):
        make_return(pair, semantics)
    if not (records[-1].terminated or records[-1].truncated):
        raise ValueError("segment must explicitly end in termination or truncation")
    return tuple(records[i:i + max_steps] for i in range(len(records)))


@dataclass(frozen=True)
class PreparedReplayBatch:
    samples: tuple[ReturnSample, ...]
    current_actor: MatchedInputs
    current_critic: MatchedInputs
    next_actor: MatchedInputs
    rewards: Any
    bootstrap_discounts: Any

    def targets(self, next_q):
        """Shared prospective calibration/update target, with no extra gamma."""
        require_torch()
        if not isinstance(next_q, torch.Tensor):
            raise TypeError("next_q must be a tensor")
        if tuple(next_q.shape) != (len(self.samples), 1):
            raise ValueError("next_q shape mismatch; no implicit broadcasting")
        if next_q.dtype != self.rewards.dtype or next_q.device != self.rewards.device:
            raise ValueError("next_q dtype/device mismatch")
        if not torch.isfinite(next_q).all().item():
            raise ValueError("next_q must be finite")
        targets = self.rewards + self.bootstrap_discounts * next_q.detach()
        if not torch.isfinite(targets).all().item():
            raise ValueError("nonfinite Bellman target")
        return targets


def _decode_actor_state(states, contract):
    schema = contract.inputs
    batch = states.shape[0]
    n, f, g = len(schema.node_ids), len(schema.node_feature_names), len(schema.global_feature_names)
    start_globals, start_links, start_anchor = n * f, n * f + g, n * f + g + n * n
    observation = ObservationBatch(
        schema, states[:, :start_globals].reshape(batch, n, f),
        states[:, start_globals:start_links],
        states[:, start_links:start_anchor].reshape(batch, n, n))
    return observation, states[:, start_anchor:]


def prepare_replay_batch(
    windows: Sequence[Sequence[OneStepRecord]], contract: ReplayInputContract,
    *, dtype, device, message_mode: str,
) -> PreparedReplayBatch:
    """Validate before converting; no loader, cache migration or Q evaluation.

    Current and bootstrap anchors are stored with their own observed state.
    Critic actions are the first recorded action, in the declared coordinates.
    Gate proposals are not guessed from post-gate behavior records.
    """
    require_torch()
    if not isinstance(contract, ReplayInputContract):
        raise TypeError("explicit ReplayInputContract required")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("explicit float32/float64 dtype required")
    samples = tuple(make_return(window, contract.replay) for window in windows)
    if not samples:
        raise ValueError("nonempty replay batch required")
    current = torch.tensor([s.state for s in samples], dtype=dtype, device=device)
    following = torch.tensor([s.next_state for s in samples], dtype=dtype, device=device)
    actions = torch.tensor([s.action for s in samples], dtype=dtype, device=device)
    current_obs, current_anchor = _decode_actor_state(current, contract)
    next_obs, next_anchor = _decode_actor_state(following, contract)
    current_actor = build_matched_inputs(current_obs, contract.inputs, current_anchor,
                                         role="actor", message_mode=message_mode)
    current_critic = build_matched_inputs(current_obs, contract.inputs, current_anchor,
                                          role="critic", action=actions, message_mode=message_mode)
    next_actor = build_matched_inputs(next_obs, contract.inputs, next_anchor,
                                      role="actor", message_mode=message_mode)
    rewards = torch.tensor([[s.reward] for s in samples], dtype=dtype, device=device)
    discounts = torch.tensor([[s.bootstrap_discount] for s in samples], dtype=dtype, device=device)
    if not torch.isfinite(rewards).all().item() or not torch.isfinite(discounts).all().item():
        raise ValueError("target fields overflow during dtype conversion")
    return PreparedReplayBatch(samples, current_actor, current_critic, next_actor, rewards, discounts)
