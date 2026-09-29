"""Opt-in replay semantics prototype, not wired into any historical agent.

Counterfactuals remain one-step records. Multi-step targets require explicit
trajectory lineage as well as exact observation adjacency. Metadata is a
collector assertion, not independent proof of simulator/RNG continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty provenance identifier")


def _finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric, not a flag or string")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    values = tuple(_finite(x, name) for x in value)
    if len(values) != size:
        raise ValueError(f"{name} dimension mismatch")
    return values


@dataclass(frozen=True)
class ReplaySemantics:
    reward_kind: str
    reward_definition_id: str
    reward_scale: float
    gamma: float
    state_schema_id: str
    action_schema_id: str
    state_dim: int
    action_dim: int
    bootstrap_on_truncation: bool

    def __post_init__(self):
        if self.reward_kind not in ("absolute_environment", "anchor_relative"):
            raise ValueError("unknown reward convention")
        for name in ("reward_definition_id", "state_schema_id", "action_schema_id"):
            _identifier(getattr(self, name), name)
        for name in ("state_dim", "action_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if _finite(self.reward_scale, "reward_scale") <= 0:
            raise ValueError("reward_scale must be positive")
        if not 0 <= _finite(self.gamma, "gamma") <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if type(self.bootstrap_on_truncation) is not bool:
            raise ValueError("truncation bootstrap policy must be explicit")


@dataclass(frozen=True)
class OneStepRecord:
    semantics: ReplaySemantics
    source_id: str
    origin: str
    trajectory_id: str | None
    step_index: int | None
    state_token: str
    next_state_token: str
    state: tuple[float, ...]
    action: tuple[float, ...]
    raw_reward: float
    next_state: tuple[float, ...]
    terminated: bool
    truncated: bool

    def __post_init__(self):
        if not isinstance(self.semantics, ReplaySemantics):
            raise TypeError("explicit ReplaySemantics required")
        for name in ("source_id", "state_token", "next_state_token"):
            _identifier(getattr(self, name), name)
        if self.origin == "trajectory":
            _identifier(self.trajectory_id, "trajectory_id")
            if type(self.step_index) is not int or self.step_index < 0:
                raise ValueError("trajectory step_index must be nonnegative integer")
        elif self.origin == "counterfactual":
            if self.trajectory_id is not None or self.step_index is not None:
                raise ValueError("independent counterfactual must not assert trajectory order")
        else:
            raise ValueError("unknown transition origin")
        for name in ("terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean")
        # Immutable numeric copies prevent caller-side array changes after validation.
        for name, size in (("state", self.semantics.state_dim),
                           ("next_state", self.semantics.state_dim),
                           ("action", self.semantics.action_dim)):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        object.__setattr__(self, "raw_reward", _finite(self.raw_reward, "raw_reward"))


@dataclass(frozen=True)
class ReturnSample:
    semantics: ReplaySemantics
    source_id: str
    origin: str
    trajectory_id: str | None
    first_step_index: int | None
    state: tuple[float, ...]
    action: tuple[float, ...]
    reward: float
    one_step_reward: float
    next_state: tuple[float, ...]
    terminated: bool
    truncated: bool
    n_steps: int

    @property
    def bootstrap_discount(self) -> float:
        if self.terminated or (self.truncated and not self.semantics.bootstrap_on_truncation):
            return 0.0
        return self.semantics.gamma ** self.n_steps


def make_return(records: Sequence[OneStepRecord], semantics: ReplaySemantics) -> ReturnSample:
    """Build one window only; the caller owns window scheduling and tail flushing.

    Raw rewards are scaled once here. The returned bootstrap discount is the
    complete factor gamma**n, not the legacy extra multiplier gamma**(n-1).
    """
    records = tuple(records)
    if not records:
        raise ValueError("at least one record required")
    if not isinstance(semantics, ReplaySemantics):
        raise TypeError("explicit expected replay semantics required")
    for record in records:
        if not isinstance(record, OneStepRecord):
            raise TypeError("OneStepRecord required; legacy arrays are not auto-upgraded")
        if record.semantics != semantics:
            raise ValueError("mixed reward/state/action/discount semantics")
    if len(records) > 1 and any(r.origin != "trajectory" for r in records):
        raise ValueError("independent counterfactuals cannot form multi-step returns")
    for previous, current in zip(records, records[1:]):
        if previous.terminated or previous.truncated:
            raise ValueError("return cannot cross an episode or collection boundary")
        if (previous.source_id != current.source_id
                or previous.trajectory_id != current.trajectory_id
                or current.step_index != previous.step_index + 1):
            raise ValueError("trajectory lineage or step adjacency mismatch")
        if previous.next_state_token != current.state_token:
            raise ValueError("full-state lineage token mismatch")
        if previous.next_state != current.state:
            raise ValueError("observed state adjacency mismatch")
    first, last = records[0], records[-1]
    try:
        reward = math.fsum((semantics.gamma ** i) * r.raw_reward * semantics.reward_scale
                           for i, r in enumerate(records))
    except OverflowError as exc:
        raise ValueError("discounted reward overflow") from exc
    reward = _finite(reward, "discounted reward")
    return ReturnSample(semantics, first.source_id, first.origin, first.trajectory_id,
                        first.step_index, first.state, first.action, reward,
                        _finite(first.raw_reward * semantics.reward_scale, "one-step reward"), last.next_state,
                        last.terminated, last.truncated, len(records))


def bellman_targets(samples: Sequence[ReturnSample], next_q: np.ndarray,
                    semantics: ReplaySemantics) -> np.ndarray:
    """One numeric reference for both calibration and ordinary update targets.

    Prospective adapter tests can compare tensor implementations to this NumPy
    reference. This function is not currently called by any training agent.
    """
    samples = tuple(samples)
    if not samples:
        raise ValueError("nonempty target batch required")
    if not isinstance(semantics, ReplaySemantics):
        raise TypeError("explicit expected replay semantics required")
    values = np.asarray(next_q)
    if values.shape != (len(samples), 1):
        raise ValueError("next_q must have shape (batch, 1), no implicit broadcasting")
    if values.dtype.kind not in "iuf":
        raise ValueError("next_q must contain real numbers, not flags or strings")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("next_q must be finite")
    targets = []
    for sample, q in zip(samples, values[:, 0]):
        if not isinstance(sample, ReturnSample) or sample.semantics != semantics:
            raise ValueError("target batch semantics mismatch")
        if type(sample.n_steps) is not int or sample.n_steps < 1:
            raise ValueError("positive integer n_steps required")
        if type(sample.terminated) is not bool or type(sample.truncated) is not bool:
            raise ValueError("explicit termination flags required")
        if sample.origin not in ("trajectory", "counterfactual"):
            raise ValueError("unknown transition origin")
        if sample.origin == "counterfactual" and sample.n_steps != 1:
            raise ValueError("counterfactual targets must remain one-step")
        target = _finite(sample.reward, "sample reward") + sample.bootstrap_discount * float(q)
        targets.append(_finite(target, "Bellman target"))
    return np.asarray(targets, dtype=np.float64).reshape(-1, 1)
