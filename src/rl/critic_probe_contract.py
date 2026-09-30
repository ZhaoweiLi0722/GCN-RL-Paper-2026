"""Data and split guards for a prospective frozen-continuation critic probe.

No environment, policy loading, optimizer or experimental launch lives here.
Monte Carlo contrast labels are not immediate rewards or replay transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ProbeSchema:
    observation_size: int = 561
    action_size: int = 80
    choices: int = 6
    horizon: int = 52
    time_index: int = 560


@dataclass(frozen=True)
class StateLineage:
    state_id: str
    snapshot_sha256: str
    continuation_sha256: str
    environment_sha256: str
    trajectory_seed: int
    decision_step: int
    usage: str


@dataclass(frozen=True)
class PublicInputs:
    observation: np.ndarray
    requests: np.ndarray
    reference_request: np.ndarray


def _finite_array(value, shape):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Invalid public input shape or nonfinite value")
    array.setflags(write=False)
    return array


def public_inputs(payload: Mapping, step: int, schema: ProbeSchema = ProbeSchema()) -> PublicInputs:
    if set(payload) != {"observation", "requests", "reference_request"}:
        raise ValueError("Public input whitelist rejects metadata, hidden state and labels")
    if not 0 <= step < schema.horizon:
        raise ValueError("Invalid decision time")
    observation = _finite_array(payload["observation"], (schema.observation_size,))
    requests = _finite_array(payload["requests"], (schema.choices, schema.action_size))
    reference = _finite_array(payload["reference_request"], (schema.action_size,))
    if np.max(np.abs(requests)) > 1.0 or not np.array_equal(requests[0], reference):
        raise ValueError("Invalid normalized requests or frozen reference")
    if not np.isclose(observation[schema.time_index], step / schema.horizon, atol=1e-7, rtol=0):
        raise ValueError("Public time coordinate differs from decision time")
    return PublicInputs(observation, requests, reference)


def validate_partitions(states: Sequence[StateLineage], assignment: Mapping[str, str]):
    """A trajectory cannot cross partitions, even under different policies."""
    ids = [row.state_id for row in states]
    if len(ids) != len(set(ids)) or set(ids) != set(assignment):
        raise ValueError("Duplicate, missing or extra state assignment")
    by_trajectory, by_snapshot, environments, per_policy = {}, {}, set(), {}
    for row in states:
        if row.usage != "prospective_unseen":
            raise ValueError("Previously inspected pilot states are engineering-only")
        role = assignment[row.state_id]
        if role not in ("train", "test"):
            raise ValueError("Only explicit train/test roles are supported")
        environments.add(row.environment_sha256)
        trajectory = (row.environment_sha256, row.trajectory_seed)
        for key, table in ((trajectory, by_trajectory), (row.snapshot_sha256, by_snapshot)):
            if key in table and table[key] != role:
                raise ValueError("Trajectory/snapshot crosses train and test")
            table[key] = role
        per_policy.setdefault(row.continuation_sha256, {"train": set(), "test": set()})[role].add(trajectory)
    if not states or len(environments) != 1:
        raise ValueError("One unchanged environment contract is required")
    if any(not roles["train"] or not roles["test"] for roles in per_policy.values()):
        raise ValueError("Each continuation needs disjoint training and test trajectories")
    return {policy: {role: len(value) for role, value in roles.items()}
            for policy, roles in per_policy.items()}


def require_single_continuation(states: Sequence[StateLineage]):
    values = {row.continuation_sha256 for row in states}
    if len(values) != 1:
        raise ValueError("Do not pool different continuation policies into one unconditioned critic")
    return next(iter(values))


def paired_advantage(candidate_costs, reference_costs, candidate_seeds, reference_seeds,
                     *, reward_scale=1e-9):
    costs = np.asarray(candidate_costs, dtype=np.float64)
    reference = np.asarray(reference_costs, dtype=np.float64)
    seeds = tuple(int(seed) for seed in candidate_seeds)
    if (costs.ndim != 1 or len(costs) < 2 or reference.shape != costs.shape
            or len(seeds) != len(costs) or seeds != tuple(reference_seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Contrast needs aligned distinct paired future RNG starts")
    if (not np.isfinite(costs).all() or not np.isfinite(reference).all()
            or np.any(costs < 0) or np.any(reference < 0)
            or not np.isfinite(reward_scale) or reward_scale <= 0):
        raise ValueError("Invalid absolute costs or common scale")
    difference = costs - reference
    advantages = -float(reward_scale) * difference
    return {"target_kind": "monte_carlo_frozen_continuation_advantage_not_immediate_reward",
            "paired_cost_deltas": difference.tolist(), "advantage_draws": advantages.tolist(),
            "mean_advantage": float(advantages.mean()),
            "advantage_mean_se": float(advantages.std(ddof=1) / np.sqrt(len(costs))),
            "common_reward_scale": float(reward_scale), "positive_means_lower_cost": True}


def _training_matrix(values, roles):
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 2 or not len(array) or len(roles) != len(array)
            or set(roles) != {"train"} or not np.isfinite(array).all()):
        raise ValueError("Fit inputs must contain finite training rows only")
    return array


def train_only_standardization(values, roles):
    array = _training_matrix(values, roles)
    mean = array.mean(axis=0)
    scale = array.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def train_only_constant(costs, support, roles):
    """Best training action family with observable-support fallback to frozen."""
    array = _training_matrix(costs, roles)
    support = np.asarray(support, dtype=bool)
    if support.shape != array.shape or not support[:, 0].all():
        raise ValueError("Support mask must retain frozen action for every state")
    realized = np.where(support, array, array[:, :1])
    means = realized.mean(axis=0)
    return int(np.argmin(means)), means


def select_supported(advantage_predictions, support):
    predictions = np.asarray(advantage_predictions, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if (predictions.ndim != 2 or predictions.shape != support.shape
            or not np.isfinite(predictions).all() or not support[:, 0].all()
            or not np.all(predictions[:, 0] == 0.0)):
        raise ValueError("Prediction contract requires a zero frozen advantage and support")
    return np.argmax(np.where(support, predictions, -np.inf), axis=1)


def ranking_counts(predictions, costs, support):
    """Exact ties are reported separately; empty comparisons are not failures."""
    predictions = np.asarray(predictions, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    choices = select_supported(predictions, support)
    if costs.shape != predictions.shape or not np.isfinite(costs).all():
        raise ValueError("Invalid evaluation labels")
    totals = dict(correct=0, incorrect=0, prediction_ties=0, label_ties=0)
    regrets = []
    for row, choice in enumerate(choices):
        valid = np.flatnonzero(support[row])
        regrets.append(float(costs[row, choice] - np.min(costs[row, valid])))
        for i, left in enumerate(valid):
            for right in valid[i + 1:]:
                truth = costs[row, right] - costs[row, left]
                prediction = predictions[row, left] - predictions[row, right]
                if truth == 0:
                    totals["label_ties"] += 1
                elif prediction == 0:
                    totals["prediction_ties"] += 1
                else:
                    totals["correct" if np.sign(truth) == np.sign(prediction) else "incorrect"] += 1
    comparable = totals["correct"] + totals["incorrect"] + totals["prediction_ties"]
    return {**totals, "comparable_pairs": comparable,
            "pairwise_accuracy": totals["correct"] / comparable if comparable else None,
            "selected_actions": choices.tolist(), "sampled_support_regret": regrets,
            "regret_reference": "optimistic held-out sampled argmin; diagnostic only"}
