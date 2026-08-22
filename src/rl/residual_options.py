"""Structured graph-residual options around a deterministic heuristic anchor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state


SUPPORTED_RESIDUAL_OPTION_GROUPS = (
    "replenishment_uniform",
    "specimen_transfer",
    "reagent_transfer",
    "combined_transfer",
    "reagent_replenishment",
    "combined_network",
    "combined_routing_network",
)


@dataclass(frozen=True)
class ResidualOptionSpec:
    group: str
    epsilon: float
    sign: float

    @property
    def is_anchor(self) -> bool:
        return self.group == "anchor"


def make_residual_option_specs(
    epsilons: Iterable[float],
    groups: Iterable[str],
    signs: Iterable[float] = (-1.0, 1.0),
) -> tuple[ResidualOptionSpec, ...]:
    """Return a fixed candidate ordering shared by training and deployment."""

    epsilon_values = tuple(float(value) for value in epsilons)
    group_values = tuple(str(group) for group in groups)
    sign_values = tuple(float(sign) for sign in signs)
    unknown = sorted(set(group_values) - set(SUPPORTED_RESIDUAL_OPTION_GROUPS))
    if unknown:
        raise ValueError(f"Unsupported residual option groups: {unknown}")
    if not epsilon_values or any(value <= 0.0 or value > 1.0 for value in epsilon_values):
        raise ValueError("Residual option epsilons must lie in (0, 1]")
    if not sign_values or any(sign not in (-1.0, 1.0) for sign in sign_values):
        raise ValueError("Residual option signs must be -1 or 1")
    specs = [ResidualOptionSpec("anchor", 0.0, 0.0)]
    for epsilon in epsilon_values:
        for sign in sign_values:
            for group in group_values:
                specs.append(ResidualOptionSpec(group, epsilon, sign))
    return tuple(specs)


def make_explicit_residual_option_specs(
    options: Iterable[dict[str, Any]],
) -> tuple[ResidualOptionSpec, ...]:
    """Return an anchor plus an explicitly ordered correction option subset."""

    specs = [ResidualOptionSpec("anchor", 0.0, 0.0)]
    seen: set[tuple[str, float, float]] = set()
    for raw_option in options:
        option = dict(raw_option)
        group = str(option.get("group", ""))
        epsilon = float(option.get("epsilon", 0.0))
        sign = float(option.get("sign", 0.0))
        if group not in SUPPORTED_RESIDUAL_OPTION_GROUPS:
            raise ValueError(f"Unsupported residual option group: {group}")
        if epsilon <= 0.0 or epsilon > 1.0:
            raise ValueError("Residual option epsilon must lie in (0, 1]")
        if sign not in (-1.0, 1.0):
            raise ValueError("Residual option sign must be -1 or 1")
        key = (group, epsilon, sign)
        if key in seen:
            raise ValueError(f"Duplicate residual option: {group}:{sign:g}:{epsilon:g}")
        seen.add(key)
        specs.append(ResidualOptionSpec(group, epsilon, sign))
    if len(specs) == 1:
        raise ValueError("Explicit residual options must include at least one correction")
    return tuple(specs)


def residual_option_actions_from_env(
    anchor_action: np.ndarray,
    env,
    specs: Sequence[ResidualOptionSpec],
) -> list[np.ndarray]:
    """Instantiate graph-residual options from the current live environment.

    The option direction is an on-hand imbalance pattern. In-transit resources
    remain part of the policy observation, so the option selector can decide
    whether a correction is useful without changing the option's meaning.
    """

    n = int(env.config.num_facilities)
    demand = np.asarray(env.demand, dtype=float)
    forecast = np.asarray(getattr(env, "demand_forecast", demand), dtype=float)
    specimens = np.asarray(env.specimens, dtype=float)
    reagents = np.asarray(env.reagents, dtype=float)
    idle_bioreactors = np.asarray(env.bioreactors[:, 0], dtype=float)
    risk = _env_vector(env, "at_risk_counts", n) + _env_vector(
        env,
        "near_expiry_counts",
        n,
    )
    resource_pattern = _centered_unit_pattern(
        demand + 0.25 * forecast + specimens - reagents + 0.5 * risk
    )
    capacity_pattern = _centered_unit_pattern(
        demand
        + 0.25 * forecast
        + specimens
        - idle_bioreactors
        + 0.5 * risk
    )
    pending_specimens = np.zeros(n, dtype=float)
    pending = getattr(env, "_pending_transfer_arrivals", None)
    if callable(pending):
        pending_specimens = np.asarray(pending()[0], dtype=float)
    specimen_pattern = _centered_unit_pattern(
        np.minimum(reagents, idle_bioreactors)
        - specimens
        - pending_specimens
    )
    return residual_option_actions_from_patterns(
        anchor_action,
        resource_pattern,
        capacity_pattern,
        specs,
        specimen_pattern=specimen_pattern,
    )


def residual_option_actions_from_state(
    state: np.ndarray,
    env_config: dict[str, Any],
    anchor_settings,
    specs: Sequence[ResidualOptionSpec],
) -> list[np.ndarray]:
    """Reconstruct the same options from a flat observation for cached labels."""

    anchor_action = facility_net_action_from_state(
        np.asarray(state, dtype=np.float32),
        env_config,
        settings=anchor_settings,
    )
    resource_pattern, capacity_pattern = residual_pressure_patterns_from_state(
        state,
        env_config,
    )
    specimen_pattern = specimen_pressure_pattern_from_state(
        state,
        env_config,
    )
    return residual_option_actions_from_patterns(
        anchor_action,
        resource_pattern,
        capacity_pattern,
        specs,
        specimen_pattern=specimen_pattern,
    )


def residual_option_actions_from_patterns(
    anchor_action: np.ndarray,
    resource_pattern: np.ndarray,
    capacity_pattern: np.ndarray,
    specs: Sequence[ResidualOptionSpec],
    *,
    specimen_pattern: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Apply each option specification to an anchor action."""

    anchor = np.asarray(anchor_action, dtype=np.float32).reshape(-1)
    resource = np.asarray(resource_pattern, dtype=np.float32).reshape(-1)
    capacity = np.asarray(capacity_pattern, dtype=np.float32).reshape(-1)
    n = int(resource.size)
    specimen = (
        np.zeros(n, dtype=np.float32)
        if specimen_pattern is None
        else np.asarray(specimen_pattern, dtype=np.float32).reshape(-1)
    )
    if (
        capacity.shape != (n,)
        or specimen.shape != (n,)
        or anchor.shape != (4 * n,)
    ):
        raise ValueError("Residual option patterns require a facility-net action layout")
    actions: list[np.ndarray] = []
    for spec in specs:
        action = anchor.copy()
        if spec.is_anchor:
            actions.append(action)
            continue
        delta = float(spec.sign) * float(spec.epsilon)
        if spec.group == "replenishment_uniform":
            action[3 * n : 4 * n] += delta
        elif spec.group == "specimen_transfer":
            action[:n] += delta * specimen
        elif spec.group == "reagent_transfer":
            action[n : 2 * n] += delta * resource
        elif spec.group == "combined_transfer":
            action[n : 2 * n] += delta * resource
            action[2 * n : 3 * n] += delta * capacity
        elif spec.group == "reagent_replenishment":
            action[n : 2 * n] += delta * resource
            action[3 * n : 4 * n] += delta * resource
        elif spec.group == "combined_network":
            action[n : 2 * n] += delta * resource
            action[2 * n : 3 * n] += delta * capacity
            action[3 * n : 4 * n] += delta * resource
        elif spec.group == "combined_routing_network":
            action[:n] += delta * specimen
            action[n : 2 * n] += delta * resource
            action[2 * n : 3 * n] += delta * capacity
            action[3 * n : 4 * n] += delta * resource
        else:  # pragma: no cover - guarded by make_residual_option_specs
            raise ValueError(f"Unsupported residual option group: {spec.group}")
        actions.append(np.clip(action, -1.0, 1.0).astype(np.float32))
    return actions


def residual_option_labels_from_actions(
    states: np.ndarray,
    actions: np.ndarray,
    env_config: dict[str, Any],
    anchor_settings,
    specs: Sequence[ResidualOptionSpec],
) -> np.ndarray:
    """Map cached continuous teacher actions to their nearest legal option."""

    state_array = np.asarray(states, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    if state_array.ndim != 2 or action_array.ndim != 2:
        raise ValueError("Residual option labels require rank-2 states and actions")
    if state_array.shape[0] != action_array.shape[0]:
        raise ValueError("Residual option label batches must have equal sample counts")
    labels = np.zeros(state_array.shape[0], dtype=np.int64)
    for index, (state, target_action) in enumerate(zip(state_array, action_array)):
        candidates = residual_option_actions_from_state(
            state,
            env_config,
            anchor_settings,
            specs,
        )
        errors = np.asarray(
            [
                float(np.mean(np.square(candidate - target_action)))
                for candidate in candidates
            ],
            dtype=float,
        )
        labels[index] = int(np.argmin(errors))
    return labels


def residual_option_labels_from_advantages(
    advantages: np.ndarray,
    feasible: np.ndarray,
    *,
    min_advantage: float = 0.0,
) -> np.ndarray:
    """Select option labels directly from dense common-random-number scores."""

    values = np.asarray(advantages, dtype=np.float32)
    mask = np.asarray(feasible, dtype=bool)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Option advantages require a non-empty rank-2 array")
    if mask.shape != values.shape:
        raise ValueError("Option feasibility must match option advantages")
    if not np.all(np.isfinite(values)):
        raise ValueError("Option advantages must be finite")
    if not np.all(mask[:, 0]):
        raise ValueError("The anchor option must be feasible in every state")
    threshold = float(min_advantage)
    if threshold < 0.0:
        raise ValueError("min_advantage must be non-negative")
    adjusted = values.copy()
    if adjusted.shape[1] > 1 and threshold > 0.0:
        adjusted[:, 1:] -= threshold
    adjusted[~mask] = -np.inf
    return np.argmax(adjusted, axis=1).astype(np.int64)


def residual_pressure_patterns_from_state(
    state: np.ndarray,
    env_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Recover teacher-aligned on-hand pressure from an observation.

    Pipeline features are deliberately excluded from the directional pattern.
    They are still visible to the learned selector, which can keep the anchor
    when in-transit resources already cover the apparent local imbalance.
    """

    values = np.asarray(state, dtype=np.float32).reshape(-1)
    n = int(env_config.get("num_facilities", 0))
    lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier = int(bool(env_config.get("include_supplier_state", False)))
    include_forecast = int(bool(env_config.get("include_demand_forecast_state", False)))
    include_pipeline = int(bool(env_config.get("include_transfer_pipeline_state", False)))
    include_history = int(
        bool(env_config.get("include_demand_history_state", False))
    )
    width = (
        3
        + lead_time
        + include_supplier
        + include_forecast
        + 3 * include_pipeline
        + 3 * include_history
    )
    base_width = n * width
    if n <= 0 or values.size < base_width:
        raise ValueError("Observation is too short for residual option pressure features")
    facility = values[:base_width].reshape(n, width)
    demand = facility[:, 0]
    specimens = facility[:, 1]
    reagents = facility[:, 2]
    idle_bioreactors = facility[:, 3]
    if include_forecast:
        forecast_col = 3 + lead_time + include_supplier
        forecast = facility[:, forecast_col]
    else:
        forecast = demand
    risk = _patient_waiting_risk_from_state(values, env_config, width)
    resource = (
        demand
        + 0.25 * forecast
        + specimens
        - reagents
        + 0.5 * risk
    )
    capacity = (
        demand
        + 0.25 * forecast
        + specimens
        - idle_bioreactors
        + 0.5 * risk
    )
    return _centered_unit_pattern(resource), _centered_unit_pattern(capacity)


def specimen_pressure_pattern_from_state(
    state: np.ndarray,
    env_config: dict[str, Any],
) -> np.ndarray:
    """Recover the patient-lot donor/receiver pressure from a flat state."""

    values = np.asarray(state, dtype=np.float32).reshape(-1)
    n = int(env_config.get("num_facilities", 0))
    lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier = int(bool(env_config.get("include_supplier_state", False)))
    include_forecast = int(
        bool(env_config.get("include_demand_forecast_state", False))
    )
    include_pipeline = int(
        bool(env_config.get("include_transfer_pipeline_state", False))
    )
    include_history = int(
        bool(env_config.get("include_demand_history_state", False))
    )
    include_sequence = int(
        bool(env_config.get("include_demand_sequence_state", False))
    )
    sequence_length = int(env_config.get("demand_sequence_length", 12))
    width = (
        3
        + lead_time
        + include_supplier
        + include_forecast
        + 3 * include_pipeline
        + 3 * include_history
        + 3 * sequence_length * include_sequence
    )
    base_width = n * width
    if n <= 0 or values.size < base_width:
        raise ValueError("Observation is too short for specimen pressure features")
    facility = values[:base_width].reshape(n, width)
    specimens = facility[:, 1]
    reagents = facility[:, 2]
    idle_bioreactors = facility[:, 3]
    pending_specimens = np.zeros(n, dtype=np.float32)
    if include_pipeline:
        pipeline_start = 3 + lead_time + include_supplier + include_forecast
        pending_specimens = facility[:, pipeline_start]
    return _centered_unit_pattern(
        np.minimum(reagents, idle_bioreactors)
        - specimens
        - pending_specimens
    )


def _patient_waiting_risk_from_state(
    state: np.ndarray,
    env_config: dict[str, Any],
    features_per_facility: int,
) -> np.ndarray:
    n = int(env_config.get("num_facilities", 0))
    if env_config.get("env_type") != "patient_condition":
        return np.zeros(n, dtype=np.float32)
    edges = tuple(float(value) for value in env_config.get(
        "survival_bucket_edges",
        (0.85, 0.90, 0.97),
    ))
    summary_width = (
        6
        + len(edges)
        + 1
        + (4 if env_config.get("include_specimen_routing_state", False) else 0)
    )
    base_width = n * int(features_per_facility)
    expected = base_width + n * summary_width
    if state.size < expected:
        return np.zeros(n, dtype=np.float32)
    summary = state[base_width:expected].reshape(n, summary_width)
    patient_config = dict(env_config.get("patient", {}))
    threshold = float(patient_config.get("eligibility_threshold", 0.80)) + float(
        env_config.get("urgency_margin", 0.10)
    )
    bucket_count = sum(edge <= threshold + 1e-8 for edge in edges)
    waiting_at_risk = (
        summary[:, 6 : 6 + bucket_count].sum(axis=1)
        if bucket_count > 0
        else np.zeros(n, dtype=np.float32)
    )
    return np.asarray(summary[:, 2] + waiting_at_risk, dtype=np.float32)


def _env_vector(env, name: str, n: int) -> np.ndarray:
    value = getattr(env, name, None)
    if callable(value):
        value = value()
    if value is None:
        return np.zeros(n, dtype=float)
    result = np.asarray(value, dtype=float).reshape(-1)
    return result if result.shape == (n,) else np.zeros(n, dtype=float)


def _centered_unit_pattern(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    centered = vector - float(vector.mean())
    scale = max(float(np.max(np.abs(centered))), 1e-6)
    return (centered / scale).astype(np.float32)
