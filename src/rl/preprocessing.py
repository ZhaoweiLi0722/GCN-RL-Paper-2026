"""Preprocessing helpers shared by learned RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.rl.networks import torch


def demand_sequence_length(env_config: dict[str, Any]) -> int:
    """Return the enabled causal demand-sequence length."""

    if not bool(env_config.get("include_demand_sequence_state", False)):
        return 0
    length = int(
        env_config.get(
            "demand_sequence_length",
            env_config.get("demand_history_window", 4),
        )
    )
    if length < 1:
        raise ValueError("demand_sequence_length must be positive")
    return length


def facility_state_width(env_config: dict[str, Any]) -> int:
    """Return the flat base-observation width for one facility."""

    lead_time = int(env_config.get("production_lead_time", 3))
    width = (
        3
        + lead_time
        + int(bool(env_config.get("include_supplier_state", False)))
        + int(bool(env_config.get("include_demand_forecast_state", False)))
        + 3
        * int(bool(env_config.get("include_transfer_pipeline_state", False)))
        + 3
        * int(bool(env_config.get("include_demand_history_state", False)))
    )
    width += 3 * demand_sequence_length(env_config)
    if bool(env_config.get("enable_overtime_control", False)):
        # [previous u_ot, outstanding overtime, static surge headroom] plus
        # fatigue when enabled; mirrors CapacityPlanningEnv.features_per_facility
        # (spec 2026-08-29-continuous-overtime-control).
        width += 3 + int(bool(env_config.get("enable_overtime_fatigue", False)))
        if bool(env_config.get("enable_intertemporal_overtime_commitment", False)):
            # Active capacity, pending commitment slots, and repeated shared
            # budget state (spec 2026-09-01-intertemporal-shared-capacity).
            width += 2 + int(env_config.get("overtime_commitment_lead_time", 2))
    return width


@dataclass(frozen=True)
class FixedObservationScaler:
    """Deterministic observation scaling from environment capacity metadata."""

    enabled: bool
    scales: np.ndarray
    clip: float = 10.0

    @classmethod
    def from_config(cls, config: dict[str, Any], state_dim: int) -> "FixedObservationScaler":
        enabled = bool(config.get("normalize_observations", False))
        clip = float(config.get("observation_clip", 10.0))
        scales = np.ones(int(state_dim), dtype=np.float32)
        if not enabled:
            return cls(enabled=False, scales=scales, clip=clip)

        env_config = dict(config.get("env", {}))
        if "num_facilities" in env_config:
            num_facilities = int(env_config["num_facilities"])
        else:
            num_facilities = _infer_num_facilities(state_dim, env_config)
        lead_time = int(env_config.get("production_lead_time", 3))
        include_supplier = bool(env_config.get("include_supplier_state", False))
        include_forecast = bool(env_config.get("include_demand_forecast_state", False))
        include_demand_history = bool(
            env_config.get("include_demand_history_state", False)
        )
        sequence_length = demand_sequence_length(env_config)
        include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
        include_time_state = bool(env_config.get("include_time_state", False))
        patient_summary_width = _patient_summary_width(env_config)
        features_per_facility = (
            3
            + lead_time
            + int(include_supplier)
            + int(include_forecast)
            + 3 * int(include_demand_history)
            + 3 * sequence_length
        )
        if include_transfer_pipeline:
            features_per_facility += 3
        expected_state_dim = (
            num_facilities * (features_per_facility + patient_summary_width)
            + int(include_time_state)
        )
        if expected_state_dim != int(state_dim):
            raise ValueError(
                "normalize_observations expected state_dim="
                f"{expected_state_dim} from env config, got {state_dim}"
            )

        demand_rates = _as_vector(env_config.get("demand_rates", 1.0), num_facilities)
        max_specimens = _as_vector(env_config.get("max_specimens", 1.0), num_facilities)
        max_reagents = _as_vector(env_config.get("max_reagents", 1.0), num_facilities)
        max_idle = _as_vector(env_config.get("max_idle_bioreactors", 1.0), num_facilities)

        base_rows: list[np.ndarray] = []
        patient_rows: list[np.ndarray] = []
        for facility in range(num_facilities):
            row = [
                max(float(demand_rates[facility]), 1.0),
                max(float(max_specimens[facility]), 1.0),
                max(float(max_reagents[facility]), 1.0),
            ]
            row.extend([max(float(max_idle[facility]), 1.0)] * lead_time)
            if include_supplier:
                row.append(1.0)
            if include_forecast:
                horizon = int(env_config.get("demand_forecast_horizon", 1))
                row.append(max(float(demand_rates[facility]) * horizon, 1.0))
            if include_transfer_pipeline:
                row.extend(
                    [
                        max(float(max_specimens[facility]), 1.0),
                        max(float(max_reagents[facility]), 1.0),
                        max(float(max_idle[facility]), 1.0),
                    ]
                )
            if include_demand_history:
                demand_scale = max(float(demand_rates[facility]), 1.0)
                row.extend([demand_scale, demand_scale, demand_scale])
            if sequence_length:
                demand_scale = max(float(demand_rates[facility]), 1.0)
                row.extend(
                    [demand_scale] * sequence_length
                    + [demand_scale] * sequence_length
                    + [1.0] * sequence_length
                )
            if patient_summary_width:
                patient_rows.append(
                    np.asarray(
                        _patient_summary_scale(
                            float(max_specimens[facility]),
                            patient_summary_width,
                        ),
                        dtype=np.float32,
                    )
                )
            base_rows.append(np.asarray(row, dtype=np.float32))
        # PatientConditionCapacityEnv appends all per-clinic patient summaries
        # after the complete base-state block; mirror that exact flat layout.
        rows = base_rows + patient_rows
        scales = np.concatenate(rows)
        if include_time_state:
            scales = np.concatenate((scales, np.ones(1, dtype=np.float32)))
        return cls(enabled=True, scales=scales, clip=clip)

    def normalize_np(self, value: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.asarray(value, dtype=np.float32)
        normalized = np.asarray(value, dtype=np.float32) / self.scales
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def normalize_tensor(self, value):
        if not self.enabled:
            return value
        if torch is None:  # pragma: no cover
            return value
        scales = torch.as_tensor(self.scales, dtype=value.dtype, device=value.device)
        return (value / scales).clamp(-self.clip, self.clip)


def reward_scale_from_config(config: dict[str, Any]) -> float:
    """Return multiplicative reward scale for value-function targets."""

    return float(config.get("reward_scale", config.get("reward_scaling", 1.0)))


def graph_node_feature_scale(config: dict[str, Any], node_feature_dim: int) -> tuple[float, ...]:
    """Build per-feature scaling for GCN node features."""

    env_config = dict(config.get("env", {}))
    lead_time = int(env_config.get("production_lead_time", 3))
    num_facilities = int(env_config.get("num_facilities", 1))
    include_supplier = bool(env_config.get("include_supplier_state", False))
    include_forecast = bool(env_config.get("include_demand_forecast_state", False))
    include_demand_history = bool(
        env_config.get("include_demand_history_state", False)
    )
    sequence_length = demand_sequence_length(env_config)
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    include_adaptive_demand_features = bool(
        config.get("include_adaptive_demand_features", False)
    )
    include_time_state = bool(env_config.get("include_time_state", False))
    include_hub = bool(env_config.get("include_central_capacity_hub", False))
    demand_rates = _as_vector(env_config.get("demand_rates", 1.0), num_facilities)
    max_specimens = _as_vector(env_config.get("max_specimens", 1.0), num_facilities)
    max_reagents = _as_vector(env_config.get("max_reagents", 1.0), num_facilities)
    max_idle = _as_vector(env_config.get("max_idle_bioreactors", 1.0), num_facilities)
    scale = [
        max(float(np.mean(demand_rates)), 1.0),
        max(float(np.max(max_specimens)), 1.0),
        max(float(np.max(max_reagents)), 1.0),
        max(float(np.max(max_idle)), 1.0),
        max(float(np.max(max_idle)) * lead_time, 1.0),
    ]
    if include_supplier:
        scale.append(1.0)
    if include_forecast:
        horizon = int(env_config.get("demand_forecast_horizon", 1))
        scale.append(max(float(np.mean(demand_rates)) * horizon, 1.0))
    if include_transfer_pipeline:
        scale.extend(
            [
                max(float(np.max(max_specimens)), 1.0),
                max(float(np.max(max_reagents)), 1.0),
                max(float(np.max(max_idle)), 1.0),
            ]
        )
    if include_demand_history:
        demand_scale = max(float(np.mean(demand_rates)), 1.0)
        scale.extend([demand_scale, demand_scale, demand_scale])
    if sequence_length:
        demand_scale = max(float(np.mean(demand_rates)), 1.0)
        scale.extend(
            [demand_scale] * sequence_length
            + [demand_scale] * sequence_length
            + [1.0] * sequence_length
        )
    if include_adaptive_demand_features:
        scale.extend([1.0, 1.0, 1.0])
    if include_time_state:
        scale.append(1.0)
    residual_config = dict(config.get("residual_action", {}))
    if bool(
        residual_config.get("enabled", False)
        and residual_config.get("include_base_action_features", False)
    ):
        scale.extend([1.0] * 4)
    if include_hub:
        scale.append(1.0)
    patient_summary_width = _patient_summary_width(env_config)
    if patient_summary_width:
        scale.extend(_patient_summary_scale(float(np.max(max_specimens)), patient_summary_width))
    while len(scale) < int(node_feature_dim):
        scale.append(1.0)
    return tuple(scale[: int(node_feature_dim)])


def _patient_summary_width(env_config: dict[str, Any]) -> int:
    if env_config.get("env_type") != "patient_condition":
        return 0
    edges = env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97))
    routing_width = 4 if env_config.get("include_specimen_routing_state", False) else 0
    return 6 + len(tuple(edges)) + 1 + routing_width


def _patient_summary_scale(max_specimens: float, summary_width: int) -> list[float]:
    count_scale = max(max_specimens, 1.0)
    # Layout: waiting count/survival/expiry, manufacturing count/survival/risk,
    # then waiting-survival buckets.
    return [
        count_scale,
        1.0,
        count_scale,
        count_scale,
        1.0,
        count_scale,
    ] + [count_scale] * max(int(summary_width) - 6, 0)


def _as_vector(values: Sequence[float] | float | int | None, length: int) -> np.ndarray:
    if values is None:
        return np.ones(length, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.shape == ():
        return np.full(length, float(array), dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"Expected vector length {length}, got shape {array.shape}")
    return array


def _infer_num_facilities(state_dim: int, env_config: dict[str, Any]) -> int:
    lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier = bool(env_config.get("include_supplier_state", False))
    include_forecast = bool(env_config.get("include_demand_forecast_state", False))
    include_demand_history = bool(
        env_config.get("include_demand_history_state", False)
    )
    sequence_length = demand_sequence_length(env_config)
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    features_per_facility = (
        3
        + lead_time
        + int(include_supplier)
        + int(include_forecast)
        + 3 * int(include_demand_history)
        + 3 * sequence_length
    )
    if include_transfer_pipeline:
        features_per_facility += 3
    if state_dim % features_per_facility != 0:
        raise ValueError("num_facilities is required for observation normalization")
    return state_dim // features_per_facility
