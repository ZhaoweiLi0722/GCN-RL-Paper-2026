"""Preprocessing helpers shared by learned RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.rl.networks import torch


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
        features_per_facility = 3 + lead_time + int(include_supplier)
        expected_state_dim = num_facilities * features_per_facility
        if expected_state_dim != int(state_dim):
            raise ValueError(
                "normalize_observations expected state_dim="
                f"{expected_state_dim} from env config, got {state_dim}"
            )

        demand_rates = _as_vector(env_config.get("demand_rates", 1.0), num_facilities)
        max_specimens = _as_vector(env_config.get("max_specimens", 1.0), num_facilities)
        max_reagents = _as_vector(env_config.get("max_reagents", 1.0), num_facilities)
        max_idle = _as_vector(env_config.get("max_idle_bioreactors", 1.0), num_facilities)

        rows: list[np.ndarray] = []
        for facility in range(num_facilities):
            row = [
                max(float(demand_rates[facility]), 1.0),
                max(float(max_specimens[facility]), 1.0),
                max(float(max_reagents[facility]), 1.0),
            ]
            row.extend([max(float(max_idle[facility]), 1.0)] * lead_time)
            if include_supplier:
                row.append(1.0)
            rows.append(np.asarray(row, dtype=np.float32))
        return cls(enabled=True, scales=np.concatenate(rows), clip=clip)

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
    while len(scale) < int(node_feature_dim):
        scale.append(1.0)
    return tuple(scale[: int(node_feature_dim)])


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
    features_per_facility = 3 + lead_time + int(include_supplier)
    if state_dim % features_per_facility != 0:
        raise ValueError("num_facilities is required for observation normalization")
    return state_dim // features_per_facility
