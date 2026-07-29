"""Stateful deployment constraints for anchored residual transfer policies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.graph.edges import k_nearest_ring_edges
from src.graph.geography import geographic_knn_edges, normalize_coordinates
from src.rl.preprocessing import facility_state_width


TRANSFER_GROUPS = ("reagent_transfer", "capacity_transfer")
SUPPORTED_GROUPS = (*TRANSFER_GROUPS, "replenishment")


class ResidualTemporalGuard:
    """Limit repeated residual corrections while preserving net-flow balance.

    The guard operates on the *scaled* residual action, so all budgets are
    expressed in normalized facility-net action units. Transfer-group L1 norms
    count both sending and receiving endpoints.
    """

    def __init__(
        self,
        *,
        num_facilities: int,
        action_dim: int,
        env_config: dict[str, Any],
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.num_facilities = int(num_facilities)
        self.action_dim = int(action_dim)
        self.env_config = dict(env_config)
        self.settings = deepcopy(dict(settings or {}))
        self.enabled = bool(self.settings.get("enabled", False))
        if self.enabled and self.action_dim != 4 * self.num_facilities:
            raise ValueError(
                "Residual temporal guard requires a facility-net action layout"
            )

        self.activation_threshold = max(
            float(self.settings.get("activation_threshold", 1e-6)),
            0.0,
        )
        self.cooldown_steps = self._group_int_values(
            self.settings.get("cooldown_steps", 0),
            default_groups=TRANSFER_GROUPS,
            name="cooldown_steps",
        )
        self.per_step_l1_limits = self._group_float_values(
            self.settings.get("per_step_l1_limits", {}),
            name="per_step_l1_limits",
            default=np.inf,
        )
        self.episode_l1_budgets = self._group_float_values(
            self.settings.get("episode_l1_budgets", {}),
            name="episode_l1_budgets",
            default=np.inf,
        )

        pipeline_config = dict(self.settings.get("pipeline_guard", {}))
        self.pipeline_guard_enabled = bool(
            self.enabled and pipeline_config.get("enabled", False)
        )
        pipeline_groups = pipeline_config.get("groups", TRANSFER_GROUPS)
        if isinstance(pipeline_groups, str):
            pipeline_groups = (pipeline_groups,)
        self.pipeline_groups = tuple(str(value) for value in pipeline_groups)
        self._validate_groups(self.pipeline_groups, "pipeline_guard.groups")
        unknown_pipeline_groups = set(self.pipeline_groups) - set(TRANSFER_GROUPS)
        if unknown_pipeline_groups:
            raise ValueError(
                "Pipeline guard supports transfer groups only: "
                + ", ".join(sorted(unknown_pipeline_groups))
            )
        self.pipeline_pending_thresholds = self._group_float_values(
            pipeline_config.get("pending_thresholds", {}),
            name="pipeline_guard.pending_thresholds",
            default=0.0,
            supported=TRANSFER_GROUPS,
        )
        if self.pipeline_guard_enabled and not bool(
            self.env_config.get("include_transfer_pipeline_state", False)
        ):
            raise ValueError(
                "pipeline_guard requires env.include_transfer_pipeline_state"
            )

        shift_config = dict(
            self.settings.get("observable_shift_gate", {})
        )
        self.observable_shift_gate_enabled = bool(
            self.enabled and shift_config.get("enabled", False)
        )
        shift_groups = shift_config.get("groups", TRANSFER_GROUPS)
        if isinstance(shift_groups, str):
            shift_groups = (shift_groups,)
        self.observable_shift_groups = tuple(
            str(value) for value in shift_groups
        )
        self._validate_groups(
            self.observable_shift_groups,
            "observable_shift_gate.groups",
        )
        self.shift_rate_deviation_threshold = max(
            float(
                shift_config.get(
                    "absolute_rate_deviation_threshold",
                    0.25,
                )
            ),
            0.0,
        )
        self.shift_trend_ratio_threshold = max(
            float(
                shift_config.get(
                    "absolute_trend_ratio_threshold",
                    0.10,
                )
            ),
            0.0,
        )
        self.shift_min_facilities = max(
            int(shift_config.get("min_facilities", 3)),
            1,
        )
        self.shift_min_time_fraction = float(
            shift_config.get("min_time_fraction", 0.0)
        )
        self.shift_detection_mode = str(
            shift_config.get("detection_mode", "relative")
        )
        if self.shift_detection_mode not in (
            "relative",
            "graph_poisson_z",
        ):
            raise ValueError(
                "observable_shift_gate.detection_mode must be "
                "'relative' or 'graph_poisson_z'"
            )
        self.shift_z_score_threshold = max(
            float(shift_config.get("z_score_threshold", 1.5)),
            0.0,
        )
        self.shift_smoothing_steps = max(
            int(shift_config.get("smoothing_steps", 1)),
            0,
        )
        if not 0.0 <= self.shift_min_time_fraction <= 1.0:
            raise ValueError(
                "observable_shift_gate.min_time_fraction must lie in [0, 1]"
            )
        if self.observable_shift_gate_enabled and not bool(
            self.env_config.get("include_demand_history_state", False)
        ):
            raise ValueError(
                "observable_shift_gate requires env.include_demand_history_state"
            )
        self.shift_adjacency = self._shift_adjacency()
        self.reset()

    def reset(self) -> None:
        self.cooldowns = {
            group: np.zeros(self.num_facilities, dtype=np.int64)
            for group in TRANSFER_GROUPS
        }
        self.used_l1 = {
            group: 0.0
            for group in SUPPORTED_GROUPS
        }
        self.last_info = self._empty_step_info()

    def snapshot(self) -> dict[str, Any]:
        """Capture mutable deployment state for isolated counterfactuals."""

        return {
            "cooldowns": {
                group: values.copy()
                for group, values in self.cooldowns.items()
            },
            "used_l1": {
                group: float(value)
                for group, value in self.used_l1.items()
            },
            "last_info": deepcopy(self.last_info),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot without sharing mutable arrays or metadata."""

        cooldowns = dict(snapshot.get("cooldowns", {}))
        used_l1 = dict(snapshot.get("used_l1", {}))
        if set(cooldowns) != set(TRANSFER_GROUPS):
            raise ValueError(
                "Temporal-guard snapshot has incompatible cooldown groups"
            )
        if set(used_l1) != set(SUPPORTED_GROUPS):
            raise ValueError(
                "Temporal-guard snapshot has incompatible budget groups"
            )
        restored_cooldowns: dict[str, np.ndarray] = {}
        for group in TRANSFER_GROUPS:
            values = np.asarray(cooldowns[group], dtype=np.int64)
            if values.shape != (self.num_facilities,):
                raise ValueError(
                    "Temporal-guard snapshot has incompatible facility count"
                )
            if np.any(values < 0):
                raise ValueError(
                    "Temporal-guard snapshot cooldowns cannot be negative"
                )
            restored_cooldowns[group] = values.copy()
        restored_used_l1 = {
            group: float(used_l1[group])
            for group in SUPPORTED_GROUPS
        }
        if any(
            not np.isfinite(value) or value < 0.0
            for value in restored_used_l1.values()
        ):
            raise ValueError(
                "Temporal-guard snapshot budgets must be finite and non-negative"
            )

        self.cooldowns = restored_cooldowns
        self.used_l1 = restored_used_l1
        self.last_info = deepcopy(
            snapshot.get("last_info", self._empty_step_info())
        )

    def apply(
        self,
        residual_action: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return a temporally admissible scaled residual correction."""

        raw = np.asarray(residual_action, dtype=np.float32)
        if raw.shape != (self.action_dim,):
            raise ValueError(
                f"Expected residual shape {(self.action_dim,)}, got {raw.shape}"
            )
        if not self.enabled:
            self.last_info = self._step_info(raw, raw, {})
            return raw.copy()

        governed = raw.copy()
        pending = self._pending_pipeline_vectors(state)
        shift_info = self._observable_shift_info(state)
        group_details: dict[str, dict[str, Any]] = {}
        next_cooldowns = {
            group: np.maximum(values - 1, 0)
            for group, values in self.cooldowns.items()
        }

        for group in SUPPORTED_GROUPS:
            group_slice = self._group_slice(group)
            values = governed[group_slice].astype(np.float64, copy=True)
            raw_values = values.copy()
            blocked = np.zeros(self.num_facilities, dtype=bool)
            positive_blocked = np.zeros(self.num_facilities, dtype=bool)
            shift_blocked = bool(
                self.observable_shift_gate_enabled
                and group in self.observable_shift_groups
                and not shift_info["active"]
            )
            if shift_blocked:
                values.fill(0.0)

            if group in TRANSFER_GROUPS:
                blocked = self.cooldowns[group] > 0
                values[blocked] = 0.0
                if self.pipeline_guard_enabled and group in self.pipeline_groups:
                    positive_blocked = (
                        pending[group]
                        > self.pipeline_pending_thresholds[group]
                    )
                values = _balance_signed_net(
                    values,
                    positive_blocked=positive_blocked,
                )

            values = self._apply_l1_limit(
                values,
                self.per_step_l1_limits[group],
            )
            remaining_budget = max(
                self.episode_l1_budgets[group] - self.used_l1[group],
                0.0,
            )
            values = self._apply_l1_limit(values, remaining_budget)
            applied_l1 = float(np.abs(values).sum())
            self.used_l1[group] += applied_l1
            governed[group_slice] = values.astype(np.float32)

            active = np.abs(values) > self.activation_threshold
            if group in TRANSFER_GROUPS:
                next_cooldowns[group][active] = self.cooldown_steps[group]
            group_details[group] = {
                "raw_l1": float(np.abs(raw_values).sum()),
                "applied_l1": applied_l1,
                "cooldown_blocked_facilities": int(blocked.sum()),
                "pipeline_blocked_receivers": int(positive_blocked.sum()),
                "observable_shift_blocked": shift_blocked,
                "episode_l1_used": float(self.used_l1[group]),
                "episode_l1_budget": float(
                    self.episode_l1_budgets[group]
                ),
            }

        self.cooldowns = next_cooldowns
        governed = np.clip(governed, -1.0, 1.0).astype(np.float32)
        self.last_info = self._step_info(
            raw,
            governed,
            group_details,
            shift_info=shift_info,
        )
        return governed

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used_l1": {
                group: float(value)
                for group, value in self.used_l1.items()
            },
            "episode_l1_budgets": {
                group: float(value)
                for group, value in self.episode_l1_budgets.items()
            },
            "cooldowns": {
                group: values.tolist()
                for group, values in self.cooldowns.items()
            },
            "last_info": deepcopy(self.last_info),
        }

    def observable_shift_info(
        self,
        state: np.ndarray,
    ) -> dict[str, Any]:
        """Return the causal shift-detector state without changing the guard."""

        return dict(self._observable_shift_info(state))

    def _pending_pipeline_vectors(
        self,
        state: np.ndarray,
    ) -> dict[str, np.ndarray]:
        zeros = {
            group: np.zeros(self.num_facilities, dtype=np.float32)
            for group in TRANSFER_GROUPS
        }
        if not self.pipeline_guard_enabled:
            return zeros

        facility_state = self._facility_state(state)
        lead_time = int(self.env_config.get("production_lead_time", 3))
        include_supplier = int(
            bool(self.env_config.get("include_supplier_state", False))
        )
        include_forecast = int(
            bool(
                self.env_config.get(
                    "include_demand_forecast_state",
                    False,
                )
            )
        )
        pending_start = (
            3 + lead_time + include_supplier + include_forecast
        )
        return {
            "reagent_transfer": facility_state[:, pending_start + 1],
            "capacity_transfer": facility_state[:, pending_start + 2],
        }

    def _observable_shift_info(
        self,
        state: np.ndarray,
    ) -> dict[str, Any]:
        if not self.observable_shift_gate_enabled:
            return {
                "enabled": False,
                "active": True,
                "directional_facilities": 0,
                "time_fraction": 0.0,
                "max_abs_rate_deviation": 0.0,
                "max_abs_trend_ratio": 0.0,
                "max_abs_smoothed_z": 0.0,
            }

        facility_state = self._facility_state(state)
        history_start = (
            3
            + int(self.env_config.get("production_lead_time", 3))
            + int(bool(self.env_config.get("include_supplier_state", False)))
            + int(
                bool(
                    self.env_config.get(
                        "include_demand_forecast_state",
                        False,
                    )
                )
            )
            + 3
            * int(
                bool(
                    self.env_config.get(
                        "include_transfer_pipeline_state",
                        False,
                    )
                )
            )
        )
        rolling_mean = facility_state[:, history_start]
        trend = facility_state[:, history_start + 1]
        prior = np.asarray(
            self.env_config.get(
                "demand_rate_estimates",
                self.env_config.get("demand_rates", 1.0),
            ),
            dtype=np.float64,
        )
        if prior.shape == ():
            prior = np.full(
                self.num_facilities,
                float(prior),
                dtype=np.float64,
            )
        if prior.shape != (self.num_facilities,):
            raise ValueError(
                "observable_shift_gate requires one demand prior per facility"
            )
        safe_prior = np.maximum(prior, 1e-6)
        rate_deviation = rolling_mean / safe_prior - 1.0
        trend_ratio = trend / safe_prior
        directional_counts = (
            int(
                np.count_nonzero(
                    rate_deviation
                    >= self.shift_rate_deviation_threshold
                )
            ),
            int(
                np.count_nonzero(
                    rate_deviation
                    <= -self.shift_rate_deviation_threshold
                )
            ),
            int(
                np.count_nonzero(
                    trend_ratio >= self.shift_trend_ratio_threshold
                )
            ),
            int(
                np.count_nonzero(
                    trend_ratio <= -self.shift_trend_ratio_threshold
                )
            ),
        )
        time_fraction = self._time_fraction(state)
        smoothed_z = self._graph_smoothed_poisson_z(
            rolling_mean,
            safe_prior,
            time_fraction=time_fraction,
        )
        if self.shift_detection_mode == "graph_poisson_z":
            directional_counts = (
                int(
                    np.count_nonzero(
                        smoothed_z >= self.shift_z_score_threshold
                    )
                ),
                int(
                    np.count_nonzero(
                        smoothed_z <= -self.shift_z_score_threshold
                    )
                ),
            )
        directional_facilities = max(directional_counts)
        active = bool(
            time_fraction + 1e-12 >= self.shift_min_time_fraction
            and directional_facilities >= self.shift_min_facilities
        )
        return {
            "enabled": True,
            "active": active,
            "directional_facilities": directional_facilities,
            "time_fraction": time_fraction,
            "max_abs_rate_deviation": float(
                np.max(np.abs(rate_deviation))
            ),
            "max_abs_trend_ratio": float(
                np.max(np.abs(trend_ratio))
            ),
            "max_abs_smoothed_z": float(
                np.max(np.abs(smoothed_z))
            ),
        }

    def _graph_smoothed_poisson_z(
        self,
        rolling_mean: np.ndarray,
        prior: np.ndarray,
        *,
        time_fraction: float,
    ) -> np.ndarray:
        horizon = max(
            int(self.env_config.get("episode_horizon", 52)),
            1,
        )
        history_window = max(
            int(self.env_config.get("demand_history_window", 1)),
            1,
        )
        observed_steps = max(
            int(round(time_fraction * max(horizon - 1, 1))) + 1,
            1,
        )
        effective_window = min(history_window, observed_steps)
        standard_error = np.sqrt(
            np.maximum(prior, 1e-6) / float(effective_window)
        )
        smoothed = (
            np.asarray(rolling_mean, dtype=np.float64) - prior
        ) / np.maximum(standard_error, 1e-6)
        for _step in range(self.shift_smoothing_steps):
            smoothed = self.shift_adjacency @ smoothed
        return smoothed

    def _shift_adjacency(self) -> np.ndarray:
        adjacency = np.eye(self.num_facilities, dtype=np.float64)
        configured_edges = self.env_config.get("information_edges")
        if configured_edges is not None:
            edges = tuple(
                (int(edge[0]), int(edge[1]))
                for edge in configured_edges
            )
        else:
            coordinates = normalize_coordinates(
                self.env_config.get("clinic_coordinates"),
                self.num_facilities,
            )
            edges = (
                geographic_knn_edges(
                    coordinates,
                    k=int(
                        self.env_config.get(
                            "geographic_neighbor_k",
                            3,
                        )
                    ),
                )
                if coordinates
                else k_nearest_ring_edges(
                    self.num_facilities,
                    k=2,
                )
            )
        for source, target in edges:
            adjacency[int(source), int(target)] = 1.0
            adjacency[int(target), int(source)] = 1.0
        row_sum = adjacency.sum(axis=1, keepdims=True)
        return adjacency / np.maximum(row_sum, 1.0)

    def _facility_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        features_per_facility = facility_state_width(self.env_config)
        base_width = self.num_facilities * features_per_facility
        if state_array.size < base_width:
            raise ValueError(
                "State is too short for residual temporal-guard features"
            )
        return state_array[:base_width].reshape(
            self.num_facilities,
            features_per_facility,
        )

    def _time_fraction(self, state: np.ndarray) -> float:
        if not bool(self.env_config.get("include_time_state", False)):
            return 1.0
        state_array = np.asarray(state, dtype=np.float32)
        return float(np.clip(state_array[-1], 0.0, 1.0))

    def _apply_l1_limit(
        self,
        values: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        if not np.isfinite(limit):
            return values
        current = float(np.abs(values).sum())
        if current <= limit + 1e-12:
            return values
        if current <= 1e-12 or limit <= 0.0:
            return np.zeros_like(values)
        return values * (float(limit) / current)

    def _group_slice(self, group: str) -> slice:
        n = self.num_facilities
        return {
            "reagent_transfer": slice(n, 2 * n),
            "capacity_transfer": slice(2 * n, 3 * n),
            "replenishment": slice(3 * n, 4 * n),
        }[group]

    def _group_int_values(
        self,
        raw: Any,
        *,
        default_groups: tuple[str, ...],
        name: str,
    ) -> dict[str, int]:
        if isinstance(raw, dict):
            values = {
                group: int(raw.get(group, 0))
                for group in SUPPORTED_GROUPS
            }
        else:
            scalar = int(raw)
            values = {
                group: scalar if group in default_groups else 0
                for group in SUPPORTED_GROUPS
            }
        if any(value < 0 for value in values.values()):
            raise ValueError(f"{name} values cannot be negative")
        return values

    def _group_float_values(
        self,
        raw: Any,
        *,
        name: str,
        default: float,
        supported: tuple[str, ...] = SUPPORTED_GROUPS,
    ) -> dict[str, float]:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be a mapping")
        self._validate_groups(tuple(str(key) for key in raw), name, supported)
        values = {
            group: float(raw.get(group, default))
            for group in supported
        }
        if any(value < 0.0 for value in values.values()):
            raise ValueError(f"{name} values cannot be negative")
        return values

    def _validate_groups(
        self,
        groups: tuple[str, ...],
        name: str,
        supported: tuple[str, ...] = SUPPORTED_GROUPS,
    ) -> None:
        unknown = set(groups) - set(supported)
        if unknown:
            raise ValueError(
                f"Unsupported {name}: " + ", ".join(sorted(unknown))
            )

    def _empty_step_info(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "raw_l1": 0.0,
            "applied_l1": 0.0,
            "suppressed_l1": 0.0,
            "groups": {},
        }

    def _step_info(
        self,
        raw: np.ndarray,
        applied: np.ndarray,
        groups: dict[str, dict[str, Any]],
        shift_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_l1 = float(np.abs(raw).sum())
        applied_l1 = float(np.abs(applied).sum())
        return {
            "enabled": self.enabled,
            "raw_l1": raw_l1,
            "applied_l1": applied_l1,
            "suppressed_l1": max(raw_l1 - applied_l1, 0.0),
            "groups": groups,
            "observable_shift": dict(shift_info or {}),
        }


def _balance_signed_net(
    values: np.ndarray,
    *,
    positive_blocked: np.ndarray | None = None,
) -> np.ndarray:
    """Balance a signed facility-net flow without increasing either side."""

    balanced = np.asarray(values, dtype=np.float64).copy()
    if positive_blocked is not None:
        blocked = np.asarray(positive_blocked, dtype=bool)
        if blocked.shape != balanced.shape:
            raise ValueError("positive_blocked must match the net-flow shape")
        balanced[blocked & (balanced > 0.0)] = 0.0

    positive = np.maximum(balanced, 0.0)
    negative = np.maximum(-balanced, 0.0)
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    if positive_total <= 1e-12 or negative_total <= 1e-12:
        return np.zeros_like(balanced)
    matched = min(positive_total, negative_total)
    return (
        positive * (matched / positive_total)
        - negative * (matched / negative_total)
    )
