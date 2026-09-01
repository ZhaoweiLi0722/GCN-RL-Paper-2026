"""Configurable PRM capacity-planning simulation environment.

This module migrates the core state-transition ideas from the legacy
two-facility DDPG environment into a small, dependency-light environment that
can be used by future GCN-DDPG, flat-state DDPG, and graph-ablation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.graph.edges import Edge, complete_undirected_edges, k_nearest_ring_edges, ring_edges
from src.graph.geography import (
    geographic_distance_matrix,
    geographic_knn_edges,
    geographic_transfer_time_matrix,
    normalize_coordinates,
)


@dataclass(frozen=True)
class CostParameters:
    """Cost coefficients for one decision epoch."""

    reagent_purchase: float = 42174.0
    reagent_holding: float = 113.5
    reagent_shortage: float = 86504.5
    bioreactor_holding: float = 14.4
    bioreactor_shortage: float = 50273.6
    specimen_transfer: float = 600.0
    bioreactor_transfer: float = 500.0
    reagent_transfer: float = 200.0


@dataclass(frozen=True)
class CapacityPlanningConfig:
    """Configuration for a distributed PRM manufacturing network."""

    num_facilities: int = 2
    production_lead_time: int = 5
    episode_horizon: int = 104
    demand_rates: Sequence[float] = (250.0 / 52.0, 250.0 / 52.0)
    demand_rate_estimates: Sequence[float] | None = None
    initial_specimens: Sequence[float] = (0.0, 0.0)
    initial_reagents: Sequence[float] = (100.0, 100.0)
    initial_idle_bioreactors: Sequence[float] = (10.0, 10.0)
    max_specimens: Sequence[float] = (100.0, 100.0)
    max_reagents: Sequence[float] = (200.0, 200.0)
    max_idle_bioreactors: Sequence[float] = (20.0, 20.0)
    max_reagent_replenishment: Sequence[float] = (100.0, 100.0)
    max_specimen_transfer: float = 100.0
    max_bioreactor_transfer: float = 20.0
    max_reagent_transfer: float = 100.0
    action_mode: str = "edge_transfer"
    include_supplier_state: bool = False
    supplier_disruption_rate: Sequence[float] | float = 0.0
    include_central_capacity_hub: bool = False
    transfer_lead_time: int = 0
    include_transfer_pipeline_state: bool = False
    include_time_state: bool = False
    demand_shock_probability: float = 0.0
    demand_shock_multiplier: float = 1.0
    demand_shock_duration: int = 0
    demand_shock_cluster_size: int = 0
    include_demand_forecast_state: bool = False
    demand_forecast_horizon: int = 1
    demand_forecast_error: float | None = None
    demand_forecast_source: str = "effective_rate"
    include_demand_history_state: bool = False
    demand_history_window: int = 4
    include_demand_sequence_state: bool = False
    demand_sequence_length: int = 12
    demand_regime_initial_multipliers: Sequence[float] | float = 1.0
    demand_regime_final_multipliers: Sequence[float] | float = 1.0
    demand_regime_change_step: int = 0
    demand_regime_transition_duration: int = 0
    # Advance referral/collection schedule used only by the intertemporal
    # shared-capacity study. The active cluster changes at declared block
    # boundaries, and the known schedule is folded into the demand forecast.
    enable_scheduled_referral_waves: bool = False
    scheduled_referral_clusters: Sequence[Sequence[int]] = ()
    scheduled_referral_start_step: int = 0
    scheduled_referral_block_length: int = 1
    scheduled_referral_peak_multiplier: float = 1.0
    scheduled_referral_repeat: bool = True
    clinic_coordinates: Sequence[Sequence[float]] | None = None
    geographic_neighbor_k: int = 3
    geographic_transfer_cost_scale: float = 0.0
    geographic_transfer_speed_mph: float = 500.0
    geographic_transfer_fixed_hours: float = 0.5
    geographic_transfer_time_cost_scale: float = 0.0
    transfer_lead_time_distance_thresholds: Sequence[float] | None = None
    regional_supplier_disruption_probability: float = 0.0
    regional_supplier_disruption_duration: int = 0
    regional_supplier_disruption_cluster_size: int = 0
    information_edges: Sequence[Edge] | None = None
    specimen_edges: Sequence[Edge] | None = None
    capacity_edges: Sequence[Edge] | None = None
    resource_edges: Sequence[Edge] | None = None
    costs: CostParameters = CostParameters()
    # Stochastic procurement lead times with order crossing
    # (spec 2026-08-29-stochastic-lead-time-regime). Flag-off behavior is
    # bit-identical to the deterministic-procurement environment.
    #
    # `reagent_purchase_lead_time` already had a CONSUMER in
    # src/baselines/heuristics.py -- which reads it via getattr and folds it
    # into the order-up-to horizon -- but no producer anywhere in src/env/, so
    # it silently defaulted to 0. This supplies the producer.
    enable_stochastic_procurement: bool = False
    reagent_purchase_lead_time: int = 0
    # Categorical distribution over arrival delays; index = epochs of delay.
    # A non-degenerate distribution permits ORDER CROSSING, which is the
    # property that breaks conventional order-up-to reasoning.
    reagent_lead_time_probabilities: Sequence[float] = ()
    include_on_order_state: bool = False
    # Analysis device, not a modelling choice: when set, lead times are still
    # DRAWN (so the random stream is untouched) and then overridden with this
    # constant. It isolates the cost of lead-time VARIABILITY from the cost of
    # the lead itself, paired exactly against the stochastic run.
    procurement_lead_override: int | None = None
    # Continuous overtime control (spec 2026-08-29-continuous-overtime-control).
    # Flag-off behavior is bit-identical to the pre-overtime environment.
    enable_overtime_control: bool = False
    max_overtime_fraction: float = 0.3
    weight_overtime_linear: float = 15_000.0
    weight_overtime_quadratic: float = 5_000.0
    enable_overtime_fatigue: bool = False
    overtime_fatigue_decay: float = 0.8
    overtime_fatigue_cost_scale: float = 1.0
    # Intertemporal shared-capacity extension. This reuses the overtime action
    # block but delays and smooths its effect; defaults preserve the original
    # same-epoch overtime contract exactly.
    enable_intertemporal_overtime_commitment: bool = False
    overtime_commitment_lead_time: int = 2
    overtime_commitment_persistence: float = 0.5
    overtime_shared_budget_fraction: float = 1.0
    weight_overtime_activation: float = 0.0
    # Decision B is dormant: the flag freezes the config surface but activation
    # is rejected until separately authorized by change control.
    enable_production_throttle: bool = False


class CapacityPlanningEnv:
    """Small NumPy environment for distributed PRM capacity planning.

    State layout per facility:
        demand, waiting specimens, reagent inventory, bioreactor pipeline,
        and optional supplier availability.

    Edge-transfer action layout:
        reagent replenishment for each facility, then specimen transfers,
        bioreactor-capacity transfers, and reagent transfers for configured
        undirected edge sets. Actions are normalized to [-1, 1].

    Facility-net action layout:
        manuscript-aligned net facility actions ``(w, e, q, p)`` for each
        facility, represented as 4N normalized continuous controls.
    """

    def __init__(self, config: CapacityPlanningConfig | None = None, seed: int | None = None):
        self.config = config or make_legacy_two_facility_config()
        self._validate_config()
        self.rng = np.random.default_rng(seed)

        n = self.config.num_facilities
        self.demand_rates = _as_vector(self.config.demand_rates, n, "demand_rates")
        self.base_demand_rates = self.demand_rates.astype(float).copy()
        self.demand_rate_estimates = (
            self.base_demand_rates.copy()
            if self.config.demand_rate_estimates is None
            else _as_vector(self.config.demand_rate_estimates, n, "demand_rate_estimates")
        )
        self.demand_regime_initial_multipliers = _as_vector(
            self.config.demand_regime_initial_multipliers,
            n,
            "demand_regime_initial_multipliers",
        )
        self.demand_regime_final_multipliers = _as_vector(
            self.config.demand_regime_final_multipliers,
            n,
            "demand_regime_final_multipliers",
        )
        self.initial_specimens = _as_vector(self.config.initial_specimens, n, "initial_specimens")
        self.initial_reagents = _as_vector(self.config.initial_reagents, n, "initial_reagents")
        self.initial_idle_bioreactors = _as_vector(
            self.config.initial_idle_bioreactors, n, "initial_idle_bioreactors"
        )
        self.max_specimens = _as_vector(self.config.max_specimens, n, "max_specimens")
        self.max_reagents = _as_vector(self.config.max_reagents, n, "max_reagents")
        self.max_idle_bioreactors = _as_vector(
            self.config.max_idle_bioreactors, n, "max_idle_bioreactors"
        )
        self.max_reagent_replenishment = _as_vector(
            self.config.max_reagent_replenishment, n, "max_reagent_replenishment"
        )
        self.supplier_disruption_rate = _as_vector(
            self.config.supplier_disruption_rate, n, "supplier_disruption_rate"
        )
        self.base_supplier_disruption_rate = self.supplier_disruption_rate.astype(float).copy()
        self.clinic_coordinates = normalize_coordinates(self.config.clinic_coordinates, n)
        self.clinic_distance_matrix = (
            np.asarray(geographic_distance_matrix(self.clinic_coordinates), dtype=float)
            if self.clinic_coordinates
            else None
        )
        self.clinic_transfer_time_hours_matrix = (
            np.asarray(
                geographic_transfer_time_matrix(
                    self.clinic_coordinates,
                    speed_mph=self.config.geographic_transfer_speed_mph,
                    fixed_handling_hours=self.config.geographic_transfer_fixed_hours,
                ),
                dtype=float,
            )
            if self.clinic_coordinates
            else None
        )
        self.transfer_delay_thresholds = tuple(
            float(value) for value in (self.config.transfer_lead_time_distance_thresholds or ())
        )
        # Mutable instance copy of the forecast error so the per-episode
        # randomization hook can vary it without touching the immutable config.
        self.demand_forecast_error = self.config.demand_forecast_error
        # Train-time domain-randomization ranges; None => fixed (eval/nominal).
        self._train_disruption_range: tuple[float, float] | None = None
        self._train_forecast_error_range: tuple[float, float] | None = None
        self._train_demand_rate_multiplier_range: tuple[float, float] | None = None

        default_edges = complete_undirected_edges(n)
        geographic_edges = (
            geographic_knn_edges(
                self.clinic_coordinates,
                k=int(self.config.geographic_neighbor_k),
            )
            if self.clinic_coordinates
            else ()
        )
        if self.config.action_mode == "facility_net":
            default_specimen_edges = geographic_edges or ring_edges(n)
            default_resource_edges = geographic_edges or ring_edges(n)
            default_capacity_edges = complete_undirected_edges(n)
        else:
            default_specimen_edges = default_edges
            default_resource_edges = default_edges
            default_capacity_edges = default_edges
        default_information_edges = geographic_edges or k_nearest_ring_edges(n, k=2)
        self.specimen_edges = _normalize_edges(self.config.specimen_edges, default_specimen_edges, n)
        self.capacity_edges = _normalize_edges(self.config.capacity_edges, default_capacity_edges, n)
        self.resource_edges = _normalize_edges(self.config.resource_edges, default_resource_edges, n)
        self.information_edges = _normalize_edges(
            self.config.information_edges, default_information_edges, n
        )
        self.specimen_transfer_priorities = self._facility_net_transfer_priorities(
            self.specimen_edges
        )
        self.capacity_transfer_priorities = self._facility_net_transfer_priorities(
            self.capacity_edges
        )
        self.reagent_transfer_priorities = self._facility_net_transfer_priorities(
            self.resource_edges
        )
        self.hub_index = n if self.config.include_central_capacity_hub else None

        self.features_per_facility = 3 + self.config.production_lead_time
        if self.config.include_supplier_state:
            self.features_per_facility += 1
        if self.config.include_demand_forecast_state:
            self.features_per_facility += 1
        if self.config.include_transfer_pipeline_state:
            self.features_per_facility += 3
        if self.config.include_demand_history_state:
            self.features_per_facility += 3
        if self.config.include_demand_sequence_state:
            self.features_per_facility += 3 * int(
                self.config.demand_sequence_length
            )
        if self.config.include_on_order_state:
            self.features_per_facility += self._procurement_pipeline_depth() + 1
        if self.config.enable_overtime_control:
            # [previous u_ot, outstanding overtime, static surge headroom]
            # plus fatigue when enabled (spec 2026-08-29).
            self.features_per_facility += 3 + int(self.config.enable_overtime_fatigue)
            if self.config.enable_intertemporal_overtime_commitment:
                # Active capacity, each pending target, and the remaining
                # shared-budget fraction (spec 2026-09-01).
                self.features_per_facility += (
                    2 + int(self.config.overtime_commitment_lead_time)
                )
        self.observation_size = (
            n * self.features_per_facility + int(self.config.include_time_state)
        )
        self.overtime_surge_headroom = (
            self.config.max_overtime_fraction * self.initial_idle_bioreactors
            if self.config.enable_overtime_control
            else np.zeros(n, dtype=float)
        )
        if self.config.action_mode == "facility_net":
            self.action_size = 4 * n + (n if self.config.enable_overtime_control else 0)
        else:
            self.action_size = (
                n + len(self.specimen_edges) + len(self.capacity_edges) + len(self.resource_edges)
            )
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.config.num_facilities
        lead_time = self.config.production_lead_time
        self.t = 0
        self.demand_shock_remaining = np.zeros(n, dtype=int)
        self.regional_supplier_disruption_remaining = np.zeros(n, dtype=int)
        self.demand_rate_multiplier = np.ones(n, dtype=float)
        self.demand_regime_multiplier = np.ones(n, dtype=float)
        self.specimen_transfer_pipeline = self._empty_transfer_pipeline()
        self.reagent_transfer_pipeline = self._empty_transfer_pipeline()
        self.capacity_transfer_pipeline = self._empty_transfer_pipeline()
        self._maybe_randomize_regime()
        self._update_demand_regime_multiplier()
        self.demand = self.rng.poisson(self._effective_demand_rates()).astype(float)
        self._advance_regional_supplier_disruptions()
        self.supplier_available = self._sample_supplier_available()
        self.demand_forecast = self._sample_demand_forecast()
        self.demand_history: list[np.ndarray] = []
        self.forecast_error_history: list[np.ndarray] = []
        self._record_demand_observation()
        self.specimens = self.initial_specimens.astype(float).copy()
        self.reagents = self.initial_reagents.astype(float).copy()
        self.bioreactors = np.zeros((n, lead_time), dtype=float)
        self.bioreactors[:, 0] = self.initial_idle_bioreactors
        self.cumulative_demand = 0.0
        self.cumulative_production = 0.0
        self.cumulative_waiting_specimens = 0.0
        self.cumulative_bioreactor_capacity = 0.0
        self.reagent_shortage_steps = 0
        self.bioreactor_shortage_steps = 0
        self.reagent_purchase_pipeline = np.zeros(
            (max(self._procurement_pipeline_depth(), 0) + 1, n), dtype=float
        )
        self.previous_overtime_fraction = np.zeros(n, dtype=float)
        self.overtime_outstanding = np.zeros(n, dtype=float)
        self.overtime_fatigue = np.zeros(n, dtype=float)
        self.overtime_active_capacity = np.zeros(n, dtype=float)
        self.overtime_commitment_pipeline = np.zeros(
            (
                int(self.config.overtime_commitment_lead_time)
                if self.config.enable_intertemporal_overtime_commitment
                else 0,
                n,
            ),
            dtype=float,
        )
        return self.observation()

    def enable_train_randomization(
        self,
        disruption_range: tuple[float, float] | None = None,
        forecast_error_range: tuple[float, float] | None = None,
        demand_rate_multiplier_range: tuple[float, float] | None = None,
    ) -> None:
        """Opt into train-time per-episode domain randomization.

        When set, each :meth:`reset` resamples the stressed parameter uniformly
        from the given ``[lo, hi]`` range *before* demand/supplier/forecast are
        drawn, so training sees a distribution of regimes. Eval envs leave this
        unset and keep the fixed config values, enabling clean train-on-range /
        test-OOD splits. ``demand_rate_multiplier_range`` randomizes the true
        Poisson demand rates while leaving ``demand_rate_estimates`` fixed, which
        mirrors the old DRL-CaP case where heuristics plan from a priori demand
        estimates but the simulator draws from a shifted ground-truth demand
        distribution. Passing ``None`` (the default) disables that lever.
        """

        def _check(name: str, rng: tuple[float, float] | None) -> tuple[float, float] | None:
            if rng is None:
                return None
            lo, hi = float(rng[0]), float(rng[1])
            if lo < 0.0 or hi < lo:
                raise ValueError(f"{name} must satisfy 0 <= lo <= hi, got {rng}")
            return (lo, hi)

        self._train_disruption_range = _check("disruption_range", disruption_range)
        self._train_forecast_error_range = _check("forecast_error_range", forecast_error_range)
        self._train_demand_rate_multiplier_range = _check(
            "demand_rate_multiplier_range",
            demand_rate_multiplier_range,
        )

    def _maybe_randomize_regime(self) -> None:
        """Resample stressed parameters from their train-time ranges, if enabled."""
        n = self.config.num_facilities
        self.demand_rates = self.base_demand_rates.astype(float).copy()
        self.supplier_disruption_rate = self.base_supplier_disruption_rate.astype(float).copy()
        self.demand_forecast_error = self.config.demand_forecast_error
        if self._train_disruption_range is not None:
            lo, hi = self._train_disruption_range
            rate = float(self.rng.uniform(lo, hi))
            self.supplier_disruption_rate = np.full(n, rate, dtype=float)
        if self._train_forecast_error_range is not None:
            lo, hi = self._train_forecast_error_range
            self.demand_forecast_error = float(self.rng.uniform(lo, hi))
        if self._train_demand_rate_multiplier_range is not None:
            lo, hi = self._train_demand_rate_multiplier_range
            multiplier = float(self.rng.uniform(lo, hi))
            self.demand_rates = self.base_demand_rates * multiplier

    def observation(self) -> np.ndarray:
        """Return a flat observation suitable for MLP baselines."""

        rows = []
        pending_specimens, pending_reagents, pending_capacity = self._pending_transfer_arrivals()
        rolling_mean, demand_trend, rolling_forecast_error = (
            self._demand_history_features()
        )
        demand_sequence, error_sequence, sequence_mask = (
            self._demand_sequence_features()
        )
        if self.config.enable_overtime_control:
            overtime_features = self._overtime_features()
        if self.config.include_on_order_state:
            on_order_features = self._on_order_features()
        for i in range(self.config.num_facilities):
            row_parts = [
                np.array([self.demand[i], self.specimens[i], self.reagents[i]], dtype=float),
                self.bioreactors[i],
            ]
            if self.config.include_supplier_state:
                row_parts.append(np.array([self.supplier_available[i]], dtype=float))
            if self.config.include_demand_forecast_state:
                row_parts.append(np.array([self.demand_forecast[i]], dtype=float))
            if self.config.include_transfer_pipeline_state:
                row_parts.append(
                    np.array(
                        [pending_specimens[i], pending_reagents[i], pending_capacity[i]],
                        dtype=float,
                    )
                )
            if self.config.include_demand_history_state:
                row_parts.append(
                    np.array(
                        [
                            rolling_mean[i],
                            demand_trend[i],
                            rolling_forecast_error[i],
                        ],
                        dtype=float,
                    )
                )
            if self.config.include_demand_sequence_state:
                row_parts.append(
                    np.concatenate(
                        (
                            demand_sequence[i],
                            error_sequence[i],
                            sequence_mask[i],
                        )
                    )
                )
            if self.config.include_on_order_state:
                row_parts.append(on_order_features[i])
            if self.config.enable_overtime_control:
                row_parts.append(overtime_features[i])
            rows.append(np.concatenate(tuple(row_parts)))
        observation = np.concatenate(rows).astype(np.float32)
        if self.config.include_time_state:
            observation = np.concatenate(
                (observation, np.asarray([self._normalized_time()], dtype=np.float32))
            )
        return observation

    def _pending_transfer_arrivals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.num_facilities
        if self.config.transfer_lead_time <= 0:
            zeros = np.zeros(n, dtype=float)
            return zeros, zeros, zeros
        return (
            self.specimen_transfer_pipeline.sum(axis=0),
            self.reagent_transfer_pipeline.sum(axis=0),
            self.capacity_transfer_pipeline.sum(axis=0),
        )

    def _receive_transfer_arrivals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        specimen_arrivals = self._pop_transfer_arrivals(self.specimen_transfer_pipeline)
        reagent_arrivals = self._pop_transfer_arrivals(self.reagent_transfer_pipeline)
        capacity_arrivals = self._pop_transfer_arrivals(self.capacity_transfer_pipeline)
        if np.any(specimen_arrivals):
            self.specimens = np.clip(self.specimens + specimen_arrivals, 0.0, self.max_specimens)
        if np.any(reagent_arrivals):
            self.reagents = np.clip(self.reagents + reagent_arrivals, 0.0, self.max_reagents)
        if np.any(capacity_arrivals):
            self.bioreactors[:, 0] = np.clip(
                self.bioreactors[:, 0] + capacity_arrivals,
                0.0,
                self.max_idle_bioreactors,
            )
        return specimen_arrivals, reagent_arrivals, capacity_arrivals

    def graph_observation(self) -> dict[str, np.ndarray]:
        """Return node features and edge sets for future GCN policies."""

        facility_columns = [
            self.demand,
            self.specimens,
            self.reagents,
            self.bioreactors[:, 0],
            self.bioreactors.sum(axis=1),
        ]
        if self.config.include_supplier_state:
            facility_columns.append(self.supplier_available)
        if self.config.include_demand_forecast_state:
            facility_columns.append(self.demand_forecast)
        if self.config.include_transfer_pipeline_state:
            facility_columns.extend(self._pending_transfer_arrivals())
        if self.config.include_demand_history_state:
            facility_columns.extend(self._demand_history_features())
        if self.config.include_demand_sequence_state:
            demand_sequence, error_sequence, sequence_mask = (
                self._demand_sequence_features()
            )
            facility_columns.extend(
                demand_sequence[:, index]
                for index in range(demand_sequence.shape[1])
            )
            facility_columns.extend(
                error_sequence[:, index]
                for index in range(error_sequence.shape[1])
            )
            facility_columns.extend(
                sequence_mask[:, index]
                for index in range(sequence_mask.shape[1])
            )
        if self.config.include_on_order_state:
            on_order = self._on_order_features()
            facility_columns.extend(
                on_order[:, index] for index in range(on_order.shape[1])
            )
        if self.config.enable_overtime_control:
            overtime_features = self._overtime_features()
            facility_columns.extend(
                overtime_features[:, index]
                for index in range(overtime_features.shape[1])
            )
        time_feature_index = None
        if self.config.include_time_state:
            time_feature_index = len(facility_columns)
            facility_columns.append(
                np.full(
                    self.config.num_facilities,
                    self._normalized_time(),
                    dtype=float,
                )
            )
        if self.config.include_central_capacity_hub:
            facility_columns.append(np.zeros(self.config.num_facilities, dtype=float))
        node_features = np.column_stack(tuple(facility_columns)).astype(np.float32)
        capacity_graph_edges = self._capacity_graph_edges()
        if self.config.include_central_capacity_hub:
            hub_features = np.zeros((1, node_features.shape[1]), dtype=np.float32)
            hub_features[0, 3] = float(self.bioreactors[:, 0].sum())
            hub_features[0, 4] = float(self.bioreactors.sum())
            if time_feature_index is not None:
                hub_features[0, time_feature_index] = self._normalized_time()
            hub_features[0, -1] = 1.0
            node_features = np.vstack((node_features, hub_features))
        graph = {
            "node_features": node_features,
            "information_edges": _edge_array(self.information_edges),
            "specimen_edges": _edge_array(self.specimen_edges),
            "capacity_edges": _edge_array(capacity_graph_edges),
            "resource_edges": _edge_array(self.resource_edges),
        }
        if self.clinic_coordinates:
            graph["clinic_coordinates"] = np.asarray(self.clinic_coordinates, dtype=np.float32)
        if self.clinic_distance_matrix is not None:
            graph["clinic_distance_matrix"] = self.clinic_distance_matrix.astype(np.float32)
        if self.clinic_transfer_time_hours_matrix is not None:
            graph["clinic_transfer_time_hours_matrix"] = (
                self.clinic_transfer_time_hours_matrix.astype(np.float32)
            )
        return graph

    def noop_action(self) -> np.ndarray:
        """Return an action with no replenishment and no transfers."""

        action = np.zeros(self.action_size, dtype=np.float32)
        if self.config.action_mode == "facility_net":
            n = self.config.num_facilities
            action[3 * n : 4 * n] = -1.0
            if self.config.enable_overtime_control:
                action[4 * n : 5 * n] = -1.0  # raw -1 maps to u_ot = 0
        else:
            action[: self.config.num_facilities] = -1.0
        return action

    def sample_random_action(self) -> np.ndarray:
        """Sample a random normalized action."""

        return self.rng.uniform(-1.0, 1.0, size=self.action_size).astype(np.float32)

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, dict[str, np.ndarray | float]]:
        """Advance one decision epoch.

        The transition follows the legacy model's ordering at a high level:
        production consumes waiting specimens, idle bioreactors, and reagents;
        current demand arrives; replenishment and sharing decisions affect the
        next state. The implementation clips transfers against post-production
        availability so inventories and idle capacity cannot become negative.
        """

        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"Expected action shape {(self.action_size,)}, got {action_array.shape}")
        normalized = np.clip(action_array, -1.0, 1.0)
        if self.config.action_mode == "facility_net":
            return self._step_facility_net(normalized)
        return self._step_edge_transfer(normalized)

    def _step_edge_transfer(
        self, normalized: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, np.ndarray | float]]:
        """Advance one epoch using legacy edge-level transfer actions."""

        n = self.config.num_facilities
        costs = self.config.costs
        specimen_arrivals, reagent_arrivals, capacity_arrivals = self._receive_transfer_arrivals()

        supplier_available = self.supplier_available.copy()
        current_demand = self.demand.copy()
        current_demand_forecast = self.demand_forecast.copy()
        current_scheduled_referral_multiplier = (
            self._scheduled_referral_multiplier_at(self.t)
        )
        replenishment = (
            ((normalized[:n] + 1.0) / 2.0)
            * self.max_reagent_replenishment
            * supplier_available
        )
        offset = n
        specimen_requests = normalized[offset : offset + len(self.specimen_edges)]
        specimen_requests = specimen_requests * self.config.max_specimen_transfer
        offset += len(self.specimen_edges)
        capacity_requests = normalized[offset : offset + len(self.capacity_edges)]
        capacity_requests = capacity_requests * self.config.max_bioreactor_transfer
        offset += len(self.capacity_edges)
        reagent_requests = normalized[offset : offset + len(self.resource_edges)]
        reagent_requests = reagent_requests * self.config.max_reagent_transfer

        idle_bioreactors = self.bioreactors[:, 0]
        under_reagents = np.maximum(self.specimens - self.reagents, 0.0)
        idle_reagents = np.maximum(self.reagents - self.specimens, 0.0)
        under_bioreactors = np.maximum(self.specimens - idle_bioreactors, 0.0)
        idle_bioreactor_counts = np.maximum(idle_bioreactors - self.specimens, 0.0)

        production = np.minimum.reduce((self.specimens, idle_bioreactors, self.reagents))

        next_specimens = self.specimens - production + current_demand
        next_reagents = self.reagents - production + replenishment
        next_bioreactors = np.zeros_like(self.bioreactors)
        next_bioreactors[:, 0] = self.bioreactors[:, 0] - production + self.bioreactors[:, 1]
        if self.config.production_lead_time > 2:
            next_bioreactors[:, 1:-1] = self.bioreactors[:, 2:]
        next_bioreactors[:, -1] = production

        if self.config.transfer_lead_time > 0:
            specimen_transfers, specimen_future_arrivals = _apply_transfers_delayed(
                next_specimens, self.specimen_edges, specimen_requests
            )
            capacity_transfers, capacity_future_arrivals = _apply_transfers_delayed(
                next_bioreactors[:, 0], self.capacity_edges, capacity_requests
            )
            reagent_transfers, reagent_future_arrivals = _apply_transfers_delayed(
                next_reagents, self.resource_edges, reagent_requests
            )
            if self._uses_geographic_transfer_delays():
                self._schedule_edge_transfer_arrivals(
                    self.specimen_transfer_pipeline, self.specimen_edges, specimen_transfers
                )
                self._schedule_edge_transfer_arrivals(
                    self.capacity_transfer_pipeline, self.capacity_edges, capacity_transfers
                )
                self._schedule_edge_transfer_arrivals(
                    self.reagent_transfer_pipeline, self.resource_edges, reagent_transfers
                )
            else:
                self._schedule_transfer_arrivals(
                    self.specimen_transfer_pipeline, specimen_future_arrivals
                )
                self._schedule_transfer_arrivals(
                    self.capacity_transfer_pipeline, capacity_future_arrivals
                )
                self._schedule_transfer_arrivals(
                    self.reagent_transfer_pipeline, reagent_future_arrivals
                )
        else:
            specimen_transfers = _apply_transfers(
                next_specimens, self.specimen_edges, specimen_requests
            )
            capacity_transfers = _apply_transfers(
                next_bioreactors[:, 0], self.capacity_edges, capacity_requests
            )
            reagent_transfers = _apply_transfers(
                next_reagents, self.resource_edges, reagent_requests
            )

        self.specimens = np.clip(next_specimens, 0.0, self.max_specimens)
        self.reagents = np.clip(next_reagents, 0.0, self.max_reagents)
        next_bioreactors[:, 0] = np.clip(next_bioreactors[:, 0], 0.0, self.max_idle_bioreactors)
        self.bioreactors = np.maximum(next_bioreactors, 0.0)
        self._update_running_metrics(current_demand, production, self.specimens, self.bioreactors)
        done = self._advance_clock()

        cost_components = self._operating_cost_components(
            replenishment=replenishment,
            idle_reagents=idle_reagents,
            under_reagents=under_reagents,
            idle_bioreactors=idle_bioreactor_counts,
            under_bioreactors=under_bioreactors,
            specimen_transfer_cost=self._edge_transfer_cost(
                costs.specimen_transfer,
                self.specimen_edges,
                specimen_transfers,
            ),
            capacity_transfer_cost=self._edge_transfer_cost(
                costs.bioreactor_transfer,
                self.capacity_edges,
                capacity_transfers,
            ),
            reagent_transfer_cost=self._edge_transfer_cost(
                costs.reagent_transfer,
                self.resource_edges,
                reagent_transfers,
            ),
        )
        cost = float(sum(cost_components.values()))

        info: dict[str, np.ndarray | float] = {
            "cost": cost,
            "base_cost": cost,
            **cost_components,
            "transshipment_cost": (
                cost_components["specimen_transfer_cost"]
                + cost_components["capacity_transfer_cost"]
                + cost_components["reagent_transfer_cost"]
            ),
            "production": production.copy(),
            "demand": current_demand.copy(),
            "demand_forecast": current_demand_forecast.copy(),
            "supplier_available": supplier_available.copy(),
            "demand_rate_multiplier": self.demand_rate_multiplier.copy(),
            "demand_regime_multiplier": self.demand_regime_multiplier.copy(),
            "regional_supplier_disruption_remaining": self.regional_supplier_disruption_remaining.copy(),
            "replenishment": replenishment.copy(),
            "specimen_transfer_arrivals": specimen_arrivals.copy(),
            "reagent_transfer_arrivals": reagent_arrivals.copy(),
            "capacity_transfer_arrivals": capacity_arrivals.copy(),
            "specimen_transfers": specimen_transfers.copy(),
            "capacity_transfers": capacity_transfers.copy(),
            "reagent_transfers": reagent_transfers.copy(),
            "under_reagents": under_reagents.copy(),
            "under_bioreactors": under_bioreactors.copy(),
        }
        if self.config.enable_scheduled_referral_waves:
            info["scheduled_referral_multiplier"] = (
                current_scheduled_referral_multiplier.copy()
            )
        info.update(self._performance_info())
        return self.observation(), -cost, done, info

    def _step_facility_net(
        self, normalized: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, np.ndarray | float]]:
        """Advance one epoch using manuscript-aligned facility net actions."""

        n = self.config.num_facilities
        costs = self.config.costs
        specimen_arrivals, reagent_arrivals, capacity_arrivals = self._receive_transfer_arrivals()
        supplier_available = self.supplier_available.copy()
        current_demand = self.demand.copy()
        current_demand_forecast = self.demand_forecast.copy()
        current_scheduled_referral_multiplier = (
            self._scheduled_referral_multiplier_at(self.t)
        )

        specimen_requests = normalized[:n] * self.config.max_specimen_transfer
        reagent_transfer_requests = normalized[n : 2 * n] * self.config.max_reagent_transfer
        capacity_requests = normalized[2 * n : 3 * n] * self.config.max_bioreactor_transfer
        replenishment = (
            ((normalized[3 * n : 4 * n] + 1.0) / 2.0)
            * self.max_reagent_replenishment
            * supplier_available
        )
        procurement_arrivals = None
        if self._procurement_pipeline_depth() > 0:
            leads = self._draw_procurement_leads()
            self._place_procurement(replenishment, leads)
            procurement_arrivals = self._receive_procurement()
        if self.config.enable_overtime_control:
            (
                overtime_fraction,
                overtime_surge,
                overtime_requested_surge,
                overtime_activation_change,
            ) = self._resolve_overtime_for_step(normalized)
            idle_now = self.bioreactors[:, 0]
            production = np.minimum.reduce(
                (self.specimens, idle_now + overtime_surge, self.reagents)
            )
            base_production = np.minimum(production, idle_now)
            overtime_production = production - base_production
            returning = self.bioreactors[:, 1]
            overtime_repaid = np.minimum(self.overtime_outstanding, returning)
            next_specimens = self.specimens - production + current_demand
            next_reagents = self.reagents - production + (
                replenishment if procurement_arrivals is None else procurement_arrivals
            )
            next_bioreactors = np.zeros_like(self.bioreactors)
            next_bioreactors[:, 0] = (
                idle_now - base_production + returning - overtime_repaid
            )
        else:
            production = np.minimum.reduce((self.specimens, self.bioreactors[:, 0], self.reagents))
            next_specimens = self.specimens - production + current_demand
            next_reagents = self.reagents - production + (
                replenishment if procurement_arrivals is None else procurement_arrivals
            )
            next_bioreactors = np.zeros_like(self.bioreactors)
            next_bioreactors[:, 0] = self.bioreactors[:, 0] - production + self.bioreactors[:, 1]
        if self.config.production_lead_time > 2:
            next_bioreactors[:, 1:-1] = self.bioreactors[:, 2:]
        next_bioreactors[:, -1] = production

        if self.config.transfer_lead_time > 0:
            specimen_net, specimen_flows, specimen_future_arrivals = _apply_net_transfers_delayed(
                next_specimens,
                self.specimen_edges,
                specimen_requests,
                edge_priorities=self.specimen_transfer_priorities,
            )
            capacity_net, capacity_flows, capacity_future_arrivals = _apply_net_transfers_delayed(
                next_bioreactors[:, 0],
                self.capacity_edges,
                capacity_requests,
                edge_priorities=self.capacity_transfer_priorities,
            )
            reagent_net, reagent_flows, reagent_future_arrivals = _apply_net_transfers_delayed(
                next_reagents,
                self.resource_edges,
                reagent_transfer_requests,
                edge_priorities=self.reagent_transfer_priorities,
            )
            if self._uses_geographic_transfer_delays():
                self._schedule_edge_transfer_arrivals(
                    self.specimen_transfer_pipeline, self.specimen_edges, specimen_flows
                )
                self._schedule_edge_transfer_arrivals(
                    self.capacity_transfer_pipeline, self.capacity_edges, capacity_flows
                )
                self._schedule_edge_transfer_arrivals(
                    self.reagent_transfer_pipeline, self.resource_edges, reagent_flows
                )
            else:
                self._schedule_transfer_arrivals(
                    self.specimen_transfer_pipeline, specimen_future_arrivals
                )
                self._schedule_transfer_arrivals(
                    self.capacity_transfer_pipeline, capacity_future_arrivals
                )
                self._schedule_transfer_arrivals(
                    self.reagent_transfer_pipeline, reagent_future_arrivals
                )
        else:
            specimen_net, specimen_flows = _apply_net_transfers(
                next_specimens,
                self.specimen_edges,
                specimen_requests,
                edge_priorities=self.specimen_transfer_priorities,
            )
            capacity_net, capacity_flows = _apply_net_transfers(
                next_bioreactors[:, 0],
                self.capacity_edges,
                capacity_requests,
                edge_priorities=self.capacity_transfer_priorities,
            )
            reagent_net, reagent_flows = _apply_net_transfers(
                next_reagents,
                self.resource_edges,
                reagent_transfer_requests,
                edge_priorities=self.reagent_transfer_priorities,
            )

        self.specimens = np.clip(next_specimens, 0.0, self.max_specimens)
        self.reagents = np.clip(next_reagents, 0.0, self.max_reagents)
        next_bioreactors[:, 0] = np.clip(next_bioreactors[:, 0], 0.0, self.max_idle_bioreactors)
        self.bioreactors = np.maximum(next_bioreactors, 0.0)

        under_reagents = np.maximum(self.specimens - self.reagents, 0.0)
        idle_reagents = np.maximum(self.reagents - self.specimens, 0.0)
        under_bioreactors = np.maximum(self.specimens - self.bioreactors[:, 0], 0.0)
        idle_bioreactor_counts = np.maximum(self.bioreactors[:, 0] - self.specimens, 0.0)

        cost_components = self._operating_cost_components(
            replenishment=replenishment,
            idle_reagents=idle_reagents,
            under_reagents=under_reagents,
            idle_bioreactors=idle_bioreactor_counts,
            under_bioreactors=under_bioreactors,
            specimen_transfer_cost=self._facility_net_transfer_cost(
                costs.specimen_transfer,
                self.specimen_edges,
                specimen_flows,
                specimen_net,
            ),
            capacity_transfer_cost=self._facility_net_transfer_cost(
                costs.bioreactor_transfer,
                self.capacity_edges,
                capacity_flows,
                capacity_net,
            ),
            reagent_transfer_cost=self._facility_net_transfer_cost(
                costs.reagent_transfer,
                self.resource_edges,
                reagent_flows,
                reagent_net,
            ),
            overtime_surge=(
                overtime_surge if self.config.enable_overtime_control else None
            ),
            overtime_activation_change=(
                overtime_activation_change
                if self.config.enable_intertemporal_overtime_commitment
                else None
            ),
        )
        cost = float(sum(cost_components.values()))
        if self.config.enable_overtime_control:
            self.overtime_outstanding = (
                self.overtime_outstanding - overtime_repaid + overtime_production
            )
            if self.config.enable_overtime_fatigue:
                self.overtime_fatigue = (
                    self.config.overtime_fatigue_decay * self.overtime_fatigue
                    + self._overtime_utilization_fraction(
                        overtime_fraction, overtime_surge
                    )
                )
            self.previous_overtime_fraction = overtime_fraction.copy()

        self._update_running_metrics(current_demand, production, self.specimens, self.bioreactors)
        if np.any(under_reagents > 0):
            self.reagent_shortage_steps += 1
        if np.any(under_bioreactors > 0):
            self.bioreactor_shortage_steps += 1
        done = self._advance_clock()

        info: dict[str, np.ndarray | float] = {
            "cost": cost,
            "base_cost": cost,
            **cost_components,
            "transshipment_cost": (
                cost_components["specimen_transfer_cost"]
                + cost_components["capacity_transfer_cost"]
                + cost_components["reagent_transfer_cost"]
            ),
            "production": production.copy(),
            "demand": current_demand.copy(),
            "demand_forecast": current_demand_forecast.copy(),
            "supplier_available": supplier_available.copy(),
            "demand_rate_multiplier": self.demand_rate_multiplier.copy(),
            "demand_regime_multiplier": self.demand_regime_multiplier.copy(),
            "regional_supplier_disruption_remaining": self.regional_supplier_disruption_remaining.copy(),
            "replenishment": replenishment.copy(),
            "specimen_transfer_arrivals": specimen_arrivals.copy(),
            "reagent_transfer_arrivals": reagent_arrivals.copy(),
            "capacity_transfer_arrivals": capacity_arrivals.copy(),
            "specimen_transfers": specimen_net.copy(),
            "capacity_transfers": capacity_net.copy(),
            "reagent_transfers": reagent_net.copy(),
            "specimen_edge_flows": specimen_flows.copy(),
            "capacity_edge_flows": capacity_flows.copy(),
            "reagent_edge_flows": reagent_flows.copy(),
            "under_reagents": under_reagents.copy(),
            "under_bioreactors": under_bioreactors.copy(),
        }
        if procurement_arrivals is not None:
            info["procurement_arrivals"] = procurement_arrivals.copy()
            info["reagents_on_order"] = self.reagent_purchase_pipeline.sum(axis=0)
        if self.config.enable_scheduled_referral_waves:
            info["scheduled_referral_multiplier"] = (
                current_scheduled_referral_multiplier.copy()
            )
        if self.config.enable_overtime_control:
            info["overtime_fraction"] = overtime_fraction.copy()
            info["overtime_surge"] = overtime_surge.copy()
            info["overtime_production"] = overtime_production.copy()
            info["overtime_outstanding"] = self.overtime_outstanding.copy()
            if self.config.enable_intertemporal_overtime_commitment:
                info["overtime_requested_fraction"] = overtime_fraction.copy()
                info["overtime_requested_surge"] = overtime_requested_surge.copy()
                info["overtime_active_capacity"] = overtime_surge.copy()
                info["overtime_commitment_change"] = (
                    overtime_activation_change.copy()
                )
        info.update(self._performance_info())
        return self.observation(), -cost, done, info

    def _procurement_pipeline_depth(self) -> int:
        """Number of future epochs an order can land in (0 = arrives next epoch)."""

        if self.config.enable_stochastic_procurement:
            return max(
                len(self.config.reagent_lead_time_probabilities) - 1,
                int(self.config.reagent_purchase_lead_time),
            )
        return int(self.config.reagent_purchase_lead_time)

    def _draw_procurement_leads(self) -> np.ndarray:
        """One lead-time draw per facility per epoch.

        Deliberately per (facility, epoch) rather than per order or per unit:
        the number of random draws must not depend on the ORDER QUANTITY,
        which is an action. Drawing per unit would make the random stream
        action-dependent and destroy the exact CRN pairing that every paired
        screen in this project relies on. Order crossing still arises, because
        orders placed in different epochs draw independent delays.
        """

        n = self.config.num_facilities
        if not self.config.enable_stochastic_procurement:
            return np.full(n, int(self.config.reagent_purchase_lead_time), dtype=int)
        probabilities = np.asarray(
            self.config.reagent_lead_time_probabilities, dtype=float
        )
        drawn = self.rng.choice(len(probabilities), size=n, p=probabilities)
        if self.config.procurement_lead_override is not None:
            # Draw first, then discard: the stream must match the stochastic run.
            return np.full(n, int(self.config.procurement_lead_override), dtype=int)
        return drawn

    def _receive_procurement(self) -> np.ndarray:
        """Pop this epoch's arrivals and age the procurement pipeline."""

        arrivals = self.reagent_purchase_pipeline[0].copy()
        self.reagent_purchase_pipeline = np.roll(
            self.reagent_purchase_pipeline, -1, axis=0
        )
        self.reagent_purchase_pipeline[-1] = 0.0
        return arrivals

    def _place_procurement(self, replenishment: np.ndarray, leads: np.ndarray) -> None:
        depth = self.reagent_purchase_pipeline.shape[0]
        for facility in range(self.config.num_facilities):
            slot = min(int(leads[facility]), depth - 1)
            self.reagent_purchase_pipeline[slot, facility] += float(
                replenishment[facility]
            )

    def _on_order_features(self) -> np.ndarray:
        """Outstanding on-order quantity by remaining age, per facility."""

        return self.reagent_purchase_pipeline.T.copy()

    def _decode_overtime(self, normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map the overtime action block to (u_ot, committed surge).

        Raw ``[-1, 1]`` values map affinely to ``u_ot ∈ [0, 1]``; the committed
        surge is ``u_ot * max_overtime_fraction * initial_idle_bioreactors``
        (spec 2026-08-29-continuous-overtime-control).
        """

        n = self.config.num_facilities
        overtime_fraction = (normalized[4 * n : 5 * n] + 1.0) / 2.0
        overtime_surge = overtime_fraction * self.overtime_surge_headroom
        return overtime_fraction, overtime_surge

    def _shared_overtime_budget(self) -> float:
        return float(
            self.config.overtime_shared_budget_fraction
            * self.overtime_surge_headroom.sum()
        )

    def _project_overtime_commitment(
        self,
        overtime_fraction: np.ndarray,
        requested_surge: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a request continuously onto the shared staffing budget."""

        requested = np.maximum(np.asarray(requested_surge, dtype=float), 0.0)
        budget = self._shared_overtime_budget()
        total = float(requested.sum())
        if total > budget and total > 0.0:
            requested = requested * (budget / total)
        projected_fraction = np.divide(
            requested,
            self.overtime_surge_headroom,
            out=np.zeros_like(requested),
            where=self.overtime_surge_headroom > 0.0,
        )
        return np.clip(projected_fraction, 0.0, 1.0), requested

    def _resolve_overtime_for_step(
        self, normalized: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return request fraction, active surge, request surge, and adjustment.

        The original same-epoch path remains a pure decode. In intertemporal
        mode the current request enters a lead-time pipeline while a matured
        target updates the persistent capacity available in this epoch.
        """

        overtime_fraction, requested_surge = self._decode_overtime(normalized)
        if not self.config.enable_intertemporal_overtime_commitment:
            return (
                overtime_fraction,
                requested_surge,
                requested_surge,
                np.zeros_like(requested_surge),
            )

        overtime_fraction, requested_surge = self._project_overtime_commitment(
            overtime_fraction, requested_surge
        )
        previous_request = (
            self.previous_overtime_fraction * self.overtime_surge_headroom
        )
        commitment_change = np.abs(requested_surge - previous_request)

        matured_request = self.overtime_commitment_pipeline[0].copy()
        if self.overtime_commitment_pipeline.shape[0] > 1:
            self.overtime_commitment_pipeline[:-1] = (
                self.overtime_commitment_pipeline[1:]
            )
        self.overtime_commitment_pipeline[-1] = requested_surge

        persistence = float(self.config.overtime_commitment_persistence)
        self.overtime_active_capacity = (
            persistence * self.overtime_active_capacity
            + (1.0 - persistence) * matured_request
        )
        return (
            overtime_fraction,
            self.overtime_active_capacity.copy(),
            requested_surge,
            commitment_change,
        )

    def _overtime_utilization_fraction(
        self, overtime_fraction: np.ndarray, active_surge: np.ndarray
    ) -> np.ndarray:
        if not self.config.enable_intertemporal_overtime_commitment:
            return overtime_fraction
        return np.divide(
            active_surge,
            self.overtime_surge_headroom,
            out=np.zeros_like(active_surge),
            where=self.overtime_surge_headroom > 0.0,
        )

    def _overtime_features(self) -> np.ndarray:
        """Per-facility overtime feature block appended to observations."""

        columns = [
            self.previous_overtime_fraction,
            self.overtime_outstanding,
            self.overtime_surge_headroom,
        ]
        if self.config.enable_intertemporal_overtime_commitment:
            active_fraction = np.divide(
                self.overtime_active_capacity,
                self.overtime_surge_headroom,
                out=np.zeros_like(self.overtime_active_capacity),
                where=self.overtime_surge_headroom > 0.0,
            )
            columns.append(active_fraction)
            for pending in self.overtime_commitment_pipeline:
                columns.append(
                    np.divide(
                        pending,
                        self.overtime_surge_headroom,
                        out=np.zeros_like(pending),
                        where=self.overtime_surge_headroom > 0.0,
                    )
                )
            budget = self._shared_overtime_budget()
            previous_request = (
                self.previous_overtime_fraction * self.overtime_surge_headroom
            )
            remaining = (
                max(budget - float(previous_request.sum()), 0.0) / budget
                if budget > 0.0
                else 0.0
            )
            columns.append(
                np.full(self.config.num_facilities, remaining, dtype=float)
            )
        if self.config.enable_overtime_fatigue:
            columns.append(self.overtime_fatigue)
        return np.column_stack(tuple(columns))

    def _operating_cost_components(
        self,
        *,
        replenishment: np.ndarray,
        idle_reagents: np.ndarray,
        under_reagents: np.ndarray,
        idle_bioreactors: np.ndarray,
        under_bioreactors: np.ndarray,
        specimen_transfer_cost: float,
        capacity_transfer_cost: float,
        reagent_transfer_cost: float,
        overtime_surge: np.ndarray | None = None,
        overtime_activation_change: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return an additive decomposition of one epoch's operating cost."""

        costs = self.config.costs
        components = {
            "reagent_purchase_cost": costs.reagent_purchase * float(replenishment.sum()),
            "reagent_holding_cost": costs.reagent_holding * float(idle_reagents.sum()),
            "reagent_shortage_cost": costs.reagent_shortage * float(under_reagents.sum()),
            "bioreactor_holding_cost": costs.bioreactor_holding
            * float(idle_bioreactors.sum()),
            "bioreactor_shortage_cost": costs.bioreactor_shortage
            * float(under_bioreactors.sum()),
            "specimen_transfer_cost": float(specimen_transfer_cost),
            "capacity_transfer_cost": float(capacity_transfer_cost),
            "reagent_transfer_cost": float(reagent_transfer_cost),
        }
        if overtime_surge is not None:
            # Charged on the committed surge so the cost stays smooth and
            # strictly monotone in u_ot even when the integer production bound
            # does not move (spec 2026-08-29-continuous-overtime-control).
            linear = self.config.weight_overtime_linear
            if self.config.enable_overtime_fatigue:
                linear = linear * (
                    1.0
                    + self.config.overtime_fatigue_cost_scale * self.overtime_fatigue
                )
            components["overtime_cost"] = float(
                np.sum(
                    linear * overtime_surge
                    + self.config.weight_overtime_quadratic * overtime_surge**2
                )
            )
        if overtime_activation_change is not None:
            components["overtime_activation_cost"] = float(
                self.config.weight_overtime_activation
                * np.asarray(overtime_activation_change, dtype=float).sum()
            )
        return components

    def _validate_config(self) -> None:
        if self.config.num_facilities < 1:
            raise ValueError("num_facilities must be positive")
        if self.config.production_lead_time < 2:
            raise ValueError("production_lead_time must be at least 2")
        if self.config.episode_horizon < 1:
            raise ValueError("episode_horizon must be positive")
        if self.config.transfer_lead_time < 0:
            raise ValueError("transfer_lead_time must be nonnegative")
        if not 0.0 <= self.config.demand_shock_probability <= 1.0:
            raise ValueError("demand_shock_probability must be between 0 and 1")
        if self.config.demand_shock_multiplier < 1.0:
            raise ValueError("demand_shock_multiplier must be at least 1")
        if self.config.demand_shock_duration < 0:
            raise ValueError("demand_shock_duration must be nonnegative")
        if self.config.demand_shock_cluster_size < 0:
            raise ValueError("demand_shock_cluster_size must be nonnegative")
        if self.config.demand_forecast_horizon < 1:
            raise ValueError("demand_forecast_horizon must be positive")
        if self.config.demand_forecast_source not in ("effective_rate", "prior_estimate"):
            raise ValueError(
                "demand_forecast_source must be 'effective_rate' or 'prior_estimate'"
            )
        if self.config.demand_history_window < 1:
            raise ValueError("demand_history_window must be positive")
        if self.config.demand_sequence_length < 1:
            raise ValueError("demand_sequence_length must be positive")
        if self.config.demand_forecast_error is not None and self.config.demand_forecast_error < 0.0:
            raise ValueError("demand_forecast_error must be nonnegative or null")
        if not 0 <= self.config.demand_regime_change_step < self.config.episode_horizon:
            raise ValueError(
                "demand_regime_change_step must be within the episode horizon"
            )
        if self.config.demand_regime_transition_duration < 0:
            raise ValueError("demand_regime_transition_duration must be nonnegative")
        for name, values in (
            (
                "demand_regime_initial_multipliers",
                self.config.demand_regime_initial_multipliers,
            ),
            (
                "demand_regime_final_multipliers",
                self.config.demand_regime_final_multipliers,
            ),
        ):
            multipliers = _as_vector(values, self.config.num_facilities, name)
            if np.any(multipliers < 0.0):
                raise ValueError(f"{name} must be nonnegative")
        if self.config.enable_scheduled_referral_waves:
            clusters = tuple(self.config.scheduled_referral_clusters)
            if not clusters or any(not tuple(cluster) for cluster in clusters):
                raise ValueError(
                    "scheduled_referral_clusters must contain nonempty clusters"
                )
            for cluster in clusters:
                indices = tuple(int(index) for index in cluster)
                if len(set(indices)) != len(indices):
                    raise ValueError(
                        "scheduled_referral_clusters cannot repeat a facility"
                    )
                if any(
                    index < 0 or index >= self.config.num_facilities
                    for index in indices
                ):
                    raise ValueError(
                        "scheduled_referral_clusters contains an invalid facility"
                    )
            if self.config.scheduled_referral_start_step < 0:
                raise ValueError("scheduled_referral_start_step must be nonnegative")
            if self.config.scheduled_referral_block_length < 1:
                raise ValueError("scheduled_referral_block_length must be positive")
            if self.config.scheduled_referral_peak_multiplier < 1.0:
                raise ValueError(
                    "scheduled_referral_peak_multiplier must be at least 1"
                )
        if self.config.geographic_neighbor_k < 1:
            raise ValueError("geographic_neighbor_k must be positive")
        if self.config.clinic_coordinates is not None:
            normalize_coordinates(self.config.clinic_coordinates, self.config.num_facilities)
        if self.config.reagent_purchase_lead_time < 0:
            raise ValueError("reagent_purchase_lead_time must be nonnegative")
        if self.config.enable_stochastic_procurement:
            probabilities = [float(p) for p in self.config.reagent_lead_time_probabilities]
            if len(probabilities) < 2:
                raise ValueError(
                    "enable_stochastic_procurement requires at least two lead-time "
                    "probabilities; a degenerate distribution is deterministic "
                    "procurement and should use reagent_purchase_lead_time instead"
                )
            if any(p < 0.0 for p in probabilities):
                raise ValueError("reagent_lead_time_probabilities must be nonnegative")
            if abs(sum(probabilities) - 1.0) > 1e-9:
                raise ValueError("reagent_lead_time_probabilities must sum to 1")
        elif self.config.reagent_lead_time_probabilities:
            raise ValueError(
                "reagent_lead_time_probabilities requires enable_stochastic_procurement"
            )
        if (
            self.config.procurement_lead_override is not None
            and not self.config.enable_stochastic_procurement
        ):
            raise ValueError(
                "procurement_lead_override is a counterfactual device for the "
                "stochastic regime and requires enable_stochastic_procurement"
            )
        if self.config.include_on_order_state and not self._procurement_pipeline_depth():
            raise ValueError(
                "include_on_order_state requires a procurement lead time or a "
                "stochastic procurement distribution"
            )
        if self.config.enable_production_throttle:
            raise ValueError(
                "enable_production_throttle is dormant (Decision B); activation "
                "requires its own change-control entry per "
                "specs/2026-08-29-continuous-overtime-control"
            )
        if not 0.0 <= self.config.max_overtime_fraction <= 1.0:
            raise ValueError("max_overtime_fraction must be within [0, 1]")
        if self.config.enable_overtime_control:
            if self.config.action_mode != "facility_net":
                raise ValueError(
                    "enable_overtime_control requires action_mode='facility_net'"
                )
            if self.config.weight_overtime_linear <= 0.0:
                raise ValueError("weight_overtime_linear must be positive")
            if self.config.weight_overtime_quadratic <= 0.0:
                raise ValueError("weight_overtime_quadratic must be positive")
        if self.config.enable_intertemporal_overtime_commitment:
            if not self.config.enable_overtime_control:
                raise ValueError(
                    "enable_intertemporal_overtime_commitment requires "
                    "enable_overtime_control"
                )
            if self.config.overtime_commitment_lead_time < 1:
                raise ValueError("overtime_commitment_lead_time must be positive")
            if not 0.0 <= self.config.overtime_commitment_persistence < 1.0:
                raise ValueError(
                    "overtime_commitment_persistence must be within [0, 1)"
                )
            if not 0.0 < self.config.overtime_shared_budget_fraction <= 1.0:
                raise ValueError(
                    "overtime_shared_budget_fraction must be within (0, 1]"
                )
            if self.config.weight_overtime_activation < 0.0:
                raise ValueError("weight_overtime_activation must be nonnegative")
        if self.config.enable_overtime_fatigue:
            if not self.config.enable_overtime_control:
                raise ValueError(
                    "enable_overtime_fatigue requires enable_overtime_control"
                )
            if not 0.0 < self.config.overtime_fatigue_decay < 1.0:
                raise ValueError("overtime_fatigue_decay must be within (0, 1)")
            if self.config.overtime_fatigue_cost_scale < 0.0:
                raise ValueError("overtime_fatigue_cost_scale must be nonnegative")
        if self.config.geographic_transfer_cost_scale < 0.0:
            raise ValueError("geographic_transfer_cost_scale must be nonnegative")
        if self.config.geographic_transfer_speed_mph <= 0.0:
            raise ValueError("geographic_transfer_speed_mph must be positive")
        if self.config.geographic_transfer_fixed_hours < 0.0:
            raise ValueError("geographic_transfer_fixed_hours must be nonnegative")
        if self.config.geographic_transfer_time_cost_scale < 0.0:
            raise ValueError("geographic_transfer_time_cost_scale must be nonnegative")
        if self.config.transfer_lead_time_distance_thresholds is not None:
            thresholds = [float(value) for value in self.config.transfer_lead_time_distance_thresholds]
            if any(value < 0.0 for value in thresholds):
                raise ValueError("transfer_lead_time_distance_thresholds must be nonnegative")
            if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
                raise ValueError("transfer_lead_time_distance_thresholds must be strictly increasing")
        if not 0.0 <= self.config.regional_supplier_disruption_probability <= 1.0:
            raise ValueError("regional_supplier_disruption_probability must be between 0 and 1")
        if self.config.regional_supplier_disruption_duration < 0:
            raise ValueError("regional_supplier_disruption_duration must be nonnegative")
        if self.config.regional_supplier_disruption_cluster_size < 0:
            raise ValueError("regional_supplier_disruption_cluster_size must be nonnegative")
        if self.config.action_mode not in ("edge_transfer", "facility_net"):
            raise ValueError("action_mode must be 'edge_transfer' or 'facility_net'")

    def _sample_supplier_available(self) -> np.ndarray:
        available = self.rng.random(self.config.num_facilities) >= self.supplier_disruption_rate
        if np.any(self.regional_supplier_disruption_remaining > 0):
            available = np.logical_and(available, self.regional_supplier_disruption_remaining <= 0)
        return available.astype(float)

    def _advance_clock(self) -> bool:
        self.t += 1
        done = self.t >= self.config.episode_horizon
        self._update_demand_regime_multiplier()
        self._advance_demand_shocks()
        self.demand = self.rng.poisson(self._effective_demand_rates()).astype(float)
        self._advance_regional_supplier_disruptions()
        self.supplier_available = self._sample_supplier_available()
        self.demand_forecast = self._sample_demand_forecast()
        self._record_demand_observation()
        return done

    def _record_demand_observation(self) -> None:
        window = max(
            int(self.config.demand_history_window),
            (
                int(self.config.demand_sequence_length)
                if self.config.include_demand_sequence_state
                else 1
            ),
            1,
        )
        per_period_forecast = self.demand_forecast / max(
            int(self.config.demand_forecast_horizon),
            1,
        )
        self.demand_history.append(self.demand.astype(float).copy())
        self.forecast_error_history.append(
            (self.demand - per_period_forecast).astype(float)
        )
        del self.demand_history[:-window]
        del self.forecast_error_history[:-window]

    def _demand_history_features(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.num_facilities
        if not getattr(self, "demand_history", None):
            zeros = np.zeros(n, dtype=float)
            return zeros, zeros.copy(), zeros.copy()
        window = max(int(self.config.demand_history_window), 1)
        history = np.stack(self.demand_history[-window:], axis=0)
        rolling_mean = history.mean(axis=0)
        if history.shape[0] <= 1:
            trend = np.zeros(n, dtype=float)
        else:
            trend = (history[-1] - history[0]) / float(history.shape[0] - 1)
        forecast_error = np.stack(
            self.forecast_error_history[-window:],
            axis=0,
        ).mean(axis=0)
        return rolling_mean, trend, forecast_error

    def _demand_sequence_features(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return left-padded causal demand/error sequences and validity masks."""

        n = self.config.num_facilities
        length = max(int(self.config.demand_sequence_length), 1)
        demand_sequence = np.zeros((n, length), dtype=float)
        error_sequence = np.zeros((n, length), dtype=float)
        sequence_mask = np.zeros((n, length), dtype=float)
        if not getattr(self, "demand_history", None):
            return demand_sequence, error_sequence, sequence_mask

        demand = np.stack(self.demand_history[-length:], axis=0)
        error = np.stack(
            self.forecast_error_history[-length:],
            axis=0,
        )
        valid = demand.shape[0]
        demand_sequence[:, -valid:] = demand.transpose(1, 0)
        error_sequence[:, -valid:] = error.transpose(1, 0)
        sequence_mask[:, -valid:] = 1.0
        return demand_sequence, error_sequence, sequence_mask

    def _normalized_time(self) -> float:
        return float(np.clip(self.t / self.config.episode_horizon, 0.0, 1.0))

    def _advance_demand_shocks(self) -> None:
        if self.config.demand_shock_duration <= 0:
            self.demand_shock_remaining[:] = 0
            self.demand_rate_multiplier[:] = 1.0
            return

        self.demand_shock_remaining = np.maximum(self.demand_shock_remaining - 1, 0)
        if self.rng.random() < self.config.demand_shock_probability:
            shocked = self._sample_demand_shock_cluster()
            self.demand_shock_remaining[shocked] = np.maximum(
                self.demand_shock_remaining[shocked],
                int(self.config.demand_shock_duration),
            )
        self.demand_rate_multiplier = np.where(
            self.demand_shock_remaining > 0,
            float(self.config.demand_shock_multiplier),
            1.0,
        )

    def _sample_demand_shock_cluster(self) -> np.ndarray:
        n = self.config.num_facilities
        cluster_size = int(self.config.demand_shock_cluster_size) or n
        cluster_size = min(max(cluster_size, 1), n)
        if cluster_size == n:
            return np.arange(n)
        start = int(self.rng.integers(0, n))
        if self.clinic_distance_matrix is not None:
            return np.argsort(self.clinic_distance_matrix[start], kind="stable")[:cluster_size]
        return (start + np.arange(cluster_size)) % n

    def _advance_regional_supplier_disruptions(self) -> None:
        if self.config.regional_supplier_disruption_duration <= 0:
            self.regional_supplier_disruption_remaining[:] = 0
            return

        self.regional_supplier_disruption_remaining = np.maximum(
            self.regional_supplier_disruption_remaining - 1,
            0,
        )
        if self.rng.random() < self.config.regional_supplier_disruption_probability:
            disrupted = self._sample_supplier_disruption_cluster()
            self.regional_supplier_disruption_remaining[disrupted] = np.maximum(
                self.regional_supplier_disruption_remaining[disrupted],
                int(self.config.regional_supplier_disruption_duration),
            )

    def _sample_supplier_disruption_cluster(self) -> np.ndarray:
        n = self.config.num_facilities
        cluster_size = int(self.config.regional_supplier_disruption_cluster_size) or n
        cluster_size = min(max(cluster_size, 1), n)
        if cluster_size == n:
            return np.arange(n)
        start = int(self.rng.integers(0, n))
        if self.clinic_distance_matrix is not None:
            return np.argsort(self.clinic_distance_matrix[start], kind="stable")[:cluster_size]
        return (start + np.arange(cluster_size)) % n

    def _effective_demand_rates(self) -> np.ndarray:
        return (
            self.demand_rates
            * self.demand_regime_multiplier
            * self.demand_rate_multiplier
            * self._scheduled_referral_multiplier_at(self.t)
        )

    def _scheduled_referral_multiplier_at(self, step: int) -> np.ndarray:
        """Return the configured referral-wave multiplier for one epoch."""

        multiplier = np.ones(self.config.num_facilities, dtype=float)
        if not self.config.enable_scheduled_referral_waves:
            return multiplier
        start = int(self.config.scheduled_referral_start_step)
        if step < start:
            return multiplier
        clusters = tuple(tuple(int(index) for index in cluster) for cluster in (
            self.config.scheduled_referral_clusters
        ))
        block_index = (int(step) - start) // int(
            self.config.scheduled_referral_block_length
        )
        if not self.config.scheduled_referral_repeat and block_index >= len(clusters):
            return multiplier
        active_cluster = clusters[block_index % len(clusters)]
        multiplier[np.asarray(active_cluster, dtype=int)] = float(
            self.config.scheduled_referral_peak_multiplier
        )
        return multiplier

    def _demand_regime_multiplier_at(self, step: int) -> np.ndarray:
        """Pure counterpart of `_update_demand_regime_multiplier`."""

        change_step = int(self.config.demand_regime_change_step)
        duration = int(self.config.demand_regime_transition_duration)
        if step < change_step:
            return self.demand_regime_initial_multipliers.astype(float).copy()
        if duration == 0:
            return self.demand_regime_final_multipliers.astype(float).copy()
        progress = float(np.clip((step - change_step) / duration, 0.0, 1.0))
        return (
            (1.0 - progress) * self.demand_regime_initial_multipliers
            + progress * self.demand_regime_final_multipliers
        )

    def _update_demand_regime_multiplier(self) -> None:
        """Update the persistent spatial demand regime for the current epoch."""

        self.demand_regime_multiplier = self._demand_regime_multiplier_at(self.t)

    def _sample_demand_forecast(self) -> np.ndarray:
        horizon = int(self.config.demand_forecast_horizon)
        if self.config.enable_scheduled_referral_waves:
            base_rates = (
                self.demand_rate_estimates
                if self.config.demand_forecast_source == "prior_estimate"
                else self.demand_rates
            )
            forecast = np.zeros(self.config.num_facilities, dtype=float)
            for offset in range(1, horizon + 1):
                step = self.t + offset
                forecast += (
                    base_rates
                    * self._demand_regime_multiplier_at(step)
                    * self.demand_rate_multiplier
                    * self._scheduled_referral_multiplier_at(step)
                )
        elif self.config.demand_forecast_source == "prior_estimate":
            forecast = self.demand_rate_estimates * horizon
        else:
            forecast = self._effective_demand_rates() * horizon
        if not self.config.include_demand_forecast_state:
            return forecast.astype(float)
        error = self.demand_forecast_error
        if error is None or float(error) == 0.0:
            return forecast.astype(float)
        noise = self.rng.normal(loc=0.0, scale=float(error), size=self.config.num_facilities)
        return np.clip(forecast * (1.0 + noise), 0.0, None).astype(float)

    def _empty_transfer_pipeline(self) -> np.ndarray:
        return np.zeros(
            (int(self.config.transfer_lead_time), self.config.num_facilities),
            dtype=float,
        )

    def _pop_transfer_arrivals(self, pipeline: np.ndarray) -> np.ndarray:
        if pipeline.shape[0] == 0:
            return np.zeros(self.config.num_facilities, dtype=float)
        arrivals = pipeline[0].copy()
        if pipeline.shape[0] > 1:
            pipeline[:-1] = pipeline[1:]
        pipeline[-1] = 0.0
        return arrivals

    def _schedule_transfer_arrivals(self, pipeline: np.ndarray, arrivals: np.ndarray) -> None:
        if pipeline.shape[0] == 0:
            return
        pipeline[-1] += arrivals

    def _schedule_edge_transfer_arrivals(
        self,
        pipeline: np.ndarray,
        edges: Sequence[Edge],
        edge_flows: np.ndarray,
    ) -> None:
        if pipeline.shape[0] == 0:
            return
        for (i, j), flow in zip(edges, edge_flows):
            amount = float(flow)
            if abs(amount) <= 1e-8:
                continue
            destination = j if amount > 0.0 else i
            delay = self._transfer_delay_for_edge((i, j))
            pipeline[delay - 1, destination] += abs(amount)

    def _uses_geographic_transfer_delays(self) -> bool:
        return (
            self.config.transfer_lead_time > 0
            and self.clinic_distance_matrix is not None
            and bool(self.transfer_delay_thresholds)
        )

    def _transfer_delay_for_edge(self, edge: Edge) -> int:
        if self.config.transfer_lead_time <= 0:
            return 0
        if self.clinic_distance_matrix is None or not self.transfer_delay_thresholds:
            return int(self.config.transfer_lead_time)
        distance = self._edge_distance(edge)
        delay = 1 + sum(distance > threshold for threshold in self.transfer_delay_thresholds)
        return min(max(int(delay), 1), int(self.config.transfer_lead_time))

    def _edge_distance(self, edge: Edge) -> float:
        if self.clinic_distance_matrix is None:
            return 0.0
        i, j = edge
        return float(self.clinic_distance_matrix[int(i), int(j)])

    def _edge_transfer_time_hours(self, edge: Edge) -> float:
        if self.clinic_transfer_time_hours_matrix is None:
            return 0.0
        i, j = edge
        return float(self.clinic_transfer_time_hours_matrix[int(i), int(j)])

    def _facility_net_transfer_priorities(
        self,
        edges: Sequence[Edge],
    ) -> tuple[float, ...] | None:
        """Rank feasible routes by discrete delay, then continuous travel time."""

        if self.clinic_transfer_time_hours_matrix is None:
            return None
        return tuple(
            1000.0 * float(self._transfer_delay_for_edge(edge))
            + self._edge_transfer_time_hours(edge)
            for edge in edges
        )

    def _edge_transfer_cost(self, base_cost: float, edges: Sequence[Edge], edge_flows: np.ndarray) -> float:
        edge_flows = np.asarray(edge_flows, dtype=float)
        if (
            self.clinic_distance_matrix is None
            and self.clinic_transfer_time_hours_matrix is None
        ):
            return float(base_cost) * float(np.abs(edge_flows).sum())
        total = 0.0
        for edge, flow in zip(edges, edge_flows):
            multiplier = (
                1.0
                + float(self.config.geographic_transfer_cost_scale)
                * (self._edge_distance(edge) / 1000.0)
                + float(self.config.geographic_transfer_time_cost_scale)
                * self._edge_transfer_time_hours(edge)
            )
            total += float(base_cost) * multiplier * abs(float(flow))
        return total

    def _facility_net_transfer_cost(
        self,
        base_cost: float,
        edges: Sequence[Edge],
        edge_flows: np.ndarray,
        net_flows: np.ndarray,
    ) -> float:
        base_total = float(base_cost) * float(np.abs(net_flows).sum())
        if (
            self.clinic_distance_matrix is None
            and self.clinic_transfer_time_hours_matrix is None
        ):
            return base_total
        distance_surcharge = 0.0
        for edge, flow in zip(edges, np.asarray(edge_flows, dtype=float)):
            distance_surcharge += (
                float(base_cost)
                * (
                    float(self.config.geographic_transfer_cost_scale)
                    * (self._edge_distance(edge) / 1000.0)
                    + float(self.config.geographic_transfer_time_cost_scale)
                    * self._edge_transfer_time_hours(edge)
                )
                * abs(float(flow))
            )
        return base_total + distance_surcharge

    def _capacity_graph_edges(self) -> tuple[Edge, ...]:
        if not self.config.include_central_capacity_hub:
            return self.capacity_edges
        if not self.capacity_edges:
            return ()
        hub = self.config.num_facilities
        return tuple((i, hub) for i in range(self.config.num_facilities))

    def _update_running_metrics(
        self,
        demand: np.ndarray,
        production: np.ndarray,
        specimens: np.ndarray,
        bioreactors: np.ndarray,
    ) -> None:
        self.cumulative_demand += float(demand.sum())
        self.cumulative_production += float(production.sum())
        self.cumulative_waiting_specimens += float(specimens.sum())
        self.cumulative_bioreactor_capacity += float(bioreactors[:, 0].sum() + production.sum())

    def _performance_info(self) -> dict[str, float]:
        steps = max(self.t, 1)
        return {
            "service_level": self.cumulative_production / max(self.cumulative_demand, 1.0),
            "average_waiting_time": self.cumulative_waiting_specimens
            / max(self.cumulative_demand, 1.0),
            "bioreactor_utilization": self.cumulative_production
            / max(self.cumulative_bioreactor_capacity, 1.0),
            "reagent_shortage_frequency": self.reagent_shortage_steps / steps,
            "bioreactor_shortage_frequency": self.bioreactor_shortage_steps / steps,
        }


def make_legacy_two_facility_config(episode_horizon: int = 104) -> CapacityPlanningConfig:
    """Return defaults matching the legacy two-facility DDPG environment."""

    return CapacityPlanningConfig(
        num_facilities=2,
        production_lead_time=5,
        episode_horizon=episode_horizon,
        demand_rates=(250.0 / 52.0, 250.0 / 52.0),
        initial_specimens=(0.0, 0.0),
        initial_reagents=(100.0, 100.0),
        initial_idle_bioreactors=(10.0, 10.0),
        max_specimens=(100.0, 100.0),
        max_reagents=(200.0, 200.0),
        max_idle_bioreactors=(20.0, 20.0),
        max_reagent_replenishment=(100.0, 100.0),
        max_specimen_transfer=100.0,
        max_bioreactor_transfer=20.0,
        max_reagent_transfer=100.0,
    )


def make_20_clinic_config(
    episode_horizon: int = 52,
    supplier_disruption_rate: float = 0.3,
) -> CapacityPlanningConfig:
    """Return the manuscript-aligned 20-clinic PRM configuration."""

    n = 20
    return CapacityPlanningConfig(
        num_facilities=n,
        production_lead_time=3,
        episode_horizon=episode_horizon,
        demand_rates=(250.0 / 52.0,) * n,
        initial_specimens=(0.0,) * n,
        initial_reagents=(100.0,) * n,
        initial_idle_bioreactors=(10.0,) * n,
        max_specimens=(100.0,) * n,
        max_reagents=(200.0,) * n,
        max_idle_bioreactors=(20.0,) * n,
        max_reagent_replenishment=(100.0,) * n,
        max_specimen_transfer=100.0,
        max_bioreactor_transfer=20.0,
        max_reagent_transfer=100.0,
        action_mode="facility_net",
        include_supplier_state=True,
        supplier_disruption_rate=supplier_disruption_rate,
        include_central_capacity_hub=True,
        information_edges=k_nearest_ring_edges(n, k=2),
        specimen_edges=ring_edges(n),
        resource_edges=ring_edges(n),
        capacity_edges=complete_undirected_edges(n),
    )


def _as_vector(values: Sequence[float] | None, length: int, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(length, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape == ():
        return np.full(length, float(array))
    if array.shape != (length,):
        raise ValueError(f"{name} must have length {length}; got shape {array.shape}")
    return array


def _normalize_edges(
    configured_edges: Sequence[Edge] | None, default_edges: tuple[Edge, ...], num_nodes: int
) -> tuple[Edge, ...]:
    edges = default_edges if configured_edges is None else tuple(configured_edges)
    normalized = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Edge must have two endpoints, got {edge}")
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            raise ValueError(f"Self-loops are not supported in sharing edges: {edge}")
        if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
            raise ValueError(f"Edge {edge} is outside the {num_nodes}-facility network")
        normalized.append((min(i, j), max(i, j)))
    return tuple(dict.fromkeys(normalized))


def _apply_transfers(values: np.ndarray, edges: Sequence[Edge], requested: np.ndarray) -> np.ndarray:
    actual = np.zeros(len(edges), dtype=float)
    for idx, ((i, j), amount) in enumerate(zip(edges, requested)):
        if amount >= 0:
            flow = min(float(amount), float(values[i]))
            values[i] -= flow
            values[j] += flow
            actual[idx] = flow
        else:
            flow = min(float(-amount), float(values[j]))
            values[j] -= flow
            values[i] += flow
            actual[idx] = -flow
    return actual


def _apply_transfers_delayed(
    values: np.ndarray, edges: Sequence[Edge], requested: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.zeros(len(edges), dtype=float)
    arrivals = np.zeros_like(values, dtype=float)
    for idx, ((i, j), amount) in enumerate(zip(edges, requested)):
        if amount >= 0:
            flow = min(float(amount), float(values[i]))
            values[i] -= flow
            arrivals[j] += flow
            actual[idx] = flow
        else:
            flow = min(float(-amount), float(values[j]))
            values[j] -= flow
            arrivals[i] += flow
            actual[idx] = -flow
    return actual, arrivals


def _apply_net_transfers(
    values: np.ndarray,
    edges: Sequence[Edge],
    requested_net: np.ndarray,
    *,
    edge_priorities: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    actual_net = np.zeros_like(values, dtype=float)
    edge_flows = np.zeros(len(edges), dtype=float)
    if not edges:
        return actual_net, edge_flows

    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    adjacency: dict[int, set[int]] = {}
    for i, j in edges:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    inbound_remaining = np.maximum(requested_net, 0.0).astype(float)
    outbound_remaining = np.maximum(-requested_net, 0.0).astype(float)
    receivers = list(np.where(inbound_remaining > 1e-8)[0])
    donors = list(np.where(outbound_remaining > 1e-8)[0])

    pairs = _ordered_net_transfer_pairs(
        receivers,
        donors,
        adjacency,
        edge_index,
        edge_priorities,
    )
    for receiver, donor in pairs:
        if inbound_remaining[receiver] <= 1e-8 or outbound_remaining[donor] <= 1e-8:
            continue
        flow = min(inbound_remaining[receiver], outbound_remaining[donor], values[donor])
        if flow <= 1e-8:
            continue
        values[donor] -= flow
        values[receiver] += flow
        actual_net[donor] -= flow
        actual_net[receiver] += flow
        edge = (min(donor, receiver), max(donor, receiver))
        sign = 1.0 if edge[0] == donor else -1.0
        edge_flows[edge_index[edge]] += sign * flow
        outbound_remaining[donor] -= flow
        inbound_remaining[receiver] -= flow

    return actual_net, edge_flows


def _apply_net_transfers_delayed(
    values: np.ndarray,
    edges: Sequence[Edge],
    requested_net: np.ndarray,
    *,
    edge_priorities: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_net = np.zeros_like(values, dtype=float)
    edge_flows = np.zeros(len(edges), dtype=float)
    arrivals = np.zeros_like(values, dtype=float)
    if not edges:
        return actual_net, edge_flows, arrivals

    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    adjacency: dict[int, set[int]] = {}
    for i, j in edges:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    inbound_remaining = np.maximum(requested_net, 0.0).astype(float)
    outbound_remaining = np.maximum(-requested_net, 0.0).astype(float)
    receivers = list(np.where(inbound_remaining > 1e-8)[0])
    donors = list(np.where(outbound_remaining > 1e-8)[0])

    pairs = _ordered_net_transfer_pairs(
        receivers,
        donors,
        adjacency,
        edge_index,
        edge_priorities,
    )
    for receiver, donor in pairs:
        if inbound_remaining[receiver] <= 1e-8 or outbound_remaining[donor] <= 1e-8:
            continue
        flow = min(inbound_remaining[receiver], outbound_remaining[donor], values[donor])
        if flow <= 1e-8:
            continue
        values[donor] -= flow
        arrivals[receiver] += flow
        actual_net[donor] -= flow
        actual_net[receiver] += flow
        edge = (min(donor, receiver), max(donor, receiver))
        sign = 1.0 if edge[0] == donor else -1.0
        edge_flows[edge_index[edge]] += sign * flow
        outbound_remaining[donor] -= flow
        inbound_remaining[receiver] -= flow

    return actual_net, edge_flows, arrivals


def _ordered_net_transfer_pairs(
    receivers: Sequence[int],
    donors: Sequence[int],
    adjacency: dict[int, set[int]],
    edge_index: dict[Edge, int],
    edge_priorities: Sequence[float] | None,
) -> list[tuple[int, int]]:
    pairs = [
        (int(receiver), int(donor))
        for receiver in receivers
        for donor in donors
        if receiver != donor and receiver in adjacency.get(donor, set())
    ]
    if edge_priorities is None:
        return pairs
    priorities = np.asarray(edge_priorities, dtype=float)
    if priorities.shape != (len(edge_index),):
        raise ValueError(
            f"Expected one edge priority per edge, got {priorities.shape} "
            f"for {len(edge_index)} edges"
        )
    if not np.all(np.isfinite(priorities)):
        raise ValueError("edge_priorities must be finite")
    return sorted(
        pairs,
        key=lambda pair: (
            float(
                priorities[
                    edge_index[
                        (min(pair[0], pair[1]), max(pair[0], pair[1]))
                    ]
                ]
            ),
            pair[0],
            pair[1],
        ),
    )


def _edge_array(edges: Sequence[Edge]) -> np.ndarray:
    return np.asarray(edges, dtype=np.int64).reshape((-1, 2))
