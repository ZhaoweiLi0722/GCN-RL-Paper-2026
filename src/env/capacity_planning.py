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
    demand_shock_probability: float = 0.0
    demand_shock_multiplier: float = 1.0
    demand_shock_duration: int = 0
    demand_shock_cluster_size: int = 0
    include_demand_forecast_state: bool = False
    demand_forecast_horizon: int = 1
    demand_forecast_error: float | None = None
    clinic_coordinates: Sequence[Sequence[float]] | None = None
    geographic_neighbor_k: int = 3
    geographic_transfer_cost_scale: float = 0.0
    transfer_lead_time_distance_thresholds: Sequence[float] | None = None
    regional_supplier_disruption_probability: float = 0.0
    regional_supplier_disruption_duration: int = 0
    regional_supplier_disruption_cluster_size: int = 0
    information_edges: Sequence[Edge] | None = None
    specimen_edges: Sequence[Edge] | None = None
    capacity_edges: Sequence[Edge] | None = None
    resource_edges: Sequence[Edge] | None = None
    costs: CostParameters = CostParameters()


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
        self.clinic_coordinates = normalize_coordinates(self.config.clinic_coordinates, n)
        self.clinic_distance_matrix = (
            np.asarray(geographic_distance_matrix(self.clinic_coordinates), dtype=float)
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
        self.hub_index = n if self.config.include_central_capacity_hub else None

        self.features_per_facility = 3 + self.config.production_lead_time
        if self.config.include_supplier_state:
            self.features_per_facility += 1
        if self.config.include_demand_forecast_state:
            self.features_per_facility += 1
        if self.config.include_transfer_pipeline_state:
            self.features_per_facility += 3
        self.observation_size = n * self.features_per_facility
        if self.config.action_mode == "facility_net":
            self.action_size = 4 * n
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
        self.specimen_transfer_pipeline = self._empty_transfer_pipeline()
        self.reagent_transfer_pipeline = self._empty_transfer_pipeline()
        self.capacity_transfer_pipeline = self._empty_transfer_pipeline()
        self._maybe_randomize_regime()
        self.demand = self.rng.poisson(self.demand_rates).astype(float)
        self._advance_regional_supplier_disruptions()
        self.supplier_available = self._sample_supplier_available()
        self.demand_forecast = self._sample_demand_forecast()
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
        return self.observation()

    def enable_train_randomization(
        self,
        disruption_range: tuple[float, float] | None = None,
        forecast_error_range: tuple[float, float] | None = None,
    ) -> None:
        """Opt into train-time per-episode domain randomization.

        When set, each :meth:`reset` resamples the stressed parameter uniformly
        from the given ``[lo, hi]`` range *before* demand/supplier/forecast are
        drawn, so training sees a distribution of regimes. Eval envs leave this
        unset and keep the fixed config values, enabling clean train-on-range /
        test-OOD splits. Passing ``None`` (the default) disables that lever.
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

    def _maybe_randomize_regime(self) -> None:
        """Resample stressed parameters from their train-time ranges, if enabled."""
        n = self.config.num_facilities
        if self._train_disruption_range is not None:
            lo, hi = self._train_disruption_range
            rate = float(self.rng.uniform(lo, hi))
            self.supplier_disruption_rate = np.full(n, rate, dtype=float)
        if self._train_forecast_error_range is not None:
            lo, hi = self._train_forecast_error_range
            self.demand_forecast_error = float(self.rng.uniform(lo, hi))

    def observation(self) -> np.ndarray:
        """Return a flat observation suitable for MLP baselines."""

        rows = []
        pending_specimens, pending_reagents, pending_capacity = self._pending_transfer_arrivals()
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
            rows.append(np.concatenate(tuple(row_parts)))
        return np.concatenate(rows).astype(np.float32)

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
        if self.config.include_central_capacity_hub:
            facility_columns.append(np.zeros(self.config.num_facilities, dtype=float))
        node_features = np.column_stack(tuple(facility_columns)).astype(np.float32)
        capacity_graph_edges = self._capacity_graph_edges()
        if self.config.include_central_capacity_hub:
            hub_features = np.zeros((1, node_features.shape[1]), dtype=np.float32)
            hub_features[0, 3] = float(self.bioreactors[:, 0].sum())
            hub_features[0, 4] = float(self.bioreactors.sum())
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
        return graph

    def noop_action(self) -> np.ndarray:
        """Return an action with no replenishment and no transfers."""

        action = np.zeros(self.action_size, dtype=np.float32)
        if self.config.action_mode == "facility_net":
            n = self.config.num_facilities
            action[3 * n : 4 * n] = -1.0
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

        cost = (
            costs.reagent_purchase * float(replenishment.sum())
            + costs.reagent_holding * float(idle_reagents.sum())
            + costs.reagent_shortage * float(under_reagents.sum())
            + costs.bioreactor_holding * float(idle_bioreactor_counts.sum())
            + costs.bioreactor_shortage * float(under_bioreactors.sum())
            + self._edge_transfer_cost(costs.specimen_transfer, self.specimen_edges, specimen_transfers)
            + self._edge_transfer_cost(costs.bioreactor_transfer, self.capacity_edges, capacity_transfers)
            + self._edge_transfer_cost(costs.reagent_transfer, self.resource_edges, reagent_transfers)
        )

        info: dict[str, np.ndarray | float] = {
            "cost": cost,
            "production": production.copy(),
            "demand": current_demand.copy(),
            "demand_forecast": current_demand_forecast.copy(),
            "supplier_available": supplier_available.copy(),
            "demand_rate_multiplier": self.demand_rate_multiplier.copy(),
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

        specimen_requests = normalized[:n] * self.config.max_specimen_transfer
        reagent_transfer_requests = normalized[n : 2 * n] * self.config.max_reagent_transfer
        capacity_requests = normalized[2 * n : 3 * n] * self.config.max_bioreactor_transfer
        replenishment = (
            ((normalized[3 * n : 4 * n] + 1.0) / 2.0)
            * self.max_reagent_replenishment
            * supplier_available
        )

        production = np.minimum.reduce((self.specimens, self.bioreactors[:, 0], self.reagents))
        next_specimens = self.specimens - production + current_demand
        next_reagents = self.reagents - production + replenishment
        next_bioreactors = np.zeros_like(self.bioreactors)
        next_bioreactors[:, 0] = self.bioreactors[:, 0] - production + self.bioreactors[:, 1]
        if self.config.production_lead_time > 2:
            next_bioreactors[:, 1:-1] = self.bioreactors[:, 2:]
        next_bioreactors[:, -1] = production

        if self.config.transfer_lead_time > 0:
            specimen_net, specimen_flows, specimen_future_arrivals = _apply_net_transfers_delayed(
                next_specimens, self.specimen_edges, specimen_requests
            )
            capacity_net, capacity_flows, capacity_future_arrivals = _apply_net_transfers_delayed(
                next_bioreactors[:, 0], self.capacity_edges, capacity_requests
            )
            reagent_net, reagent_flows, reagent_future_arrivals = _apply_net_transfers_delayed(
                next_reagents, self.resource_edges, reagent_transfer_requests
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
                next_specimens, self.specimen_edges, specimen_requests
            )
            capacity_net, capacity_flows = _apply_net_transfers(
                next_bioreactors[:, 0], self.capacity_edges, capacity_requests
            )
            reagent_net, reagent_flows = _apply_net_transfers(
                next_reagents, self.resource_edges, reagent_transfer_requests
            )

        self.specimens = np.clip(next_specimens, 0.0, self.max_specimens)
        self.reagents = np.clip(next_reagents, 0.0, self.max_reagents)
        next_bioreactors[:, 0] = np.clip(next_bioreactors[:, 0], 0.0, self.max_idle_bioreactors)
        self.bioreactors = np.maximum(next_bioreactors, 0.0)

        under_reagents = np.maximum(self.specimens - self.reagents, 0.0)
        idle_reagents = np.maximum(self.reagents - self.specimens, 0.0)
        under_bioreactors = np.maximum(self.specimens - self.bioreactors[:, 0], 0.0)
        idle_bioreactor_counts = np.maximum(self.bioreactors[:, 0] - self.specimens, 0.0)

        cost = (
            costs.reagent_purchase * float(replenishment.sum())
            + costs.reagent_holding * float(idle_reagents.sum())
            + costs.reagent_shortage * float(under_reagents.sum())
            + costs.bioreactor_holding * float(idle_bioreactor_counts.sum())
            + costs.bioreactor_shortage * float(under_bioreactors.sum())
            + self._facility_net_transfer_cost(
                costs.specimen_transfer,
                self.specimen_edges,
                specimen_flows,
                specimen_net,
            )
            + self._facility_net_transfer_cost(
                costs.bioreactor_transfer,
                self.capacity_edges,
                capacity_flows,
                capacity_net,
            )
            + self._facility_net_transfer_cost(
                costs.reagent_transfer,
                self.resource_edges,
                reagent_flows,
                reagent_net,
            )
        )

        self._update_running_metrics(current_demand, production, self.specimens, self.bioreactors)
        if np.any(under_reagents > 0):
            self.reagent_shortage_steps += 1
        if np.any(under_bioreactors > 0):
            self.bioreactor_shortage_steps += 1
        done = self._advance_clock()

        info: dict[str, np.ndarray | float] = {
            "cost": cost,
            "production": production.copy(),
            "demand": current_demand.copy(),
            "demand_forecast": current_demand_forecast.copy(),
            "supplier_available": supplier_available.copy(),
            "demand_rate_multiplier": self.demand_rate_multiplier.copy(),
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
        info.update(self._performance_info())
        return self.observation(), -cost, done, info

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
        if self.config.demand_forecast_error is not None and self.config.demand_forecast_error < 0.0:
            raise ValueError("demand_forecast_error must be nonnegative or null")
        if self.config.geographic_neighbor_k < 1:
            raise ValueError("geographic_neighbor_k must be positive")
        if self.config.clinic_coordinates is not None:
            normalize_coordinates(self.config.clinic_coordinates, self.config.num_facilities)
        if self.config.geographic_transfer_cost_scale < 0.0:
            raise ValueError("geographic_transfer_cost_scale must be nonnegative")
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
        self._advance_demand_shocks()
        self.demand = self.rng.poisson(self._effective_demand_rates()).astype(float)
        self._advance_regional_supplier_disruptions()
        self.supplier_available = self._sample_supplier_available()
        self.demand_forecast = self._sample_demand_forecast()
        return done

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
        return self.demand_rates * self.demand_rate_multiplier

    def _sample_demand_forecast(self) -> np.ndarray:
        horizon = int(self.config.demand_forecast_horizon)
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

    def _edge_transfer_cost(self, base_cost: float, edges: Sequence[Edge], edge_flows: np.ndarray) -> float:
        edge_flows = np.asarray(edge_flows, dtype=float)
        if self.clinic_distance_matrix is None or self.config.geographic_transfer_cost_scale == 0.0:
            return float(base_cost) * float(np.abs(edge_flows).sum())
        total = 0.0
        for edge, flow in zip(edges, edge_flows):
            multiplier = 1.0 + float(self.config.geographic_transfer_cost_scale) * (
                self._edge_distance(edge) / 1000.0
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
        if self.clinic_distance_matrix is None or self.config.geographic_transfer_cost_scale == 0.0:
            return base_total
        distance_surcharge = 0.0
        for edge, flow in zip(edges, np.asarray(edge_flows, dtype=float)):
            distance_surcharge += (
                float(base_cost)
                * float(self.config.geographic_transfer_cost_scale)
                * (self._edge_distance(edge) / 1000.0)
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
    values: np.ndarray, edges: Sequence[Edge], requested_net: np.ndarray
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

    for receiver in receivers:
        for donor in donors:
            if inbound_remaining[receiver] <= 1e-8:
                break
            if outbound_remaining[donor] <= 1e-8 or receiver == donor:
                continue
            if receiver not in adjacency.get(donor, set()):
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
    values: np.ndarray, edges: Sequence[Edge], requested_net: np.ndarray
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

    for receiver in receivers:
        for donor in donors:
            if inbound_remaining[receiver] <= 1e-8:
                break
            if outbound_remaining[donor] <= 1e-8 or receiver == donor:
                continue
            if receiver not in adjacency.get(donor, set()):
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


def _edge_array(edges: Sequence[Edge]) -> np.ndarray:
    return np.asarray(edges, dtype=np.int64).reshape((-1, 2))
