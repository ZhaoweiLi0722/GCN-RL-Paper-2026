"""Configurable PRM capacity-planning simulation environment.

This module migrates the core state-transition ideas from the legacy
two-facility DDPG environment into a small, dependency-light environment that
can be used by future GCN-DDPG, flat-state DDPG, and graph-ablation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.graph.edges import Edge, complete_undirected_edges


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
    specimen_edges: Sequence[Edge] | None = None
    capacity_edges: Sequence[Edge] | None = None
    resource_edges: Sequence[Edge] | None = None
    costs: CostParameters = CostParameters()


class CapacityPlanningEnv:
    """Small NumPy environment for distributed PRM capacity planning.

    State layout per facility:
        demand, waiting specimens, reagent inventory, bioreactor pipeline.

    Action layout:
        reagent replenishment for each facility, then specimen transfers,
        bioreactor-capacity transfers, and reagent transfers for configured
        undirected edge sets. Actions are normalized to [-1, 1].
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

        default_edges = complete_undirected_edges(n)
        self.specimen_edges = _normalize_edges(self.config.specimen_edges, default_edges, n)
        self.capacity_edges = _normalize_edges(self.config.capacity_edges, default_edges, n)
        self.resource_edges = _normalize_edges(self.config.resource_edges, default_edges, n)

        self.observation_size = n * (3 + self.config.production_lead_time)
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
        self.demand = self.rng.poisson(self.demand_rates).astype(float)
        self.specimens = self.initial_specimens.astype(float).copy()
        self.reagents = self.initial_reagents.astype(float).copy()
        self.bioreactors = np.zeros((n, lead_time), dtype=float)
        self.bioreactors[:, 0] = self.initial_idle_bioreactors
        return self.observation()

    def observation(self) -> np.ndarray:
        """Return a flat observation suitable for MLP baselines."""

        rows = []
        for i in range(self.config.num_facilities):
            rows.append(
                np.concatenate(
                    (
                        np.array([self.demand[i], self.specimens[i], self.reagents[i]]),
                        self.bioreactors[i],
                    )
                )
            )
        return np.concatenate(rows).astype(np.float32)

    def graph_observation(self) -> dict[str, np.ndarray]:
        """Return node features and edge sets for future GCN policies."""

        node_features = np.column_stack(
            (
                self.demand,
                self.specimens,
                self.reagents,
                self.bioreactors[:, 0],
                self.bioreactors.sum(axis=1),
            )
        ).astype(np.float32)
        return {
            "node_features": node_features,
            "specimen_edges": _edge_array(self.specimen_edges),
            "capacity_edges": _edge_array(self.capacity_edges),
            "resource_edges": _edge_array(self.resource_edges),
        }

    def noop_action(self) -> np.ndarray:
        """Return an action with no replenishment and no transfers."""

        action = np.zeros(self.action_size, dtype=np.float32)
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
        n = self.config.num_facilities
        costs = self.config.costs

        replenishment = ((normalized[:n] + 1.0) / 2.0) * self.max_reagent_replenishment
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

        next_specimens = self.specimens - production + self.demand
        next_reagents = self.reagents - production + replenishment
        next_bioreactors = np.zeros_like(self.bioreactors)
        next_bioreactors[:, 0] = self.bioreactors[:, 0] - production + self.bioreactors[:, 1]
        if self.config.production_lead_time > 2:
            next_bioreactors[:, 1:-1] = self.bioreactors[:, 2:]
        next_bioreactors[:, -1] = production

        specimen_transfers = _apply_transfers(next_specimens, self.specimen_edges, specimen_requests)
        capacity_transfers = _apply_transfers(
            next_bioreactors[:, 0], self.capacity_edges, capacity_requests
        )
        reagent_transfers = _apply_transfers(next_reagents, self.resource_edges, reagent_requests)

        self.specimens = np.clip(next_specimens, 0.0, self.max_specimens)
        self.reagents = np.clip(next_reagents, 0.0, self.max_reagents)
        next_bioreactors[:, 0] = np.clip(next_bioreactors[:, 0], 0.0, self.max_idle_bioreactors)
        self.bioreactors = np.maximum(next_bioreactors, 0.0)
        self.t += 1
        done = self.t >= self.config.episode_horizon
        self.demand = self.rng.poisson(self.demand_rates).astype(float)

        cost = (
            costs.reagent_purchase * float(replenishment.sum())
            + costs.reagent_holding * float(idle_reagents.sum())
            + costs.reagent_shortage * float(under_reagents.sum())
            + costs.bioreactor_holding * float(idle_bioreactor_counts.sum())
            + costs.bioreactor_shortage * float(under_bioreactors.sum())
            + costs.specimen_transfer * float(np.abs(specimen_transfers).sum())
            + costs.bioreactor_transfer * float(np.abs(capacity_transfers).sum())
            + costs.reagent_transfer * float(np.abs(reagent_transfers).sum())
        )

        info: dict[str, np.ndarray | float] = {
            "cost": cost,
            "production": production.copy(),
            "replenishment": replenishment.copy(),
            "specimen_transfers": specimen_transfers.copy(),
            "capacity_transfers": capacity_transfers.copy(),
            "reagent_transfers": reagent_transfers.copy(),
            "under_reagents": under_reagents.copy(),
            "under_bioreactors": under_bioreactors.copy(),
        }
        return self.observation(), -cost, done, info

    def _validate_config(self) -> None:
        if self.config.num_facilities < 1:
            raise ValueError("num_facilities must be positive")
        if self.config.production_lead_time < 2:
            raise ValueError("production_lead_time must be at least 2")
        if self.config.episode_horizon < 1:
            raise ValueError("episode_horizon must be positive")


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


def _as_vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
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


def _edge_array(edges: Sequence[Edge]) -> np.ndarray:
    return np.asarray(edges, dtype=np.int64).reshape((-1, 2))
