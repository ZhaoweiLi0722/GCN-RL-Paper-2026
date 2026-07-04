"""Deterministic heuristic baselines for PRM capacity planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv
from src.graph.edges import Edge
from src.rl.action_projection import project_action


@dataclass(frozen=True)
class HeuristicSettings:
    """Policy knobs shared by the deterministic benchmark heuristics."""

    lookahead_periods: int = 0
    allow_sharing: bool = True
    local_order_up_to_multiplier: float = 1.0


class CapacityHeuristicPolicy:
    """Base policy for MYO, ISO, MDL-1, and MDL-2 benchmarks.

    The policy emits the manuscript-aligned facility-net action layout
    ``(w, e, q, p)`` for each facility:

    - ``w``: net specimen transfer request.
    - ``e``: net reagent transfer request.
    - ``q``: net idle-bioreactor transfer request.
    - ``p``: reagent purchase request.

    Positive transfer values request inbound flow to a facility and negative
    values request outbound flow. The environment applies edge feasibility and
    inventory/capacity clipping.
    """

    algorithm = "heuristic"

    def __init__(self, state_dim: int | None = None, action_dim: int | None = None, config: dict[str, Any] | None = None):
        del state_dim, action_dim
        config = config or {}
        self.settings = HeuristicSettings(
            lookahead_periods=int(config.get("lookahead_periods", self.default_lookahead_periods())),
            allow_sharing=bool(config.get("allow_sharing", self.default_allow_sharing())),
            local_order_up_to_multiplier=float(config.get("local_order_up_to_multiplier", 1.0)),
        )

    def default_lookahead_periods(self) -> int:
        return 0

    def default_allow_sharing(self) -> bool:
        return True

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = False, env: CapacityPlanningEnv | None = None) -> np.ndarray:
        del state, explore
        if env is None:
            raise ValueError("Heuristic policies require the current environment via env=...")
        if env.config.action_mode != "facility_net":
            raise ValueError("Heuristic policies currently require action_mode='facility_net'")

        action = self._facility_net_action(env)
        return project_action(action, env_state=env, action_space_info=env.action_size).action

    def observe(self, *args, **kwargs) -> None:
        return None

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{self.algorithm}\n")

    def load_actor(self, path: str | Path) -> None:
        return None

    def _facility_net_action(self, env: CapacityPlanningEnv) -> np.ndarray:
        n = env.config.num_facilities
        production = np.minimum.reduce((env.specimens, env.bioreactors[:, 0], env.reagents))
        next_specimens = env.specimens - production + env.demand
        next_reagents = env.reagents - production
        next_idle_bioreactors = env.bioreactors[:, 0] - production + env.bioreactors[:, 1]

        lookahead_demand = self.settings.lookahead_periods * env.demand_rates
        target_workload = np.maximum(next_specimens, 0.0) + lookahead_demand
        target_workload = target_workload * self.settings.local_order_up_to_multiplier

        replenishment = np.clip(
            target_workload - next_reagents,
            0.0,
            env.max_reagent_replenishment,
        )
        replenishment = replenishment * env.supplier_available
        estimated_reagents = next_reagents + replenishment

        if self.settings.allow_sharing:
            reagent_net = _balance_shortage_surplus(
                shortage=np.maximum(target_workload - estimated_reagents, 0.0),
                surplus=np.maximum(estimated_reagents - target_workload, 0.0),
                edges=env.resource_edges,
                max_abs=float(env.config.max_reagent_transfer),
            )
            capacity_net = _balance_shortage_surplus(
                shortage=np.maximum(target_workload - next_idle_bioreactors, 0.0),
                surplus=np.maximum(next_idle_bioreactors - target_workload, 0.0),
                edges=env.capacity_edges,
                max_abs=float(env.config.max_bioreactor_transfer),
            )
            spare_processing = np.maximum(np.minimum(estimated_reagents, next_idle_bioreactors) - next_specimens, 0.0)
            excess_specimens = np.maximum(next_specimens - np.minimum(estimated_reagents, next_idle_bioreactors), 0.0)
            specimen_net = _balance_shortage_surplus(
                shortage=spare_processing,
                surplus=excess_specimens,
                edges=env.specimen_edges,
                max_abs=float(env.config.max_specimen_transfer),
            )
        else:
            specimen_net = np.zeros(n, dtype=float)
            reagent_net = np.zeros(n, dtype=float)
            capacity_net = np.zeros(n, dtype=float)

        action = np.zeros(env.action_size, dtype=np.float32)
        action[:n] = _normalize_signed(specimen_net, float(env.config.max_specimen_transfer))
        action[n : 2 * n] = _normalize_signed(reagent_net, float(env.config.max_reagent_transfer))
        action[2 * n : 3 * n] = _normalize_signed(capacity_net, float(env.config.max_bioreactor_transfer))
        action[3 * n : 4 * n] = _normalize_replenishment(replenishment, env.max_reagent_replenishment)
        return action


class MyopicPolicy(CapacityHeuristicPolicy):
    """MYO: current-period balancing with network sharing enabled."""

    algorithm = "myo"


class IsolatedPolicy(CapacityHeuristicPolicy):
    """ISO: local replenishment only, with all sharing actions disabled."""

    algorithm = "iso"

    def default_lookahead_periods(self) -> int:
        return 1

    def default_allow_sharing(self) -> bool:
        return False


class MeanDemandLookahead1Policy(CapacityHeuristicPolicy):
    """MDL-1: one-period mean-demand lookahead with network sharing."""

    algorithm = "mdl1"

    def default_lookahead_periods(self) -> int:
        return 1


class MeanDemandLookahead2Policy(CapacityHeuristicPolicy):
    """MDL-2: two-period mean-demand lookahead with network sharing."""

    algorithm = "mdl2"

    def default_lookahead_periods(self) -> int:
        return 2


HEURISTIC_POLICIES = {
    "myo": MyopicPolicy,
    "iso": IsolatedPolicy,
    "mdl1": MeanDemandLookahead1Policy,
    "mdl2": MeanDemandLookahead2Policy,
}


def available_heuristics() -> tuple[str, ...]:
    return tuple(HEURISTIC_POLICIES)


def get_heuristic_class(algorithm: str):
    try:
        return HEURISTIC_POLICIES[algorithm]
    except KeyError as exc:
        raise ValueError(f"Unsupported heuristic algorithm: {algorithm}") from exc


def _balance_shortage_surplus(
    shortage: np.ndarray,
    surplus: np.ndarray,
    edges: Sequence[Edge],
    max_abs: float,
) -> np.ndarray:
    net = np.zeros_like(shortage, dtype=float)
    if not edges or max_abs <= 0.0:
        return net

    shortage_remaining = np.asarray(shortage, dtype=float).copy()
    surplus_remaining = np.asarray(surplus, dtype=float).copy()
    adjacency = _adjacency(edges)
    receivers = list(np.argsort(-shortage_remaining))

    for receiver in receivers:
        if shortage_remaining[receiver] <= 1e-8:
            continue
        donors = sorted(
            adjacency.get(int(receiver), ()),
            key=lambda node: surplus_remaining[node],
            reverse=True,
        )
        for donor in donors:
            if shortage_remaining[receiver] <= 1e-8:
                break
            if surplus_remaining[donor] <= 1e-8:
                continue
            flow = min(shortage_remaining[receiver], surplus_remaining[donor], max_abs)
            if flow <= 1e-8:
                continue
            net[receiver] += flow
            net[donor] -= flow
            shortage_remaining[receiver] -= flow
            surplus_remaining[donor] -= flow

    return np.clip(net, -max_abs, max_abs)


def _adjacency(edges: Sequence[Edge]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for i, j in edges:
        adjacency.setdefault(int(i), set()).add(int(j))
        adjacency.setdefault(int(j), set()).add(int(i))
    return adjacency


def _normalize_signed(values: np.ndarray, max_abs: float) -> np.ndarray:
    if max_abs <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / max_abs, -1.0, 1.0).astype(np.float32)


def _normalize_replenishment(values: np.ndarray, max_replenishment: np.ndarray) -> np.ndarray:
    normalized = np.full_like(values, -1.0, dtype=float)
    positive = max_replenishment > 0
    normalized[positive] = 2.0 * np.clip(values[positive] / max_replenishment[positive], 0.0, 1.0) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)

