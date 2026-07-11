"""Deterministic heuristic baselines for PRM capacity planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv
from src.graph.edges import Edge, complete_undirected_edges, ring_edges
from src.rl.action_projection import project_action


@dataclass(frozen=True)
class HeuristicSettings:
    """Policy knobs shared by the deterministic benchmark heuristics."""

    lookahead_periods: int = 0
    allow_sharing: bool = True
    local_order_up_to_multiplier: float = 1.0
    use_demand_forecast: bool = False


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
            use_demand_forecast=bool(config.get("use_demand_forecast", self.default_use_demand_forecast())),
        )

    def default_lookahead_periods(self) -> int:
        return 0

    def default_allow_sharing(self) -> bool:
        return True

    def default_use_demand_forecast(self) -> bool:
        return False

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
        return facility_net_action_from_arrays(
            demand=env.demand,
            specimens=env.specimens,
            reagents=env.reagents,
            bioreactors=env.bioreactors,
            supplier_available=env.supplier_available,
            demand_forecast=getattr(env, "demand_forecast", None),
            demand_rates=env.demand_rates,
            max_reagent_replenishment=env.max_reagent_replenishment,
            max_specimen_transfer=float(env.config.max_specimen_transfer),
            max_bioreactor_transfer=float(env.config.max_bioreactor_transfer),
            max_reagent_transfer=float(env.config.max_reagent_transfer),
            specimen_edges=env.specimen_edges,
            capacity_edges=env.capacity_edges,
            resource_edges=env.resource_edges,
            settings=self.settings,
        )


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


class ForecastMyopicPolicy(CapacityHeuristicPolicy):
    """F-MYO: current-period balancing with patient/demand forecast lookahead."""

    algorithm = "fmyo"

    def default_use_demand_forecast(self) -> bool:
        return True


class UrgencyAwareMyopicPolicy(MyopicPolicy):
    """uMYO: myopic balancing, then surge replenishment and inbound capacity
    toward clinics with many at-risk / near-expiry patients.

    Condition-aware baseline: it reacts to deteriorating patients, unlike the
    condition-blind heuristics. On the base (non-patient) environment it degrades
    gracefully to plain myopic behaviour.
    """

    algorithm = "umyo"

    def __init__(self, state_dim=None, action_dim=None, config=None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.urgency_surge = float(config.get("urgency_surge", 1.0))

    def _facility_net_action(self, env: CapacityPlanningEnv) -> np.ndarray:
        action = super()._facility_net_action(env)
        if not hasattr(env, "at_risk_counts"):
            return action  # base env: no patient signal -> plain myopic
        n = env.config.num_facilities
        waiting = env.waiting_counts()
        urgency = np.clip(
            (env.at_risk_counts() + env.near_expiry_counts()) / np.maximum(waiting, 1.0),
            0.0,
            1.0,
        )
        surge = self.urgency_surge * urgency
        action = action.copy()
        action[2 * n : 3 * n] = np.clip(action[2 * n : 3 * n] + surge, -1.0, 1.0)  # inbound capacity (q)
        action[3 * n : 4 * n] = np.clip(action[3 * n : 4 * n] + surge, -1.0, 1.0)  # replenishment (p)
        return action


HEURISTIC_POLICIES = {
    "myo": MyopicPolicy,
    "iso": IsolatedPolicy,
    "mdl1": MeanDemandLookahead1Policy,
    "mdl2": MeanDemandLookahead2Policy,
    "fmyo": ForecastMyopicPolicy,
    "umyo": UrgencyAwareMyopicPolicy,
}


def available_heuristics() -> tuple[str, ...]:
    return tuple(HEURISTIC_POLICIES)


def get_heuristic_class(algorithm: str):
    try:
        return HEURISTIC_POLICIES[algorithm]
    except KeyError as exc:
        raise ValueError(f"Unsupported heuristic algorithm: {algorithm}") from exc


def heuristic_settings_for_policy(
    algorithm: str,
    config: dict[str, Any] | None = None,
) -> HeuristicSettings:
    """Return default or configured settings for a named heuristic policy."""

    policy = get_heuristic_class(algorithm)(config=config or {})
    return policy.settings


def facility_net_action_from_state(
    state: Sequence[float],
    env_config: dict[str, Any],
    *,
    settings: HeuristicSettings,
) -> np.ndarray:
    """Compute a facility-net heuristic action directly from a flat state.

    This mirrors :meth:`CapacityHeuristicPolicy._facility_net_action` without
    requiring a live environment object, so learned residual policies can use a
    heuristic anchor inside actor/target updates on replay-buffer states.
    """

    n = int(env_config.get("num_facilities", 0))
    if n <= 0:
        raise ValueError("env_config['num_facilities'] must be positive")
    lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier = bool(env_config.get("include_supplier_state", False))
    include_forecast = bool(env_config.get("include_demand_forecast_state", False))
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    features_per_facility = (
        3
        + lead_time
        + int(include_supplier)
        + int(include_forecast)
        + 3 * int(include_transfer_pipeline)
    )

    state_array = np.asarray(state, dtype=np.float32).reshape(n, features_per_facility)
    demand = state_array[:, 0]
    specimens = state_array[:, 1]
    reagents = state_array[:, 2]
    bioreactors = state_array[:, 3 : 3 + lead_time]
    if include_supplier:
        supplier_available = state_array[:, 3 + lead_time]
    else:
        supplier_available = np.ones(n, dtype=np.float32)
    if include_forecast:
        forecast_start = 3 + lead_time + int(include_supplier)
        demand_forecast = state_array[:, forecast_start]
    else:
        demand_forecast = None

    return facility_net_action_from_arrays(
        demand=demand,
        specimens=specimens,
        reagents=reagents,
        bioreactors=bioreactors,
        supplier_available=supplier_available,
        demand_forecast=demand_forecast,
        demand_rates=_config_vector(env_config.get("demand_rates", 0.0), n, "demand_rates"),
        max_reagent_replenishment=_config_vector(
            env_config.get("max_reagent_replenishment", 0.0),
            n,
            "max_reagent_replenishment",
        ),
        max_specimen_transfer=float(env_config.get("max_specimen_transfer", 0.0)),
        max_bioreactor_transfer=float(env_config.get("max_bioreactor_transfer", 0.0)),
        max_reagent_transfer=float(env_config.get("max_reagent_transfer", 0.0)),
        specimen_edges=_resolve_facility_edges(env_config, "specimen_edges", n),
        capacity_edges=_resolve_facility_edges(env_config, "capacity_edges", n),
        resource_edges=_resolve_facility_edges(env_config, "resource_edges", n),
        settings=settings,
    )


def facility_net_action_from_arrays(
    *,
    demand: np.ndarray,
    specimens: np.ndarray,
    reagents: np.ndarray,
    bioreactors: np.ndarray,
    supplier_available: np.ndarray,
    demand_forecast: np.ndarray | None,
    demand_rates: np.ndarray,
    max_reagent_replenishment: np.ndarray,
    max_specimen_transfer: float,
    max_bioreactor_transfer: float,
    max_reagent_transfer: float,
    specimen_edges: Sequence[Edge],
    capacity_edges: Sequence[Edge],
    resource_edges: Sequence[Edge],
    settings: HeuristicSettings,
) -> np.ndarray:
    """Compute normalized ``(w, e, q, p)`` facility-net actions."""

    n = int(np.asarray(demand).shape[0])
    bioreactors = np.asarray(bioreactors, dtype=float)
    idle_bioreactors = bioreactors[:, 0]
    next_stage_bioreactors = bioreactors[:, 1] if bioreactors.shape[1] > 1 else np.zeros(n)
    production = np.minimum.reduce((specimens, idle_bioreactors, reagents))
    next_specimens = specimens - production + demand
    next_reagents = reagents - production
    next_idle_bioreactors = idle_bioreactors - production + next_stage_bioreactors

    if settings.use_demand_forecast and demand_forecast is not None:
        lookahead_demand = np.asarray(demand_forecast, dtype=float)
    else:
        lookahead_demand = settings.lookahead_periods * demand_rates
    target_workload = np.maximum(next_specimens, 0.0) + lookahead_demand
    target_workload = target_workload * settings.local_order_up_to_multiplier

    replenishment = np.clip(
        target_workload - next_reagents,
        0.0,
        max_reagent_replenishment,
    )
    replenishment = replenishment * supplier_available
    estimated_reagents = next_reagents + replenishment

    if settings.allow_sharing:
        reagent_net = _balance_shortage_surplus(
            shortage=np.maximum(target_workload - estimated_reagents, 0.0),
            surplus=np.maximum(estimated_reagents - target_workload, 0.0),
            edges=resource_edges,
            max_abs=max_reagent_transfer,
        )
        capacity_net = _balance_shortage_surplus(
            shortage=np.maximum(target_workload - next_idle_bioreactors, 0.0),
            surplus=np.maximum(next_idle_bioreactors - target_workload, 0.0),
            edges=capacity_edges,
            max_abs=max_bioreactor_transfer,
        )
        spare_processing = np.maximum(
            np.minimum(estimated_reagents, next_idle_bioreactors) - next_specimens,
            0.0,
        )
        excess_specimens = np.maximum(
            next_specimens - np.minimum(estimated_reagents, next_idle_bioreactors),
            0.0,
        )
        specimen_net = _balance_shortage_surplus(
            shortage=spare_processing,
            surplus=excess_specimens,
            edges=specimen_edges,
            max_abs=max_specimen_transfer,
        )
    else:
        specimen_net = np.zeros(n, dtype=float)
        reagent_net = np.zeros(n, dtype=float)
        capacity_net = np.zeros(n, dtype=float)

    action = np.zeros(4 * n, dtype=np.float32)
    action[:n] = _normalize_signed(specimen_net, max_specimen_transfer)
    action[n : 2 * n] = _normalize_signed(reagent_net, max_reagent_transfer)
    action[2 * n : 3 * n] = _normalize_signed(capacity_net, max_bioreactor_transfer)
    action[3 * n : 4 * n] = _normalize_replenishment(replenishment, max_reagent_replenishment)
    return action


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


def _resolve_facility_edges(
    env_config: dict[str, Any],
    key: str,
    num_facilities: int,
) -> tuple[Edge, ...]:
    action_mode = str(env_config.get("action_mode", "edge_transfer"))
    if action_mode == "facility_net" and key in ("specimen_edges", "resource_edges"):
        default_edges = ring_edges(num_facilities)
    else:
        default_edges = complete_undirected_edges(num_facilities)
    configured = env_config.get(key)
    edges = default_edges if configured is None else tuple(configured)
    normalized = []
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            continue
        if i < 0 or j < 0 or i >= num_facilities or j >= num_facilities:
            raise ValueError(f"{key} edge {(i, j)} is outside {num_facilities} facilities")
        normalized.append((min(i, j), max(i, j)))
    return tuple(dict.fromkeys(normalized))


def _config_vector(values: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == ():
        return np.full(length, float(array), dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must have length {length}; got shape {array.shape}")
    return array.astype(float)


def _normalize_signed(values: np.ndarray, max_abs: float) -> np.ndarray:
    if max_abs <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / max_abs, -1.0, 1.0).astype(np.float32)


def _normalize_replenishment(values: np.ndarray, max_replenishment: np.ndarray) -> np.ndarray:
    normalized = np.full_like(values, -1.0, dtype=float)
    positive = max_replenishment > 0
    normalized[positive] = 2.0 * np.clip(values[positive] / max_replenishment[positive], 0.0, 1.0) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)
