"""Deterministic heuristic baselines for PRM capacity planning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv
from src.graph.edges import Edge, complete_undirected_edges, ring_edges
from src.graph.geography import geographic_knn_edges, normalize_coordinates
from src.rl.action_projection import project_action


@dataclass(frozen=True)
class HeuristicSettings:
    """Policy knobs shared by the deterministic benchmark heuristics."""

    lookahead_periods: int = 0
    allow_sharing: bool = True
    local_order_up_to_multiplier: float = 1.0
    use_demand_forecast: bool = False
    use_patient_priority: bool = False
    patient_priority_weight: float = 0.5
    near_expiry_weight: float = 1.0


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
            demand_rates=getattr(env, "demand_rate_estimates", env.demand_rates),
            max_reagent_replenishment=env.max_reagent_replenishment,
            max_specimen_transfer=float(env.config.max_specimen_transfer),
            max_bioreactor_transfer=float(env.config.max_bioreactor_transfer),
            max_reagent_transfer=float(env.config.max_reagent_transfer),
            specimen_edges=env.specimen_edges,
            capacity_edges=env.capacity_edges,
            resource_edges=env.resource_edges,
            settings=self.settings,
            patient_priority=patient_priority_from_env(env, self.settings),
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


class PatientPriorityMyopicPolicy(CapacityHeuristicPolicy):
    """P-MYO: myopic balancing with a bounded patient-risk workload uplift.

    The priority signal is converted into additional target workload before
    the usual sharing/replenishment calculation, so the policy reacts to
    critical and near-expiry patients without blindly over-ordering everywhere.
    """

    algorithm = "pmyo"

    def __init__(self, state_dim: int | None = None, action_dim: int | None = None, config: dict[str, Any] | None = None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.settings = HeuristicSettings(
            lookahead_periods=int(config.get("lookahead_periods", self.default_lookahead_periods())),
            allow_sharing=bool(config.get("allow_sharing", self.default_allow_sharing())),
            local_order_up_to_multiplier=float(config.get("local_order_up_to_multiplier", 1.0)),
            use_demand_forecast=bool(config.get("use_demand_forecast", self.default_use_demand_forecast())),
            use_patient_priority=True,
            patient_priority_weight=float(config.get("patient_priority_weight", 0.5)),
            near_expiry_weight=float(config.get("near_expiry_weight", 1.0)),
        )


class ShieldedPatientPriorityMyopicPolicy(PatientPriorityMyopicPolicy):
    """P-MYO plus online patient-facing rollout shield.

    This is a diagnostic post-decision policy, not a learned controller: it
    starts from pMYO, evaluates a small set of candidate corrections on a copied
    environment, and deploys a correction only when the short lookahead is
    service-safe and improves the patient-facing scalar score.
    """

    algorithm = "pmyo_shield"

    def default_anchor_policy(self) -> str:
        return "pmyo"

    def __init__(self, state_dim=None, action_dim=None, config=None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.anchor_policy_name = str(config.get("anchor_policy", self.default_anchor_policy()))
        self.shield_lookahead = int(config.get("shield_lookahead", 3))
        self.shield_epsilons = tuple(float(value) for value in config.get("shield_epsilons", (0.005, 0.01)))
        self.min_service_level_delta = float(config.get("min_service_level_delta", 0.0))
        self.min_score_improvement = float(config.get("min_score_improvement", 0.0))
        self.service_level_weight = float(config.get("service_level_weight", 100_000_000.0))
        self.eligibility_rate_weight = float(config.get("eligibility_rate_weight", 100_000_000.0))
        self.at_risk_unserved_weight = float(config.get("at_risk_unserved_weight", 50_000.0))
        self.patients_lost_weight = float(config.get("patients_lost_weight", 500_000.0))
        self.candidate_groups = tuple(
            str(group)
            for group in config.get(
                "candidate_groups",
                (
                    "replenishment_patient_risk_pressure",
                    "replenishment_positive_pressure",
                    "reagent_transfer",
                    "capacity_transfer",
                    "combined_transfer",
                ),
            )
        )
        self._anchor_policy = self._make_anchor_policy(state_dim, action_dim, config)

    def _make_anchor_policy(self, state_dim, action_dim, config):
        policy_map = {
            "myo": MyopicPolicy,
            "iso": IsolatedPolicy,
            "mdl1": MeanDemandLookahead1Policy,
            "mdl2": MeanDemandLookahead2Policy,
            "fmyo": ForecastMyopicPolicy,
            "umyo": UrgencyAwareMyopicPolicy,
            "pmyo": PatientPriorityMyopicPolicy,
        }
        try:
            policy_class = policy_map[self.anchor_policy_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported shield anchor policy: {self.anchor_policy_name}") from exc
        return policy_class(state_dim, action_dim, config)

    def select_action(self, state: np.ndarray, explore: bool = False, env: CapacityPlanningEnv | None = None) -> np.ndarray:
        if env is None:
            raise ValueError("Shielded pMYO requires the current environment via env=...")
        anchor_action = self._anchor_policy.select_action(state, explore=False, env=env)
        candidates = shield_candidate_actions(
            anchor_action,
            env,
            epsilons=self.shield_epsilons,
            candidate_groups=self.candidate_groups,
        )
        if len(candidates) <= 1 or self.shield_lookahead <= 0:
            return anchor_action
        candidate_metrics = [
            shield_rollout_metrics(copy.deepcopy(env), self._anchor_policy, action, horizon=self.shield_lookahead)
            for action in candidates
        ]
        best_index = select_shield_candidate_index(self, candidate_metrics)
        return project_action(candidates[best_index], env_state=env, action_space_info=env.action_size).action


class ShieldedMeanDemandLookahead2Policy(ShieldedPatientPriorityMyopicPolicy):
    """MDL-2 plus the same rollout shield used by pMYO-shield."""

    algorithm = "mdl2_shield"

    def default_anchor_policy(self) -> str:
        return "mdl2"


HEURISTIC_POLICIES = {
    "myo": MyopicPolicy,
    "iso": IsolatedPolicy,
    "mdl1": MeanDemandLookahead1Policy,
    "mdl2": MeanDemandLookahead2Policy,
    "fmyo": ForecastMyopicPolicy,
    "umyo": UrgencyAwareMyopicPolicy,
    "pmyo": PatientPriorityMyopicPolicy,
    "mdl2_shield": ShieldedMeanDemandLookahead2Policy,
    "pmyo_shield": ShieldedPatientPriorityMyopicPolicy,
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

    base_width = n * features_per_facility
    state_vector = np.asarray(state, dtype=np.float32)
    if state_vector.size < base_width:
        raise ValueError(
            f"facility_net_action_from_state expected at least {base_width} state values, "
            f"got {state_vector.size}"
        )
    state_array = state_vector[:base_width].reshape(n, features_per_facility)
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
    patient_priority = patient_priority_from_state(state_vector, env_config, settings)

    demand_rate_estimates = env_config.get("demand_rate_estimates")
    if demand_rate_estimates is None:
        demand_rate_estimates = env_config.get("demand_rates", 0.0)
    return facility_net_action_from_arrays(
        demand=demand,
        specimens=specimens,
        reagents=reagents,
        bioreactors=bioreactors,
        supplier_available=supplier_available,
        demand_forecast=demand_forecast,
        demand_rates=_config_vector(demand_rate_estimates, n, "demand_rate_estimates"),
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
        patient_priority=patient_priority,
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
    patient_priority: np.ndarray | None = None,
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
    if settings.use_patient_priority and patient_priority is not None:
        priority = np.asarray(patient_priority, dtype=float)
        if priority.shape != (n,):
            raise ValueError(f"patient_priority must have length {n}; got shape {priority.shape}")
        target_workload = target_workload + settings.patient_priority_weight * np.maximum(priority, 0.0)

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
    clinic_coordinates = normalize_coordinates(env_config.get("clinic_coordinates"), num_facilities)
    geographic_edges = (
        geographic_knn_edges(
            clinic_coordinates,
            k=int(env_config.get("geographic_neighbor_k", 3)),
        )
        if clinic_coordinates
        else ()
    )
    if action_mode == "facility_net" and key in ("specimen_edges", "resource_edges"):
        default_edges = geographic_edges or ring_edges(num_facilities)
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


def patient_priority_from_env(
    env: CapacityPlanningEnv,
    settings: HeuristicSettings,
) -> np.ndarray | None:
    """Return a bounded priority workload signal for patient-condition envs."""

    if not settings.use_patient_priority or not hasattr(env, "at_risk_counts"):
        return None
    at_risk = np.asarray(env.at_risk_counts(), dtype=float)
    near_expiry = np.asarray(env.near_expiry_counts(), dtype=float)
    waiting = np.maximum(np.asarray(env.waiting_counts(), dtype=float), 1.0)
    priority = at_risk + settings.near_expiry_weight * near_expiry
    return np.clip(priority, 0.0, waiting)


def patient_priority_from_state(
    state: Sequence[float],
    env_config: dict[str, Any],
    settings: HeuristicSettings,
) -> np.ndarray | None:
    """Read the patient summary tail and build the same priority signal for replay states."""

    if not settings.use_patient_priority or env_config.get("env_type") != "patient_condition":
        return None
    n = int(env_config.get("num_facilities", 0))
    if n <= 0:
        return None
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
    summary_edges = tuple(env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97)))
    summary_width = 3 + len(summary_edges) + 1
    base_width = n * features_per_facility
    state_vector = np.asarray(state, dtype=np.float32)
    expected_width = base_width + n * summary_width
    if state_vector.size < expected_width:
        return None
    summary = state_vector[base_width:expected_width].reshape(n, summary_width)
    waiting = np.maximum(summary[:, 0], 1.0)
    near_expiry = summary[:, 2]
    histogram = summary[:, 3:]
    patient_cfg = dict(env_config.get("patient", {}))
    risk_cutoff = float(patient_cfg.get("eligibility_threshold", 0.80)) + float(
        env_config.get("urgency_margin", 0.1)
    )
    bucket_upper_bounds = np.asarray(tuple(summary_edges) + (float("inf"),), dtype=float)
    at_risk_mask = bucket_upper_bounds <= risk_cutoff + 1e-12
    if not np.any(at_risk_mask) and histogram.shape[1] > 0:
        at_risk_mask[0] = True
    at_risk = histogram[:, at_risk_mask].sum(axis=1)
    priority = at_risk + settings.near_expiry_weight * near_expiry
    return np.clip(priority, 0.0, waiting)


def shield_candidate_actions(
    anchor_action: np.ndarray,
    env: CapacityPlanningEnv,
    *,
    epsilons: Sequence[float],
    candidate_groups: Sequence[str],
) -> list[np.ndarray]:
    """Small patient-facing correction set around a pMYO anchor action."""

    action = np.asarray(anchor_action, dtype=np.float32)
    n = int(env.config.num_facilities)
    groups = set(candidate_groups)
    candidates = [action.copy()]
    _pending_specimens, pending_reagents, pending_capacity = _pending_transfer_vectors(env, n)
    resource_pressure = (
        np.asarray(env.demand, dtype=float)
        + 0.25 * np.asarray(getattr(env, "demand_forecast", env.demand), dtype=float)
        + np.asarray(env.specimens, dtype=float)
        - np.asarray(env.reagents, dtype=float)
        - pending_reagents
    )
    capacity_pressure = (
        np.asarray(env.demand, dtype=float)
        + 0.25 * np.asarray(getattr(env, "demand_forecast", env.demand), dtype=float)
        + np.asarray(env.specimens, dtype=float)
        - np.asarray(env.bioreactors[:, 0], dtype=float)
        - pending_capacity
    )
    patient_risk = _env_vector(env, "at_risk_counts", n) + _env_vector(env, "near_expiry_counts", n)
    resource_pattern = _centered_unit_pattern(resource_pressure)
    capacity_pattern = _centered_unit_pattern(capacity_pressure)
    positive_resource_pattern = np.maximum(resource_pattern, 0.0)
    patient_risk_pattern = _positive_unit_pattern(patient_risk)
    patient_risk_pressure_pattern = _positive_unit_pattern(
        np.maximum(patient_risk, 0.0) * (1.0 + np.maximum(resource_pressure, 0.0))
    )

    for epsilon in epsilons:
        epsilon = float(epsilon)
        if "replenishment_patient_risk" in groups:
            candidate = action.copy()
            candidate[3 * n : 4 * n] = np.clip(
                candidate[3 * n : 4 * n] + epsilon * patient_risk_pattern,
                -1.0,
                1.0,
            )
            candidates.append(candidate)
        if "replenishment_patient_risk_pressure" in groups:
            candidate = action.copy()
            candidate[3 * n : 4 * n] = np.clip(
                candidate[3 * n : 4 * n] + epsilon * patient_risk_pressure_pattern,
                -1.0,
                1.0,
            )
            candidates.append(candidate)
        if "replenishment_positive_pressure" in groups:
            candidate = action.copy()
            candidate[3 * n : 4 * n] = np.clip(
                candidate[3 * n : 4 * n] + epsilon * positive_resource_pattern,
                -1.0,
                1.0,
            )
            candidates.append(candidate)
        for sign in (-1.0, 1.0):
            if "reagent_transfer" in groups:
                candidate = action.copy()
                candidate[n : 2 * n] = np.clip(
                    candidate[n : 2 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                candidates.append(candidate)
            if "capacity_transfer" in groups:
                candidate = action.copy()
                candidate[2 * n : 3 * n] = np.clip(
                    candidate[2 * n : 3 * n] + sign * epsilon * capacity_pattern,
                    -1.0,
                    1.0,
                )
                candidates.append(candidate)
            if "combined_transfer" in groups:
                candidate = action.copy()
                candidate[n : 2 * n] = np.clip(
                    candidate[n : 2 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                candidate[2 * n : 3 * n] = np.clip(
                    candidate[2 * n : 3 * n] + sign * epsilon * capacity_pattern,
                    -1.0,
                    1.0,
                )
                candidates.append(candidate)
    return [candidate.astype(np.float32) for candidate in candidates]


def shield_rollout_metrics(
    env: CapacityPlanningEnv,
    anchor_policy: CapacityHeuristicPolicy,
    action: np.ndarray,
    *,
    horizon: int,
) -> dict[str, float]:
    from src.rl.experiment import EpisodeMetrics

    state, _reward, done, info = env.step(action)
    metrics = EpisodeMetrics()
    metrics.update(info)
    steps = 1
    while not done and steps < max(int(horizon), 1):
        followup = anchor_policy.select_action(state, explore=False, env=env)
        state, _reward, done, info = env.step(followup)
        metrics.update(info)
        steps += 1
    return {
        "total_cost": float(metrics.total_cost),
        "service_level": float(metrics.service_level),
        "eligibility_rate": float(metrics.eligibility_rate_mean)
        if metrics.has_patient_metrics
        else 0.0,
        "at_risk_unserved": float(metrics.at_risk_unserved),
        "patients_lost": float(metrics.patients_lost),
    }


def shield_metric_score(policy: ShieldedPatientPriorityMyopicPolicy, metrics: dict[str, float]) -> float:
    return (
        float(metrics["total_cost"])
        - policy.service_level_weight * float(metrics.get("service_level", 0.0))
        - policy.eligibility_rate_weight * float(metrics.get("eligibility_rate", 0.0))
        + policy.at_risk_unserved_weight * float(metrics.get("at_risk_unserved", 0.0))
        + policy.patients_lost_weight * float(metrics.get("patients_lost", 0.0))
    )


def select_shield_candidate_index(
    policy: ShieldedPatientPriorityMyopicPolicy,
    candidate_metrics: Sequence[dict[str, float]],
) -> int:
    """Return the service-safe candidate with the best shield score."""

    if not candidate_metrics:
        raise ValueError("candidate_metrics must contain at least the anchor candidate")
    anchor_metrics = candidate_metrics[0]
    anchor_score = shield_metric_score(policy, anchor_metrics)
    service_threshold = float(anchor_metrics.get("service_level", 0.0)) + policy.min_service_level_delta
    best_index = 0
    best_score = anchor_score
    for index, metrics in enumerate(candidate_metrics[1:], start=1):
        service_level = float(metrics.get("service_level", 0.0))
        if service_level + 1e-12 < service_threshold:
            continue
        score = shield_metric_score(policy, metrics)
        if score < best_score - policy.min_score_improvement:
            best_index = index
            best_score = score
    return best_index


def _pending_transfer_vectors(env: CapacityPlanningEnv, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pending = getattr(env, "_pending_transfer_arrivals", None)
    if callable(pending):
        vectors = pending()
        if len(vectors) == 3:
            return tuple(np.asarray(vector, dtype=float) for vector in vectors)  # type: ignore[return-value]
    zeros = np.zeros(int(length), dtype=float)
    return (
        _pipeline_pending_vector(env, "specimen_transfer_pipeline", length, zeros),
        _pipeline_pending_vector(env, "reagent_transfer_pipeline", length, zeros),
        _pipeline_pending_vector(env, "capacity_transfer_pipeline", length, zeros),
    )


def _pipeline_pending_vector(
    env: CapacityPlanningEnv,
    name: str,
    length: int,
    default: np.ndarray,
) -> np.ndarray:
    pipeline = getattr(env, name, None)
    if pipeline is None:
        return default.copy()
    array = np.asarray(pipeline, dtype=float)
    if array.ndim != 2 or array.shape[1] != int(length):
        return default.copy()
    return array.sum(axis=0)


def _env_vector(env: CapacityPlanningEnv, name: str, length: int) -> np.ndarray:
    value = getattr(env, name, None)
    if value is None:
        return np.zeros(int(length), dtype=float)
    if callable(value):
        value = value()
    vector = np.asarray(value, dtype=float)
    if vector.shape != (int(length),):
        return np.zeros(int(length), dtype=float)
    return vector


def _centered_unit_pattern(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denominator = max(float(np.max(np.abs(centered))), 1e-6)
    return centered / denominator


def _positive_unit_pattern(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    denominator = max(float(np.max(positive)), 1e-6)
    return positive / denominator


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
