"""Deterministic heuristic baselines for PRM capacity planning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv
from src.graph.edges import Edge, complete_undirected_edges, ring_edges
from src.graph.geography import geographic_knn_edges, normalize_coordinates
from src.rl.action_projection import project_action
from src.rl.preprocessing import facility_state_width


@dataclass(frozen=True)
class HeuristicSettings:
    """Policy knobs shared by the deterministic benchmark heuristics."""

    lookahead_periods: int = 0
    allow_sharing: bool = True
    local_order_up_to_multiplier: float = 1.0
    use_demand_forecast: bool = False
    use_demand_history: bool = False
    demand_history_weight: float = 1.0
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
            use_demand_history=bool(
                config.get(
                    "use_demand_history",
                    self.default_use_demand_history(),
                )
            ),
            demand_history_weight=float(
                config.get(
                    "demand_history_weight",
                    self.default_demand_history_weight(),
                )
            ),
        )

    def default_lookahead_periods(self) -> int:
        return 0

    def default_allow_sharing(self) -> bool:
        return True

    def default_use_demand_forecast(self) -> bool:
        return False

    def default_use_demand_history(self) -> bool:
        return False

    def default_demand_history_weight(self) -> float:
        return 1.0

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = False, env: CapacityPlanningEnv | None = None) -> np.ndarray:
        del state, explore
        if env is None:
            raise ValueError("Heuristic policies require the current environment via env=...")
        if env.config.action_mode != "facility_net":
            raise ValueError("Heuristic policies currently require action_mode='facility_net'")

        action = self._facility_net_action(env)
        if getattr(env.config, "enable_overtime_control", False):
            action = np.concatenate(
                [np.asarray(action, dtype=np.float32), self._overtime_action_block(env)]
            )
        return project_action(action, env_state=env, action_space_info=env.action_size).action

    def _overtime_action_block(self, env: CapacityPlanningEnv) -> np.ndarray:
        """Raw overtime block in [-1, 1]; the default heuristic uses none."""

        return np.full(env.config.num_facilities, -1.0, dtype=np.float32)

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
            demand_history_mean=(
                env._demand_history_features()[0]
                if env.config.include_demand_history_state
                else None
            ),
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


class ForecastMeanDemandLookahead2Policy(MeanDemandLookahead2Policy):
    """fMDL-2: MDL-2 network sharing driven by the observable forecast."""

    algorithm = "fmdl2"

    def default_use_demand_forecast(self) -> bool:
        return True


class RollingMeanDemandLookahead2Policy(MeanDemandLookahead2Policy):
    """rMDL-2: MDL-2 driven by the causal rolling demand mean."""

    algorithm = "rmdl2"

    def default_use_demand_history(self) -> bool:
        return True

    def default_demand_history_weight(self) -> float:
        return 0.05


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
        self.shield_rollout_replications = max(
            int(config.get("shield_rollout_replications", 1)),
            1,
        )
        self.shield_seed = int(config.get("shield_seed", 1970000))
        self._shield_decision_index = 0
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

    def reset(self) -> None:
        super().reset()
        self._anchor_policy.reset()
        self._shield_decision_index = 0

    def _make_anchor_policy(self, state_dim, action_dim, config):
        policy_map = {
            "myo": MyopicPolicy,
            "iso": IsolatedPolicy,
            "mdl1": MeanDemandLookahead1Policy,
            "mdl2": MeanDemandLookahead2Policy,
            "fmyo": ForecastMyopicPolicy,
            "fmdl2": ForecastMeanDemandLookahead2Policy,
            "rmdl2": RollingMeanDemandLookahead2Policy,
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
        candidate_metrics = self._evaluate_candidates(env, candidates)
        best_index = select_shield_candidate_index(self, candidate_metrics)
        return project_action(candidates[best_index], env_state=env, action_space_info=env.action_size).action

    def _evaluate_candidates(
        self,
        env: CapacityPlanningEnv,
        candidates: Sequence[np.ndarray],
    ) -> list[dict[str, float]]:
        start = self.shield_seed + self._shield_decision_index * self.shield_rollout_replications
        rollout_seeds = tuple(
            start + replication
            for replication in range(self.shield_rollout_replications)
        )
        self._shield_decision_index += 1
        return [
            mean_shield_rollout_metrics(
                env,
                self._anchor_policy,
                action,
                horizon=self.shield_lookahead,
                rollout_seeds=rollout_seeds,
            )
            for action in candidates
        ]


class ShieldedMeanDemandLookahead2Policy(ShieldedPatientPriorityMyopicPolicy):
    """MDL-2 plus the same rollout shield used by pMYO-shield."""

    algorithm = "mdl2_shield"

    def default_anchor_policy(self) -> str:
        return "mdl2"


class LeadTimeAwareMeanDemandLookahead2Policy(MeanDemandLookahead2Policy):
    """MDL-2-LT: MDL-2 that knows orders take time to arrive.

    Plain MDL-2 sets reagent orders from ON-HAND inventory and covers only its
    lookahead window. Under a procurement lead time both assumptions break, and
    the first one breaks badly: ignoring stock already in transit means
    re-ordering the same shortfall every epoch until it lands, so the planner
    systematically over-orders. Comparing a learned policy against that
    unmodified planner would be a strawman, which is why this comparator is
    built and tuned before any learned work.

    Two corrections, both standard inventory practice:

    1. Order against the inventory POSITION (on hand plus on order), not the
       on-hand level.
    2. Cover demand over the lead plus the lookahead window, not the lookahead
       window alone, with a tunable safety multiplier for lead variability.
    """

    algorithm = "mdl2_lt"

    def __init__(self, state_dim=None, action_dim=None, config=None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.safety_multiplier = float(config.get("safety_multiplier", 1.0))
        if self.safety_multiplier < 0.0:
            raise ValueError("safety_multiplier must be nonnegative")

    def expected_lead(self, env: CapacityPlanningEnv) -> float:
        probabilities = getattr(env.config, "reagent_lead_time_probabilities", ())
        if getattr(env.config, "enable_stochastic_procurement", False) and probabilities:
            weights = np.asarray(probabilities, dtype=float)
            return float((np.arange(len(weights)) * weights).sum())
        return float(getattr(env.config, "reagent_purchase_lead_time", 0))

    def _facility_net_action(self, env: CapacityPlanningEnv) -> np.ndarray:
        action = super()._facility_net_action(env)
        pipeline = getattr(env, "reagent_purchase_pipeline", None)
        if pipeline is None or pipeline.shape[0] <= 1:
            return action  # no lead time configured: identical to MDL-2

        n = env.config.num_facilities
        specimens = np.asarray(env.specimens, dtype=float)
        reagents = np.asarray(env.reagents, dtype=float)
        idle = np.asarray(env.bioreactors[:, 0], dtype=float)
        production = np.minimum.reduce((specimens, idle, reagents))
        next_specimens = np.maximum(specimens - production + np.asarray(env.demand, dtype=float), 0.0)
        on_hand_after_production = reagents - production
        on_order = pipeline.sum(axis=0)

        rate = np.asarray(
            getattr(env, "demand_rate_estimates", env.demand_rates), dtype=float
        )
        coverage = float(self.settings.lookahead_periods) + self.expected_lead(env)
        target = (next_specimens + coverage * rate) * self.safety_multiplier
        target = target * float(self.settings.local_order_up_to_multiplier)

        position = on_hand_after_production + on_order
        replenishment = np.clip(
            target - position, 0.0, np.asarray(env.max_reagent_replenishment, dtype=float)
        ) * np.asarray(env.supplier_available, dtype=float)

        action = action.copy()
        action[3 * n : 4 * n] = _normalize_replenishment(
            replenishment, np.asarray(env.max_reagent_replenishment, dtype=float)
        )
        return action


def _overtime_shortfall_and_headroom(
    env: CapacityPlanningEnv,
) -> tuple[np.ndarray, np.ndarray]:
    """Reagent-sufficient capacity shortfall and per-facility surge headroom.

    Because production is ``min(waiting, idle + surge, reagents)``, surging
    while reagents bind burns overtime cost for zero production, so the usable
    shortfall is capped by the reagent bound
    (spec 2026-08-29-continuous-overtime-control).
    """

    waiting = env.waiting_counts() if hasattr(env, "waiting_counts") else env.specimens
    idle = env.bioreactors[:, 0]
    usable = np.minimum(np.asarray(waiting, dtype=float), env.reagents)
    shortfall = np.maximum(usable - idle, 0.0)
    headroom = np.asarray(env.overtime_surge_headroom, dtype=float)
    return shortfall, headroom


def _overtime_block_from_surge(
    surge_units: np.ndarray, headroom: np.ndarray
) -> np.ndarray:
    fraction = np.divide(
        surge_units,
        headroom,
        out=np.zeros_like(surge_units),
        where=headroom > 0.0,
    )
    return np.clip(2.0 * fraction - 1.0, -1.0, 1.0).astype(np.float32)


def _bounded_proportional_allocation(
    weights: np.ndarray,
    caps: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Allocate a shared continuous budget proportionally with local caps."""

    allocation = np.zeros_like(np.asarray(caps, dtype=float))
    remaining = max(float(budget), 0.0)
    active = np.asarray(caps, dtype=float) > 0.0
    positive_weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    while remaining > 1e-12 and np.any(active):
        current_weights = np.where(active, positive_weights, 0.0)
        if float(current_weights.sum()) <= 0.0:
            current_weights = active.astype(float)
        proposal = remaining * current_weights / float(current_weights.sum())
        headroom = np.maximum(np.asarray(caps, dtype=float) - allocation, 0.0)
        accepted = np.minimum(proposal, headroom)
        allocation += accepted
        used = float(accepted.sum())
        remaining -= used
        active = headroom - accepted > 1e-12
        if used <= 1e-12:
            break
    return allocation


class IntertemporalForecastOvertimePolicy(MeanDemandLookahead2Policy):
    """Allocate the shared future staffing pool from the visible forecast."""

    algorithm = "forecast_iot"

    def __init__(self, state_dim=None, action_dim=None, config=None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.forecast_overtime_budget_fraction = float(
            config.get("forecast_overtime_budget_fraction", 1.0)
        )
        if not 0.0 <= self.forecast_overtime_budget_fraction <= 1.0:
            raise ValueError(
                "forecast_overtime_budget_fraction must be within [0, 1]"
            )

    def _forecast_weights(self, env: CapacityPlanningEnv) -> np.ndarray:
        waiting = (
            env.waiting_counts() if hasattr(env, "waiting_counts") else env.specimens
        )
        weights = np.asarray(env.demand_forecast, dtype=float) + np.asarray(
            waiting, dtype=float
        )
        if hasattr(env, "at_risk_counts"):
            weights += 2.0 * np.asarray(env.at_risk_counts(), dtype=float)
        return np.maximum(weights, 0.0)

    def _overtime_action_block(self, env: CapacityPlanningEnv) -> np.ndarray:
        if not env.config.enable_intertemporal_overtime_commitment:
            raise ValueError(
                "forecast_iot requires enable_intertemporal_overtime_commitment"
            )
        headroom = np.asarray(env.overtime_surge_headroom, dtype=float)
        budget = (
            env._shared_overtime_budget()
            * self.forecast_overtime_budget_fraction
        )
        allocation = _bounded_proportional_allocation(
            self._forecast_weights(env), headroom, budget
        )
        return _overtime_block_from_surge(allocation, headroom)


class GraphSmoothedIntertemporalForecastPolicy(
    IntertemporalForecastOvertimePolicy
):
    """Forecast allocator with one-hop information-graph smoothing."""

    algorithm = "graph_forecast_iot"

    def __init__(self, state_dim=None, action_dim=None, config=None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.graph_forecast_smoothing = float(
            config.get("graph_forecast_smoothing", 0.5)
        )
        if not 0.0 <= self.graph_forecast_smoothing <= 1.0:
            raise ValueError("graph_forecast_smoothing must be within [0, 1]")

    def _forecast_weights(self, env: CapacityPlanningEnv) -> np.ndarray:
        base = super()._forecast_weights(env)
        neighbor_sum = np.zeros_like(base)
        neighbor_count = np.zeros_like(base)
        for left, right in env.information_edges:
            neighbor_sum[left] += base[right]
            neighbor_sum[right] += base[left]
            neighbor_count[left] += 1.0
            neighbor_count[right] += 1.0
        neighbor_mean = np.divide(
            neighbor_sum,
            neighbor_count,
            out=base.copy(),
            where=neighbor_count > 0.0,
        )
        smoothing = self.graph_forecast_smoothing
        return (1.0 - smoothing) * base + smoothing * neighbor_mean


class OvertimeMeanDemandLookahead2Policy(MeanDemandLookahead2Policy):
    """MDL-2-OT: MDL-2 plus a closed-form myopic overtime rule.

    Surges only when (a) eligible-waiting exceeds idle capacity, (b) reagents
    suffice to use the surge, and (c) the modeled marginal shortage/patient
    risk exceeds the marginal convex overtime cost. No simulation at decision
    time.
    """

    algorithm = "mdl2_ot"

    def _overtime_action_block(self, env: CapacityPlanningEnv) -> np.ndarray:
        shortfall, headroom = _overtime_shortfall_and_headroom(env)
        target = np.minimum(shortfall, headroom)
        # Marginal modeled benefit of one surge unit: avoided bioreactor
        # shortage, plus patient-loss pressure when the environment carries a
        # patient layer.
        benefit = np.full_like(target, float(env.config.costs.bioreactor_shortage))
        if hasattr(env, "at_risk_counts") and hasattr(env, "waiting_counts"):
            waiting = np.maximum(np.asarray(env.waiting_counts(), dtype=float), 1.0)
            urgency = np.clip(np.asarray(env.at_risk_counts(), dtype=float) / waiting, 0.0, 1.0)
            weight_patient_lost = float(
                getattr(getattr(env, "env_config", None), "weight_patient_lost", 0.0)
            )
            benefit = benefit + weight_patient_lost * urgency
        marginal_cost = (
            float(env.config.weight_overtime_linear)
            + 2.0 * float(env.config.weight_overtime_quadratic) * target
        )
        surge_units = np.where(benefit >= marginal_cost, target, 0.0)
        return _overtime_block_from_surge(surge_units, headroom)


class OvertimeUrgencyAwareMyopicPolicy(UrgencyAwareMyopicPolicy):
    """uMYO-OT: urgency surge mapped onto the continuous overtime channel."""

    algorithm = "umyo_ot"

    def _overtime_action_block(self, env: CapacityPlanningEnv) -> np.ndarray:
        shortfall, headroom = _overtime_shortfall_and_headroom(env)
        if not hasattr(env, "at_risk_counts"):
            return _overtime_block_from_surge(np.zeros_like(shortfall), headroom)
        waiting = np.maximum(np.asarray(env.waiting_counts(), dtype=float), 1.0)
        urgency = np.clip(
            (np.asarray(env.at_risk_counts(), dtype=float)
             + np.asarray(env.near_expiry_counts(), dtype=float)) / waiting,
            0.0,
            1.0,
        )
        surge_units = np.minimum(shortfall, headroom) * urgency
        return _overtime_block_from_surge(surge_units, headroom)


class StaticOvertimePolicy(MeanDemandLookahead2Policy):
    """static_ot: MDL-2 plus a constant overtime fraction (tuned-scalar reference)."""

    algorithm = "static_ot"

    def __init__(self, state_dim=None, action_dim=None, config=None):
        super().__init__(state_dim, action_dim, config)
        config = config or {}
        self.static_overtime_fraction = float(
            config.get("static_overtime_fraction", 0.0)
        )
        if not 0.0 <= self.static_overtime_fraction <= 1.0:
            raise ValueError("static_overtime_fraction must be within [0, 1]")

    def _overtime_action_block(self, env: CapacityPlanningEnv) -> np.ndarray:
        n = env.config.num_facilities
        return np.full(
            n, 2.0 * self.static_overtime_fraction - 1.0, dtype=np.float32
        )


HEURISTIC_POLICIES = {
    "myo": MyopicPolicy,
    "iso": IsolatedPolicy,
    "mdl1": MeanDemandLookahead1Policy,
    "mdl2": MeanDemandLookahead2Policy,
    "fmyo": ForecastMyopicPolicy,
    "fmdl2": ForecastMeanDemandLookahead2Policy,
    "rmdl2": RollingMeanDemandLookahead2Policy,
    "umyo": UrgencyAwareMyopicPolicy,
    "pmyo": PatientPriorityMyopicPolicy,
    "mdl2_shield": ShieldedMeanDemandLookahead2Policy,
    "pmyo_shield": ShieldedPatientPriorityMyopicPolicy,
    "mdl2_ot": OvertimeMeanDemandLookahead2Policy,
    "umyo_ot": OvertimeUrgencyAwareMyopicPolicy,
    "static_ot": StaticOvertimePolicy,
    "forecast_iot": IntertemporalForecastOvertimePolicy,
    "graph_forecast_iot": GraphSmoothedIntertemporalForecastPolicy,
    "mdl2_lt": LeadTimeAwareMeanDemandLookahead2Policy,
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
    include_demand_history = bool(
        env_config.get("include_demand_history_state", False)
    )
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    features_per_facility = facility_state_width(env_config)

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
    if include_demand_history:
        history_start = (
            3
            + lead_time
            + int(include_supplier)
            + int(include_forecast)
            + 3 * int(include_transfer_pipeline)
        )
        demand_history_mean = state_array[:, history_start]
    else:
        demand_history_mean = None
    patient_priority = patient_priority_from_state(state_vector, env_config, settings)

    demand_rate_estimates = env_config.get("demand_rate_estimates")
    if demand_rate_estimates is None:
        demand_rate_estimates = env_config.get("demand_rates", 0.0)
    specimen_edges, capacity_edges, resource_edges = _resolve_facility_edge_sets(
        env_config,
        n,
    )
    return facility_net_action_from_arrays(
        demand=demand,
        specimens=specimens,
        reagents=reagents,
        bioreactors=bioreactors,
        supplier_available=supplier_available,
        demand_forecast=demand_forecast,
        demand_history_mean=demand_history_mean,
        demand_rates=_config_vector(demand_rate_estimates, n, "demand_rate_estimates"),
        max_reagent_replenishment=_config_vector(
            env_config.get("max_reagent_replenishment", 0.0),
            n,
            "max_reagent_replenishment",
        ),
        max_specimen_transfer=float(env_config.get("max_specimen_transfer", 0.0)),
        max_bioreactor_transfer=float(env_config.get("max_bioreactor_transfer", 0.0)),
        max_reagent_transfer=float(env_config.get("max_reagent_transfer", 0.0)),
        specimen_edges=specimen_edges,
        capacity_edges=capacity_edges,
        resource_edges=resource_edges,
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
    demand_history_mean: np.ndarray | None,
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

    # Replay states are stored as float32. Normalize live-environment inputs to
    # the same precision so equal-pressure routing ties resolve identically in
    # training, validation, and deployment.
    demand = np.asarray(demand, dtype=np.float32)
    specimens = np.asarray(specimens, dtype=np.float32)
    reagents = np.asarray(reagents, dtype=np.float32)
    bioreactors = np.asarray(bioreactors, dtype=np.float32)
    supplier_available = np.asarray(supplier_available, dtype=np.float32)
    demand_rates = np.asarray(demand_rates, dtype=np.float32)
    if demand_forecast is not None:
        demand_forecast = np.asarray(demand_forecast, dtype=np.float32)
    if demand_history_mean is not None:
        demand_history_mean = np.asarray(
            demand_history_mean,
            dtype=np.float32,
        )
    n = int(demand.shape[0])
    idle_bioreactors = bioreactors[:, 0]
    next_stage_bioreactors = bioreactors[:, 1] if bioreactors.shape[1] > 1 else np.zeros(n)
    production = np.minimum.reduce((specimens, idle_bioreactors, reagents))
    next_specimens = specimens - production + demand
    next_reagents = reagents - production
    next_idle_bioreactors = idle_bioreactors - production + next_stage_bioreactors

    if settings.use_demand_forecast and demand_forecast is not None:
        lookahead_demand = np.asarray(demand_forecast, dtype=float)
    elif settings.use_demand_history and demand_history_mean is not None:
        history_weight = float(
            np.clip(settings.demand_history_weight, 0.0, 1.0)
        )
        estimated_rate = (
            (1.0 - history_weight) * demand_rates
            + history_weight * np.asarray(demand_history_mean, dtype=float)
        )
        lookahead_demand = (
            settings.lookahead_periods
            * estimated_rate
        )
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
    shortage_remaining = np.asarray(shortage, dtype=float).copy()
    surplus_remaining = np.asarray(surplus, dtype=float).copy()
    if shortage_remaining.ndim != 1:
        raise ValueError("shortage must be a one-dimensional finite array")
    if surplus_remaining.shape != shortage_remaining.shape:
        raise ValueError("shortage and surplus must have identical shapes")
    if not (
        np.all(np.isfinite(shortage_remaining))
        and np.all(np.isfinite(surplus_remaining))
    ):
        raise ValueError("shortage and surplus must contain only finite values")
    if not np.isfinite(max_abs):
        raise ValueError("max_abs must be finite")

    net = np.zeros_like(shortage_remaining, dtype=float)
    if not edges or max_abs <= 0.0:
        return net

    adjacency = _adjacency(edges)
    # This runs in every residual-policy update on small facility vectors.
    # Keep the ordering in Python to avoid repeatedly entering NumPy's native
    # sort implementation and make equal-shortage ordering deterministic.
    receivers = sorted(
        range(shortage_remaining.size),
        key=lambda index: (-float(shortage_remaining[index]), index),
    )

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
    edge_sets = _resolve_facility_edge_sets(env_config, num_facilities)
    index = {
        "specimen_edges": 0,
        "capacity_edges": 1,
        "resource_edges": 2,
    }
    try:
        return edge_sets[index[key]]
    except KeyError as exc:
        raise ValueError(f"Unsupported facility edge key: {key}") from exc


def _resolve_facility_edge_sets(
    env_config: dict[str, Any],
    num_facilities: int,
) -> tuple[tuple[Edge, ...], tuple[Edge, ...], tuple[Edge, ...]]:
    action_mode = str(env_config.get("action_mode", "edge_transfer"))
    clinic_coordinates = normalize_coordinates(
        env_config.get("clinic_coordinates"),
        num_facilities,
    )
    geographic_neighbor_k = int(env_config.get("geographic_neighbor_k", 3))
    configured = tuple(
        _normalize_facility_edges(env_config.get(key), key, num_facilities)
        for key in ("specimen_edges", "capacity_edges", "resource_edges")
    )
    return _cached_facility_edge_sets(
        action_mode,
        clinic_coordinates,
        geographic_neighbor_k,
        int(num_facilities),
        configured[0],
        configured[1],
        configured[2],
    )


@lru_cache(maxsize=256)
def _cached_facility_edge_sets(
    action_mode: str,
    clinic_coordinates: tuple[tuple[float, float], ...],
    geographic_neighbor_k: int,
    num_facilities: int,
    specimen_edges: tuple[Edge, ...] | None,
    capacity_edges: tuple[Edge, ...] | None,
    resource_edges: tuple[Edge, ...] | None,
) -> tuple[tuple[Edge, ...], tuple[Edge, ...], tuple[Edge, ...]]:
    configured_by_key = {
        "specimen_edges": specimen_edges,
        "capacity_edges": capacity_edges,
        "resource_edges": resource_edges,
    }
    needs_geographic_edges = (
        action_mode == "facility_net"
        and (
            specimen_edges is None
            or resource_edges is None
        )
    )
    geographic_edges = (
        geographic_knn_edges(
            clinic_coordinates,
            k=geographic_neighbor_k,
        )
        if clinic_coordinates and needs_geographic_edges
        else ()
    )
    facility_net_default = geographic_edges or ring_edges(num_facilities)
    complete_default = complete_undirected_edges(num_facilities)
    resolved: list[tuple[Edge, ...]] = []
    for key in ("specimen_edges", "capacity_edges", "resource_edges"):
        configured_edges = configured_by_key[key]
        if configured_edges is not None:
            resolved.append(configured_edges)
        elif action_mode == "facility_net" and key in (
            "specimen_edges",
            "resource_edges",
        ):
            resolved.append(facility_net_default)
        else:
            resolved.append(complete_default)
    return resolved[0], resolved[1], resolved[2]


def _normalize_facility_edges(
    edges: Sequence[Sequence[int]] | None,
    key: str,
    num_facilities: int,
) -> tuple[Edge, ...] | None:
    if edges is None:
        return None
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
    include_demand_history = bool(
        env_config.get("include_demand_history_state", False)
    )
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    features_per_facility = facility_state_width(env_config)
    summary_edges = tuple(env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97)))
    summary_width = (
        6
        + len(summary_edges)
        + 1
        + (4 if env_config.get("include_specimen_routing_state", False) else 0)
    )
    base_width = n * features_per_facility
    state_vector = np.asarray(state, dtype=np.float32)
    expected_width = base_width + n * summary_width
    if state_vector.size < expected_width:
        return None
    summary = state_vector[base_width:expected_width].reshape(n, summary_width)
    waiting = np.maximum(summary[:, 0], 1.0)
    near_expiry = summary[:, 2]
    histogram = summary[:, 6:]
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
    rollout_seed: int | None = None,
) -> dict[str, float]:
    from src.rl.experiment import EpisodeMetrics

    if rollout_seed is not None:
        env.rng = np.random.default_rng(int(rollout_seed))
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


def mean_shield_rollout_metrics(
    env: CapacityPlanningEnv,
    anchor_policy: CapacityHeuristicPolicy,
    action: np.ndarray,
    *,
    horizon: int,
    rollout_seeds: Sequence[int],
) -> dict[str, float]:
    """Average a shield candidate over CRN draws independent of the live episode."""

    seeds = tuple(int(seed) for seed in rollout_seeds)
    if not seeds:
        raise ValueError("rollout_seeds must contain at least one seed")
    rows = [
        shield_rollout_metrics(
            copy.deepcopy(env),
            anchor_policy,
            action,
            horizon=horizon,
            rollout_seed=seed,
        )
        for seed in seeds
    ]
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
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
