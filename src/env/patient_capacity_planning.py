"""Patient-condition capacity-planning environment (Phase 4, task group 3).

Extends the base PRM capacity-planning environment so that **patients drive
demand** (approach a): each clinic holds a queue of individual patients whose
health deteriorates while they wait (`patient_condition`). The count of *eligible
waiting patients* is the specimen supply the base manufacturing dynamics consume.

Two loss channels are added on top of the base cost:
- **Patients lost** — survival falls below the eligibility threshold before the
  patient is served.
- **Material wasted** — a waiting specimen ages past its shelf life, or finished
  product expires before delivery (`aging_inventory`), plus an urgency penalty on
  at-risk patients left unserved.

Modeling notes:
- Specimens are **identity-bound** (autologous): one patient's material cannot be
  pooled for another, so the base specimen-transfer action is intentionally
  ignored here. Only reagents and bioreactor capacity are shareable across
  clinics. Reagent and capacity transfers may arrive through the base delayed
  transfer pipeline when ``transfer_lead_time > 0``.
- Inter-clinic patient routing (with the cold-chain viability hook) is deferred.
- Requires ``action_mode == "facility_net"``.
- The base env (`capacity_planning.py`) is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.env.aging_inventory import AgingInventory
from src.env.capacity_planning import (
    CapacityPlanningConfig,
    CapacityPlanningEnv,
    _apply_net_transfers,
    _apply_net_transfers_delayed,
    make_20_clinic_config,
)
from src.env.patient_condition import (
    PatientConditionConfig,
    PatientConditionModel,
    PatientStatus,
)


@dataclass(frozen=True)
class PatientEnvConfig:
    """Bundles the base network config with patient-condition parameters."""

    base: CapacityPlanningConfig = field(default_factory=make_20_clinic_config)
    patient: PatientConditionConfig = PatientConditionConfig()
    material_shelf_life: int = 6          # epochs a waiting specimen stays usable
    finished_shelf_life: int = 2          # epochs finished product stays deliverable
    weight_patient_lost: float = 50_000.0
    weight_expiry: float = 40_000.0
    weight_urgency: float = 5_000.0
    urgency_margin: float = 0.1           # "at risk" = survival < threshold + margin
    enable_viability_hook: bool = False
    # Decision-aligned interior survival-bucket edges for the observation summary.
    # Default gives 4 buckets over waiting patients (all have survival >=
    # threshold): [thr, 0.85) critical, [0.85, 0.90) watch, [0.90, 0.97) healthy,
    # [0.97, 1.0] fresh. Config-driven so the Phase 7 pilot can sweep granularity.
    survival_bucket_edges: tuple[float, ...] = (0.85, 0.90, 0.97)
    expiry_warning_margin: int = 1        # "near expiry" = age >= shelf_life - margin


class PatientConditionCapacityEnv(CapacityPlanningEnv):
    """Base capacity-planning env with per-patient deterioration and expiry."""

    def __init__(self, config: PatientEnvConfig | None = None, seed: int | None = None):
        self.env_config = config or PatientEnvConfig()
        self.patient_model = PatientConditionModel(self.env_config.patient)
        self._summary_edges = np.asarray(self.env_config.survival_bucket_edges, dtype=float)
        if self._summary_edges.ndim != 1 or (np.diff(self._summary_edges) <= 0).any():
            raise ValueError("survival_bucket_edges must be strictly increasing")
        # 3 scalars (waiting count, mean survival, near-expiry count) + histogram.
        self.summary_width = 3 + len(self._summary_edges) + 1
        # base __init__ calls self.reset(), which needs the attributes above.
        super().__init__(self.env_config.base, seed)
        if self.config.action_mode != "facility_net":
            raise ValueError("PatientConditionCapacityEnv requires action_mode='facility_net'")
        # The patient summary is appended as a per-clinic block; extend the sizes.
        self.base_observation_size = self.config.num_facilities * self.features_per_facility
        self.observation_size = (
            self.base_observation_size + self.config.num_facilities * self.summary_width
        )

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> np.ndarray:
        super().reset(seed)
        n = self.config.num_facilities
        viability_fn = self._viability_fn if self.env_config.enable_viability_hook else None
        self.patient_queues = [[] for _ in range(n)]
        self.finished_product = [
            AgingInventory(self.env_config.finished_shelf_life, viability_fn) for _ in range(n)
        ]
        # Seed initial waiting patients from the base initial specimen counts.
        for i in range(n):
            for _ in range(int(round(float(self.specimens[i])))):
                self.patient_queues[i].append(self.patient_model.enroll(self.rng, epoch=0))
        self.specimens = self._waiting_counts()
        self.cumulative_enrolled = float(self.specimens.sum())
        self.cumulative_lost = 0.0
        self.cumulative_served = 0.0
        return self.observation()

    # ------------------------------------------------------------ observation
    def observation(self) -> np.ndarray:
        """Base flat observation with a per-clinic patient summary appended.

        The summary is added as a trailing block (not interleaved), so
        ``normalize_observations`` — which assumes the base per-facility layout —
        must stay off for this env (it is off in all shipped configs).
        """

        base = super().observation()
        if not hasattr(self, "patient_queues"):  # during base __init__/reset setup
            return base
        summary = self._patient_summary().reshape(-1)
        return np.concatenate([base, summary]).astype(np.float32)

    def graph_observation(self) -> dict[str, np.ndarray]:
        """Base graph observation with patient-summary columns on each clinic node."""

        data = super().graph_observation()
        if not hasattr(self, "patient_queues"):  # during base __init__/reset setup
            return data
        node_features = data["node_features"]
        summary = self._patient_summary()
        if node_features.shape[0] > summary.shape[0]:  # central capacity hub row
            pad = np.zeros((node_features.shape[0] - summary.shape[0], summary.shape[1]))
            summary = np.vstack([summary, pad])
        data["node_features"] = np.hstack([node_features, summary]).astype(np.float32)
        return data

    def _patient_summary(self) -> np.ndarray:
        """Per-clinic fixed-width summary: [count, mean_survival, near_expiry, hist...]."""

        n = self.config.num_facilities
        buckets = len(self._summary_edges) + 1
        near_expiry_age = self.env_config.material_shelf_life - self.env_config.expiry_warning_margin
        rows = np.zeros((n, self.summary_width), dtype=float)
        for i, queue in enumerate(self.patient_queues):
            if not queue:
                continue
            survivals = np.array([p.survival for p in queue], dtype=float)
            ages = np.array([p.age for p in queue], dtype=float)
            histogram = np.bincount(
                np.digitize(survivals, self._summary_edges), minlength=buckets
            )[:buckets].astype(float)
            rows[i, 0] = float(len(queue))
            rows[i, 1] = float(survivals.mean())
            rows[i, 2] = float((ages >= near_expiry_age).sum())
            rows[i, 3:] = histogram
        return rows

    # ------------------------------------------------------------------- step
    def step(self, action):
        n = self.config.num_facilities
        costs = self.config.costs
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"Expected action shape {(self.action_size,)}, got {action_array.shape}")
        normalized = np.clip(action_array, -1.0, 1.0)

        specimen_arrivals, reagent_arrivals, capacity_arrivals = self._receive_transfer_arrivals()
        supplier_available = self.supplier_available.copy()
        reagent_transfer_requests = normalized[n : 2 * n] * self.config.max_reagent_transfer
        capacity_requests = normalized[2 * n : 3 * n] * self.config.max_bioreactor_transfer
        replenishment = (
            ((normalized[3 * n : 4 * n] + 1.0) / 2.0)
            * self.max_reagent_replenishment
            * supplier_available
        )

        # 1) Age waiting patients; apply the loss channels (expiry then eligibility).
        lost_ineligible, lost_expired = self._age_and_gate_patients()

        # 2) Production consumes the most-urgent eligible patients, bounded by
        #    idle bioreactors and reagents.
        idle_bioreactors = self.bioreactors[:, 0]
        waiting = self._waiting_counts()
        production = np.floor(
            np.minimum.reduce((waiting, idle_bioreactors, self.reagents))
        ).astype(float)
        self._start_production(production)

        # 3) Base resource bookkeeping (reagents + bioreactor pipeline), then
        #    reagent/capacity transfers (specimens are identity-bound: no pooling).
        next_reagents = self.reagents - production + replenishment
        next_bioreactors = np.zeros_like(self.bioreactors)
        next_bioreactors[:, 0] = self.bioreactors[:, 0] - production + self.bioreactors[:, 1]
        if self.config.production_lead_time > 2:
            next_bioreactors[:, 1:-1] = self.bioreactors[:, 2:]
        next_bioreactors[:, -1] = production

        specimen_net = np.zeros(n, dtype=float)
        specimen_flows = np.zeros(len(self.specimen_edges), dtype=float)
        if self.config.transfer_lead_time > 0:
            reagent_net, reagent_flows, reagent_future_arrivals = _apply_net_transfers_delayed(
                next_reagents, self.resource_edges, reagent_transfer_requests
            )
            capacity_net, capacity_flows, capacity_future_arrivals = _apply_net_transfers_delayed(
                next_bioreactors[:, 0], self.capacity_edges, capacity_requests
            )
            if self._uses_geographic_transfer_delays():
                self._schedule_edge_transfer_arrivals(
                    self.reagent_transfer_pipeline, self.resource_edges, reagent_flows
                )
                self._schedule_edge_transfer_arrivals(
                    self.capacity_transfer_pipeline, self.capacity_edges, capacity_flows
                )
            else:
                self._schedule_transfer_arrivals(
                    self.reagent_transfer_pipeline, reagent_future_arrivals
                )
                self._schedule_transfer_arrivals(
                    self.capacity_transfer_pipeline, capacity_future_arrivals
                )
        else:
            reagent_net, reagent_flows = _apply_net_transfers(
                next_reagents, self.resource_edges, reagent_transfer_requests
            )
            capacity_net, capacity_flows = _apply_net_transfers(
                next_bioreactors[:, 0], self.capacity_edges, capacity_requests
            )

        self.reagents = np.clip(next_reagents, 0.0, self.max_reagents)
        next_bioreactors[:, 0] = np.clip(next_bioreactors[:, 0], 0.0, self.max_idle_bioreactors)
        self.bioreactors = np.maximum(next_bioreactors, 0.0)

        # 4) Finished product ages and is delivered; expired product is wasted.
        finished_expired = self._age_and_deliver_finished()

        # 5) New patient arrivals (demand) enroll into the queues.
        current_demand = self.demand.copy()
        self._enroll_arrivals(current_demand)
        self.specimens = self._waiting_counts()

        # 6) Costs: base terms + patient/expiry/urgency terms.
        under_reagents = np.maximum(self.specimens - self.reagents, 0.0)
        idle_reagents = np.maximum(self.reagents - self.specimens, 0.0)
        under_bioreactors = np.maximum(self.specimens - self.bioreactors[:, 0], 0.0)
        idle_bioreactor_counts = np.maximum(self.bioreactors[:, 0] - self.specimens, 0.0)

        patients_lost = lost_ineligible + lost_expired
        material_wasted = lost_expired + finished_expired
        at_risk_unserved = self._at_risk_unserved_counts()
        reagent_transfer_cost = self._facility_net_transfer_cost(
            costs.reagent_transfer,
            self.resource_edges,
            reagent_flows,
            reagent_net,
        )
        capacity_transfer_cost = self._facility_net_transfer_cost(
            costs.bioreactor_transfer,
            self.capacity_edges,
            capacity_flows,
            capacity_net,
        )

        base_cost = (
            costs.reagent_purchase * float(replenishment.sum())
            + costs.reagent_holding * float(idle_reagents.sum())
            + costs.reagent_shortage * float(under_reagents.sum())
            + costs.bioreactor_holding * float(idle_bioreactor_counts.sum())
            + costs.bioreactor_shortage * float(under_bioreactors.sum())
            + reagent_transfer_cost
            + capacity_transfer_cost
        )
        cost = (
            base_cost
            + self.env_config.weight_patient_lost * float(patients_lost.sum())
            + self.env_config.weight_expiry * float(material_wasted.sum())
            + self.env_config.weight_urgency * float(at_risk_unserved.sum())
        )

        self.cumulative_lost += float(patients_lost.sum())
        self.cumulative_served += float(production.sum())
        self._update_running_metrics(current_demand, production, self.specimens, self.bioreactors)
        if np.any(under_reagents > 0):
            self.reagent_shortage_steps += 1
        if np.any(under_bioreactors > 0):
            self.bioreactor_shortage_steps += 1
        done = self._advance_clock()

        info: dict[str, np.ndarray | float] = {
            "cost": cost,
            "base_cost": base_cost,
            "production": production.copy(),
            "demand": current_demand.copy(),
            "replenishment": replenishment.copy(),
            "patients_lost": patients_lost.copy(),
            "patients_lost_ineligible": lost_ineligible.copy(),
            "patients_lost_expired": lost_expired.copy(),
            "material_wasted": material_wasted.copy(),
            "finished_expired": finished_expired.copy(),
            "at_risk_unserved": at_risk_unserved.copy(),
            "risk_type_counts": self.risk_type_counts().copy(),
            "waiting_patients": self.specimens.copy(),
            "eligibility_rate": self._eligibility_rate(),
            "under_reagents": under_reagents.copy(),
            "under_bioreactors": under_bioreactors.copy(),
            "specimen_transfer_arrivals": specimen_arrivals.copy(),
            "reagent_transfer_arrivals": reagent_arrivals.copy(),
            "capacity_transfer_arrivals": capacity_arrivals.copy(),
            "specimen_transfers": specimen_net.copy(),
            "capacity_transfers": capacity_net.copy(),
            "reagent_transfers": reagent_net.copy(),
            "transshipment_cost": reagent_transfer_cost + capacity_transfer_cost,
        }
        info.update(self._performance_info())
        return self.observation(), -cost, done, info

    # -------------------------------------------------------------- internals
    def _waiting_counts(self) -> np.ndarray:
        return np.array([float(len(q)) for q in self.patient_queues], dtype=float)

    def _age_and_gate_patients(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.config.num_facilities
        lost_ineligible = np.zeros(n, dtype=float)
        lost_expired = np.zeros(n, dtype=float)
        shelf = self.env_config.material_shelf_life
        for i, queue in enumerate(self.patient_queues):
            survivors = []
            for patient in queue:
                self.patient_model.advance(patient)
                if patient.age >= shelf:
                    patient.status = PatientStatus.LOST
                    lost_expired[i] += 1.0
                elif not self.patient_model.is_eligible(patient):
                    patient.status = PatientStatus.LOST
                    lost_ineligible[i] += 1.0
                else:
                    survivors.append(patient)
            # Most-urgent (lowest survival) first, for priority production.
            survivors.sort(key=lambda p: p.survival)
            self.patient_queues[i] = survivors
        return lost_ineligible, lost_expired

    def _start_production(self, production: np.ndarray) -> None:
        for i, count in enumerate(production.astype(int)):
            served = self.patient_queues[i][:count]
            for patient in served:
                patient.status = PatientStatus.IN_PRODUCTION
            self.finished_product[i].add(float(count))
            self.patient_queues[i] = self.patient_queues[i][count:]

    def _age_and_deliver_finished(self) -> np.ndarray:
        n = self.config.num_facilities
        expired = np.zeros(n, dtype=float)
        for i, inventory in enumerate(self.finished_product):
            expired[i] = inventory.advance()
            # MVP: deliver all available finished product this epoch.
            inventory.consume(inventory.total())
        return expired

    def _enroll_arrivals(self, demand: np.ndarray) -> None:
        for i, count in enumerate(demand.astype(int)):
            for _ in range(int(count)):
                self.patient_queues[i].append(self.patient_model.enroll(self.rng, epoch=self.t))
            self.cumulative_enrolled += float(count)

    def _at_risk_unserved_counts(self) -> np.ndarray:
        threshold = self.env_config.patient.eligibility_threshold + self.env_config.urgency_margin
        return np.array(
            [float(sum(1 for p in q if p.survival < threshold)) for q in self.patient_queues],
            dtype=float,
        )

    # -- Public accessors for condition-aware policies -----------------------
    def waiting_counts(self) -> np.ndarray:
        """Per-clinic number of eligible waiting patients."""

        return self._waiting_counts()

    def at_risk_counts(self) -> np.ndarray:
        """Per-clinic count of waiting patients within the urgency margin."""

        return self._at_risk_unserved_counts()

    def near_expiry_counts(self) -> np.ndarray:
        """Per-clinic count of waiting patients close to material expiry."""

        near_expiry_age = self.env_config.material_shelf_life - self.env_config.expiry_warning_margin
        return np.array(
            [float(sum(1 for p in q if p.age >= near_expiry_age)) for q in self.patient_queues],
            dtype=float,
        )

    def risk_type_counts(self) -> np.ndarray:
        """Per-clinic waiting-patient counts by risk type."""

        risk_types = len(self.patient_model.risk_decay_multipliers)
        counts = np.zeros((self.config.num_facilities, risk_types), dtype=float)
        for i, queue in enumerate(self.patient_queues):
            for patient in queue:
                counts[i, int(patient.risk_type)] += 1.0
        return counts

    def _eligibility_rate(self) -> float:
        resolved = self.cumulative_served + self.cumulative_lost
        return self.cumulative_served / max(resolved, 1.0)

    def _viability_fn(self, age: int, transport_time: float) -> float:
        # Placeholder cold-chain curve; only used when enable_viability_hook=True.
        return float(np.exp(-0.15 * (age + transport_time)))


_PATIENT_CONFIG_KEYS = frozenset(
    {
        "patient",
        "material_shelf_life",
        "finished_shelf_life",
        "weight_patient_lost",
        "weight_expiry",
        "weight_urgency",
        "urgency_margin",
        "enable_viability_hook",
        "survival_bucket_edges",
        "expiry_warning_margin",
    }
)


def _to_tuple(value):
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    return value


def patient_env_config_from_dict(env_config: dict) -> PatientEnvConfig:
    """Build a ``PatientEnvConfig`` from a flat JSON-style env dict.

    Keys in ``_PATIENT_CONFIG_KEYS`` configure the patient layer (``patient`` is
    a nested dict for :class:`PatientConditionConfig`); all other keys configure
    the base :class:`CapacityPlanningConfig`.
    """

    cfg = dict(env_config)
    cfg.pop("env_type", None)
    patient_kwargs: dict = {}
    base_kwargs: dict = {}
    for key, value in cfg.items():
        if key in _PATIENT_CONFIG_KEYS:
            patient_kwargs[key] = value
        else:
            base_kwargs[key] = _to_tuple(value)

    patient_sub = patient_kwargs.pop("patient", None)
    patient_cond = PatientConditionConfig(**patient_sub) if patient_sub else PatientConditionConfig()
    if "survival_bucket_edges" in patient_kwargs:
        patient_kwargs["survival_bucket_edges"] = tuple(patient_kwargs["survival_bucket_edges"])
    base = CapacityPlanningConfig(**base_kwargs)
    return PatientEnvConfig(base=base, patient=patient_cond, **patient_kwargs)
