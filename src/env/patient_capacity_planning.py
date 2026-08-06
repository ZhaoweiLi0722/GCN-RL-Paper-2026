"""Patient-condition capacity-planning environment (Phase 4, task group 3).

Extends the base PRM capacity-planning environment so that **patients drive
demand** (approach a): each clinic holds individual patients whose health
deteriorates both while they wait and while their autologous therapy is in
manufacturing (`patient_condition`). The count of *eligible waiting patients* is
the specimen supply the manufacturing dynamics consume.

Two loss channels are added on top of the base cost:
- **Patients lost** — survival falls below the eligibility threshold before
  infusion, including during manufacturing.
- **Material wasted** — a waiting specimen ages past its shelf life, or finished
  product expires before delivery (`aging_inventory`), or an in-process therapy
  is discarded after the patient becomes ineligible, plus an urgency penalty on
  at-risk patients left unserved.

Modeling notes:
- Specimens are **identity-bound** (autologous): one patient's material cannot be
  pooled, split, or substituted for another. Patient-indexed pre-manufacturing
  routing is available behind ``enable_specimen_routing`` and is disabled by
  default for backward-compatible no-routing experiments.
- Reagent and capacity transfers may arrive through the base delayed transfer
  pipeline when ``transfer_lead_time > 0``. Specimen routing has its own explicit
  0/1-epoch lead time and never reuses that generic resource delay.
- A bioreactor attached to a discarded therapy is cleaned during the current
  epoch and returns to the idle pool at the next decision epoch.
- Requires ``action_mode == "facility_net"``.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

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
    PatientState,
    PatientStatus,
)
from src.env.specimen_routing import (
    ProductReturnTransit,
    RoutingExecution,
    SpecimenTransit,
    execute_patient_indexed_routes,
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
    enable_specimen_routing: bool = False
    include_specimen_routing_state: bool = False
    specimen_routing_lead_time_epochs: int = 1
    max_specimen_transfers_per_patient: int = 1
    specimen_transit_loss_probability: float = 0.0
    finished_product_return_lead_time_epochs: int = 0
    finished_product_return_assumption: str = "return_to_collection_facility"
    # Decision-aligned interior survival-bucket edges for the observation summary.
    # Default gives 4 buckets over waiting patients (all have survival >=
    # threshold): [thr, 0.85) critical, [0.85, 0.90) watch, [0.90, 0.97) healthy,
    # [0.97, 1.0] fresh. Config-driven so the Phase 7 pilot can sweep granularity.
    survival_bucket_edges: tuple[float, ...] = (0.85, 0.90, 0.97)
    expiry_warning_margin: int = 1        # "near expiry" = age >= shelf_life - margin


class PatientConditionCapacityEnv(CapacityPlanningEnv):
    """Base capacity-planning env with per-patient deterioration and expiry."""

    def __init__(self, config: PatientEnvConfig | None = None, seed: int | None = None):
        self._construction_seed = seed
        self.env_config = config or PatientEnvConfig()
        self.patient_model = PatientConditionModel(self.env_config.patient)
        self._validate_patient_routing_config()
        self._summary_edges = np.asarray(self.env_config.survival_bucket_edges, dtype=float)
        if self._summary_edges.ndim != 1 or (np.diff(self._summary_edges) <= 0).any():
            raise ValueError("survival_bucket_edges must be strictly increasing")
        # Six legacy scalars, a waiting histogram, and an optional matched
        # routing-state block used by both routing and no-routing control arms.
        self.routing_summary_width = (
            4 if self.env_config.include_specimen_routing_state else 0
        )
        self.summary_width = (
            6 + len(self._summary_edges) + 1 + self.routing_summary_width
        )
        # base __init__ calls self.reset(), which needs the attributes above.
        super().__init__(self.env_config.base, seed)
        if self.config.action_mode != "facility_net":
            raise ValueError("PatientConditionCapacityEnv requires action_mode='facility_net'")
        # The patient summary is appended as a per-clinic block; extend the sizes.
        self.base_observation_size = self.config.num_facilities * self.features_per_facility
        self.observation_size = (
            self.base_observation_size
            + self.config.num_facilities * self.summary_width
            + int(self.config.include_time_state)
        )

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> np.ndarray:
        super().reset(seed)
        n = self.config.num_facilities
        self._episode_seed = int(
            seed
            if seed is not None
            else (self._construction_seed if self._construction_seed is not None else 0)
        )
        self._next_patient_sequence = 0
        viability_fn = self._viability_fn if self.env_config.enable_viability_hook else None
        self.patient_queues = [[] for _ in range(n)]
        self.patient_registry: dict[str, PatientState] = {}
        self.specimen_transits: list[SpecimenTransit] = []
        self.product_return_transits: list[ProductReturnTransit] = []
        self.route_history: list[dict[str, object]] = []
        lead_time = self.config.production_lead_time
        self.in_production_patients: list[list[list[PatientState]]] = [
            [[] for _ in range(lead_time)] for _ in range(n)
        ]
        self.finished_product = [
            AgingInventory(self.env_config.finished_shelf_life, viability_fn) for _ in range(n)
        ]
        # Seed initial waiting patients from the base initial specimen counts.
        for i in range(n):
            for _ in range(int(round(float(self.specimens[i])))):
                self.patient_queues[i].append(self._enroll_patient(i, epoch=0))
        self.specimens = self._waiting_counts()
        self.cumulative_enrolled = float(self.specimens.sum())
        self.cumulative_lost = 0.0
        self.cumulative_served = 0.0
        self.cumulative_started = 0.0
        self.cumulative_manufacturing_lost = 0.0
        self.cumulative_turnaround_time = 0.0
        self.cumulative_specimen_routes = 0.0
        self.cumulative_specimen_route_distance = 0.0
        self.cumulative_specimen_route_time_hours = 0.0
        self.cumulative_specimen_route_cost = 0.0
        self.cumulative_blocked_specimen_requests = 0.0
        self.cumulative_transit_loss = 0.0
        self.cumulative_transit_expiry = 0.0
        self.risk_type_count_recoveries = 0
        self.assert_identity_conservation()
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
        if self.config.include_time_state:
            base, time_state = base[:-1], base[-1:]
        else:
            time_state = np.empty(0, dtype=np.float32)
        summary = self._patient_summary().reshape(-1)
        return np.concatenate([base, summary, time_state]).astype(np.float32)

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

    def _pending_transfer_arrivals(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pending_specimens, pending_reagents, pending_capacity = (
            super()._pending_transfer_arrivals()
        )
        if (
            self.env_config.enable_specimen_routing
            and hasattr(self, "specimen_transits")
        ):
            pending_specimens = self._in_transit_counts()
        return pending_specimens, pending_reagents, pending_capacity

    def _patient_summary(self) -> np.ndarray:
        """Per-clinic waiting/manufacturing risk summary.

        Columns are ``[waiting_count, waiting_mean_survival, near_expiry_count,
        in_production_count, in_production_mean_survival,
        in_production_at_risk_count, waiting_survival_histogram...]``. When
        enabled, the trailing routing block is ``[in_transit_count,
        in_transit_mean_survival, transferred_waiting_count,
        route_eligible_waiting_count]``.
        """

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
            rows[i, 6 : 6 + buckets] = histogram
        for i in range(n):
            in_production = [
                patient
                for stage in self.in_production_patients[i][1:]
                for patient in stage
            ]
            if not in_production:
                continue
            survivals = np.array([p.survival for p in in_production], dtype=float)
            threshold = (
                self.env_config.patient.eligibility_threshold + self.env_config.urgency_margin
            )
            rows[i, 3] = float(len(in_production))
            rows[i, 4] = float(survivals.mean())
            rows[i, 5] = float((survivals < threshold).sum())
        if self.routing_summary_width:
            routing_start = 6 + buckets
            transits_by_destination: list[list[PatientState]] = [
                [] for _ in range(n)
            ]
            for transit in self.specimen_transits:
                transits_by_destination[transit.destination_facility].append(
                    self.patient_registry[transit.patient_id]
                )
            for i, queue in enumerate(self.patient_queues):
                transit_patients = transits_by_destination[i]
                rows[i, routing_start] = float(len(transit_patients))
                if transit_patients:
                    rows[i, routing_start + 1] = float(
                        np.mean([patient.survival for patient in transit_patients])
                    )
                rows[i, routing_start + 2] = float(
                    sum(1 for patient in queue if patient.transfer_count > 0)
                )
                rows[i, routing_start + 3] = float(
                    sum(
                        1
                        for patient in queue
                        if patient.transfer_count
                        < self.env_config.max_specimen_transfers_per_patient
                    )
                )
        return rows

    # ------------------------------------------------------------------- step
    def step(self, action):
        n = self.config.num_facilities
        costs = self.config.costs
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"Expected action shape {(self.action_size,)}, got {action_array.shape}")
        normalized = np.clip(action_array, -1.0, 1.0)

        base_specimen_arrivals, reagent_arrivals, capacity_arrivals = (
            self._receive_transfer_arrivals()
        )
        routed_specimen_arrivals = self._receive_specimen_route_arrivals()
        supplier_available = self.supplier_available.copy()
        specimen_requests = normalized[:n] * self.config.max_specimen_transfer
        reagent_transfer_requests = normalized[n : 2 * n] * self.config.max_reagent_transfer
        capacity_requests = normalized[2 * n : 3 * n] * self.config.max_bioreactor_transfer
        replenishment = (
            ((normalized[3 * n : 4 * n] + 1.0) / 2.0)
            * self.max_reagent_replenishment
            * supplier_available
        )

        # Decode facility-net specimen pressure into concrete patient lots
        # before production. Lead-time 0 routes are available immediately;
        # lead-time 1 routes age exactly once in the dedicated transit queue and
        # arrive before the next epoch's production decision.
        routing = self._execute_specimen_routing(specimen_requests)
        self.specimen_transits.extend(routing.transits)
        self.route_history.extend(routing.events)

        # 1) Start the most-urgent eligible patients, bounded by idle
        #    bioreactors and reagents. A start is not counted as treatment until
        #    the therapy completes manufacturing and is infused.
        idle_bioreactors = self.bioreactors[:, 0]
        waiting = self._waiting_counts()
        production = np.floor(
            np.minimum.reduce((waiting, idle_bioreactors, self.reagents))
        ).astype(float)
        started_patients = self._start_production(production)

        # 2) During the epoch, waiting and in-production patients both
        #    deteriorate. In-process therapies are discarded when their matched
        #    patient becomes ineligible; the associated bioreactor is cleaned
        #    and returns idle at the next decision epoch.
        lost_waiting_ineligible, lost_expired = self._age_and_gate_patients()
        (
            lost_transit_ineligible,
            lost_transit_expired,
            lost_transit_transport,
        ) = self._age_and_gate_transit_patients()
        completed_patients, lost_manufacturing = self._advance_manufacturing_patients(
            started_patients
        )
        completed_counts = np.array(
            [float(len(patients)) for patients in completed_patients], dtype=float
        )

        # 3) Resource bookkeeping (reagents + patient-aligned bioreactor
        #    pipeline), then reagent/capacity transfers. Specimens remain
        #    identity-bound and cannot be pooled.
        next_reagents = self.reagents - production + replenishment
        next_bioreactors = np.zeros_like(self.bioreactors)
        next_bioreactors[:, 0] = (
            self.bioreactors[:, 0] - production + completed_counts + lost_manufacturing
        )
        for i in range(n):
            for stage in range(1, self.config.production_lead_time):
                next_bioreactors[i, stage] = float(
                    len(self.in_production_patients[i][stage])
                )

        specimen_net = routing.actual_net
        specimen_flows = routing.edge_flows
        if self.config.transfer_lead_time > 0:
            reagent_net, reagent_flows, reagent_future_arrivals = _apply_net_transfers_delayed(
                next_reagents,
                self.resource_edges,
                reagent_transfer_requests,
                edge_priorities=self.reagent_transfer_priorities,
            )
            capacity_net, capacity_flows, capacity_future_arrivals = _apply_net_transfers_delayed(
                next_bioreactors[:, 0],
                self.capacity_edges,
                capacity_requests,
                edge_priorities=self.capacity_transfer_priorities,
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
                next_reagents,
                self.resource_edges,
                reagent_transfer_requests,
                edge_priorities=self.reagent_transfer_priorities,
            )
            capacity_net, capacity_flows = _apply_net_transfers(
                next_bioreactors[:, 0],
                self.capacity_edges,
                capacity_requests,
                edge_priorities=self.capacity_transfer_priorities,
            )

        self.reagents = np.clip(next_reagents, 0.0, self.max_reagents)
        next_bioreactors[:, 0] = np.clip(next_bioreactors[:, 0], 0.0, self.max_idle_bioreactors)
        self.bioreactors = np.maximum(next_bioreactors, 0.0)

        # 4) Completed therapies enter finished inventory and are infused
        #    locally. Existing inventory ages first, so a future delivery
        #    constraint can use the same hook without changing patient identity.
        (
            finished_expired,
            delivered_counts,
            lost_return_ineligible,
            delivered_turnaround_time,
        ) = self._age_and_deliver_finished(completed_patients)

        # 5) New patient arrivals (demand) enroll into the queues.
        current_demand = self.demand.copy()
        self._enroll_arrivals(current_demand)
        self.specimens = self._waiting_counts()

        # 6) Costs: base terms + patient/expiry/urgency terms.
        under_reagents = np.maximum(self.specimens - self.reagents, 0.0)
        idle_reagents = np.maximum(self.reagents - self.specimens, 0.0)
        under_bioreactors = np.maximum(self.specimens - self.bioreactors[:, 0], 0.0)
        idle_bioreactor_counts = np.maximum(self.bioreactors[:, 0] - self.specimens, 0.0)

        patients_lost_ineligible = (
            lost_waiting_ineligible
            + lost_transit_ineligible
            + lost_manufacturing
            + lost_return_ineligible
        )
        patients_lost_expired = (
            lost_expired + lost_transit_expired + finished_expired
        )
        patients_lost = (
            patients_lost_ineligible
            + patients_lost_expired
            + lost_transit_transport
        )
        material_wasted = (
            lost_expired
            + lost_transit_expired
            + lost_transit_transport
            + finished_expired
            + lost_manufacturing
        )
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
        specimen_transfer_cost = float(
            sum(float(event["transfer_cost"]) for event in routing.events)
        )

        operating_cost_components = self._operating_cost_components(
            replenishment=replenishment,
            idle_reagents=idle_reagents,
            under_reagents=under_reagents,
            idle_bioreactors=idle_bioreactor_counts,
            under_bioreactors=under_bioreactors,
            specimen_transfer_cost=specimen_transfer_cost,
            capacity_transfer_cost=capacity_transfer_cost,
            reagent_transfer_cost=reagent_transfer_cost,
        )
        base_cost = float(sum(operating_cost_components.values()))
        patient_cost_components = {
            "patient_loss_cost": self.env_config.weight_patient_lost
            * float(patients_lost.sum()),
            "expiry_cost": self.env_config.weight_expiry * float(material_wasted.sum()),
            "urgency_cost": self.env_config.weight_urgency
            * float(at_risk_unserved.sum()),
        }
        cost = float(base_cost + sum(patient_cost_components.values()))

        self.cumulative_lost += float(patients_lost.sum())
        self.cumulative_started += float(production.sum())
        self.cumulative_manufacturing_lost += float(lost_manufacturing.sum())
        self.cumulative_served += float(delivered_counts.sum())
        self.cumulative_turnaround_time += float(delivered_turnaround_time)
        self.cumulative_specimen_routes += float(len(routing.events))
        self.cumulative_specimen_route_distance += float(
            sum(float(event["distance_miles"]) for event in routing.events)
        )
        self.cumulative_specimen_route_time_hours += float(
            sum(float(event["transport_time_hours"]) for event in routing.events)
        )
        self.cumulative_specimen_route_cost += specimen_transfer_cost
        self.cumulative_blocked_specimen_requests += float(
            routing.blocked_requests
        )
        self.cumulative_transit_loss += float(lost_transit_transport.sum())
        self.cumulative_transit_expiry += float(lost_transit_expired.sum())
        self._update_running_metrics(current_demand, production, self.specimens, self.bioreactors)
        if np.any(under_reagents > 0):
            self.reagent_shortage_steps += 1
        if np.any(under_bioreactors > 0):
            self.bioreactor_shortage_steps += 1
        done = self._advance_clock()
        identity_audit = self.assert_identity_conservation()

        specimen_arrivals = (
            base_specimen_arrivals
            + routed_specimen_arrivals
            + routing.immediate_arrivals
        )

        info: dict[str, Any] = {
            "cost": cost,
            "base_cost": base_cost,
            **operating_cost_components,
            **patient_cost_components,
            "production": production.copy(),
            "demand": current_demand.copy(),
            "replenishment": replenishment.copy(),
            "patients_lost": patients_lost.copy(),
            "patients_lost_ineligible": patients_lost_ineligible.copy(),
            "patients_lost_waiting_ineligible": lost_waiting_ineligible.copy(),
            "patients_lost_manufacturing": lost_manufacturing.copy(),
            "patients_lost_expired": patients_lost_expired.copy(),
            "patients_lost_waiting_expired": lost_expired.copy(),
            "patients_lost_transit_ineligible": lost_transit_ineligible.copy(),
            "patients_lost_transit_expired": lost_transit_expired.copy(),
            "patients_lost_transit_transport": lost_transit_transport.copy(),
            "patients_lost_return_ineligible": lost_return_ineligible.copy(),
            "patients_started": production.copy(),
            "patients_completed": delivered_counts.copy(),
            "therapies_discarded": lost_manufacturing.copy(),
            "bioreactors_released_after_cleaning": lost_manufacturing.copy(),
            "material_wasted": material_wasted.copy(),
            "finished_expired": finished_expired.copy(),
            "at_risk_unserved": at_risk_unserved.copy(),
            "risk_type_counts": self.risk_type_counts().copy(),
            "in_production_risk_type_counts": self.in_production_risk_type_counts().copy(),
            "risk_type_count_recoveries": float(self.risk_type_count_recoveries),
            "waiting_patients": self.specimens.copy(),
            "in_production_patients": self._in_production_counts(),
            "eligibility_rate": self._eligibility_rate(),
            "completion_service_level": self._completion_service_level(),
            "patient_ineligibility_during_manufacturing_rate": (
                self._manufacturing_loss_rate()
            ),
            # Compatibility alias for pre-calibration result readers.
            "manufacturing_loss_rate": self._manufacturing_loss_rate(),
            "average_turnaround_time": self._average_turnaround_time(),
            "under_reagents": under_reagents.copy(),
            "under_bioreactors": under_bioreactors.copy(),
            "specimen_transfer_arrivals": specimen_arrivals.copy(),
            "reagent_transfer_arrivals": reagent_arrivals.copy(),
            "capacity_transfer_arrivals": capacity_arrivals.copy(),
            "specimen_transfers": specimen_net.copy(),
            "specimen_edge_flows": specimen_flows.copy(),
            "specimen_requested_integer_net": routing.requested_integer_net.copy(),
            "specimen_route_count": float(len(routing.events)),
            "specimen_route_count_cumulative": self.cumulative_specimen_routes,
            "specimen_route_distance_miles": float(
                sum(float(event["distance_miles"]) for event in routing.events)
            ),
            "specimen_route_time_hours": float(
                sum(float(event["transport_time_hours"]) for event in routing.events)
            ),
            "specimen_route_cost": specimen_transfer_cost,
            "blocked_specimen_requests": float(routing.blocked_requests),
            "blocked_specimen_inbound_requests": float(
                routing.blocked_inbound_requests
            ),
            "blocked_specimen_outbound_requests": float(
                routing.blocked_outbound_requests
            ),
            "transferred_patient_ids": tuple(
                str(event["patient_id"]) for event in routing.events
            ),
            "specimen_route_events": tuple(dict(event) for event in routing.events),
            "transit_loss": lost_transit_transport.copy(),
            "transit_expiry": lost_transit_expired.copy(),
            "specimen_in_transit": self._in_transit_counts(),
            "finished_product_return_assumption": (
                self.env_config.finished_product_return_assumption
            ),
            "finished_product_return_lead_time_epochs": float(
                self.env_config.finished_product_return_lead_time_epochs
            ),
            "identity_active_count": float(identity_audit["active_count"]),
            "identity_terminal_count": float(identity_audit["terminal_count"]),
            "capacity_transfers": capacity_net.copy(),
            "reagent_transfers": reagent_net.copy(),
            "transshipment_cost": (
                specimen_transfer_cost
                + reagent_transfer_cost
                + capacity_transfer_cost
            ),
        }
        info.update(self._performance_info())
        return self.observation(), -cost, done, info

    # -------------------------------------------------------------- internals
    def _waiting_counts(self) -> np.ndarray:
        return np.array([float(len(q)) for q in self.patient_queues], dtype=float)

    def _in_transit_counts(self) -> np.ndarray:
        counts = np.zeros(self.config.num_facilities, dtype=float)
        for transit in self.specimen_transits:
            counts[transit.destination_facility] += 1.0
        return counts

    def _execute_specimen_routing(
        self,
        specimen_requests: np.ndarray,
    ) -> RoutingExecution:
        n = self.config.num_facilities
        if not self.env_config.enable_specimen_routing:
            zeros = np.zeros(n, dtype=float)
            return RoutingExecution(
                requested_integer_net=zeros.copy(),
                actual_net=zeros.copy(),
                edge_flows=np.zeros(len(self.specimen_edges), dtype=float),
                immediate_arrivals=zeros.copy(),
                transits=(),
                events=(),
                blocked_requests=0,
                blocked_inbound_requests=0,
                blocked_outbound_requests=0,
            )
        edge_distances = tuple(
            self._edge_distance(edge) for edge in self.specimen_edges
        )
        edge_transport_hours = tuple(
            self._edge_transfer_time_hours(edge) for edge in self.specimen_edges
        )
        edge_costs = tuple(
            float(self.config.costs.specimen_transfer)
            * (
                1.0
                + float(self.config.geographic_transfer_cost_scale)
                * (float(distance) / 1000.0)
                + float(self.config.geographic_transfer_time_cost_scale)
                * float(transport_hours)
            )
            for distance, transport_hours in zip(
                edge_distances,
                edge_transport_hours,
            )
        )
        return execute_patient_indexed_routes(
            self.patient_queues,
            specimen_requests,
            self.specimen_edges,
            edge_priorities=self.specimen_transfer_priorities,
            edge_distances=edge_distances,
            edge_transport_hours=edge_transport_hours,
            edge_transfer_costs=edge_costs,
            lead_time_epochs=self.env_config.specimen_routing_lead_time_epochs,
            max_transfers_per_patient=(
                self.env_config.max_specimen_transfers_per_patient
            ),
            route_epoch=self.t,
        )

    def _receive_specimen_route_arrivals(self) -> np.ndarray:
        arrivals = np.zeros(self.config.num_facilities, dtype=float)
        remaining: list[SpecimenTransit] = []
        for transit in self.specimen_transits:
            if transit.remaining_epochs > 0:
                remaining.append(transit)
                continue
            patient = self.patient_registry[transit.patient_id]
            if patient.status is not PatientStatus.IN_TRANSIT:
                raise RuntimeError(
                    f"Transit patient has invalid status: {patient.patient_id}"
                )
            patient.status = PatientStatus.WAITING
            patient.material_facility = transit.destination_facility
            self.patient_queues[transit.destination_facility].append(patient)
            arrivals[transit.destination_facility] += 1.0
        self.specimen_transits = remaining
        return arrivals

    def _age_and_gate_patients(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.config.num_facilities
        lost_ineligible = np.zeros(n, dtype=float)
        lost_expired = np.zeros(n, dtype=float)
        shelf = self.env_config.material_shelf_life
        for i, queue in enumerate(self.patient_queues):
            survivors = []
            for patient in queue:
                self.patient_model.advance(patient)
                patient.specimen_age += 1
                if patient.specimen_age >= shelf:
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

    def _age_and_gate_transit_patients(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.num_facilities
        lost_ineligible = np.zeros(n, dtype=float)
        lost_expired = np.zeros(n, dtype=float)
        lost_transport = np.zeros(n, dtype=float)
        survivors: list[SpecimenTransit] = []
        for transit in self.specimen_transits:
            patient = self.patient_registry[transit.patient_id]
            destination = transit.destination_facility
            if patient.status is not PatientStatus.IN_TRANSIT:
                raise RuntimeError(
                    f"Transit patient has invalid status: {patient.patient_id}"
                )
            self.patient_model.advance(patient)
            patient.specimen_age += 1
            if patient.specimen_age >= self.env_config.material_shelf_life:
                patient.status = PatientStatus.LOST
                lost_expired[destination] += 1.0
                continue
            if not self.patient_model.is_eligible(patient):
                patient.status = PatientStatus.LOST
                lost_ineligible[destination] += 1.0
                continue
            if self._specimen_transport_is_lost(patient, transit):
                patient.status = PatientStatus.LOST
                lost_transport[destination] += 1.0
                continue
            survivors.append(
                SpecimenTransit(
                    patient_id=transit.patient_id,
                    origin_facility=transit.origin_facility,
                    destination_facility=transit.destination_facility,
                    remaining_epochs=max(transit.remaining_epochs - 1, 0),
                    lead_time_epochs=transit.lead_time_epochs,
                    distance_miles=transit.distance_miles,
                    transport_time_hours=transit.transport_time_hours,
                    transfer_cost=transit.transfer_cost,
                    route_epoch=transit.route_epoch,
                )
            )
        self.specimen_transits = survivors
        return lost_ineligible, lost_expired, lost_transport

    def _start_production(self, production: np.ndarray) -> list[list[PatientState]]:
        started: list[list[PatientState]] = []
        for i, count in enumerate(production.astype(int)):
            served = self.patient_queues[i][:count]
            for patient in served:
                patient.status = PatientStatus.IN_PRODUCTION
                patient.material_facility = i
                patient.manufacturing_facility = i
            started.append(served)
            self.patient_queues[i] = self.patient_queues[i][count:]
        return started

    def _advance_manufacturing_patients(
        self,
        started_patients: list[list[PatientState]],
    ) -> tuple[list[list[PatientState]], np.ndarray]:
        """Advance patient-aligned therapies by one manufacturing epoch."""

        n = self.config.num_facilities
        lead_time = self.config.production_lead_time
        next_pipeline: list[list[list[PatientState]]] = [
            [[] for _ in range(lead_time)] for _ in range(n)
        ]
        completed: list[list[PatientState]] = [[] for _ in range(n)]
        lost = np.zeros(n, dtype=float)

        def advance_one(patient: PatientState, facility: int, next_stage: int) -> None:
            self.patient_model.advance(patient)
            if not self.patient_model.is_eligible(patient):
                patient.status = PatientStatus.LOST
                lost[facility] += 1.0
            elif next_stage == 0:
                patient.status = PatientStatus.FINISHED
                completed[facility].append(patient)
            else:
                next_pipeline[facility][next_stage].append(patient)

        for i in range(n):
            for stage in range(1, lead_time):
                for patient in self.in_production_patients[i][stage]:
                    advance_one(patient, i, stage - 1)
            for patient in started_patients[i]:
                advance_one(patient, i, lead_time - 1)

        self.in_production_patients = next_pipeline
        return completed, lost

    def _age_and_deliver_finished(
        self,
        completed_patients: list[list[PatientState]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n = self.config.num_facilities
        expired = np.zeros(n, dtype=float)
        delivered = np.zeros(n, dtype=float)
        lost_ineligible = np.zeros(n, dtype=float)
        delivered_turnaround_time = 0.0
        surviving_returns: list[ProductReturnTransit] = []
        for transit in self.product_return_transits:
            patient = self.patient_registry[transit.patient_id]
            infusion_facility = transit.infusion_facility
            self.patient_model.advance(patient)
            next_age = int(transit.age_epochs) + 1
            if next_age >= self.env_config.finished_shelf_life:
                patient.status = PatientStatus.LOST
                expired[infusion_facility] += 1.0
                continue
            if not self.patient_model.is_eligible(patient):
                patient.status = PatientStatus.LOST
                lost_ineligible[infusion_facility] += 1.0
                continue
            remaining = max(int(transit.remaining_epochs) - 1, 0)
            if remaining == 0:
                patient.status = PatientStatus.DELIVERED
                delivered[infusion_facility] += 1.0
                delivered_turnaround_time += float(patient.age)
            else:
                surviving_returns.append(
                    ProductReturnTransit(
                        patient_id=transit.patient_id,
                        manufacturing_facility=transit.manufacturing_facility,
                        infusion_facility=transit.infusion_facility,
                        remaining_epochs=remaining,
                        lead_time_epochs=transit.lead_time_epochs,
                        age_epochs=next_age,
                    )
                )
        self.product_return_transits = surviving_returns

        return_lead_time = int(
            self.env_config.finished_product_return_lead_time_epochs
        )
        for manufacturing_facility, patients in enumerate(completed_patients):
            inventory = self.finished_product[manufacturing_facility]
            inventory.advance()
            inventory.add(float(len(patients)))
            inventory.consume(inventory.total())
            for patient in patients:
                infusion_facility = int(patient.collection_facility)
                if return_lead_time == 0:
                    patient.status = PatientStatus.DELIVERED
                    delivered[infusion_facility] += 1.0
                    delivered_turnaround_time += float(patient.age)
                else:
                    patient.status = PatientStatus.FINISHED
                    self.product_return_transits.append(
                        ProductReturnTransit(
                            patient_id=patient.patient_id,
                            manufacturing_facility=manufacturing_facility,
                            infusion_facility=infusion_facility,
                            remaining_epochs=return_lead_time,
                            lead_time_epochs=return_lead_time,
                        )
                    )
        return (
            expired,
            delivered,
            lost_ineligible,
            delivered_turnaround_time,
        )

    def _enroll_arrivals(self, demand: np.ndarray) -> None:
        for i, count in enumerate(demand.astype(int)):
            for _ in range(int(count)):
                self.patient_queues[i].append(
                    self._enroll_patient(i, epoch=self.t)
                )
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
                risk_type = self._patient_risk_type_index(
                    patient,
                    facility=i,
                    stage="waiting",
                )
                counts[i, risk_type] = float(counts[i, risk_type]) + 1.0
        return counts

    def in_production_risk_type_counts(self) -> np.ndarray:
        """Per-clinic in-production patient counts by risk type."""

        risk_types = len(self.patient_model.risk_decay_multipliers)
        counts = np.zeros((self.config.num_facilities, risk_types), dtype=float)
        for i, stages in enumerate(self.in_production_patients):
            for stage_index, stage in enumerate(stages[1:], start=1):
                for patient in stage:
                    risk_type = self._patient_risk_type_index(
                        patient,
                        facility=i,
                        stage=stage_index,
                    )
                    counts[i, risk_type] = float(counts[i, risk_type]) + 1.0
        return counts

    def _patient_risk_type_index(
        self,
        patient: PatientState,
        *,
        facility: int,
        stage: int | str,
    ) -> int:
        """Return a valid risk index, using the active multiplier as a safe fallback."""

        configured = np.asarray(
            self.patient_model.risk_decay_multipliers,
            dtype=float,
        )
        value = patient.risk_type
        if isinstance(value, (int, np.integer)) and not isinstance(
            value,
            (bool, np.bool_),
        ):
            index = int(value)
            if 0 <= index < configured.size:
                return index

        multiplier = patient.risk_multiplier
        try:
            multiplier_value = float(multiplier)
        except (TypeError, ValueError, OverflowError):
            matches = np.empty(0, dtype=int)
        else:
            matches = np.flatnonzero(
                np.isclose(
                    configured,
                    multiplier_value,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        if matches.size == 1:
            self.risk_type_count_recoveries += 1
            return int(matches[0])

        patient_type = f"{type(patient).__module__}.{type(patient).__qualname__}"
        raise TypeError(
            "Cannot resolve patient risk type for info accounting: "
            f"epoch={self.t}, facility={facility}, stage={stage!r}, "
            f"patient_type={patient_type}, risk_type={value!r}, "
            f"risk_type_type={type(value).__name__}, "
            f"risk_multiplier={multiplier!r}, "
            f"risk_multiplier_type={type(multiplier).__name__}, "
            f"configured_multipliers={configured.tolist()}"
        )

    def _in_production_counts(self) -> np.ndarray:
        return np.array(
            [
                float(sum(len(stage) for stage in stages[1:]))
                for stages in self.in_production_patients
            ],
            dtype=float,
        )

    def _eligibility_rate(self) -> float:
        resolved = self.cumulative_served + self.cumulative_lost
        return self.cumulative_served / max(resolved, 1.0)

    def _completion_service_level(self) -> float:
        return self.cumulative_served / max(self.cumulative_enrolled, 1.0)

    def _manufacturing_loss_rate(self) -> float:
        return self.cumulative_manufacturing_lost / max(self.cumulative_started, 1.0)

    def _average_turnaround_time(self) -> float:
        return self.cumulative_turnaround_time / max(self.cumulative_served, 1.0)

    def _enroll_patient(self, facility: int, *, epoch: int) -> PatientState:
        patient_id = (
            f"seed{int(self._episode_seed)}:patient"
            f"{int(self._next_patient_sequence):08d}"
        )
        self._next_patient_sequence += 1
        if patient_id in self.patient_registry:
            raise RuntimeError(f"Duplicate patient ID generated: {patient_id}")
        patient = self.patient_model.enroll(
            self.rng,
            epoch=epoch,
            patient_id=patient_id,
            collection_facility=facility,
        )
        self.patient_registry[patient_id] = patient
        return patient

    def _specimen_transport_is_lost(
        self,
        patient: PatientState,
        transit: SpecimenTransit,
    ) -> bool:
        base_probability = float(
            self.env_config.specimen_transit_loss_probability
        )
        viability = 1.0
        if self.env_config.enable_viability_hook:
            transport_weeks = float(transit.transport_time_hours) / (24.0 * 7.0)
            viability = self._viability_fn(
                patient.specimen_age,
                transport_weeks,
            )
        loss_probability = 1.0 - (1.0 - base_probability) * viability
        if loss_probability <= 0.0:
            return False
        payload = (
            f"{self._episode_seed}|{patient.patient_id}|"
            f"{transit.route_epoch}|{patient.transfer_count}|transport"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return draw < loss_probability

    def assert_identity_conservation(self) -> dict[str, int]:
        """Validate that every enrolled identity occupies exactly one state."""

        active_occurrences: dict[str, list[str]] = {}

        def record(patient_id: str, location: str) -> None:
            active_occurrences.setdefault(str(patient_id), []).append(location)

        for facility, queue in enumerate(self.patient_queues):
            for patient in queue:
                record(patient.patient_id, f"waiting:{facility}")
                if patient.status is not PatientStatus.WAITING:
                    raise RuntimeError(
                        f"Waiting patient has invalid status: {patient.patient_id}"
                    )
                if patient.material_facility != facility:
                    raise RuntimeError(
                        f"Waiting patient material location mismatch: {patient.patient_id}"
                    )
        for facility, stages in enumerate(self.in_production_patients):
            for stage, patients in enumerate(stages[1:], start=1):
                for patient in patients:
                    record(patient.patient_id, f"production:{facility}:{stage}")
                    if patient.status is not PatientStatus.IN_PRODUCTION:
                        raise RuntimeError(
                            f"Manufacturing patient has invalid status: {patient.patient_id}"
                        )
                    if patient.manufacturing_facility != facility:
                        raise RuntimeError(
                            f"Manufacturing identity location mismatch: {patient.patient_id}"
                        )
        for transit in self.specimen_transits:
            record(transit.patient_id, "specimen_transit")
            patient = self.patient_registry[transit.patient_id]
            if patient.status is not PatientStatus.IN_TRANSIT:
                raise RuntimeError(
                    f"Specimen-transit patient has invalid status: {patient.patient_id}"
                )
            if patient.material_facility is not None:
                raise RuntimeError(
                    f"In-transit specimen cannot occupy a facility: {patient.patient_id}"
                )
        for transit in self.product_return_transits:
            record(transit.patient_id, "product_return")
            patient = self.patient_registry[transit.patient_id]
            if patient.status is not PatientStatus.FINISHED:
                raise RuntimeError(
                    f"Finished-product return has invalid status: {patient.patient_id}"
                )

        terminal_count = 0
        for patient_id, patient in self.patient_registry.items():
            if patient_id != patient.patient_id or patient_id != patient.specimen_id:
                raise RuntimeError(
                    f"Patient/specimen identity substitution detected: {patient_id}"
                )
            if patient.transfer_count < 0 or patient.transfer_count > int(
                self.env_config.max_specimen_transfers_per_patient
            ):
                raise RuntimeError(
                    f"Patient transfer-count contract violated: {patient_id}"
                )
            occurrences = active_occurrences.get(patient_id, [])
            if patient.status in (PatientStatus.DELIVERED, PatientStatus.LOST):
                terminal_count += 1
                if occurrences:
                    raise RuntimeError(
                        f"Terminal patient remains active: {patient_id} {occurrences}"
                    )
            elif len(occurrences) != 1:
                raise RuntimeError(
                    f"Active patient must occupy exactly one state: "
                    f"{patient_id} {occurrences}"
                )
        unknown = set(active_occurrences) - set(self.patient_registry)
        if unknown:
            raise RuntimeError(f"Unregistered patient identities are active: {sorted(unknown)}")
        if len(self.patient_registry) != int(round(self.cumulative_enrolled)):
            raise RuntimeError(
                "Patient registry count does not match cumulative enrollment"
            )
        return {
            "enrolled_count": len(self.patient_registry),
            "active_count": len(active_occurrences),
            "terminal_count": terminal_count,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a full environment snapshot for exact mid-episode recovery."""

        self.assert_identity_conservation()
        arrays = {
            name: np.asarray(getattr(self, name)).copy()
            for name in (
                "demand_shock_remaining",
                "regional_supplier_disruption_remaining",
                "demand_rate_multiplier",
                "demand_regime_multiplier",
                "demand",
                "supplier_available",
                "demand_forecast",
                "specimen_transfer_pipeline",
                "reagent_transfer_pipeline",
                "capacity_transfer_pipeline",
                "specimens",
                "reagents",
                "bioreactors",
                "demand_rates",
                "supplier_disruption_rate",
            )
        }
        scalars = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                "t",
                "demand_forecast_error",
                "cumulative_demand",
                "cumulative_production",
                "cumulative_waiting_specimens",
                "cumulative_bioreactor_capacity",
                "reagent_shortage_steps",
                "bioreactor_shortage_steps",
                "cumulative_enrolled",
                "cumulative_lost",
                "cumulative_served",
                "cumulative_started",
                "cumulative_manufacturing_lost",
                "cumulative_turnaround_time",
                "cumulative_specimen_routes",
                "cumulative_specimen_route_distance",
                "cumulative_specimen_route_time_hours",
                "cumulative_specimen_route_cost",
                "cumulative_blocked_specimen_requests",
                "cumulative_transit_loss",
                "cumulative_transit_expiry",
                "risk_type_count_recoveries",
                "_episode_seed",
                "_next_patient_sequence",
            )
        }
        return {
            "format_version": 1,
            "num_facilities": int(self.config.num_facilities),
            "enable_specimen_routing": bool(
                self.env_config.enable_specimen_routing
            ),
            "include_specimen_routing_state": bool(
                self.env_config.include_specimen_routing_state
            ),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "arrays": arrays,
            "scalars": scalars,
            "demand_history": [value.copy() for value in self.demand_history],
            "forecast_error_history": [
                value.copy() for value in self.forecast_error_history
            ],
            "patients": {
                patient_id: self._patient_to_dict(patient)
                for patient_id, patient in self.patient_registry.items()
            },
            "patient_queues": [
                [patient.patient_id for patient in queue]
                for queue in self.patient_queues
            ],
            "in_production_patients": [
                [
                    [patient.patient_id for patient in stage]
                    for stage in stages
                ]
                for stages in self.in_production_patients
            ],
            "specimen_transits": [transit.__dict__.copy() for transit in self.specimen_transits],
            "product_return_transits": [
                transit.__dict__.copy() for transit in self.product_return_transits
            ],
            "route_history": [dict(event) for event in self.route_history],
            "finished_product_buckets": [
                inventory.age_buckets() for inventory in self.finished_product
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""

        if int(state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported patient-environment state format")
        if int(state.get("num_facilities", -1)) != self.config.num_facilities:
            raise ValueError("Patient-environment state facility count does not match")
        for key, expected in (
            ("enable_specimen_routing", self.env_config.enable_specimen_routing),
            (
                "include_specimen_routing_state",
                self.env_config.include_specimen_routing_state,
            ),
        ):
            if bool(state.get(key)) != bool(expected):
                raise ValueError(f"Patient-environment state contract mismatch: {key}")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        for name, value in dict(state["arrays"]).items():
            setattr(self, name, np.asarray(value).copy())
        for name, value in dict(state["scalars"]).items():
            setattr(self, name, copy.deepcopy(value))
        self.demand_history = [
            np.asarray(value).copy() for value in state["demand_history"]
        ]
        self.forecast_error_history = [
            np.asarray(value).copy()
            for value in state["forecast_error_history"]
        ]
        self.patient_registry = {
            str(patient_id): self._patient_from_dict(dict(payload))
            for patient_id, payload in dict(state["patients"]).items()
        }
        self.patient_queues = [
            [self.patient_registry[str(patient_id)] for patient_id in queue]
            for queue in state["patient_queues"]
        ]
        self.in_production_patients = [
            [
                [self.patient_registry[str(patient_id)] for patient_id in stage]
                for stage in stages
            ]
            for stages in state["in_production_patients"]
        ]
        self.specimen_transits = [
            SpecimenTransit(**dict(payload))
            for payload in state["specimen_transits"]
        ]
        self.product_return_transits = [
            ProductReturnTransit(**dict(payload))
            for payload in state["product_return_transits"]
        ]
        self.route_history = [dict(event) for event in state["route_history"]]
        for inventory, buckets in zip(
            self.finished_product,
            state["finished_product_buckets"],
        ):
            inventory._buckets = np.asarray(buckets, dtype=float).copy()
        self.assert_identity_conservation()

    @staticmethod
    def _patient_to_dict(patient: PatientState) -> dict[str, Any]:
        return {
            "health_index": float(patient.health_index),
            "deterioration_epoch": float(patient.deterioration_epoch),
            "enrollment_epoch": int(patient.enrollment_epoch),
            "patient_id": str(patient.patient_id),
            "specimen_id": str(patient.specimen_id),
            "collection_facility": patient.collection_facility,
            "material_facility": patient.material_facility,
            "manufacturing_facility": patient.manufacturing_facility,
            "transfer_count": int(patient.transfer_count),
            "risk_type": int(patient.risk_type),
            "risk_multiplier": float(patient.risk_multiplier),
            "age": int(patient.age),
            "specimen_age": int(patient.specimen_age),
            "survival": float(patient.survival),
            "status": patient.status.value,
        }

    @staticmethod
    def _patient_from_dict(payload: dict[str, Any]) -> PatientState:
        payload["status"] = PatientStatus(str(payload["status"]))
        return PatientState(**payload)

    def _validate_patient_routing_config(self) -> None:
        if self.env_config.material_shelf_life < 1:
            raise ValueError("material_shelf_life must be positive")
        if self.env_config.finished_shelf_life < 1:
            raise ValueError("finished_shelf_life must be positive")
        if self.env_config.specimen_routing_lead_time_epochs not in (0, 1):
            raise ValueError(
                "specimen_routing_lead_time_epochs must be 0 or 1"
            )
        if self.env_config.finished_product_return_lead_time_epochs not in (0, 1):
            raise ValueError(
                "finished_product_return_lead_time_epochs must be 0 or 1"
            )
        if self.env_config.max_specimen_transfers_per_patient < 1:
            raise ValueError(
                "max_specimen_transfers_per_patient must be positive"
            )
        probability = float(self.env_config.specimen_transit_loss_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "specimen_transit_loss_probability must lie in [0, 1]"
            )
        if self.env_config.finished_product_return_assumption != (
            "return_to_collection_facility"
        ):
            raise ValueError(
                "finished_product_return_assumption must be "
                "'return_to_collection_facility'"
            )
        if self.env_config.enable_specimen_routing:
            if not self.env_config.include_specimen_routing_state:
                raise ValueError(
                    "routing-enabled experiments must expose specimen-routing state"
                )
            if self.env_config.base.specimen_edges is None:
                raise ValueError(
                    "routing-enabled experiments require explicit qualified "
                    "specimen_edges"
                )
            if not tuple(self.env_config.base.specimen_edges):
                raise ValueError(
                    "routing-enabled experiments require at least one qualified "
                    "specimen edge"
                )

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
        "enable_specimen_routing",
        "include_specimen_routing_state",
        "specimen_routing_lead_time_epochs",
        "max_specimen_transfers_per_patient",
        "specimen_transit_loss_probability",
        "finished_product_return_lead_time_epochs",
        "finished_product_return_assumption",
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
