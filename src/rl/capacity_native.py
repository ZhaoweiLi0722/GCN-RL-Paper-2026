"""Opt-in binding of the fixed capacity pilot to native patient mechanics.

Importing and constructing configs performs no science. The native constructor
requires an admission callback; the final runner owns its one-attempt ledger.
Only immutable public arrays are passed to the common MDL-2 decision helper.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import numpy as np

from src.baselines.heuristics import HeuristicSettings, facility_net_action_from_arrays
from src.env.capacity_planning import CapacityPlanningConfig, CostParameters
from src.env.patient_capacity_planning import PatientEnvConfig
from src.env.patient_condition import PatientConditionConfig, PatientState, PatientStatus
from src.env.patient_support_public import PublicPatientSupportCapacityEnv
from src.env.patient_support_work import PatientSupportConfig
from src.env.service_effort_mechanics import ServiceEffortConfig
from src.rl.public_support_input import PublicSupportControlInput


def keyed_seed(proposal, world, purpose):
    payload = {"namespace": proposal["design"]["rng_namespace"], "phase": world["phase"],
               "b": world["block"], "c": world["condition"], "j": world["replicate"], "purpose": purpose}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def native_configs(proposal):
    s, h = proposal["synthetic_system"], proposal["synthetic_system"]["support"]
    base_names = {f.name for f in fields(CapacityPlanningConfig)}
    base = {k: copy.deepcopy(v) for k, v in s.items() if k in base_names}
    base.update(num_facilities=len(s["site_ids"]), episode_horizon=s["host_total_horizon"],
                action_mode="facility_net", include_supplier_state=True,
                include_transfer_pipeline_state=True, include_on_order_state=True,
                include_demand_forecast_state=True, include_time_state=True,
                transfer_lead_time=s["resource_transfer_lead_time"],
                supplier_disruption_rate=s["supplier_disruption_rate"])
    edges = tuple(tuple(x) for x in s["transport_resource_information_edges"])
    base.update({k: edges for k in ("specimen_edges", "resource_edges", "capacity_edges", "information_edges")})
    base["costs"] = CostParameters(**{f.name: s["base_costs"][f.name] for f in fields(CostParameters)})
    patient = PatientConditionConfig(**{f.name: s["patient"][f.name] for f in fields(PatientConditionConfig)})
    env = PatientEnvConfig(base=CapacityPlanningConfig(**base), patient=patient,
        material_shelf_life=s["material_shelf_life"], finished_shelf_life=s["finished_shelf_life"],
        weight_patient_lost=s["base_costs"]["weight_patient_lost"],
        weight_expiry=s["base_costs"]["weight_expiry"], weight_urgency=s["base_costs"]["weight_urgency"],
        urgency_margin=s["patient"]["urgency_margin"], enable_viability_hook=False,
        enable_specimen_routing=True, include_specimen_routing_state=True,
        specimen_routing_lead_time_epochs=s["specimen_routing_lead_time_epochs"],
        max_specimen_transfers_per_patient=s["max_specimen_transfers_per_patient"],
        specimen_transit_loss_probability=s["specimen_transit_loss_probability"],
        finished_product_return_lead_time_epochs=s["finished_product_return_lead_time_epochs"],
        finished_product_return_assumption=s["finished_product_return_assumption"],
        survival_bucket_edges=tuple(s["patient"]["survival_bucket_edges"]),
        expiry_warning_margin=s["patient"]["expiry_warning_margin"])
    effort = ServiceEffortConfig(tuple(h["site_hour_caps"]), h["shared_hour_budget"],
        h["commitment_lead_steps"], h["hourly_cost"], h["quadratic_cost"], h["switching_cost"])
    if h["ordinary_hourly_cost"] != effort.hourly_cost:
        raise ValueError("support ledger requires the declared equal ordinary hourly expense")
    support = PatientSupportConfig(effort, tuple(s["site_ids"]), tuple(h["ordinary_hours_per_site"]),
        h["work_per_patient"], proposal["information"]["history_window"], s["host_total_horizon"])
    return env, support


@dataclass(frozen=True)
class CapacityWorldTape:
    """Privileged exogenous tape, never a controller input."""
    world: dict
    arrivals: tuple[tuple[int, ...], ...]
    responses: tuple[tuple[float, ...], ...]
    patient_attributes: tuple[tuple[float, float], ...]
    change_epoch: int | None

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()


def make_world_tape(proposal, world):
    s = proposal["synthetic_system"]
    horizon, n = s["host_total_horizon"], len(s["site_ids"])
    rng = np.random.default_rng(keyed_seed(proposal, world, "demand"))
    arrivals = np.zeros((horizon, n), dtype=int)
    arrivals[:s["control_epochs"]] = rng.poisson(s["demand_rates"], size=(s["control_epochs"], n))
    r = np.random.default_rng(keyed_seed(proposal, world, "response_change"))
    eta, change = s["support"]["response_values"], None
    if world["condition"] == 0:
        responses = np.tile(r.choice(eta, n), (horizon, 1))
    elif world["condition"] == 1:
        bounds = proposal["conditions"][1]["change_epoch_uniform_inclusive"]
        change = int(r.integers(bounds[0], bounds[1] + 1))
        responses = np.ones((horizon, n))
        responses[change:] = r.permutation([0.5, 0.75, 1.25, 1.5])
    elif world["condition"] == 2:
        responses = r.choice(eta, size=(horizon, n))
    else:
        raise ValueError("unsupported prospective condition")
    patient_rng = np.random.default_rng(keyed_seed(proposal, world, "patient_identity_attributes"))
    count, p = int(arrivals.sum()), s["patient"]
    attributes = tuple((float(patient_rng.uniform()), float(p["weibull_scale"] * patient_rng.weibull(p["weibull_shape"])))
                       for _ in range(count))
    return CapacityWorldTape(copy.deepcopy(world), tuple(tuple(int(x) for x in row) for row in arrivals),
        tuple(tuple(float(x) for x in row) for row in responses), attributes, change)


def common_operation(view: PublicSupportControlInput, proposal, tail=False):
    """Same public MDL-2 array implementation, without a privileged env object."""
    v, s = view.operations, proposal["synthetic_system"]
    n = len(v.site_ids)
    counts = np.asarray([len(x) for x in v.waiting_order], dtype=float)
    purchases = np.asarray(s["max_reagent_replenishment"], dtype=float)
    if tail:
        order = np.zeros(n)
        if v.epoch <= proposal["settlement"]["last_purchase_epoch"]:
            on_order = np.asarray(v.reagent_orders).sum(axis=0) if v.reagent_orders else np.zeros(n)
            order = np.minimum(purchases, np.maximum(0, counts - v.reagents - on_order))
        action = np.zeros(4 * n)
        action[3*n:] = 2 * order / purchases - 1
        return action
    if len(v.current_arrivals) != n:
        raise ValueError("common MDL-2 needs the same public current-demand field as the native policy")
    edges = tuple(tuple(x) for x in s["transport_resource_information_edges"])
    settings = HeuristicSettings(lookahead_periods=s["common_operations"]["lookahead_periods"],
        allow_sharing=s["common_operations"]["allow_sharing"],
        local_order_up_to_multiplier=s["common_operations"]["local_order_up_to_multiplier"])
    return facility_net_action_from_arrays(demand=np.asarray(v.current_arrivals), specimens=counts,
        reagents=np.asarray(v.reagents), bioreactors=np.asarray(v.bioreactors),
        supplier_available=np.asarray(v.supplier_available), demand_forecast=np.asarray(v.demand_forecast),
        demand_history_mean=None, demand_rates=np.asarray(s["demand_rates"]),
        max_reagent_replenishment=purchases, max_specimen_transfer=s["max_specimen_transfer"],
        max_bioreactor_transfer=s["max_bioreactor_transfer"], max_reagent_transfer=s["max_reagent_transfer"],
        specimen_edges=edges, capacity_edges=edges, resource_edges=edges, settings=settings).astype(np.float64)


def resource_totals(host):
    return (float(np.sum(host.reagents) + np.sum(host.reagent_transfer_pipeline)
                  + np.sum(host.reagent_purchase_pipeline)),
            float(np.sum(host.bioreactors) + np.sum(host.capacity_transfer_pipeline)))


class CapacityPilotEnv(PublicPatientSupportCapacityEnv):
    """Fresh versioned native producer; admitted construction/reset/step once."""
    support_format = "capacity-adaptation-native-v1"

    def __init__(self, proposal, tape, before_native):
        if not callable(before_native):
            raise TypeError("an active admission/budget callback is required")
        self._proposal, self._tape = copy.deepcopy(proposal), tape
        self._before_native, self._reset_consumed = before_native, False
        self._tape_digest, self._failed = tape.digest(), False
        self._initial_reactor_total = float(sum(proposal["synthetic_system"]["initial_idle_bioreactors"]))
        self._cumulative_purchases = self._cumulative_consumption = 0.0
        config, support = native_configs(proposal)
        before_native("construction")
        super().__init__(config, support_config=support, response_tape=tape.responses, seed=int(tape.world["seed"]))

    def reset(self, seed=None):
        if self._reset_consumed:
            raise RuntimeError("no additional reset or resume in this single attempt")
        self._reset_consumed = True
        self._before_native("construction_reset")
        super().reset(seed)
        self.demand = np.asarray(self._tape.arrivals[0], dtype=float)
        self.demand_forecast = np.asarray(self._proposal["synthetic_system"]["demand_rates"], dtype=float)
        self.demand_history, self.forecast_error_history = [], []
        self._record_demand_observation()
        return self.observation()

    def _advance_clock(self):
        self.t += 1
        s = self._proposal["synthetic_system"]
        self.demand = (np.asarray(self._tape.arrivals[self.t], dtype=float)
                       if self.t < len(self._tape.arrivals) else np.zeros(len(s["site_ids"])))
        self.demand_forecast = np.asarray(s["demand_rates"] if self.t < s["control_epochs"]
                                          else s["tail_arrival_rates"], dtype=float)
        self.supplier_available = np.ones(len(s["site_ids"]))
        self._record_demand_observation()
        return self.t >= s["host_total_horizon"]

    def _enroll_patient(self, facility, *, epoch):
        index = self._next_patient_sequence
        health, shock = self._tape.patient_attributes[index]
        pid = f"world{self._tape.world['seed']}:patient{index:08d}"
        self._next_patient_sequence += 1
        p = PatientState(health_index=health, deterioration_epoch=shock, enrollment_epoch=int(epoch),
                         patient_id=pid, specimen_id=pid, collection_facility=int(facility),
                         material_facility=int(facility), risk_type=0, risk_multiplier=1.0)
        if pid in self.patient_registry:
            raise ValueError("duplicate exogenous patient identity")
        self.patient_registry[pid] = p
        return p

    def step(self, action):
        if self._failed:
            raise RuntimeError("failed native attempt cannot retry")
        self._before_native("step")
        try:
            before_reagent, before_reactor = resource_totals(self)
            work_before = np.asarray(self.support._delivered, dtype=np.float64)
            result = super().step(action)
            info = result[3]
            bought, started = float(np.sum(info["replenishment"])), float(np.sum(info["production"]))
            after_reagent, after_reactor = resource_totals(self)
            if (not math.isclose(before_reagent + bought - started, after_reagent, rel_tol=1e-10, abs_tol=1e-8)
                    or not math.isclose(before_reactor, after_reactor, rel_tol=1e-10, abs_tol=1e-8)):
                raise ValueError("unaccounted resource clipping or loss; preserve incurred evidence")
            self._cumulative_purchases += bought
            self._cumulative_consumption += started
            work = np.asarray(self.support._delivered, dtype=np.float64) - work_before
            exposure = np.asarray(info["support_public_receipt"]["ordinary_hours"]) + np.asarray(info["support_public_receipt"]["applied_hours"])
            productive = work / np.asarray(self._tape.responses[self.t-1])
            if np.any(productive < -1e-9) or np.any(productive > exposure+1e-9):
                raise ValueError("productive support exposure is not conserved")
            info["support_private_audit"] = dict(performed_work=work.tolist(),
                productive_total_hours=productive.tolist(), idle_total_hours=(exposure-productive).tolist(),
                controller_input=False, ordinary_flexible_productive_split_identified=False)
            return result
        except BaseException:
            self._failed = True
            raise

    def state_dict(self):
        return {**super().state_dict(), "world_tape_sha256": self._tape_digest,
                "cumulative_purchases": self._cumulative_purchases,
                "cumulative_consumption": self._cumulative_consumption}

    def _restore(self, state):
        if state["world_tape_sha256"] != self._tape_digest:
            raise ValueError("different exogenous tape")
        super()._restore(state)
        self._cumulative_purchases = float(state["cumulative_purchases"])
        self._cumulative_consumption = float(state["cumulative_consumption"])

    def settlement(self):
        if self.t != self._proposal["synthetic_system"]["host_total_horizon"]:
            raise ValueError("not the fixed settlement boundary")
        self.assert_identity_conservation()
        liabilities = self.terminal_support_liabilities()
        live = [p.patient_id for p in self.patient_registry.values()
                if p.status not in (PatientStatus.LOST, PatientStatus.DELIVERED)]
        pending = sum(float(np.sum(x)) for x in (self.support.history.pending_hours,
            self.reagent_purchase_pipeline, self.reagent_transfer_pipeline, self.capacity_transfer_pipeline,
            self.specimen_transfer_pipeline))
        pending += len(self.specimen_transits) + len(self.product_return_transits)
        reagent, reactor = resource_totals(self)
        expected = sum(self._proposal["synthetic_system"]["initial_reagents"]) + self._cumulative_purchases - self._cumulative_consumption
        reconciled = (math.isclose(reagent, expected, abs_tol=1e-8, rel_tol=1e-10)
                      and math.isclose(reactor, self._initial_reactor_total, abs_tol=1e-8))
        return {"settled": not live and pending == 0 and reconciled,
                "live_ids": live, "pending_obligations": pending, "support": liabilities,
                "closing_reagents": reagent, "closing_reactors": reactor,
                "resource_conservation": reconciled, "enrolled": len(self.patient_registry),
                "lost": sum(p.status is PatientStatus.LOST for p in self.patient_registry.values()),
                "delivered": sum(p.status is PatientStatus.DELIVERED for p in self.patient_registry.values())}
