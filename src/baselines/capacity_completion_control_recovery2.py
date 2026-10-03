"""Resource-conserving predictor correction; prior experiment code is immutable.

Unregistered preparation only. This version does not authorize a new trial.
Integer patient starts cannot borrow fractional inventory, and transfers reject
invalid stocks rather than propagating them to another facility.
"""

from dataclasses import asdict
import copy
import math

import numpy as np

from src.baselines.capacity_completion_control import (
    PublicPatientRecord, PublicSupportServiceEvent, _ForecastPatient,
    _TERMINAL, _admit, _routes, project_hours,
)
from src.baselines.capacity_completion_control_recovery1 import (
    CapacityCompletionControl as LegacyControl,
    PublicPatientForecast as LegacyForecast,
    _condition_prefix,
)


def whole_patient_starts(ready, reagents, idle):
    """Match the native integer floor without creating an inventory overdraft."""
    if type(ready) is not int or ready < 0:
        raise ValueError("nonnegative integer ready count required")
    if any(not math.isfinite(x) or x < 0 for x in (reagents, idle)):
        raise ValueError("finite nonnegative inventory required before production")
    return math.floor(min(ready, reagents, idle))


class PublicPatientForecast(LegacyForecast):
    """Strict inventory conservation plus recovery1 service-prefix arithmetic."""

    def _validate_resources(self):
        for name in ("reagents", "idle", "orders", "reagent_transfers",
                     "capacity_transfers", "pending"):
            values = np.asarray(getattr(self, name))
            invalid = np.argwhere(~np.isfinite(values) | (values < 0))
            if invalid.size:
                index = tuple(int(x) for x in invalid[0])
                raise ValueError(
                    f"forecast resource {name}{index}={values[index]!r} "
                    f"at epoch {self.epoch}")

    def public_view(self):
        self._validate_resources()
        return super().public_view()

    def _resource_transfer(self, stock, request, pipeline):
        # Negative availability must never become a negative transfer.
        if (not np.isfinite(stock).all() or (stock < 0).any()
                or not np.isfinite(request).all()
                or not np.isfinite(pipeline).all() or (pipeline < 0).any()):
            raise ValueError("finite nonnegative stock/pipeline required before transfer")
        return super()._resource_transfer(stock, request, pipeline)

    def step(self, hours, base_control):
        s, costs = self.proposal["synthetic_system"], self.proposal["synthetic_system"]["base_costs"]
        tail = self.epoch >= s["control_epochs"]
        hours = np.zeros(4) if tail else project_hours(hours, self.proposal)
        view = self.public_view()
        action = np.asarray(base_control(view, self.proposal, tail=tail), dtype=float)
        if action.shape != (16,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError("public common operation must return normalized 4N vector")
        destinations = {pid: p.destination for pid, p in self.patients.items() if p.destination is not None}
        transferred = {pid for pid, p in self.patients.items() if p.transferred}
        moves, queues = _routes(view, self.proposal, action, destinations, transferred)
        if tail and (moves or np.any(action[4:12])):
            raise ValueError("tail public operation must not open transfers")
        self.queues = queues
        self.reagents += self._receive(self.reagent_transfers)
        self.idle += self._receive(self.capacity_transfers)
        for pid, p in self.patients.items():
            if p.record["status"] == "in_transit":
                p.record.update(status="waiting", material_site=p.destination)
            if pid in moves:
                p.destination, p.transferred = moves[pid], True
                p.record.update(status="in_transit", material_site=None)
        ordinary = np.asarray(s["support"]["ordinary_hours_per_site"])
        applied = self._receive(self.pending)
        self.pending[-1] += hours
        eligible, completed = [], []
        for site, queue in enumerate(self.queues):
            eligible.append(tuple(pid for pid in queue if not self.patients[pid].record["support_complete"]))
            capacity = self.eta[site] * (ordinary[site] + applied[site])
            available = capacity
            prior_intervals = {pid: self.patients[pid].interval for pid in eligible[-1]}
            complete = []
            prefix_work = []
            for pid in eligible[-1]:
                p = self.patients[pid]
                midpoint = sum(p.interval) / 2
                prefix_work.append(midpoint)
                residual = math.fsum([*prefix_work, -available])
                p.work = min(midpoint, max(0., residual))
                if residual <= 0.:
                    p.work, p.interval = 0., (0., 0.)
                    p.record["support_complete"] = True
                    complete.append(pid)
            compatibility, posterior = _condition_prefix(prior_intervals, eligible[-1], complete, available)
            if compatibility == 0:
                raise ValueError("midpoint forecast incompatible with its public work intervals")
            for pid, interval in posterior.items():
                self.patients[pid].interval = interval
                self.patients[pid].work = sum(interval) / 2
            completed.append(tuple(complete))
            ready = [pid for pid in queue if self.patients[pid].record["support_complete"]]
            count = whole_patient_starts(len(ready), self.reagents[site], self.idle[site])
            for pid in ready[:count]:
                p = self.patients[pid]
                p.record.update(status="in_production", material_site=site, manufacturing_site=site)
                p.stage = s["production_lead_time"]
                queue.remove(pid)
            self.reagents[site] -= count
            self.idle[site] -= count
        lost, wasted = 0, 0
        decay = math.exp(-self.proposal["id_mpc"]["predictive_patient_decay_rate"])
        for pid, p in self.patients.items():
            r, status = p.record, p.record["status"]
            if status in _TERMINAL:
                continue
            r["age"] += 1
            r["survival"] *= decay
            expired = False
            if status in {"waiting", "in_transit"}:
                r["specimen_age"] += 1
                expired = r["specimen_age"] >= s["material_shelf_life"]
            elif status == "finished":
                p.finished_age += 1
                expired = p.finished_age >= s["finished_shelf_life"]
            dead = expired or r["survival"] < s["patient"]["eligibility_threshold"]
            if status == "in_production":
                p.stage -= 1
                if dead or p.stage == 0:
                    self.idle[r["manufacturing_site"]] += 1
                    if not dead:
                        r["status"] = "delivered"
            if dead:
                r["status"] = "lost"
                lost += 1
                wasted += int(expired or status == "in_production")
                if status == "waiting":
                    self.queues[r["material_site"]].remove(pid)
            elif status == "finished":
                r["status"] = "delivered"
        purchase = (action[12:16] + 1) / 2 * np.asarray(s["max_reagent_replenishment"]) * self.supplier
        if tail and self.epoch > self.proposal["settlement"]["last_purchase_epoch"] and np.any(purchase):
            raise ValueError("tail public operation purchases after cutoff")
        if len(self.orders) < 2:
            raise ValueError("public fixed lead-one purchase pipeline requires two slots")
        self.orders[1] += purchase
        self.reagents += self._receive(self.orders)
        reagent_flow = self._resource_transfer(self.reagents, action[4:8] * s["max_reagent_transfer"], self.reagent_transfers)
        reactor_flow = self._resource_transfer(self.idle, action[8:12] * s["max_bioreactor_transfer"], self.capacity_transfers)
        if (np.any(self.reagents > np.asarray(s["max_reagents"]) + 1e-9)
                or np.any(self.idle > np.asarray(s["max_idle_bioreactors"]) + 1e-9)
                or np.any(self.reagents < -1e-9) or np.any(self.idle < -1e-9)):
            raise ValueError("forecast resource conservation would require unaccounted clipping")
        for queue in self.queues:
            queue.sort(key=lambda pid: self.patients[pid].record["survival"])
        if self.epoch <= s["arrival_last_epoch_inclusive"]:
            demand = self.current_arrivals if self.epoch == self.initial_epoch else np.asarray(s["demand_rates"])
            self.arrival_carry += demand
            arrivals = np.floor(self.arrival_carry + 1e-12).astype(int)
            self.arrival_carry -= arrivals
            for site, count in enumerate(arrivals):
                for index in range(count):
                    pid = f"__forecast__{self.initial_epoch}:{self.epoch}:{site}:{index}"
                    if pid in self.patients:
                        raise ValueError("public ID collides with synthetic forecast identity")
                    record = asdict(PublicPatientRecord(pid, self.epoch, "waiting", site, site, None, 0, 0, 1., False))
                    work = s["support"]["work_per_patient"]
                    self.patients[pid] = _ForecastPatient(record, work, (work, work))
                    self.queues[site].append(pid)
        if any(len(q) > s["max_specimens"][site] for site, q in enumerate(self.queues)):
            raise ValueError("forecast specimen capacity overflow")
        waiting = np.asarray([len(q) for q in self.queues])
        urgent = sum(self.patients[pid].record["survival"] < s["patient"]["eligibility_threshold"] + s["patient"]["urgency_margin"]
                     for q in self.queues for pid in q)
        sup = s["support"]
        components = {
            "reagent_purchase": costs["reagent_purchase"] * float(purchase.sum()),
            "reagent_holding": costs["reagent_holding"] * float(np.maximum(self.reagents - waiting, 0).sum()),
            "reagent_shortage": costs["reagent_shortage"] * float(np.maximum(waiting - self.reagents, 0).sum()),
            "bioreactor_holding": costs["bioreactor_holding"] * float(np.maximum(self.idle - waiting, 0).sum()),
            "bioreactor_shortage": costs["bioreactor_shortage"] * float(np.maximum(waiting - self.idle, 0).sum()),
            "specimen_transfer": costs["specimen_transfer"] * len(moves),
            "bioreactor_transfer": costs["bioreactor_transfer"] * reactor_flow,
            "reagent_transfer": costs["reagent_transfer"] * reagent_flow,
            "patient_loss": costs["weight_patient_lost"] * lost,
            "expiry": costs["weight_expiry"] * wasted,
            "urgency": costs["weight_urgency"] * urgent,
            "ordinary_labor": sup["ordinary_hourly_cost"] * float(ordinary.sum()),
            "flexible_labor": float(sup["hourly_cost"] * hours.sum() + sup["quadratic_cost"] * (hours ** 2).sum()),
            "switching": sup["switching_cost"] * float(np.abs(hours - self.previous).sum()),
        }
        self.last_service = PublicSupportServiceEvent(self.epoch, self.epoch + 1, tuple(eligible), tuple(completed), tuple(ordinary), tuple(applied))
        self.history = tuple(tuple(rows[1:]) + ((hours[i], applied[i], ordinary[i], len(eligible[i]), len(completed[i]), 1.),)
                             for i, rows in enumerate(self.history))
        self.previous = hours.copy()
        self.epoch += 1
        self.current_arrivals = np.asarray(s["demand_rates"] if self.epoch < s["control_epochs"] else [0.] * 4)
        for key, value in components.items():
            self.total_components[key] = self.total_components.get(key, 0.) + value
        self._validate_resources()
        return math.fsum(components.values())



class CapacityCompletionControl(LegacyControl):
    """Identical controller recipe bound to the corrected public forecast."""

    def act(self, view, *, role="adaptive_rule", before_query=None, before_filter=None):
        self.observe(view, before_filter=before_filter)
        if view.common.epoch >= self.proposal["synthetic_system"]["control_epochs"]:
            return np.zeros(4)
        if role == "adaptive_rule":
            return self.adaptive(view)
        if role == "fixed_allocation_reference":
            return np.full(4, 2.)
        if role != "id_mpc":
            raise ValueError("unknown completion-control role")
        mpc = self.proposal["id_mpc"]
        candidates = self.candidates(view)
        scores, detail = [], []
        for candidate, (initial, switch) in enumerate(candidates):
            quantile_costs, components = [], []
            for q in mpc["response_quantiles"]:
                model = None
                total = 0.
                for horizon in range(mpc["predictive_horizon"]):
                    _admit(before_query, self.scientific,
                           {"epoch": view.common.epoch, "candidate": candidate, "quantile": q,
                            "horizon": horizon, "model_epochs": 1})
                    if model is None:
                        model = PublicPatientForecast(view, self.proposal, self.filter, self.lifecycle, q)
                    hours = model.adaptive() if switch and horizon >= 2 and model.epoch < self.proposal["synthetic_system"]["control_epochs"] else initial
                    total += model.step(hours, self.base_control)
                terminal = model.terminal_value()
                quantile_costs.append(total + terminal)
                components.append({**model.total_components, "terminal_live_patient_value": terminal})
            score = float(np.dot(mpc["summary_weights"], quantile_costs))
            if not math.isfinite(score):
                raise ValueError("nonfinite approximate planner score")
            scores.append(score)
            detail.append({"quantile_costs": quantile_costs, "components": components})
        best = min(range(len(scores)), key=scores.__getitem__)
        self.last_plan = {"epoch": view.common.epoch, "scores": scores, "chosen": best,
                          "detail": detail, "approximate_public_forecast": True,
                          "model_epochs": len(candidates) * len(mpc["response_quantiles"]) * mpc["predictive_horizon"]}
        return candidates[best][0].copy()


    def state_dict(self):
        state = super().state_dict()
        state["format"] = "capacity-completion-control-recovery2"
        return state

    def load_state_dict(self, state):
        if state["format"] != "capacity-completion-control-recovery2":
            raise ValueError("controller recovery restore format mismatch")
        restored = copy.deepcopy(state)
        restored["format"] = "capacity-completion-control-recovery1"
        super().load_state_dict(restored)

    def load_recovery1_boundary(self, state):
        """Explicit public-state migration, without a model or environment call."""
        if state["format"] != "capacity-completion-control-recovery1":
            raise ValueError("expected recovery1 controller boundary")
        super().load_state_dict(copy.deepcopy(state))

