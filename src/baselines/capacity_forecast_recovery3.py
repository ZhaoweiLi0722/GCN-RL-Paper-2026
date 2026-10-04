"""Versioned forecast repair for rounded midpoint completion boundaries.

The live public filter, environment and historical predictors are unchanged.
Only ambiguous floating-point midpoint prefix comparisons use exact arithmetic.
"""

from dataclasses import asdict
from fractions import Fraction
import math

import numpy as np

from src.baselines.capacity_completion_control import (
    PublicPatientRecord, PublicSupportServiceEvent, _ForecastPatient,
    _TERMINAL, _routes,
)
from src.baselines.capacity_completion_control_recovery1 import _condition_prefix
from src.baselines.capacity_completion_control_recovery2 import (
    PublicPatientForecast as PreviousForecast, whole_patient_starts,
)
from src.baselines.capacity_completion_control import project_hours


def midpoint_residual(intervals, capacity):
    """Return remaining work and completion, resolving near-zero sums exactly."""
    if not math.isfinite(capacity) or capacity < 0:
        raise ValueError("finite nonnegative capacity required")
    if any(not math.isfinite(x) or x < 0 for pair in intervals for x in pair):
        raise ValueError("finite nonnegative work intervals required")
    if any(lo > hi for lo, hi in intervals):
        raise ValueError("ordered work intervals required")
    mids = [lo / 2 + hi / 2 for lo, hi in intervals]
    residual = math.fsum([*mids, -capacity])
    # This bound only selects exact computation; it never relaxes feasibility.
    rounding_bound = math.fsum(math.ulp(x) for x in mids)
    if abs(residual) <= rounding_bound:
        exact = sum((Fraction(lo) + Fraction(hi)) / 2 for lo, hi in intervals)
        exact -= Fraction(capacity)
        return float(exact), exact <= 0
    return residual, residual <= 0


class PublicPatientForecast(PreviousForecast):
    """Keep native mechanics; do not round a positive midpoint residual to done."""

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
            prefix_intervals = []
            for pid in eligible[-1]:
                p = self.patients[pid]
                midpoint = sum(p.interval) / 2
                prefix_intervals.append(p.interval)
                residual, finished = midpoint_residual(prefix_intervals, available)
                p.work = min(midpoint, max(0., residual))
                if finished:
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
