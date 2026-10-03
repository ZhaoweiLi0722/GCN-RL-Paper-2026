"""Completion-only public controls; no native environment or neural execution.

The filter is a conservative box approximation, not an exact posterior. For a
prefix of k completed jobs with residuals x and capacity c, condition on
sum(x[:k]) <= c < sum(x[:k+1]); omit the right inequality when backlog limited.
Intersect the coordinate hull with these halfspaces and multiply retained
interval fractions (a feasible point contributes one). Correlations discarded
by this box likelihood and by hypothesis merging are deliberately not inferred.

The planner is an explicit approximate patient model, NOT a simulator clone:
response-conditional interval midpoints, constant response quantiles, exponential
mean survival decay, indivisible mean-arrival rounding with fractional carry,
public deterministic routing, five-stage manufacture, and resource bookkeeping.
No future tape, true residual work, latent health, or native loss ledger is read.
The 16 x 3 x 8 forecast epochs are each admitted before dispatch. The adaptive
rule uses an analytic two-epoch backlog forecast, not a model rollout.

Integration: inject common_operation(view, proposal, tail=False), returning the
normalized 4N facility-net action. Call record_operation for EVERY executed
public operation, including other arms and the tail. This recovers destinations
omitted from public in-transit records. Observe every boundary from epoch zero,
or restore state; a single late snapshot cannot identify production stages.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import json
import math

import numpy as np

from src.env.patient_support_public import (
    PublicPatientRecord, PublicSupportOperations, PublicSupportServiceEvent,
    validate_public_operations,
)
from src.rl.public_support_collector import PublicSupportInput
from src.rl.public_support_input import PublicSupportControlInput


NODE_SUMMARY_FIELDS = (
    "eta_mean", "eta_sd", "unresolved_count", "work_low", "work_mean",
    "work_high", "ready_count", "urgent_fraction", "reset_count",
)
_TERMINAL = {"delivered", "lost"}


def _config(proposal):
    p = copy.deepcopy(proposal)
    s, f, a, m = (p["synthetic_system"], p["information"]["estimator"],
                  p["adaptive_rule"], p["id_mpc"])
    if (len(s["site_ids"]) != 4 or f["response_values"] != [.5, .75, 1, 1.25, 1.5]
            or f["pairs_per_site_receipt"] != 25
            or f["transition"] != "0.95_identity_plus_0.05_uniform"
            or a["iterations"] != 20 or a["step_size"] != .1
            or a["projection_bisection_iterations"] != 32
            or m["candidate_sequences_per_decision"] != 16
            or m["predictive_horizon"] != 8 or m["response_quantiles"] != [.1, .5, .9]
            or m["summary_weights"] != [.25, .5, .25]
            or s["support"]["commitment_lead_steps"] != 2
            or s["production_lead_time"] != 5
            or s["resource_transfer_lead_time"] != 1
            or s["specimen_routing_lead_time_epochs"] != 1
            or s["finished_product_return_lead_time_epochs"] != 0):
        raise ValueError("controller supports the fixed pilot contract only")
    return p


def _admit(callback, scientific, payload):
    if callback is None:
        if scientific:
            raise ValueError("scientific work requires a before-work admission callback")
    else:
        callback(dict(payload))


def _check_view(view, proposal):
    if type(view) is not PublicSupportControlInput:
        raise TypeError("PublicSupportControlInput required; never pass a host")
    op = validate_public_operations(view.operations)
    c = view.common
    if (c.epoch != op.epoch or c.site_ids != op.site_ids
            or list(c.site_ids) != proposal["synthetic_system"]["site_ids"]
            or len(c.pending_hours) != 2 or any(len(r) != 4 for r in c.pending_hours)
            or len(c.capacity_history) != 4):
        raise ValueError("public boundary/site/pending mismatch")
    by_id = {p.patient_id: p for p in op.patients}
    if tuple(sum(by_id[x].support_complete for x in q) for q in op.waiting_order) != c.ready_waiting_counts:
        raise ValueError("ready identity/count mismatch")
    if op.last_service is not None:
        e = op.last_service
        for i, history in enumerate(c.capacity_history):
            if tuple(history[-1][1:]) != (e.applied_hours[i], e.ordinary_hours[i],
                                          len(e.eligible_order[i]), len(e.completed_ids[i]), 1.):
                raise ValueError("public aggregate/identity receipt mismatch")
    if any(not math.isfinite(x) or x < 0 for r in c.pending_hours for x in r):
        raise ValueError("invalid pending staff")
    return op


def _hull(bounds):
    return (min(x[0] for x in bounds), max(x[1] for x in bounds))


def _condition_prefix(prior, order, completed, capacity):
    """Return (box compatibility, posterior residual box), including censoring.

    Two nested sum constraints give the coordinate hull below. A strict head
    inequality has a closed conservative hull, but equality-only intersections
    have zero likelihood. No zero-width division is performed.
    """
    k = len(completed)
    if tuple(completed) != tuple(order[:k]):
        raise ValueError("completed IDs are not a service prefix")
    result = dict(prior)
    done = [prior[pid] for pid in order[:k]]
    lo_sum, hi_sum = (math.fsum(x[j] for x in done) for j in (0, 1))
    if lo_sum > capacity:
        return 0., result
    head = prior[order[k]] if k < len(order) else None
    if head is not None and hi_sum + head[1] <= capacity:
        return 0., result
    clipped = []
    for lo, hi in done:
        lower = lo if head is None else max(lo, capacity - head[1] - (hi_sum - hi))
        upper = min(hi, capacity - (lo_sum - lo))
        clipped.append((lower, upper))
    head_clip = None if head is None else (max(head[0], capacity - hi_sum), head[1])
    weight = 1.
    original = done + ([] if head is None else [head])
    constrained = clipped + ([] if head_clip is None else [head_clip])
    for (lo, hi), (left, right) in zip(original, constrained):
        if left > right:
            return 0., result
        if hi > lo:
            weight *= max(0., right - left) / (hi - lo)
    if weight == 0:
        return 0., result
    for pid in order[:k]:
        result[pid] = (0., 0.)
    if head is not None:
        low = max(0., math.fsum(x[0] for x in clipped) + head_clip[0] - capacity)
        high = min(head_clip[1], math.fsum(x[1] for x in clipped) + head_clip[1] - capacity)
        result[order[k]] = (low, max(low, high))
    return weight, result


class CompletionIntervalFilter:
    """Five response hypotheses per site, with all unresolved IDs retained."""

    summary_fields = NODE_SUMMARY_FIELDS

    def __init__(self, proposal, *, scientific=True):
        self.proposal = _config(proposal)
        self.scientific = scientific
        self.responses = np.asarray(proposal["information"]["estimator"]["response_values"], dtype=float)
        self.work = float(proposal["synthetic_system"]["support"]["work_per_patient"])
        self._weights = np.tile(proposal["information"]["estimator"]["initial_weights"], (4, 1)).astype(float)
        self._boxes = [[{} for _ in range(5)] for _ in range(4)]
        self._location = {}
        self._closed = set()
        self._patients = {}
        self._epoch = -1
        self._last_view = None
        self.reset_events = []

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def intervals(self):
        return {pid: {"site": site, "bounds_by_response":
                      tuple(tuple(self._boxes[site][h][pid]) for h in range(5))}
                for pid, site in sorted(self._location.items())}

    @property
    def residual_means(self):
        return {pid: sum(self._weights[site, h] * sum(self._boxes[site][h][pid]) / 2 for h in range(5))
                for pid, site in self._location.items()}

    def _move(self, pid, site):
        old = self._location.get(pid)
        if old == site:
            return
        if old is None:
            interval = (self.work, self.work)
        else:
            interval = _hull([self._boxes[old][h][pid] for h in range(5)
                              if self._weights[old, h] > 0])
            for box in self._boxes[old]:
                del box[pid]
        self._location[pid] = site
        for box in self._boxes[site]:
            box[pid] = interval

    def _close(self, pid):
        old = self._location.pop(pid, None)
        if old is not None:
            for box in self._boxes[old]:
                box.pop(pid, None)
        self._closed.add(pid)

    def update(self, view, *, before_filter=None):
        op = _check_view(view, self.proposal)
        if op.epoch == self._epoch:
            if json.dumps(asdict(view), sort_keys=True) != json.dumps(self._last_view, sort_keys=True):
                raise ValueError("conflicting repeated public boundary")
            return self.node_summaries()
        if op.epoch != self._epoch + 1:
            raise ValueError("filter needs consecutive receipts from zero or restored state")
        present = {p.patient_id for p in op.patients}
        if not set(self._location).issubset(present):
            raise ValueError("unresolved public identities disappeared without exit records")
        event = op.last_service
        service_site = {} if event is None else {pid: i for i, q in enumerate(event.eligible_order) for pid in q}
        for p in op.patients:
            pid = p.patient_id
            if pid in self._closed:
                if pid in service_site or (p.status not in _TERMINAL and not p.support_complete):
                    raise ValueError("completed/retired identity regained work")
                continue
            site = service_site.get(pid, p.material_site)
            if pid not in self._location and pid in self._patients and site is None:
                raise ValueError("unresolved identity lost its source interval")
            if site is not None and (pid in service_site or not p.support_complete and p.status not in _TERMINAL):
                self._move(pid, site)
            elif pid not in self._location and not p.support_complete and p.status not in _TERMINAL:
                raise ValueError("new in-transit identity lacks public work origin")
            if p.status == "in_transit" and pid in self._location:
                # Once a patient leaves, later source evidence must not narrow
                # its transported work through an obsolete response association.
                source = self._location[pid]
                carried = _hull([self._boxes[source][h][pid] for h in range(5)
                                 if self._weights[source, h] > 0])
                for box in self._boxes[source]:
                    box[pid] = carried
        if event is not None:
            transition = .95 * np.eye(5) + .05 / 5
            for site in range(4):
                prior = self._boxes[site]
                merged = [{} for _ in range(5)]
                mass = np.zeros(5)
                for old in range(5):
                    for new, eta in enumerate(self.responses):
                        _admit(before_filter, self.scientific,
                               {"epoch": event.epoch, "site": site, "old_response_index": old,
                                "new_response_index": new, "hypothesis_transitions": 1})
                        likelihood, posterior = _condition_prefix(
                            prior[old], event.eligible_order[site], event.completed_ids[site],
                            eta * (event.ordinary_hours[site] + event.applied_hours[site]))
                        pair_mass = self._weights[site, old] * transition[old, new] * likelihood
                        mass[new] += pair_mass
                        if pair_mass > 0:
                            for pid, interval in posterior.items():
                                merged[new][pid] = _hull((merged[new][pid], interval)) if pid in merged[new] else interval
                if mass.sum() == 0:
                    unresolved = {p.patient_id for p in op.patients if not p.support_complete and p.status not in _TERMINAL}
                    affected = sorted(set(prior[0]) & unresolved)
                    self.reset_events.append({"epoch": event.epoch, "site": site,
                                              "reason": "all_hypotheses_incompatible", "patient_ids": affected})
                    self._weights[site] = np.asarray(self.proposal["information"]["estimator"]["initial_weights"])
                    self._boxes[site] = [{pid: (0., self.work) for pid in prior[0]} for _ in range(5)]
                else:
                    self._weights[site] = mass / mass.sum()
                    # Zero-weight hypotheses still have a schema-complete box;
                    # their intervals cannot affect a positive-weight hull.
                    fallback = {pid: _hull([b[pid] for b in merged if pid in b]) for pid in prior[0]}
                    self._boxes[site] = [b if mass[h] > 0 else dict(fallback) for h, b in enumerate(merged)]
        for p in op.patients:
            if p.support_complete or p.status in _TERMINAL:
                self._close(p.patient_id)
            elif p.material_site is not None:
                self._move(p.patient_id, p.material_site)
        self._patients = {p.patient_id: asdict(p) for p in op.patients}
        self._epoch, self._last_view = op.epoch, asdict(view)
        return self.node_summaries()

    def node_summaries(self):
        out = np.zeros((4, len(NODE_SUMMARY_FIELDS)), dtype=float)
        means = self._weights @ self.responses
        for site in range(4):
            out[site, :2] = (means[site], math.sqrt(float(self._weights[site] @ (self.responses - means[site]) ** 2)))
            for pid, loc in self._location.items():
                if loc != site:
                    continue
                boxes = [self._boxes[site][h][pid] for h in range(5)]
                live = [boxes[h] for h in range(5) if self._weights[site, h] > 0]
                low, high = _hull(live)
                mean = sum(self._weights[site, h] * sum(boxes[h]) / 2 for h in range(5))
                out[site, 2:6] += (1, low, mean, high)
            waiting = [p for p in self._patients.values() if p["status"] == "waiting" and p["material_site"] == site]
            out[site, 6] = sum(p["support_complete"] for p in waiting)
            out[site, 7] = sum(p["survival"] < .85 for p in waiting) / max(1, len(waiting))
            out[site, 8] = sum(e["site"] == site for e in self.reset_events)
        return out

    def state_dict(self):
        return copy.deepcopy({"format": "completion-interval-filter-v1", "proposal": self.proposal,
                              "weights": self._weights.tolist(), "boxes": self._boxes,
                              "location": self._location, "closed": sorted(self._closed),
                              "patients": self._patients, "epoch": self._epoch,
                              "last_view": self._last_view, "reset_events": self.reset_events})

    def load_state_dict(self, state):
        s = copy.deepcopy(state)
        if s["format"] != "completion-interval-filter-v1" or s["proposal"] != self.proposal:
            raise ValueError("filter restore contract mismatch")
        weights = np.asarray(s["weights"], dtype=float)
        if (weights.shape != (4, 5) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0)):
            raise ValueError("invalid restored weights")
        boxes = s["boxes"]
        if len(boxes) != 4 or any(len(b) != 5 for b in boxes):
            raise ValueError("invalid restored interval dimensions")
        if set(s["location"]) & set(s["closed"]):
            raise ValueError("closed restored identity has work")
        if (type(s["epoch"]) is not int or s["epoch"] < -1
                or any(type(site) is not int or not 0 <= site < 4 for site in s["location"].values())
                or not set(s["location"]).issubset(s["patients"])):
            raise ValueError("invalid restored public identity/boundary")
        for site in range(4):
            expected = {p for p, loc in s["location"].items() if loc == site}
            for box in boxes[site]:
                if set(box) != expected:
                    raise ValueError("restored identity/site mismatch")
                for pid, pair in box.items():
                    if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or not 0 <= pair[0] <= pair[1] <= self.work:
                        raise ValueError("invalid restored interval")
                    box[pid] = tuple(pair)
        self._weights, self._boxes = weights, boxes
        self._location, self._closed = s["location"], set(s["closed"])
        self._patients, self._epoch = s["patients"], s["epoch"]
        self._last_view, self.reset_events = s["last_view"], s["reset_events"]


def project_hours(hours, proposal):
    s = proposal["synthetic_system"]["support"]
    h = np.asarray(hours, dtype=float)
    if h.shape != (4,) or not np.isfinite(h).all():
        raise ValueError("four finite hour commitments required")
    h = np.clip(h, 0, s["site_hour_caps"])
    return h * min(1., s["shared_hour_budget"] / h.sum()) if h.sum() > 0 else h


def _previous(view):
    return np.asarray([rows[-1][0] if rows[-1][-1] else 0. for rows in view.common.capacity_history])


def proximal_hours(deficit, urgency, eta, previous, proposal):
    """Exactly 20 steps of .1 and 32 feasible-endpoint dual bisections/step.

    f'(h) = -v*eta*max(0,d-eta*h)/8 + .01 + .001*h.
    prox = clip(p + soft(y-lambda-p, .0002), 0, caps).
    Unlike radial projection this is the joint L1/capped-shared-budget prox.
    """
    d, v, eta, p = (np.asarray(x, dtype=float) for x in (deficit, urgency, eta, previous))
    if any(x.shape != (4,) or not np.isfinite(x).all() for x in (d, v, eta, p)):
        raise ValueError("invalid proximal inputs")
    support, rule = proposal["synthetic_system"]["support"], proposal["adaptive_rule"]
    cap, budget = np.asarray(support["site_hour_caps"]), support["shared_hour_budget"]
    h = project_hours(p, proposal)
    for _ in range(rule["iterations"]):
        y = h - rule["step_size"] * (-v * eta * np.maximum(0, d - eta * h) / 8 + .01 + .001 * h)
        def prox(lam):
            z = y - lam - p
            return np.clip(p + np.sign(z) * np.maximum(np.abs(z) - .0002, 0), 0, cap)
        unconstrained = prox(0.)
        low, high = 0., max(0., float(np.max(y)) + .0002)
        for _ in range(rule["projection_bisection_iterations"]):
            middle = (low + high) / 2
            if prox(middle).sum() > budget:
                low = middle
            else:
                high = middle
        h = unconstrained if unconstrained.sum() <= budget else prox(high)
    return h


def _routes(view, proposal, action, destinations, transferred):
    """Public deterministic ID routes, including arrivals before dispatch."""
    op, s = view.operations, proposal["synthetic_system"]
    queues = [list(q) for q in op.waiting_order]
    rows = {p.patient_id: p for p in op.patients}
    for p in op.patients:
        if p.status == "in_transit":
            if p.patient_id not in destinations:
                raise ValueError("in-transit destination missing; record_operation every epoch")
            queues[destinations[p.patient_id]].append(p.patient_id)
    requested = np.asarray(action[:4]) * s["max_specimen_transfer"]
    requested = np.sign(requested) * np.floor(np.abs(requested) + .5)
    inbound, outbound = np.maximum(requested, 0), np.maximum(-requested, 0)
    edges = {tuple(sorted(e)) for e in s["transport_resource_information_edges"]}
    moves = {}
    for receiver in range(4):
        for donor in range(4):
            if tuple(sorted((donor, receiver))) not in edges:
                continue
            count = int(min(inbound[receiver], outbound[donor]))
            candidates = sorted((pid for pid in queues[donor] if pid not in transferred),
                                key=lambda pid: (rows[pid].survival, -rows[pid].specimen_age,
                                                 rows[pid].enrollment_epoch, pid))[:count]
            for pid in candidates:
                queues[donor].remove(pid)
                moves[pid] = receiver
            inbound[receiver] -= len(candidates)
            outbound[donor] -= len(candidates)
    return moves, queues


class _Lifecycle:
    """Causal public stage/destination ledger, shared across all control roles."""

    def __init__(self):
        self.epoch = -1
        self.records = {}
        self.stages = {}
        self.destinations = {}
        self.transferred = set()
        self.recorded_epoch = -1

    def observe(self, view, proposal):
        op, s = view.operations, proposal["synthetic_system"]
        if self.epoch == op.epoch:
            return
        if op.epoch != self.epoch + 1:
            raise ValueError("patient lifecycle requires consecutive public history or restore")
        stages = {}
        for p in op.patients:
            old = self.records.get(p.patient_id)
            if p.status == "in_production":
                if old is None:
                    raise ValueError("production stage unidentifiable without earlier public status")
                stages[p.patient_id] = (self.stages[p.patient_id] - 1 if old["status"] == "in_production"
                                         else s["production_lead_time"] - 1)
                if stages[p.patient_id] <= 0:
                    raise ValueError("public production exceeds declared duration")
            if p.status == "in_transit" and p.patient_id not in self.destinations:
                if p.material_site is None:
                    raise ValueError("public transit destination absent; missing record_operation")
                self.destinations[p.patient_id] = p.material_site
                self.transferred.add(p.patient_id)
        counts = np.zeros((4, s["production_lead_time"]))
        for p in op.patients:
            if p.patient_id in stages:
                counts[p.manufacturing_site, stages[p.patient_id]] += 1
        reactors = np.asarray(op.bioreactors)
        if reactors.shape != counts.shape or not np.allclose(reactors[:, 1:], counts[:, 1:], atol=1e-9, rtol=0):
            raise ValueError("public individual production stages disagree with resource inventory")
        self.stages, self.records = stages, {p.patient_id: asdict(p) for p in op.patients}
        self.epoch = op.epoch

    def record_operation(self, view, action, proposal):
        if self.epoch != view.common.epoch:
            raise ValueError("observe public boundary before record_operation")
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError("normalized public 4N facility-net action required")
        if self.recorded_epoch == self.epoch:
            raise ValueError("operation already recorded for this epoch")
        moves, _ = _routes(view, proposal, action, self.destinations, self.transferred)
        self.destinations.update(moves)
        self.transferred.update(moves)
        self.recorded_epoch = self.epoch

    def state_dict(self):
        return {"epoch": self.epoch, "records": copy.deepcopy(self.records), "stages": dict(self.stages),
                "destinations": dict(self.destinations), "transferred": sorted(self.transferred),
                "recorded_epoch": self.recorded_epoch}

    def load_state_dict(self, state):
        s = copy.deepcopy(state)
        if any(type(x) is not int or not 1 <= x <= 4 for x in s["stages"].values()):
            raise ValueError("invalid public production restore")
        if any(type(x) is not int or not 0 <= x < 4 for x in s["destinations"].values()):
            raise ValueError("invalid public destination restore")
        self.epoch, self.records, self.stages = s["epoch"], s["records"], s["stages"]
        self.destinations, self.transferred = s["destinations"], set(s["transferred"])
        self.recorded_epoch = s["recorded_epoch"]


def _known_arrivals(view, proposal):
    values = tuple(getattr(view.operations, "current_arrivals", ()))
    if len(values) != len(proposal["synthetic_system"]["site_ids"]):
        raise ValueError("public current_arrivals required; do not replace known demand with a guess")
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid public current arrivals")
    return np.asarray(values, dtype=float)


def _adaptive_inputs(view, proposal, intervals, weights, destinations):
    """Analytic public lead forecast: serve existing IDs, then end arrivals.

    d is residual work at the maturation epoch after its ordinary service.
    Future routing is not forecast here; known one-step transits keep their
    recorded destinations. Eligibility/expiry are projected once per epoch.
    """
    s, lead = proposal["synthetic_system"], proposal["adaptive_rule"]["prediction_lead"]
    eta = weights @ np.asarray(proposal["information"]["estimator"]["response_values"])
    jobs = [[] for _ in range(4)]
    for p in view.operations.patients:
        if p.status not in {"waiting", "in_transit"}:
            continue
        site = p.material_site if p.status == "waiting" else destinations.get(p.patient_id)
        if site is None:
            raise ValueError("adaptive forecast needs public transit destination")
        residual = 0.
        if not p.support_complete:
            row = intervals[p.patient_id]
            residual = sum(weights[row["site"], h] * sum(b) / 2 for h, b in enumerate(row["bounds_by_response"]))
        jobs[site].append([residual, p.survival, p.specimen_age])
    ordinary = np.asarray(s["support"]["ordinary_hours_per_site"])
    deficit, urgency = np.zeros(4), np.ones(4)
    rate = proposal["id_mpc"]["predictive_patient_decay_rate"]
    for offset in range(lead + 1):
        for site in range(4):
            service = eta[site] * (ordinary[site] + (view.common.pending_hours[offset][site] if offset < lead else 0.))
            jobs[site].sort(key=lambda j: j[1])
            for job in jobs[site]:
                used = min(service, job[0])
                job[0] -= used
                service -= used
            if offset == lead:
                deficit[site] = sum(j[0] for j in jobs[site])
                urgency[site] += 2 * sum(j[1] < .85 for j in jobs[site]) / max(1, len(jobs[site]))
            else:
                jobs[site] = [[r, survival * math.exp(-rate), age + 1] for r, survival, age in jobs[site]
                              if age + 1 < s["material_shelf_life"]
                              and survival * math.exp(-rate) >= s["patient"]["eligibility_threshold"]]
                if view.common.epoch + offset <= s["arrival_last_epoch_inclusive"]:
                    demand = _known_arrivals(view, proposal)[site] if offset == 0 else s["demand_rates"][site]
                    # Analytic rule needs expected work only, not synthetic IDs.
                    jobs[site].extend([[s["support"]["work_per_patient"], 1., 0] for _ in range(int(demand))])
                    if demand % 1:
                        jobs[site].append([s["support"]["work_per_patient"] * (demand % 1), 1., 0])
    return deficit, urgency, eta


@dataclass
class _ForecastPatient:
    record: dict
    work: float
    interval: tuple[float, float]
    stage: int = 0
    destination: int | None = None
    transferred: bool = False
    finished_age: int = 0


class PublicPatientForecast:
    """Identity/inventory approximate forecast, isolated from live filter state."""

    def __init__(self, view, proposal, interval_filter, lifecycle, quantile):
        self.proposal, self.epoch = proposal, view.common.epoch
        self.initial_epoch = self.epoch
        self.lifecycle = copy.deepcopy(lifecycle)
        self.weights = interval_filter.weights
        indices = [min(4, int(np.searchsorted(np.cumsum(w), quantile))) for w in self.weights]
        self.eta = interval_filter.responses[indices]
        self.weights = np.eye(5)[indices]
        self.patients = {}
        intervals = interval_filter.intervals
        for p in view.operations.patients:
            b = (0., 0.)
            if p.patient_id in intervals:
                row = intervals[p.patient_id]
                b = row["bounds_by_response"][indices[row["site"]]]
            self.patients[p.patient_id] = _ForecastPatient(asdict(p), sum(b) / 2, tuple(b),
                                                        lifecycle.stages.get(p.patient_id, 0),
                                                        lifecycle.destinations.get(p.patient_id),
                                                        p.patient_id in lifecycle.transferred)
        op = view.operations
        self.queues = [list(q) for q in op.waiting_order]
        self.reagents = np.array(op.reagents, dtype=float)
        self.idle = np.asarray(op.bioreactors, dtype=float)[:, 0].copy()
        self.reagent_transfers = np.array(op.reagent_transfers, dtype=float)
        self.capacity_transfers = np.array(op.capacity_transfers, dtype=float)
        self.orders = np.array(op.reagent_orders, dtype=float)
        self.supplier = np.array(op.supplier_available, dtype=float)
        self.pending = np.array(view.common.pending_hours, dtype=float)
        self.history = copy.deepcopy(view.common.capacity_history)
        self.previous = _previous(view)
        self.last_service = op.last_service
        self.current_arrivals = _known_arrivals(view, proposal)
        self.arrival_carry = np.zeros(4)
        self.total_components = {}

    def public_view(self):
        s = self.proposal["synthetic_system"]
        reactors = np.zeros((4, s["production_lead_time"]))
        reactors[:, 0] = self.idle
        for p in self.patients.values():
            if p.record["status"] == "in_production":
                reactors[p.record["manufacturing_site"], p.stage] += 1
        kwargs = {"epoch": self.epoch, "site_ids": tuple(s["site_ids"]),
                  "patients": tuple(PublicPatientRecord(**p.record) for p in self.patients.values()),
                  "waiting_order": tuple(tuple(q) for q in self.queues), "reagents": tuple(self.reagents),
                  "bioreactors": tuple(map(tuple, reactors)), "reagent_transfers": tuple(map(tuple, self.reagent_transfers)),
                  "capacity_transfers": tuple(map(tuple, self.capacity_transfers)),
                  "reagent_orders": tuple(map(tuple, self.orders)), "supplier_available": tuple(self.supplier),
                  "demand_forecast": tuple(s["demand_rates"] if self.epoch < s["control_epochs"] else [0.] * 4),
                  "last_service": self.last_service}
        if "current_arrivals" in {f.name for f in fields(PublicSupportOperations)}:
            kwargs["current_arrivals"] = tuple(self.current_arrivals)
        op = PublicSupportOperations(**kwargs)
        # Base observation is intentionally not copied from the real world:
        # the injected adapter must consume the explicit public arrays/IDs.
        common = PublicSupportInput(self.epoch, tuple(s["site_ids"]), (), self.history,
                                    tuple(map(tuple, self.pending)),
                                    tuple(sum(self.patients[pid].record["support_complete"] for pid in q) for q in self.queues))
        return PublicSupportControlInput(common, op)

    def adaptive(self):
        intervals = {pid: {"site": p.record["material_site"] if p.record["material_site"] is not None else p.destination,
                           "bounds_by_response": (p.interval,) * 5}
                     for pid, p in self.patients.items() if p.record["status"] in {"waiting", "in_transit"}
                     and not p.record["support_complete"]}
        d, v, _ = _adaptive_inputs(self.public_view(), self.proposal, intervals, self.weights,
                                   {pid: p.destination for pid, p in self.patients.items() if p.destination is not None})
        # Scenario feedback uses its fixed response quantile, not new evidence.
        return proximal_hours(d, v, self.eta, self.previous, self.proposal)

    @staticmethod
    def _receive(pipeline):
        if len(pipeline) == 0:
            raise ValueError("public pipeline must expose its declared lead slots")
        arrivals = pipeline[0].copy()
        pipeline[:-1] = pipeline[1:]
        pipeline[-1] = 0
        return arrivals

    def _resource_transfer(self, stock, request, pipeline):
        inbound, outbound = np.maximum(request, 0), np.maximum(-request, 0)
        edges = {tuple(sorted(e)) for e in self.proposal["synthetic_system"]["transport_resource_information_edges"]}
        amount = 0.
        for receiver in range(4):
            for donor in range(4):
                if tuple(sorted((receiver, donor))) not in edges:
                    continue
                flow = min(inbound[receiver], outbound[donor], stock[donor])
                stock[donor] -= flow
                pipeline[-1, receiver] += flow
                inbound[receiver] -= flow
                outbound[donor] -= flow
                amount += flow
        return amount

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
            for pid in eligible[-1]:
                p = self.patients[pid]
                p.work = sum(p.interval) / 2
                used = min(capacity, p.work)
                p.work -= used
                capacity -= used
                if p.work == 0.:
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
            count = int(math.floor(min(len(ready), self.reagents[site], self.idle[site]) + 1e-12))
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
        return math.fsum(components.values())

    def terminal_value(self):
        weight = self.proposal["synthetic_system"]["base_costs"]["weight_patient_lost"]
        # Every pending booking was already charged at commitment, including
        # commitments in the initial history. No second payment at maturation.
        return math.fsum(weight * (1 - p.record["survival"]) for p in self.patients.values()
                         if p.record["status"] not in _TERMINAL)


class CapacityCompletionControl:
    """Usable adaptive/MPC comparator with a common public operation injection."""

    def __init__(self, proposal, *, base_control, scientific=True):
        if not callable(base_control):
            raise TypeError("inject public common_operation(view, proposal, tail=False)")
        self.proposal = _config(proposal)
        self.base_control, self.scientific = base_control, scientific
        self.filter = CompletionIntervalFilter(proposal, scientific=scientific)
        self.lifecycle = _Lifecycle()
        self.last_plan = None

    def observe(self, view, *, before_filter=None):
        _check_view(view, self.proposal)
        self.lifecycle.observe(view, self.proposal)
        return self.filter.update(view, before_filter=before_filter)

    def record_operation(self, view, base_action):
        self.lifecycle.record_operation(view, base_action, self.proposal)

    def adaptive(self, view):
        d, v, eta = _adaptive_inputs(view, self.proposal, self.filter.intervals, self.filter.weights,
                                     self.lifecycle.destinations)
        return proximal_hours(d, v, eta, _previous(view), self.proposal)

    def candidates(self, view):
        base = [np.zeros(4), np.full(4, 2.), self.adaptive(view), _previous(view)]
        for site in range(4):
            focused = np.full(4, 4 / 3)
            focused[site] = 4
            base.append(focused)
        return tuple((project_hours(h, self.proposal), switch) for switch in (False, True) for h in base)

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
        return copy.deepcopy({"format": "capacity-completion-control-v1", "filter": self.filter.state_dict(),
                              "lifecycle": self.lifecycle.state_dict(), "last_plan": self.last_plan})

    def load_state_dict(self, state):
        if state["format"] != "capacity-completion-control-v1":
            raise ValueError("controller restore format mismatch")
        restored = CompletionIntervalFilter(self.proposal, scientific=self.scientific)
        restored.load_state_dict(state["filter"])
        life = _Lifecycle()
        life.load_state_dict(state["lifecycle"])
        if life.epoch != restored._epoch:
            raise ValueError("filter/lifecycle restore boundary mismatch")
        self.filter, self.lifecycle, self.last_plan = restored, life, copy.deepcopy(state["last_plan"])
