"""Shared, typed observation boundary for the six planned capacity controls.

No learner, estimator, simulator or scientific controller is instantiated here.
Private tapes, exact remaining support work and environment snapshots are not
accepted as policy input. Raw costs remain float64, without reward rescaling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

from src.env.patient_support_capacity import PatientSupportAction


CAPACITY_CONTROL_ROLES = (
    "frozen_history", "frozen_matched_exploration", "online_matched_fork",
    "adaptive_rule", "id_mpc", "fixed_allocation_reference",
)


def _floats(values):
    result = tuple(float(x) for x in values)
    if not result or any(not math.isfinite(x) for x in result):
        raise ValueError("nonempty finite public feature vector required")
    return result


@dataclass(frozen=True)
class PublicSupportInput:
    epoch: int
    site_ids: tuple[str, ...]
    base_observation: tuple[float, ...]
    capacity_history: tuple[tuple[tuple[float, ...], ...], ...]
    pending_hours: tuple[tuple[float, ...], ...]
    ready_waiting_counts: tuple[int, ...]

    @classmethod
    def capture(cls, host):
        # These two explicit public methods are the entire collection boundary.
        base = _floats(host.observation())
        capacity = host.public_capacity()
        if set(capacity) != {"epoch", "site_ids", "history", "pending_hours", "ready_waiting_counts"}:
            raise ValueError("unexpected public capacity fields")
        epoch, sites = capacity["epoch"], tuple(capacity["site_ids"])
        if (type(epoch) is not int or epoch < 0 or not sites or len(set(sites)) != len(sites)
                or any(not isinstance(x, str) or not x for x in sites)):
            raise ValueError("invalid public epoch/sites")
        history = tuple(tuple(_floats(row) for row in site) for site in capacity["history"])
        pending = tuple(_floats(row) for row in capacity["pending_hours"])
        ready = tuple(capacity["ready_waiting_counts"])
        if (len(history) != len(sites) or not history[0]
                or len({len(site) for site in history}) != 1
                or any(len(row) != 6 for site in history for row in site)
                or not pending or any(len(row) != len(sites) for row in pending)
                or any(x < 0 for row in pending for x in row)
                or len(ready) != len(sites) or any(type(x) is not int or x < 0 for x in ready)):
            raise ValueError("invalid public history/pipeline/count dimensions")
        for site in history:
            for row in site:
                if any(x < 0 for x in row) or row[-1] not in (0.0, 1.0):
                    raise ValueError("invalid public history feature")
                if row[-1] == 0 and any(row):
                    raise ValueError("missing history must be zero masked")
        return cls(epoch, sites, base, history, pending, ready)


@dataclass(frozen=True)
class PublicSupportTransition:
    before: PublicSupportInput
    after: PublicSupportInput
    action: PatientSupportAction
    raw_cost: float
    reward: float
    done: bool


def collect_support_step(host, *, before, base_action, requested_hours, record_raw):
    """Single common dispatch; the caller owns nonrefundable budget admission.

    Full native cost/patient info goes only to the required audit recorder, not
    the policy transition. No role can request privileged info here. A post-step validation error does
    not refund the caller's environment debit or authorize an automatic retry.
    Collector/runtime implementations must preserve the failure and stop.
    """
    if not callable(record_raw):
        raise TypeError("separate raw cost/patient recorder required")
    if type(before) is not PublicSupportInput or PublicSupportInput.capture(host) != before:
        raise ValueError("stale or non-public policy input")
    action = PatientSupportAction(base_action, requested_hours)
    _, reward, done, info = host.step(action)
    record_raw(copy.deepcopy(info))
    after = PublicSupportInput.capture(host)
    cost, reward = float(info["cost"]), float(reward)
    if (after.epoch != before.epoch + 1 or after.site_ids != before.site_ids
            or not math.isfinite(cost) or not math.isfinite(reward)
            or not math.isclose(reward, -cost, rel_tol=1e-12, abs_tol=1e-9)
            or type(done) is not bool):
        raise ValueError("invalid public transition or raw objective")
    return PublicSupportTransition(before, after, action, cost, reward, done)
