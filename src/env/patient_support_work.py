"""Identity-bound continuous support work, separate from biological processing.

Opt-in mechanics only. No environments, random streams or learners are created.
Snapshots are privileged; controllers receive only the public receipt history.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math

from src.env.service_effort_mechanics import ServiceEffortConfig
from src.rl.capacity_adaptation_interface import (
    PublicCapacityHistory, PublicCapacityReceipt, project_effort,
)


def _nonnegative(values, n, label):
    result = tuple(values)
    if len(result) != n or any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or x < 0 for x in result
    ):
        raise ValueError(f"invalid {label}")
    return tuple(float(x) for x in result)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class PatientSupportConfig:
    effort: ServiceEffortConfig
    site_ids: tuple[str, ...]
    ordinary_hours: tuple[float, ...]
    work_per_patient: float
    history_window: int
    max_epochs: int

    def __post_init__(self):
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        object.__setattr__(self, "ordinary_hours", _nonnegative(
            self.ordinary_hours, len(self.site_ids), "ordinary hours"))
        PublicCapacityHistory(self.effort, site_ids=self.site_ids,
                              window=self.history_window, max_epochs=self.max_epochs)
        if (isinstance(self.work_per_patient, bool)
                or not isinstance(self.work_per_patient, (int, float))
                or not math.isfinite(self.work_per_patient) or self.work_per_patient <= 0):
            raise ValueError("positive finite work per patient required")


class PatientSupportWork:
    """Partial work follows patient ID, including across inter-facility transit.

    The host owns aging, death, transport, production and release. This ledger
    must never remove an unready patient from that host's waiting/aging queues.
    Ordinary and matured flexible hours have the same private site response.
    Idle booked effort is still billed; unfinished work is not a completed job.
    """

    format = "patient-support-work-v1"

    def __init__(self, config: PatientSupportConfig):
        if type(config) is not PatientSupportConfig:
            raise TypeError("PatientSupportConfig required")
        self.config = config
        self.history = PublicCapacityHistory(
            config.effort, site_ids=config.site_ids,
            window=config.history_window, max_epochs=config.max_epochs)
        self._tasks = {}
        self._delivered = (0.0,) * len(config.site_ids)

    def is_ready(self, patient_id):
        row = self._tasks.get(patient_id)
        return row is not None and row["state"] == "waiting" and row["remaining"] == 0.0

    def advance(self, *, epoch, waiting_ids, raw_hours, response):
        """Produce one end-of-step receipt; all checks precede state mutation.

        `response` is private environment input, never a policy observation.
        Queue order is supplied by the host's existing patient-priority rule.
        """
        n = len(self.config.site_ids)
        rows = tuple(tuple(row) for row in waiting_ids)
        ids = tuple(pid for row in rows for pid in row)
        if (type(epoch) is not int or epoch != self.history.next_decision_epoch
                or epoch >= self.config.max_epochs or len(rows) != n
                or any(not isinstance(pid, str) or not pid for pid in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("invalid epoch or duplicate/malformed patient queues")
        response = _nonnegative(response, n, "private response")
        raw = _nonnegative(raw_hours, n, "raw hours")
        committed = project_effort(raw, self.config.effort)
        applied = self.history.pending_hours[0]
        tasks = copy.deepcopy(self._tasks)
        delivered = list(self._delivered)
        eligible, completed = [], []
        for site, queue in enumerate(rows):
            for pid in queue:
                tasks.setdefault(pid, {"remaining": float(self.config.work_per_patient),
                                       "state": "waiting"})
                if tasks[pid]["state"] != "waiting":
                    raise ValueError("started/retired patient returned to support queue")
            eligible.append(sum(tasks[pid]["remaining"] > 0 for pid in queue))
            work = (self.config.ordinary_hours[site] + applied[site]) * response[site]
            if not math.isfinite(work):
                raise ValueError("nonfinite available support work")
            count = 0
            for pid in queue:
                remaining = tasks[pid]["remaining"]
                if remaining == 0:
                    continue
                used = min(work, remaining)
                tasks[pid]["remaining"] = remaining - used
                work -= used
                delivered[site] += used
                count += int(tasks[pid]["remaining"] == 0.0)
            completed.append(count)
        effort = self.config.effort
        previous = self.history.receipts[-1].committed_hours if epoch else (0.0,) * n
        receipt = PublicCapacityReceipt(
            epoch, epoch + 1, raw, committed, applied, self.config.ordinary_hours,
            tuple(eligible), tuple(completed),
            math.fsum(effort.hourly_cost * x + effort.quadratic_cost * x * x for x in committed),
            effort.switching_cost * math.fsum(abs(x - y) for x, y in zip(committed, previous)))
        self._validate_work(tasks, delivered)
        self.history.observe(receipt, decision_epoch=epoch + 1)
        self._tasks, self._delivered = tasks, tuple(delivered)
        return receipt

    def mark_started(self, patient_ids):
        ids = tuple(patient_ids)
        if len(set(ids)) != len(ids) or not all(self.is_ready(pid) for pid in ids):
            raise ValueError("each start requires its own completed support work")
        for pid in ids:
            self._tasks[pid]["state"] = "started"

    def retire(self, patient_ids):
        # Keep residual work in the ledger; losses must not erase obligations.
        for pid in patient_ids:
            if pid in self._tasks and self._tasks[pid]["state"] == "waiting":
                self._tasks[pid]["state"] = "retired"

    def ordinary_cost(self):
        cost = self.config.effort.hourly_cost * math.fsum(self.config.ordinary_hours)
        if not math.isfinite(cost):
            raise ValueError("nonfinite ordinary staff cost")
        return cost

    def liabilities(self):
        """Privileged raw ledger, not a terminal value or a public observation."""
        return {"unstarted_ids": sorted(pid for pid, r in self._tasks.items() if r["state"] == "waiting"),
                "unfinished_work": math.fsum(r["remaining"] for r in self._tasks.values()
                                              if r["state"] == "waiting"),
                "retired_unfinished_work": math.fsum(r["remaining"] for r in self._tasks.values()
                                                      if r["state"] == "retired"),
                "paid_pending_hours": self.history.pending_hours,
                "settled": False}

    def _validate_work(self, tasks, delivered):
        delivered = _nonnegative(delivered, len(self.config.site_ids), "delivered work")
        for pid, row in tasks.items():
            if (not isinstance(pid, str) or not pid or not isinstance(row, dict)
                    or set(row) != {"remaining", "state"}
                    or row["state"] not in ("waiting", "started", "retired")):
                raise ValueError("invalid support identity/state")
            remaining = _nonnegative((row["remaining"],), 1, "remaining work")[0]
            if (remaining > self.config.work_per_patient
                    or row["state"] == "started" and remaining != 0):
                raise ValueError("invalid support work balance")
        performed = math.fsum(self.config.work_per_patient - r["remaining"] for r in tasks.values())
        if not math.isclose(performed, math.fsum(delivered), rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError("support work conservation failure")

    def snapshot(self):
        return {"format": self.format, "config": asdict(self.config),
                "history": self.history.snapshot(), "tasks": copy.deepcopy(self._tasks),
                "delivered_by_site": list(self._delivered)}

    @classmethod
    def restore(cls, snapshot, *, config):
        result = cls(config)
        if (not isinstance(snapshot, dict) or set(snapshot) != set(result.snapshot())
                or snapshot["format"] != cls.format
                or _canonical(snapshot["config"]) != _canonical(asdict(config))
                or not isinstance(snapshot["tasks"], dict)):
            raise ValueError("support snapshot contract mismatch")
        result.history = PublicCapacityHistory.restore(
            snapshot["history"], config=config.effort, site_ids=config.site_ids,
            window=config.history_window, max_epochs=config.max_epochs)
        result._validate_work(snapshot["tasks"], snapshot["delivered_by_site"])
        result._tasks = copy.deepcopy(snapshot["tasks"])
        result._delivered = tuple(snapshot["delivered_by_site"])
        return result
