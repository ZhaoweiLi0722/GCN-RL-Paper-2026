"""Isolated synthetic service-work fixture, not a patient production model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


def _vector(values, n, name, *, positive=False):
    result = tuple(float(x) for x in values)
    if len(result) != n or any(not math.isfinite(x) or x < 0 for x in result):
        raise ValueError(f"invalid {name}")
    if positive and any(x <= 0 for x in result):
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ServiceEffortConfig:
    site_hour_caps: tuple[float, ...]
    shared_hour_budget: float
    commitment_lead_steps: int
    hourly_cost: float
    quadratic_cost: float
    switching_cost: float

    def __post_init__(self):
        caps = _vector(self.site_hour_caps, len(self.site_hour_caps), "caps", positive=True)
        object.__setattr__(self, "site_hour_caps", caps)
        if not caps or type(self.commitment_lead_steps) is not int or self.commitment_lead_steps < 1:
            raise ValueError("nonempty sites and a positive integer lead are required")
        if not math.isfinite(self.shared_hour_budget) or not 0 < self.shared_hour_budget <= sum(caps):
            raise ValueError("invalid shared hour budget")
        for cost in (self.hourly_cost, self.quadratic_cost, self.switching_cost):
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("costs must be finite and nonnegative")


@dataclass(frozen=True)
class PublicServiceObservation:
    epoch: int
    remaining_work: tuple[float, ...]
    unfinished_jobs: tuple[int, ...]
    completed_jobs: tuple[int, ...]
    pending_hours: tuple[tuple[float, ...], ...]
    previous_commitment: tuple[float, ...]


@dataclass(frozen=True)
class PublicServiceReceipt:
    epoch: int
    committed_hours: tuple[float, ...]
    applied_hours: tuple[float, ...]
    available_work: tuple[float, ...]
    delivered_work: tuple[float, ...]
    completed_jobs: tuple[int, ...]
    observation_kind: tuple[str, ...]
    labor_cost: float
    change_cost: float


class ServiceEffortMechanics:
    """FIFO work orders with delayed, budgeted effort and private response tape.

    Work units and effort are synthetic. Effort does not accelerate any existing
    patient manufacturing stage. No arrivals, routing, mortality or reward is
    modeled here. Only observation()/step() returns may cross a policy interface.
    """

    def __init__(self, config: ServiceEffortConfig, jobs, response_tape):
        self.config = config
        n = len(config.site_hour_caps)
        if len(jobs) != n or not response_tape:
            raise ValueError("jobs must match sites and response tape must be nonempty")
        self._remaining = [list(_vector(row, len(row), "job work", positive=True)) for row in jobs]
        self._response_tape = tuple(_vector(row, n, "response", positive=True) for row in response_tape)
        self._initial_work = tuple(sum(row) for row in self._remaining)
        self._initial_jobs = tuple(len(row) for row in self._remaining)
        self._delivered = [0.0] * n
        self._completed = [0] * n
        self._pipeline = [(0.0,) * n for _ in range(config.commitment_lead_steps)]
        self._previous = (0.0,) * n
        self.epoch = 0
        self.total_committed_hours = 0.0
        self.total_applied_hours = 0.0
        self.total_labor_cost = 0.0
        self.total_change_cost = 0.0

    def observation(self) -> PublicServiceObservation:
        return PublicServiceObservation(
            self.epoch, tuple(sum(row) for row in self._remaining),
            tuple(len(row) for row in self._remaining), tuple(self._completed),
            tuple(self._pipeline), self._previous,
        )

    def step(self, committed_hours: Sequence[float]):
        if self.epoch >= len(self._response_tape):
            raise ValueError("fixture tape exhausted")
        n = len(self.config.site_hour_caps)
        action = _vector(committed_hours, n, "committed hours")
        if any(x > cap for x, cap in zip(action, self.config.site_hour_caps)):
            raise ValueError("site hour cap exceeded")
        if sum(action) > self.config.shared_hour_budget + 1e-12:
            raise ValueError("shared hour budget exceeded")
        labor_cost = sum(self.config.hourly_cost * x + self.config.quadratic_cost * x * x for x in action)
        change_cost = self.config.switching_cost * sum(abs(x - p) for x, p in zip(action, self._previous))
        if not math.isfinite(labor_cost + change_cost):
            raise ValueError("nonfinite cost")
        before = self.observation()
        applied = self._pipeline.pop(0)
        self._pipeline.append(action)
        delivered, completed, kinds = [], [], []
        for i, hours in enumerate(applied):
            capacity = hours * self._response_tape[self.epoch][i]
            amount = min(capacity, before.remaining_work[i])
            residual = amount
            count = 0
            while residual > 0 and self._remaining[i]:
                work = self._remaining[i][0]
                if residual >= work:
                    self._remaining[i].pop(0)
                    residual -= work
                    count += 1
                else:
                    self._remaining[i][0] -= residual
                    residual = 0.0
            self._delivered[i] += amount
            self._completed[i] += count
            delivered.append(amount)
            completed.append(count)
            # Empty/backlog-limited service gives a lower bound, not an exact
            # efficiency sample. Equality is conservatively treated as censored.
            kind = "no_effort" if hours == 0 else (
                "backlog_limited" if amount >= before.remaining_work[i] - 1e-12 else "uncensored"
            )
            kinds.append(kind)
        receipt = PublicServiceReceipt(
            self.epoch, action, applied, before.remaining_work, tuple(delivered),
            tuple(completed), tuple(kinds), labor_cost, change_cost,
        )
        self.total_committed_hours += sum(action)
        self.total_applied_hours += sum(applied)
        self.total_labor_cost += labor_cost
        self.total_change_cost += change_cost
        self._previous = action
        self.epoch += 1
        self.audit_conservation()
        return self.observation(), receipt

    def audit_conservation(self):
        observation = self.observation()
        for i in range(len(self._remaining)):
            if not math.isclose(self._initial_work[i], self._delivered[i] + observation.remaining_work[i], abs_tol=1e-10):
                raise AssertionError("work not conserved")
            if self._initial_jobs[i] != observation.unfinished_jobs[i] + observation.completed_jobs[i]:
                raise AssertionError("job identities not conserved")
        pending = sum(sum(row) for row in self._pipeline)
        if not math.isclose(self.total_committed_hours, self.total_applied_hours + pending, abs_tol=1e-10):
            raise AssertionError("committed hours not conserved")

    def terminal_ledger(self):
        observation = self.observation()
        return {
            "settled": False,
            "performance_comparison_allowed": False,
            "reason": "mechanics fixture: unfinished jobs and prepaid pending hours are reported, not valued",
            "remaining_work": observation.remaining_work,
            "unfinished_jobs": observation.unfinished_jobs,
            "pending_prepaid_hours": observation.pending_hours,
            "committed_hours": self.total_committed_hours,
            "applied_hours": self.total_applied_hours,
            "labor_cost": self.total_labor_cost,
            "change_cost": self.total_change_cost,
        }
