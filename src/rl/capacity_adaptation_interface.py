"""Opt-in public capacity interface; not a patient collector or experiment.

The projection follows the existing radial shared-effort budget convention.
Receipts disclose completion counts, not latent efficiency or remaining work.
This module creates no environments, models, optimizers or random streams.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from src.env.service_effort_mechanics import ServiceEffortConfig


def _hours(values, n, name):
    values = tuple(values)
    if len(values) != n or any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or x < 0 for x in values
    ):
        raise ValueError(f"invalid {name}")
    return tuple(float(x) for x in values)


def project_effort(request, config: ServiceEffortConfig):
    """Float64 physical hours, with zero purchase and unused budget allowed.

    Radial budget scaling is not a Euclidean projection. Every feasible input
    is a fixed point; no integer rounding or forced full-budget spending occurs.
    """
    raw = _hours(request, len(config.site_hour_caps), "request")
    capped = tuple(min(x, cap) for x, cap in zip(raw, config.site_hour_caps))
    total = math.fsum(capped)
    scale = min(1.0, config.shared_hour_budget / total) if total else 1.0
    return tuple(x * scale for x in capped)


def project_effort_tensor(request, config: ServiceEffortConfig):
    """Batched differentiable counterpart in physical, not normalized, hours."""
    from src.rl.networks import require_torch, torch

    require_torch()
    if (not isinstance(request, torch.Tensor) or request.ndim < 1
            or request.shape[-1] != len(config.site_hour_caps)
            or request.dtype not in (torch.float32, torch.float64)):
        raise ValueError("float32/float64 tensor with final site dimension required")
    if not torch.isfinite(request).all().item() or (request < 0).any().item():
        raise ValueError("finite nonnegative physical-hour requests required")
    caps = request.new_tensor(config.site_hour_caps)
    budget = request.new_tensor(config.shared_hour_budget)
    if (not torch.isfinite(caps).all().item() or (caps <= 0).any().item()
            or not torch.isfinite(budget).item() or budget <= 0):
        raise ValueError("effort limits are not representable in tensor dtype")
    capped = torch.minimum(request, caps)
    total = capped.sum(dim=-1, keepdim=True)
    if not torch.isfinite(total).all().item():
        raise ValueError("nonfinite requested effort total")
    return capped * (budget / torch.maximum(total, budget))


@dataclass(frozen=True)
class PublicCapacityReceipt:
    epoch: int
    known_at: int
    raw_requested_hours: tuple[float, ...]
    committed_hours: tuple[float, ...]
    applied_hours: tuple[float, ...]
    ordinary_hours: tuple[float, ...]
    eligible_jobs: tuple[int, ...]
    completed_jobs: tuple[int, ...]
    labor_cost: float
    switching_cost: float

    def __post_init__(self):
        # Canonical tuples prevent caller-owned mutable lists changing history.
        for name in ("raw_requested_hours", "committed_hours", "applied_hours", "ordinary_hours",
                     "eligible_jobs", "completed_jobs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


class PublicCapacityHistory:
    """Same causal receipt channel for frozen, online, heuristic and ID-MPC.

    Supports only a fixed integer commitment lead and end-of-interval receipts.
    Completed jobs may be zero despite positive effort; this is not evidence
    that efficiency is zero. A future estimator must handle that censoring.
    Full public history is retained for restoration; feature windows are masked.
    This validates a receipt's structure, not the physical truth of its source.
    """

    format = "public-capacity-history-v1"
    feature_names = ("committed_hours", "applied_hours", "ordinary_hours", "eligible_jobs",
                     "completed_jobs", "observed")

    def __init__(self, config: ServiceEffortConfig, *, site_ids, window, max_epochs):
        if type(config) is not ServiceEffortConfig:
            raise TypeError("explicit ServiceEffortConfig required")
        sites = tuple(site_ids)
        if (len(sites) != len(config.site_hour_caps)
                or any(not isinstance(x, str) or not x for x in sites)
                or len(set(sites)) != len(sites)):
            raise ValueError("distinct nonempty site IDs must match effort limits")
        if (type(window) is not int or type(max_epochs) is not int
                or not 1 <= window <= max_epochs):
            raise ValueError("positive bounded history window required")
        self.config, self.site_ids = config, sites
        self.window, self.max_epochs = window, max_epochs
        self._receipts = ()
        self._pending = ((0.0,) * len(sites),) * config.commitment_lead_steps

    @property
    def next_decision_epoch(self):
        return len(self._receipts)

    @property
    def receipts(self):
        return self._receipts

    @property
    def pending_hours(self):
        return self._pending

    def observe(self, receipt: PublicCapacityReceipt, *, decision_epoch):
        if type(receipt) is not PublicCapacityReceipt:
            raise TypeError("public capacity receipt required, not an environment/info dict")
        t, n = self.next_decision_epoch, len(self.site_ids)
        if (type(receipt.epoch) is not int or type(receipt.known_at) is not int
                or type(decision_epoch) is not int or receipt.epoch != t
                or receipt.known_at != t + 1 or decision_epoch != receipt.known_at
                or t >= self.max_epochs):
            raise ValueError("future, duplicate, missing or over-budget receipt")
        projected = project_effort(receipt.raw_requested_hours, self.config)
        committed = _hours(receipt.committed_hours, n, "commitment")
        applied = _hours(receipt.applied_hours, n, "applied effort")
        ordinary = _hours(receipt.ordinary_hours, n, "ordinary staffed hours")
        for actual, expected in ((committed, projected), (applied, self._pending[0])):
            if any(not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
                   for x, y in zip(actual, expected)):
                raise ValueError("projection or commitment-time mismatch")
        if any(len(row) != n for row in (receipt.eligible_jobs, receipt.completed_jobs)):
            raise ValueError("completion count shape mismatch")
        for eligible, completed, effort, base in zip(
                receipt.eligible_jobs, receipt.completed_jobs, applied, ordinary):
            if (type(eligible) is not int or type(completed) is not int
                    or not 0 <= completed <= eligible
                    or (effort == 0 and base == 0 and completed != 0)):
                raise ValueError("invalid indivisible support completion count")
        previous = self._receipts[-1].committed_hours if t else (0.0,) * n
        labor = math.fsum(self.config.hourly_cost * x + self.config.quadratic_cost * x * x
                          for x in committed)
        switch = self.config.switching_cost * math.fsum(abs(x - y) for x, y in zip(committed, previous))
        for recorded, expected in ((receipt.labor_cost, labor), (receipt.switching_cost, switch)):
            if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                    or not math.isfinite(recorded) or not math.isfinite(expected)
                    or not math.isclose(recorded, expected, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("committed effort billing mismatch")
        # All validation precedes mutation, including costs and delayed exposure.
        self._receipts += (receipt,)
        self._pending = self._pending[1:] + (committed,)

    def node_history(self, *, decision_epoch):
        if type(decision_epoch) is not int or decision_epoch != self.next_decision_epoch:
            raise ValueError("history only available at the current decision boundary")
        rows = self._receipts[-self.window:]
        padding = ((0.0,) * len(self.feature_names),) * (self.window - len(rows))
        return tuple(padding + tuple((r.committed_hours[i], r.applied_hours[i],
                                      r.ordinary_hours[i], r.eligible_jobs[i], r.completed_jobs[i], 1.0)
                                     for r in rows) for i in range(len(self.site_ids)))

    def snapshot(self):
        return {"format": self.format, "config": asdict(self.config),
                "site_ids": list(self.site_ids), "window": self.window,
                "max_epochs": self.max_epochs,
                "receipts": [asdict(r) for r in self._receipts]}

    @classmethod
    def restore(cls, snapshot, *, config, site_ids, window, max_epochs):
        restored = cls(config, site_ids=site_ids, window=window, max_epochs=max_epochs)
        expected = restored.snapshot()
        if not isinstance(snapshot, dict) or set(snapshot) != set(expected):
            raise ValueError("invalid public history snapshot fields")
        # Normalize JSON tuple/list representation without permitting extra fields.
        import json
        for key in expected.keys() - {"receipts"}:
            if json.dumps(snapshot[key], sort_keys=True, allow_nan=False) != json.dumps(
                    expected[key], sort_keys=True, allow_nan=False):
                raise ValueError(f"history contract mismatch: {key}")
        if not isinstance(snapshot["receipts"], list) or len(snapshot["receipts"]) > max_epochs:
            raise ValueError("invalid or over-budget receipt history")
        for row in snapshot["receipts"]:
            receipt = PublicCapacityReceipt(**row)
            restored.observe(receipt, decision_epoch=receipt.known_at)
        return restored
