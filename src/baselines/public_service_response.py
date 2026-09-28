"""Receipt-only response estimator for the isolated service-work fixture."""

from __future__ import annotations

import math

from src.env.service_effort_mechanics import PublicServiceReceipt


class PublicServiceResponseEstimator:
    """An EWMA diagnostic, not an adaptive control or RL policy."""

    def __init__(self, prior, alpha):
        self._estimate = tuple(float(x) for x in prior)
        if not self._estimate or any(not math.isfinite(x) or x <= 0 for x in self._estimate):
            raise ValueError("positive finite prior required")
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.samples = [0] * len(self._estimate)
        self.last_epoch = -1

    @property
    def estimate(self):
        return self._estimate

    def observe(self, receipt: PublicServiceReceipt):
        if not isinstance(receipt, PublicServiceReceipt):
            raise TypeError("public receipt required, not environment state")
        n = len(self._estimate)
        fields = (receipt.applied_hours, receipt.delivered_work, receipt.available_work, receipt.observation_kind)
        if any(len(row) != n for row in fields) or receipt.epoch != self.last_epoch + 1:
            raise ValueError("receipt shape or chronology mismatch")
        updated = list(self._estimate)
        samples = self.samples.copy()
        for i, (hours, work, available, kind) in enumerate(zip(*fields)):
            if any(not math.isfinite(x) or x < 0 for x in (hours, work, available)) or work > available + 1e-12:
                raise ValueError("invalid public measurement")
            expected = "no_effort" if hours == 0 else ("backlog_limited" if work >= available - 1e-12 else "uncensored")
            if kind != expected or (hours == 0 and work != 0):
                raise ValueError("inconsistent censoring indicator")
            if kind == "uncensored":
                response = work / hours
                if not math.isfinite(response):
                    raise ValueError("nonfinite response sample")
                updated[i] = (1 - self.alpha) * updated[i] + self.alpha * response
                samples[i] += 1
        self._estimate = tuple(updated)
        self.samples = samples
        self.last_epoch = receipt.epoch
