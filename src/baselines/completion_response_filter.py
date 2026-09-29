"""Receipt-only finite-grid inference; no neural weights or control policy."""

import math

from src.env.completion_feedback_queue import CompletionReceipt, completion_probability


class CompletionResponseFilter:
    def __init__(self, rates, prior, refresh_probability):
        self.rates, self.prior = tuple(rates), tuple(prior)
        if not self.rates or len(self.rates) != len(self.prior) or len(set(self.rates)) != len(self.rates):
            raise ValueError("distinct rates and matching prior required")
        for rate in self.rates:
            completion_probability(rate, 1)
        if any(not math.isfinite(p) or p <= 0 for p in self.prior) or not math.isclose(sum(self.prior), 1, abs_tol=1e-12):
            raise ValueError("strictly positive normalized prior required")
        if not math.isfinite(refresh_probability) or not 0 <= refresh_probability <= 1:
            raise ValueError("refresh probability outside [0,1]")
        self.refresh_probability = refresh_probability
        self.posterior = (self.prior, self.prior)
        self.exposed_intervals = (0, 0)
        self.last_epoch = -1

    @property
    def mean(self):
        return tuple(sum(rate * p for rate, p in zip(self.rates, row)) for row in self.posterior)

    def observe(self, receipt):
        if not isinstance(receipt, CompletionReceipt):
            raise TypeError("public CompletionReceipt required")
        fields = (receipt.requested_hours, receipt.applied_hours, receipt.eligible_tasks,
                  receipt.exposure_hours, receipt.completed, receipt.observation_kind)
        if (receipt.epoch != self.last_epoch + 1 or receipt.known_at != receipt.epoch + 1 or
                any(len(values) != 2 for values in fields)):
            raise ValueError("receipt chronology or shape mismatch")
        if any(not math.isfinite(x) or x < 0 for x in receipt.requested_hours + receipt.applied_hours):
            raise ValueError("invalid availability measurement")
        if any(not math.isfinite(x) or x < 0 for x in (receipt.labor_cost, receipt.switching_cost)):
            raise ValueError("invalid receipt cost")
        updated, counts = [], list(self.exposed_intervals)
        for site, (applied, task, hours, completed, kind) in enumerate(zip(*fields[1:])):
            if task is not None and (not isinstance(task, str) or not task):
                raise ValueError("invalid eligible task identity")
            if type(completed) is not bool or not math.isfinite(hours) or hours < 0:
                raise ValueError("invalid completion/exposure measurement")
            if hours != (applied if task is not None else 0.0) or (hours == 0 and completed):
                raise ValueError("inconsistent exposure/completion")
            expected = "idle" if task is None else ("no_effort" if hours == 0 else
                                                    ("completed" if completed else "right_censored"))
            if kind != expected:
                raise ValueError("inconsistent observation kind")
            predicted = [(1 - self.refresh_probability) * p + self.refresh_probability * base
                         for p, base in zip(self.posterior[site], self.prior)]
            logs = [math.log(p) if p > 0 else -math.inf for p in predicted]
            if hours > 0:
                for k, rate in enumerate(self.rates):
                    probability = completion_probability(rate, hours)
                    logs[k] += math.log(probability) if completed else -rate * hours
                counts[site] += 1
            peak = max(logs)
            relative = [math.exp(value - peak) for value in logs]
            denominator = sum(relative)
            normalized = tuple(value / denominator for value in relative)
            if any(not math.isfinite(p) for p in normalized):
                raise ValueError("nonfinite posterior")
            updated.append(normalized)
        # Commit only after both sites validate; a bad second site leaves no partial update.
        self.posterior, self.exposed_intervals = tuple(updated), tuple(counts)
        self.last_epoch = receipt.epoch
        return self.posterior
