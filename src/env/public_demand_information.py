"""Opt-in public-history forecast; not wired into frozen experiments.

Demand for epoch t is observed before the existing simulator's action at t.
Forecasts issued at t cover t+1 through t+horizon. Callers must supply only
announced exposure multipliers, never latent future demand-regime multipliers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


def _vector(values: Sequence[float], n: int, *, positive: bool = False) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != n or any(not math.isfinite(v) or v < 0 for v in result):
        raise ValueError("expected a finite, nonnegative facility vector")
    if positive and any(v == 0 for v in result):
        raise ValueError("announced exposure must be positive")
    return result


@dataclass(frozen=True)
class IssuedDemandForecast:
    issued_at: int
    per_epoch: tuple[tuple[float, ...], ...]

    @property
    def total(self) -> tuple[float, ...]:
        return tuple(sum(row[i] for row in self.per_epoch) for i in range(len(self.per_epoch[0])))


class PublicDemandForecaster:
    """A transparent rolling estimate, not a proposed learned policy.

    Both adaptive and frozen modes retain the same public history. Only the
    rate estimate differs. State contains no simulator reference or RNG.
    """

    def __init__(self, prior_rates: Sequence[float], *, window: int, horizon: int, adaptive: bool):
        if type(window) is not int or window < 1 or type(horizon) is not int or horizon < 1:
            raise ValueError("window and horizon must be positive integers")
        if type(adaptive) is not bool or not len(prior_rates):
            raise ValueError("explicit Boolean mode and nonempty prior are required")
        self._prior = _vector(prior_rates, len(prior_rates))
        self._window, self._horizon, self._adaptive = window, horizon, adaptive
        self._last_epoch = -1
        self._history: list[dict[str, Any]] = []

    def observe(self, *, epoch: int, arrivals: Sequence[float], announced_exposure: Sequence[float]) -> None:
        if type(epoch) is not int or epoch != self._last_epoch + 1:
            raise ValueError("observations must arrive exactly once, in epoch order starting at zero")
        values = _vector(arrivals, len(self._prior))
        exposure = _vector(announced_exposure, len(self._prior), positive=True)
        if any(not math.isfinite(v / e) for v, e in zip(values, exposure)):
            raise ValueError("exposure-normalized arrival rate overflow")
        self._history.append({"epoch": epoch, "arrivals": values, "announced_exposure": exposure})
        self._history = self._history[-self._window:]
        self._last_epoch = epoch

    def issue(self, *, epoch: int, future_announced_exposure: Sequence[Sequence[float]]) -> IssuedDemandForecast:
        if type(epoch) is not int or epoch != self._last_epoch or epoch < 0:
            raise ValueError("issue time must equal the latest observed epoch")
        if len(future_announced_exposure) != self._horizon:
            raise ValueError("announced schedule must cover the exact future horizon")
        schedule = tuple(_vector(v, len(self._prior), positive=True) for v in future_announced_exposure)
        rates = self._prior
        if self._adaptive:
            rates = tuple(
                sum(r["arrivals"][i] / r["announced_exposure"][i] / len(self._history) for r in self._history)
                for i in range(len(self._prior))
            )
        prediction = tuple(tuple(r * e for r, e in zip(rates, exposure)) for exposure in schedule)
        if any(not math.isfinite(v) for row in prediction for v in row) or any(
            not math.isfinite(sum(row[i] for row in prediction)) for i in range(len(self._prior))
        ):
            raise ValueError("forecast overflow")
        return IssuedDemandForecast(epoch, prediction)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "prior_rates": list(self._prior), "window": self._window,
            "horizon": self._horizon, "adaptive": self._adaptive, "last_epoch": self._last_epoch,
            "history": [{"epoch": r["epoch"], "arrivals": list(r["arrivals"]),
                         "announced_exposure": list(r["announced_exposure"])} for r in self._history],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        if set(state) != set(expected) or any(state[k] != expected[k] for k in (
            "version", "prior_rates", "window", "horizon", "adaptive"
        )) or type(state["adaptive"]) is not bool or any(
            type(state[k]) is not int for k in ("version", "window", "horizon")
        ):
            raise ValueError("forecast checkpoint contract mismatch")
        last = state["last_epoch"]
        if type(last) is not int or last < -1:
            raise ValueError("invalid last observed epoch")
        history = state["history"]
        if not isinstance(history, list) or len(history) != min(last + 1, self._window):
            raise ValueError("incomplete history checkpoint")
        checked = []
        for epoch, row in zip(range(last - len(history) + 1, last + 1), history):
            if set(row) != {"epoch", "arrivals", "announced_exposure"} or type(row["epoch"]) is not int or row["epoch"] != epoch:
                raise ValueError("invalid history ordering or fields")
            arrivals = _vector(row["arrivals"], len(self._prior))
            exposure = _vector(row["announced_exposure"], len(self._prior), positive=True)
            if any(not math.isfinite(v / e) for v, e in zip(arrivals, exposure)):
                raise ValueError("invalid normalized history")
            checked.append({"epoch": epoch, "arrivals": arrivals, "announced_exposure": exposure})
        self._last_epoch, self._history = last, checked
