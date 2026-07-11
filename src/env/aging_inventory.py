"""Age-bucketed, expiry-aware inventory for perishable autologous material.

Autologous specimens and finished product are perishable: an item is usable only
within a shelf-life window, after which it is wasted. This mirrors the base
environment's bioreactor production pipeline (an age-advancing array), applied to
material that can expire.

A disabled-by-default ``viability_fn`` seam lets a later phase couple product
viability to age and inter-clinic transport time (cold chain) without
re-architecting. With no ``viability_fn`` configured the container behaves as a
plain expiry buffer — the viability factor is exactly 1.0 (see the Phase 4 spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# age (epochs since creation), transport_time (epochs in transit) -> factor in [0, 1]
ViabilityFn = Callable[[int, float], float]


@dataclass(frozen=True)
class ConsumeResult:
    """Outcome of a consume call."""

    delivered: float          # effective usable amount after viability scaling
    consumed: float           # raw amount removed from inventory
    viability_loss: float     # consumed - delivered (0.0 when the hook is inert)


class AgingInventory:
    """Quantities bucketed by integer age; items expire past the shelf life.

    Buckets index ``0 .. shelf_life - 1`` hold the quantity of items at each age.
    An item survives ``shelf_life`` calls to :meth:`advance`, then expires.
    """

    def __init__(self, shelf_life: int, viability_fn: ViabilityFn | None = None):
        if int(shelf_life) < 1:
            raise ValueError("shelf_life must be >= 1")
        self.shelf_life = int(shelf_life)
        self.viability_fn = viability_fn
        self._buckets = np.zeros(self.shelf_life, dtype=np.float64)

    def add(self, quantity: float) -> None:
        """Add fresh items (age 0)."""

        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        self._buckets[0] += float(quantity)

    def advance(self) -> float:
        """Age all items by one epoch; return the quantity that just expired."""

        waste = float(self._buckets[self.shelf_life - 1])
        self._buckets[1:] = self._buckets[:-1]
        self._buckets[0] = 0.0
        return waste

    def consume(self, quantity: float, transport_time: float = 0.0) -> ConsumeResult:
        """Remove up to ``quantity``, oldest-first (FIFO by age, to avoid waste).

        Each removed item is scaled by ``viability_fn(age, transport_time)`` if a
        hook is configured; otherwise the factor is 1.0 and delivered == consumed.
        """

        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        remaining = float(quantity)
        consumed = 0.0
        delivered = 0.0
        for age in range(self.shelf_life - 1, -1, -1):
            if remaining <= 0.0:
                break
            take = min(remaining, float(self._buckets[age]))
            if take <= 0.0:
                continue
            self._buckets[age] -= take
            factor = self._viability_factor(age, transport_time)
            consumed += take
            delivered += take * factor
            remaining -= take
        return ConsumeResult(delivered=delivered, consumed=consumed, viability_loss=consumed - delivered)

    def total(self) -> float:
        """Total quantity currently held (all ages)."""

        return float(self._buckets.sum())

    def age_buckets(self) -> np.ndarray:
        """Copy of the per-age quantity vector (for observation summaries)."""

        return self._buckets.copy()

    def _viability_factor(self, age: int, transport_time: float) -> float:
        if self.viability_fn is None:
            return 1.0
        return float(np.clip(self.viability_fn(age, transport_time), 0.0, 1.0))
