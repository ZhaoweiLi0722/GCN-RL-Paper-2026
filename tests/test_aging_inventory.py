"""Tests for the aging, expiry-aware material inventory (Phase 4, task group 2)."""

from __future__ import annotations

import unittest

from src.env.aging_inventory import AgingInventory


class AgingInventoryTests(unittest.TestCase):
    def test_items_expire_at_shelf_life(self) -> None:
        inv = AgingInventory(shelf_life=3)
        inv.add(10.0)
        # Survives shelf_life-1 advances without waste, then expires on the next.
        self.assertEqual(inv.advance(), 0.0)  # age 0 -> 1
        self.assertEqual(inv.advance(), 0.0)  # age 1 -> 2
        self.assertEqual(inv.advance(), 10.0)  # age 2 (max) -> expired
        self.assertEqual(inv.total(), 0.0)

    def test_fifo_consumes_oldest_first(self) -> None:
        inv = AgingInventory(shelf_life=5)
        inv.add(5.0)          # this batch is older...
        inv.advance()
        inv.advance()
        inv.add(7.0)          # ...than this fresh batch
        result = inv.consume(5.0)
        # All 5 taken from the older batch; the fresh batch of 7 remains intact.
        self.assertAlmostEqual(result.consumed, 5.0)
        self.assertAlmostEqual(inv.total(), 7.0)

    def test_consume_capped_by_availability(self) -> None:
        inv = AgingInventory(shelf_life=3)
        inv.add(4.0)
        result = inv.consume(10.0)
        self.assertAlmostEqual(result.consumed, 4.0)
        self.assertAlmostEqual(inv.total(), 0.0)

    def test_viability_hook_inert_by_default(self) -> None:
        inv = AgingInventory(shelf_life=4)  # no viability_fn
        inv.add(6.0)
        inv.advance()
        inv.advance()
        result = inv.consume(6.0, transport_time=99.0)  # transport ignored when inert
        self.assertAlmostEqual(result.delivered, 6.0)
        self.assertAlmostEqual(result.viability_loss, 0.0)

    def test_viability_hook_reduces_delivery_when_enabled(self) -> None:
        # A stub hook: viability drops with age + transport time.
        inv = AgingInventory(
            shelf_life=6,
            viability_fn=lambda age, tt: max(0.0, 1.0 - 0.1 * (age + tt)),
        )
        inv.add(10.0)
        for _ in range(3):
            inv.advance()  # items now at age 3
        result = inv.consume(10.0, transport_time=2.0)
        # factor = 1 - 0.1*(3 + 2) = 0.5 -> delivered 5, loss 5.
        self.assertAlmostEqual(result.consumed, 10.0)
        self.assertAlmostEqual(result.delivered, 5.0)
        self.assertAlmostEqual(result.viability_loss, 5.0)

    def test_viability_factor_is_clipped(self) -> None:
        inv = AgingInventory(shelf_life=3, viability_fn=lambda age, tt: 5.0)
        inv.add(2.0)
        result = inv.consume(2.0)
        self.assertAlmostEqual(result.delivered, 2.0)  # factor clipped to 1.0

    def test_rejects_bad_shelf_life(self) -> None:
        with self.assertRaises(ValueError):
            AgingInventory(shelf_life=0)


if __name__ == "__main__":
    unittest.main()
