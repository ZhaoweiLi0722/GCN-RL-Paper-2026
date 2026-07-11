"""Projection-load measurement + stability report (Phase 7, group 5)."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.aggregate_stats import stability_report
from src.rl.action_projection import project_action, projection_repair_magnitude


class ProjectionRepairTests(unittest.TestCase):
    def test_zero_when_already_feasible(self) -> None:
        self.assertEqual(projection_repair_magnitude([0.5, -0.5, 1.0, -1.0]), 0.0)

    def test_positive_when_out_of_bounds(self) -> None:
        # raw = 1.5 -> clipped to 1.0, repair = 0.5; raw = -2.0 -> repair 1.0.
        mag = projection_repair_magnitude([1.5, -2.0, 0.0])
        self.assertAlmostEqual(mag, float(np.sqrt(0.5**2 + 1.0**2)))
        self.assertGreater(mag, 0.0)

    def test_projected_action_carries_repair_magnitude(self) -> None:
        result = project_action([2.0, 0.0], action_space_info=2)
        self.assertTrue(result.clipped)
        self.assertAlmostEqual(result.repair_magnitude, 1.0)
        result2 = project_action([0.3, -0.3], action_space_info=2)
        self.assertFalse(result2.clipped)
        self.assertEqual(result2.repair_magnitude, 0.0)


class StabilityReportTests(unittest.TestCase):
    def test_flags_divergent_seed_and_computes_iqm(self) -> None:
        rows = [
            {"algorithm": "flat_ddpg", "seed": s, "total_cost": c}
            for s, c in zip(range(5), [100.0, 101.0, 99.0, 102.0, 5000.0])  # seed 4 diverges
        ]
        report = stability_report(rows, metric="total_cost")
        self.assertEqual(len(report), 1)
        row = report[0]
        self.assertEqual(row["algorithm"], "flat_ddpg")
        self.assertEqual(row["n_seeds"], 5)
        self.assertIn(4, __import__("json").loads(row["divergent_seeds"]))
        # IQM ignores the blown-up seed; spread captures it.
        self.assertLess(row["iqm"], 200.0)
        self.assertGreater(row["spread"], 4000.0)


if __name__ == "__main__":
    unittest.main()
