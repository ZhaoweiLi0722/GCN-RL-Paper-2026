"""IQM + bootstrap-CI aggregation (Phase 7, group 2)."""

from __future__ import annotations

import unittest

from evaluation.aggregate_stats import (
    aggregate_iqm,
    bootstrap_ci,
    divergent_seeds,
    interquartile_mean,
)


class InterquartileMeanTests(unittest.TestCase):
    def test_matches_hand_computation(self) -> None:
        # n=5, trim 25% -> drop 1 from each end -> mean of middle 3.
        self.assertAlmostEqual(interquartile_mean([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)

    def test_resists_single_divergent_seed(self) -> None:
        # The DDPG motivation: four near-optimal seeds + one blown-up seed.
        values = [0.98, 0.97, 0.99, 0.96, -70.0]
        mean = sum(values) / len(values)
        iqm = interquartile_mean(values)
        self.assertLess(mean, 0.0)          # the mean is wrecked by the outlier
        self.assertGreater(iqm, 0.9)        # the IQM is not
        self.assertIn(4, divergent_seeds(values))  # and the seed is flagged divergent


class BootstrapCITests(unittest.TestCase):
    def test_ci_brackets_point_estimate_and_is_reproducible(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo1, hi1 = bootstrap_ci(values, seed=0)
        lo2, hi2 = bootstrap_ci(values, seed=0)
        self.assertEqual((lo1, hi1), (lo2, hi2))         # deterministic under seed
        self.assertLessEqual(lo1, interquartile_mean(values))
        self.assertGreaterEqual(hi1, interquartile_mean(values))

    def test_different_seed_may_differ_but_stays_in_range(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(values, seed=7)
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, min(values))
        self.assertLessEqual(hi, max(values))


class AggregateIqmTests(unittest.TestCase):
    def test_two_level_aggregation(self) -> None:
        # 2 seeds x 2 replications for one algorithm; per-seed mean then IQM.
        rows = [
            {"algorithm": "gcn_sac", "scenario": "s", "graph_ablation": "full_graph", "seed": 0, "eligibility_rate": 0.8},
            {"algorithm": "gcn_sac", "scenario": "s", "graph_ablation": "full_graph", "seed": 0, "eligibility_rate": 0.9},
            {"algorithm": "gcn_sac", "scenario": "s", "graph_ablation": "full_graph", "seed": 1, "eligibility_rate": 0.7},
            {"algorithm": "gcn_sac", "scenario": "s", "graph_ablation": "full_graph", "seed": 1, "eligibility_rate": 0.5},
        ]
        summary = aggregate_iqm(rows, metrics=("eligibility_rate",))
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["algorithm"], "gcn_sac")
        # per-seed means: seed0 -> 0.85, seed1 -> 0.60; mean over seeds = 0.725.
        self.assertAlmostEqual(row["eligibility_rate_mean"], 0.725)
        self.assertEqual(row["eligibility_rate_n_seeds"], 2)
        self.assertIn("eligibility_rate_iqm", row)
        self.assertIn("eligibility_rate_per_seed", row)


if __name__ == "__main__":
    unittest.main()
