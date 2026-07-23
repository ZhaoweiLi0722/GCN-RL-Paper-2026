"""IQM + bootstrap-CI aggregation (Phase 7, group 2)."""

from __future__ import annotations

import unittest

from evaluation.aggregate_stats import (
    aggregate_iqm,
    bootstrap_ci,
    divergent_seeds,
    interquartile_mean,
    paired_two_level_summary,
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


class PairedTwoLevelTests(unittest.TestCase):
    def test_pairs_replications_and_bootstraps_training_seeds(self) -> None:
        candidate = []
        baseline = []
        for training_seed, differences in ((0, (-2.0, -1.0)), (1, (0.0, 1.0))):
            for replication, difference in enumerate(differences):
                common = {
                    "training_seed": training_seed,
                    "evaluation_seed": 100 + replication,
                    "replication": replication,
                }
                candidate.append({**common, "total_cost": 10.0 + difference})
                baseline.append({**common, "total_cost": 10.0})

        summary = paired_two_level_summary(
            candidate,
            baseline,
            resamples=500,
            seed=7,
        )

        self.assertEqual(summary["n_training_seeds"], 2)
        self.assertEqual(summary["n_pairs"], 4)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["ties"], 1)
        self.assertAlmostEqual(summary["mean_difference"], -0.5)
        self.assertAlmostEqual(summary["mean_gap_pct"], -5.0)
        self.assertEqual(summary["per_seed_mean_difference"], {"0": -1.5, "1": 0.5})
        self.assertLessEqual(summary["ci_low"], summary["mean_difference"])
        self.assertGreaterEqual(summary["ci_high"], summary["mean_difference"])

    def test_rejects_unpaired_rows(self) -> None:
        candidate = [
            {"training_seed": 0, "evaluation_seed": 1, "replication": 0, "total_cost": 9.0}
        ]
        baseline = [
            {"training_seed": 0, "evaluation_seed": 2, "replication": 0, "total_cost": 10.0}
        ]
        with self.assertRaises(ValueError):
            paired_two_level_summary(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
