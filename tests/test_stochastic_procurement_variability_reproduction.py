from __future__ import annotations

import copy
import unittest
from pathlib import Path

from evaluation.reproduce_stochastic_procurement_variability import (
    fixed_mean_lead,
    load_config,
    metric_summary,
    pooled_seed_cluster_summary,
    seed_values,
    smoke_config,
    validate_config,
)


CONFIG = Path(
    "experiments/configs/stochastic_procurement_variability_reproduction.json"
)


class ReproductionConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)

    def test_locked_config_is_valid_and_uses_fresh_disjoint_seeds(self) -> None:
        validate_config(self.config)
        development = set(seed_values(self.config["development_seeds"]))
        evaluation = set(seed_values(self.config["evaluation_seeds"]))
        self.assertFalse(development & evaluation)
        self.assertGreater(min(development | evaluation), 99_000_000)

    def test_role_cannot_be_mistaken_for_formal_confirmation(self) -> None:
        role = self.config["experimental_role"].lower()
        self.assertIn("post-hoc", role)
        self.assertIn("not a formal confirmation", role)
        self.assertIn("no policy training", role)

    def test_fixed_mean_is_exactly_two_epochs(self) -> None:
        self.assertEqual(fixed_mean_lead(self.config), 2)

    def test_overlap_is_rejected(self) -> None:
        broken = copy.deepcopy(self.config)
        broken["evaluation_seeds"] = dict(broken["development_seeds"])
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_config(broken)

    def test_noninteger_mean_is_rejected(self) -> None:
        broken = copy.deepcopy(self.config)
        broken["lead_time_env_overrides"]["reagent_lead_time_probabilities"] = [
            0.5,
            0.5,
        ]
        with self.assertRaisesRegex(ValueError, "integer expected lead"):
            validate_config(broken)

    def test_smoke_config_is_small_and_still_valid(self) -> None:
        smoke = smoke_config(self.config)
        validate_config(smoke)
        self.assertEqual(len(smoke["scenarios"]), 1)
        self.assertEqual(smoke["safety_multiplier_grid"], [1.0])
        self.assertEqual(smoke["development_seeds"]["count"], 1)
        self.assertEqual(smoke["evaluation_seeds"]["count"], 2)

    def test_smoke_allows_private_tmp_but_scientific_run_does_not(self) -> None:
        smoke = smoke_config(self.config)
        smoke["output_root"] = "/private/tmp/stochastic_procurement_smoke"
        validate_config(smoke)
        scientific = copy.deepcopy(self.config)
        scientific["output_root"] = "/private/tmp/not_allowed"
        with self.assertRaisesRegex(ValueError, "output_root must be under results"):
            validate_config(scientific)

    def test_metric_summary(self) -> None:
        summary = metric_summary([0.0, 0.01, 0.02])
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean"], 0.01)
        self.assertGreater(summary["normal_95_half_width"], 0.0)

    def test_pooled_interval_clusters_reused_seeds(self) -> None:
        rows = [
            {"seed": 1, "gap_fraction": 0.00},
            {"seed": 1, "gap_fraction": 0.02},
            {"seed": 2, "gap_fraction": 0.01},
            {"seed": 2, "gap_fraction": 0.03},
        ]
        summary = pooled_seed_cluster_summary(rows, expected_scenarios=2)
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["independent_seed_clusters"], 2)
        self.assertAlmostEqual(summary["mean"], 0.015)
        self.assertEqual(
            summary["interval_unit"], "evaluation_seed_mean_across_scenarios"
        )

    def test_pooled_interval_rejects_incomplete_seed_clusters(self) -> None:
        rows = [
            {"seed": 1, "gap_fraction": 0.00},
            {"seed": 2, "gap_fraction": 0.01},
            {"seed": 2, "gap_fraction": 0.03},
        ]
        with self.assertRaisesRegex(ValueError, "clusters are incomplete"):
            pooled_seed_cluster_summary(rows, expected_scenarios=2)


if __name__ == "__main__":
    unittest.main()
