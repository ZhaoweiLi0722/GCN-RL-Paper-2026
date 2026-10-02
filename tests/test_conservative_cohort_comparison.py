"""Whole five-role comparison from invented, persisted raw JSON only."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.conservative_cohort_comparison import verify_bundle
from src.rl.conservative_cohort_plan import streams
from tests.test_conservative_cohort_plan import config
from tests.test_paired_cohort_comparison import fixture_config, invented_episode, no_scientific_execution
from tests.test_cohort_bundle_verification import ACTION, save_episode


class ComparisonTests(unittest.TestCase):
    def test_whole_matrix_direct_costs_patients_actions_and_primary_screen(self):
        cfg = config()
        _, _, backend = fixture_config()
        seeds = streams(cfg, {str(b): 100 + b for b in cfg["blocks"]})
        with tempfile.TemporaryDirectory() as directory, no_scientific_execution(), \
                patch("src.rl.cohort_bundle_verification.facility_net_action_from_state", return_value=np.asarray(ACTION)):
            root, index = Path(directory), []
            for b in cfg["blocks"]:
                for role in cfg["evaluation_controllers"]:
                    for w in range(12):
                        values = invented_episode(cfg, seeds, backend, block=b, role=role, world=w,
                                                  cost=90 if role == "paired_cost" else 100)
                        index.append(save_episode(root, f"b{b}/{role}/w{w}", values))
            report = verify_bundle(root, index, cfg, backend, seeds)
            self.assertEqual(len(report["outcomes"]), 180)
            primary = report["analysis"]["contrasts"][0]
            self.assertAlmostEqual(primary["equal_block_mean_differences"]["cost"], -10)
            self.assertEqual(primary["equal_block_mean_differences"]["losses"], 0)
            self.assertEqual(report["analysis"]["decision"], "promising_development_only")
            with self.assertRaises(ValueError):
                verify_bundle(root, index[:-1], cfg, backend, seeds)
            with self.assertRaises(ValueError):
                verify_bundle(root, index + [index[0]], cfg, backend, seeds)


if __name__ == "__main__":
    unittest.main()
