import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.evaluate_multiscenario_network_residual import (
    ResidualUsageMonitor,
    normalized_deployment_candidate,
    resolve_checkpoint_variants,
    rows_by_scenario,
    select_deployment_candidate,
)


def _result(
    scale: float,
    cost_difference: float,
    *,
    aggregate_safe: bool = True,
    all_scenarios_safe: bool = True,
) -> dict:
    return {
        "candidate": {
            "scale": scale,
            "group_thresholds": [0.7, 0.8, 1.0],
        },
        "aggregate": {
            "cost_difference_mean": cost_difference,
            "clinically_noninferior": aggregate_safe,
        },
        "all_scenarios_clinically_noninferior": all_scenarios_safe,
    }


class MultiScenarioResidualEvaluationTests(unittest.TestCase):
    def test_residual_usage_monitor_counts_action_groups(self) -> None:
        class Agent:
            def reset(self):
                self.last_residual_action = np.zeros(8, dtype=np.float32)

            def select_action(self, state, explore=False, env=None):
                self.last_residual_action = np.asarray(
                    [0, 0, 0.2, 0, 0, 0, 0, 0],
                    dtype=np.float32,
                )
                return np.zeros(8, dtype=np.float32)

        monitor = ResidualUsageMonitor(Agent(), num_facilities=2)
        monitor.reset()
        monitor.select_action(np.zeros(3, dtype=np.float32))
        summary = monitor.summary()

        self.assertEqual(summary["corrected_decisions"], 1)
        self.assertEqual(
            summary["group_correction_rates"]["reagent_transfer"],
            1.0,
        )
        self.assertEqual(
            summary["group_correction_rates"]["capacity_transfer"],
            0.0,
        )

    def test_candidate_validation_rejects_bad_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            normalized_deployment_candidate(
                {
                    "scale": 0.5,
                    "group_thresholds": [0.7, 1.2, 1.0],
                }
            )

    def test_selection_uses_validation_cost_with_clinical_guardrail(self) -> None:
        selected = select_deployment_candidate(
            [
                _result(0.0, 0.0),
                _result(0.25, -10.0),
                _result(0.5, -20.0, aggregate_safe=False),
            ],
            strict_scenario_guardrail=False,
        )

        self.assertEqual(selected["candidate"]["scale"], 0.25)
        self.assertEqual(selected["selection_source"], "validation_only")

    def test_strict_selection_rejects_single_scenario_harm(self) -> None:
        selected = select_deployment_candidate(
            [
                _result(0.0, 0.0),
                _result(0.25, -10.0, all_scenarios_safe=False),
            ],
            strict_scenario_guardrail=True,
        )

        self.assertEqual(selected["candidate"]["scale"], 0.0)

    def test_selection_can_choose_pretrain_over_online_final(self) -> None:
        final = _result(0.25, -5.0)
        final["checkpoint_variant"] = "final"
        pretrain = _result(0.25, -10.0)
        pretrain["checkpoint_variant"] = "pretrain"

        selected = select_deployment_candidate(
            [final, pretrain],
            strict_scenario_guardrail=False,
        )

        self.assertEqual(selected["checkpoint_variant"], "pretrain")

    def test_checkpoint_variants_deduplicate_identical_files(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"test")

            variants = resolve_checkpoint_variants(
                {
                    "checkpoint": str(checkpoint),
                    "pretrain_checkpoint": str(checkpoint),
                },
                ("pretrain", "final"),
            )

        self.assertEqual(tuple(variants), ("pretrain",))

    def test_anchor_rows_are_grouped_without_aliasing(self) -> None:
        rows = [
            {"scenario": "nominal", "total_cost": 1.0},
            {"scenario": "drift", "total_cost": 2.0},
        ]

        grouped = rows_by_scenario(rows)
        grouped["nominal"][0]["total_cost"] = 99.0

        self.assertEqual(rows[0]["total_cost"], 1.0)
        self.assertEqual(tuple(grouped), ("nominal", "drift"))


if __name__ == "__main__":
    unittest.main()
