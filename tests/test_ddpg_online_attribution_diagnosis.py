import unittest

import numpy as np

from evaluation.diagnose_ddpg_online_attribution import (
    paired_final_pretrain_decomposition,
    sampled_advantage_metrics,
    teacher_support_summary,
)


class DDPGOnlineAttributionDiagnosisTests(unittest.TestCase):
    def test_teacher_support_reports_non_executable_global_winners(self) -> None:
        demonstrations = {
            "option_advantages": np.asarray(
                [
                    [0.0, 2.0, 1.0],
                    [0.0, 1.0, 3.0],
                    [0.0, -1.0, -2.0],
                    [0.0, 4.0, 2.0],
                ]
            ),
            "option_feasible": np.ones((4, 3), dtype=bool),
            "option_groups": np.asarray(
                ["anchor", "specimen_transfer", "reagent_transfer"]
            ),
        }
        result = teacher_support_summary(
            demonstrations,
            allowed_option_groups=("specimen_transfer",),
        )

        self.assertEqual(result["policy_supported_samples"], 3)
        self.assertEqual(result["policy_unsupported_samples"], 1)
        self.assertAlmostEqual(result["policy_unsupported_fraction"], 0.25)
        self.assertEqual(
            result["globally_best_group_counts"],
            {"anchor": 1, "reagent_transfer": 1, "specimen_transfer": 2},
        )

    def test_sampled_metrics_distinguish_correct_and_reversed_rankings(self) -> None:
        target = np.linspace(-1.0, 1.0, 200)
        correct = sampled_advantage_metrics(
            target,
            target,
            pairwise_samples=20_000,
            seed=7,
        )
        reversed_result = sampled_advantage_metrics(
            target,
            -target,
            pairwise_samples=20_000,
            seed=7,
        )

        self.assertAlmostEqual(correct["spearman"], 1.0)
        self.assertAlmostEqual(correct["pairwise_order_accuracy"], 1.0)
        self.assertAlmostEqual(reversed_result["spearman"], -1.0)
        self.assertAlmostEqual(
            reversed_result["pairwise_order_accuracy"], 0.0
        )

    def test_component_decomposition_pairs_rows_by_crn_identity(self) -> None:
        fields = (
            "total_cost",
            "base_cost",
            "specimen_transfer_cost",
            "capacity_transfer_cost",
            "reagent_transfer_cost",
            "patient_loss_cost",
            "expiry_cost",
            "urgency_cost",
            "reagent_purchase_cost",
            "reagent_holding_cost",
            "reagent_shortage_cost",
            "bioreactor_holding_cost",
            "bioreactor_shortage_cost",
            "patients_lost",
            "patients_completed",
            "completion_service_level",
            "specimen_route_count",
            "specimen_route_distance_miles",
            "specimen_route_time_hours",
        )

        def row(replication: int, total: float, routes: float) -> dict[str, object]:
            result: dict[str, object] = {
                "training_seed": "10",
                "scenario": "nominal",
                "evaluation_seed": "200",
                "replication": str(replication),
            }
            result.update({field: 0.0 for field in fields})
            result["total_cost"] = total
            result["base_cost"] = total
            result["specimen_route_count"] = routes
            return result

        pretrain = [row(0, 100.0, 1.0), row(1, 200.0, 1.0)]
        final = [row(1, 197.0, 2.0), row(0, 99.0, 1.0)]
        phases = {
            "pretrain": {("gcn", 10): {"rows": pretrain}},
            "episode100": {("gcn", 10): {"rows": final}},
        }

        result = paired_final_pretrain_decomposition(
            phase_runs=phases,
            algorithms=("gcn",),
        )["gcn"]

        self.assertAlmostEqual(result["mean_differences"]["total_cost"], -2.0)
        self.assertAlmostEqual(result["changed_total_cost_fraction"], 1.0)
        self.assertAlmostEqual(result["changed_routing_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
