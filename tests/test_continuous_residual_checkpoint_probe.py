from __future__ import annotations

import unittest

from evaluation.probe_continuous_residual_checkpoints import (
    add_training_seed,
    paired_candidate_summary,
    select_best_candidate,
)


class ContinuousResidualCheckpointProbeTests(unittest.TestCase):
    def test_paired_summary_uses_crn_differences_and_clinical_guardrails(self) -> None:
        anchor = [
            {
                "total_cost": 100.0,
                "completion_service_level": 0.90,
                "patients_lost": 10.0,
                "patient_ineligibility_during_manufacturing_rate": 0.10,
            },
            {
                "total_cost": 120.0,
                "completion_service_level": 0.80,
                "patients_lost": 12.0,
                "patient_ineligibility_during_manufacturing_rate": 0.12,
            },
        ]
        candidate = [
            {
                "total_cost": 90.0,
                "completion_service_level": 0.91,
                "patients_lost": 9.0,
                "patient_ineligibility_during_manufacturing_rate": 0.09,
            },
            {
                "total_cost": 110.0,
                "completion_service_level": 0.81,
                "patients_lost": 11.0,
                "patient_ineligibility_during_manufacturing_rate": 0.11,
            },
        ]

        summary = paired_candidate_summary(candidate, anchor)

        self.assertAlmostEqual(summary["cost_difference_mean"], -10.0)
        self.assertAlmostEqual(summary["cost_gap_pct"], -1000.0 / 110.0)
        self.assertTrue(summary["clinically_noninferior"])

    def test_best_candidate_requires_clinical_noninferiority(self) -> None:
        unsafe = {
            "candidate_cost_mean": 80.0,
            "clinically_noninferior": False,
        }
        safe = {
            "candidate_cost_mean": 90.0,
            "clinically_noninferior": True,
        }

        self.assertIs(select_best_candidate([unsafe, safe]), safe)

    def test_training_seed_is_kept_separate_from_evaluation_seed(self) -> None:
        rows = [{"seed": 230000, "replication": 0}]

        add_training_seed(rows, 2)

        self.assertEqual(rows[0]["seed"], 230000)
        self.assertEqual(rows[0]["training_seed"], 2)
        self.assertEqual(rows[0]["evaluation_seed"], 230000)


if __name__ == "__main__":
    unittest.main()
