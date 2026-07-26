from __future__ import annotations

import unittest

from evaluation.calibrate_patient_lifecycle import (
    reweighted_cost_mean,
    scaled_patient_config,
)


class PatientLifecycleCalibrationTests(unittest.TestCase):
    def test_scaled_patient_config_only_changes_selected_decay_fields(self) -> None:
        reference = {
            "healthy_decay_rate": 0.01,
            "frail_decay_rate": 0.12,
            "waiting_time_decay_rate": 0.01,
            "eligibility_threshold": 0.8,
        }

        actual = scaled_patient_config(
            reference,
            ["healthy_decay_rate", "frail_decay_rate", "waiting_time_decay_rate"],
            0.13,
        )

        self.assertAlmostEqual(actual["healthy_decay_rate"], 0.0013)
        self.assertAlmostEqual(actual["frail_decay_rate"], 0.0156)
        self.assertAlmostEqual(actual["waiting_time_decay_rate"], 0.0013)
        self.assertEqual(actual["eligibility_threshold"], 0.8)
        self.assertEqual(reference["healthy_decay_rate"], 0.01)

    def test_reweighted_cost_uses_patient_facing_components(self) -> None:
        rows = [
            {
                "base_cost": 100.0,
                "patients_lost": 2.0,
                "material_wasted": 3.0,
                "at_risk_unserved": 4.0,
            },
            {
                "base_cost": 200.0,
                "patients_lost": 1.0,
                "material_wasted": 2.0,
                "at_risk_unserved": 3.0,
            },
        ]
        profile = {
            "weight_patient_lost": 10.0,
            "weight_expiry": 5.0,
            "weight_urgency": 2.0,
        }

        self.assertEqual(reweighted_cost_mean(rows, profile), 184.5)


if __name__ == "__main__":
    unittest.main()
