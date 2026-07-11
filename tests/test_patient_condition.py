"""Tests for the patient-condition survival model (Phase 4, task group 1)."""

from __future__ import annotations

import unittest

import numpy as np

from src.env.patient_condition import (
    PatientConditionConfig,
    PatientConditionModel,
    PatientStatus,
)


class PatientConditionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = PatientConditionModel()

    def test_survival_starts_near_one(self) -> None:
        patient = self.model.enroll(np.random.default_rng(0), epoch=0)
        self.assertAlmostEqual(patient.survival, 1.0, places=6)
        self.assertEqual(patient.status, PatientStatus.WAITING)

    def test_survival_decreases_monotonically_with_age(self) -> None:
        patient = self.model.enroll(np.random.default_rng(1), epoch=0)
        previous = patient.survival
        for _ in range(30):
            self.model.advance(patient)
            self.assertLessEqual(patient.survival, previous + 1e-12)
            previous = patient.survival

    def test_survival_stays_in_unit_interval(self) -> None:
        patient = self.model.enroll(np.random.default_rng(2), epoch=0)
        for _ in range(200):
            self.model.advance(patient)
            self.assertGreaterEqual(patient.survival, 0.0)
            self.assertLessEqual(patient.survival, 1.0)

    def test_decay_accelerates_after_deterioration_shock(self) -> None:
        # Fixed health index and a known shock epoch; compare per-epoch drops
        # well before vs. well after the shock.
        model = PatientConditionModel()
        h, t_d = 0.5, 5.0

        def drop(age: int) -> float:
            return model.survival_at(age, h, t_d) - model.survival_at(age + 1, h, t_d)

        pre_shock_drop = drop(2)   # age 2 -> 3, before the shock
        post_shock_drop = drop(8)  # age 8 -> 9, after the shock
        self.assertGreater(post_shock_drop, pre_shock_drop)

    def test_eligibility_flips_below_threshold(self) -> None:
        # A frail patient (low health index) should eventually become ineligible.
        model = PatientConditionModel(PatientConditionConfig(eligibility_threshold=0.75))
        from src.env.patient_condition import PatientState

        patient = PatientState(health_index=0.0, deterioration_epoch=3.0, enrollment_epoch=0)
        model.advance(patient, epochs=0)
        self.assertTrue(model.is_eligible(patient))  # survival == 1.0 at age 0
        model.advance(patient, epochs=60)
        self.assertLess(patient.survival, 0.75)
        self.assertFalse(model.is_eligible(patient))

    def test_deterministic_under_seed(self) -> None:
        p1 = self.model.enroll(np.random.default_rng(123), epoch=0)
        p2 = self.model.enroll(np.random.default_rng(123), epoch=0)
        self.assertEqual(p1.health_index, p2.health_index)
        self.assertEqual(p1.deterioration_epoch, p2.deterioration_epoch)
        for _ in range(10):
            self.model.advance(p1)
            self.model.advance(p2)
            self.assertEqual(p1.survival, p2.survival)

    def test_config_validation_rejects_bad_rates(self) -> None:
        with self.assertRaises(ValueError):
            PatientConditionModel(
                PatientConditionConfig(healthy_decay_rate=0.1, frail_decay_rate=0.01)
            )
        with self.assertRaises(ValueError):
            PatientConditionModel(PatientConditionConfig(eligibility_threshold=1.5))


if __name__ == "__main__":
    unittest.main()
