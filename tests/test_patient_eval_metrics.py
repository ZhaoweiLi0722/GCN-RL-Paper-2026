"""Patient-condition metrics in the formal evaluator (Phase 7, group 1)."""

from __future__ import annotations

import unittest

from evaluation.evaluate_formal import PATIENT_METRICS, evaluate_agent, summarize_rows
from src.baselines.heuristics import get_heuristic_class
from src.rl.config import load_config
from src.rl.experiment import build_env

PATIENT_CONFIG = "experiments/configs/2_clinic_patient_condition.json"
BASE_CONFIG = "experiments/configs/20_clinic_disruption_0_3.json"


class PatientEvalMetricsTests(unittest.TestCase):
    def _rows(self, config_path: str):
        env = build_env({"env": load_config(config_path)}, seed=0)
        agent = get_heuristic_class("myo")(env.observation_size, env.action_size, {})
        return evaluate_agent(agent, env, algorithm="myo", seed=0, replications=3, max_steps=None)

    def test_patient_rows_have_valid_metrics(self) -> None:
        rows = self._rows(PATIENT_CONFIG)
        self.assertTrue(rows)
        for row in rows:
            for metric in PATIENT_METRICS:
                self.assertIn(metric, row)
            self.assertGreaterEqual(row["eligibility_rate"], 0.0)
            self.assertLessEqual(row["eligibility_rate"], 1.0)
            self.assertGreaterEqual(row["eligibility_rate_mean"], 0.0)
            self.assertLessEqual(row["eligibility_rate_mean"], 1.0)
            self.assertGreaterEqual(row["patients_lost"], 0.0)
            self.assertGreaterEqual(row["material_wasted"], 0.0)
            self.assertGreaterEqual(row["at_risk_unserved"], 0.0)
            # env defines patients_lost = ineligible + expired.
            self.assertAlmostEqual(
                row["patients_lost"],
                row["patients_lost_ineligible"] + row["patients_lost_expired"],
                places=6,
            )

    def test_patient_summary_includes_eligibility(self) -> None:
        summary = summarize_rows(self._rows(PATIENT_CONFIG))
        self.assertIn("eligibility_rate_mean_mean", summary)
        self.assertIn("patients_lost_mean", summary)

    def test_base_env_schema_unchanged(self) -> None:
        rows = self._rows(BASE_CONFIG)
        self.assertTrue(rows)
        for row in rows:
            for metric in PATIENT_METRICS:
                self.assertNotIn(metric, row)
        summary = summarize_rows(rows)
        self.assertNotIn("eligibility_rate_mean_mean", summary)


if __name__ == "__main__":
    unittest.main()
