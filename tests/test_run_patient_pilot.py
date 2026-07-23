"""Two-stage pilot runner smoke test (Phase 7, group 4)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class RunPatientPilotSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")

    def test_two_stage_smoke_produces_ranking_and_flagship(self) -> None:
        from evaluation.run_patient_pilot import run_two_stage_pilot

        with tempfile.TemporaryDirectory() as tmp:
            result = run_two_stage_pilot(
                stage_a_roster=["mdl2", "gcn_sac", "gcn_td3", "gcn_ddpg", "flat_ddpg"],
                seeds=(0, 1),
                target_steps=120,
                eval_replications=2,
                out_dir=tmp,
                run_stage_b=True,
            )
            # Stage A ranked everything; a graph backbone was promoted.
            self.assertTrue(result["stage_a_ranking"])
            self.assertIn(result["top_backbone"], ("gcn_sac", "gcn_td3"))
            # Stage B ran and picked a flagship among the candidates.
            self.assertIsNotNone(result["stage_b_ranking"])
            self.assertIn(result["flagship"], ("gcn_sac", "gcn_td3"))
            # Ranking rows carry the eligibility IQM column used to sort.
            self.assertIn("eligibility_rate_mean_iqm", result["stage_a_ranking"][0])
            # Artifacts written.
            self.assertTrue((Path(tmp) / "stage_A_ranking.csv").exists())
            self.assertTrue((Path(tmp) / "pilot_result.json").exists())

    def test_resume_skips_completed_work(self) -> None:
        from evaluation.run_patient_pilot import run_two_stage_pilot

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(
                stage_a_roster=["mdl2", "gcn_sac"],
                seeds=(0,),
                target_steps=100,
                eval_replications=2,
                out_dir=tmp,
                run_stage_b=False,
            )
            first = run_two_stage_pilot(**kwargs)
            # Second run with resume reuses the per-(alg,seed) CSVs; same ranking.
            second = run_two_stage_pilot(resume=True, **kwargs)
            self.assertEqual(
                [r["algorithm"] for r in first["stage_a_ranking"]],
                [r["algorithm"] for r in second["stage_a_ranking"]],
            )


if __name__ == "__main__":
    unittest.main()
