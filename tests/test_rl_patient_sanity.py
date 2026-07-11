"""RL patient-env sanity: does a learned agent beat random? (Phase 5, group 3)."""

from __future__ import annotations

import unittest

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"


class RLPatientSanityTest(unittest.TestCase):
    def test_flat_ddpg_beats_random_on_patient_env(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")

        from evaluation.verify_algorithms import patient_env_sanity

        result = patient_env_sanity("flat_ddpg", DEV_CONFIG, train_steps=1500, eval_episodes=8, seed=0)
        self.assertTrue(result["beats_random"], result)
        # Wide, non-flaky margin: flat-DDPG beats random by ~100x here.
        self.assertLess(result["learned_cost"], 0.5 * result["random_cost"])


if __name__ == "__main__":
    unittest.main()
