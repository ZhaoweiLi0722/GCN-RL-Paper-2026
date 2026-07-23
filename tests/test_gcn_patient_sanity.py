"""Graph-family patient-env sanity: each GNN agent beats random (Phase 6, group 6).

Complements the encoder component tests: those certify the graph nets in
isolation; this certifies the *composed* agents actually learn on our problem.
Verified margins at this budget (seed 0): TD3 ~0.18x, SAC ~0.01x, PPO ~0.04x of
the random-policy cost — the 0.5x assertion is deliberately wide and non-flaky.
"""

from __future__ import annotations

import unittest

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"
GRAPH_ALGORITHMS = ("gcn_td3", "gcn_sac", "gcn_ppo")


class GCNPatientSanityTest(unittest.TestCase):
    def test_graph_agents_beat_random_on_patient_env(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")

        from evaluation.verify_algorithms import patient_env_sanity

        for algorithm in GRAPH_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = patient_env_sanity(
                    algorithm, DEV_CONFIG, train_steps=2000, eval_episodes=8, seed=0
                )
                self.assertTrue(result["beats_random"], result)
                self.assertLess(result["learned_cost"], 0.5 * result["random_cost"], result)


if __name__ == "__main__":
    unittest.main()
