from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from evaluation.audit_ddpg_online_identifiability import (
    discounted_episode_remainders,
    legal_ranking_metrics,
    manifold_crn_seed_sequence,
    ordered_online_replay_arrays,
    recompute_n_step_targets,
    replay_target_metrics,
    stage_f0_decision,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_online_identifiability_f0.json"
)


class DDPGOnlineIdentifiabilityF0Tests(unittest.TestCase):
    def test_committed_config_uses_fresh_disjoint_diagnostic_crns(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        validate_config(config)

        used = manifold_crn_seed_sequence(config)
        self.assertEqual(len(used), len(set(used)))
        self.assertNotIn(91_100_000, used)
        self.assertTrue(all(seed >= 95_900_000 for seed in used))

    def test_formal_holdout_seed_is_rejected(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(config)
        invalid["manifold"]["live_seed"] = 91_100_000

        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_config(invalid)

    def test_reconstructs_n_step_and_episode_remainder_targets(self) -> None:
        one_step = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        remainder = discounted_episode_remainders(
            one_step,
            gamma=0.5,
            episode_length=4,
        )
        n_step, multipliers = recompute_n_step_targets(
            one_step,
            gamma=0.5,
            episode_length=4,
            n_step_horizon=2,
        )

        np.testing.assert_allclose(remainder, [3.25, 4.5, 5.0, 4.0])
        np.testing.assert_allclose(n_step, [2.0, 3.5, 5.0, 4.0])
        np.testing.assert_allclose(multipliers, [0.5, 0.5, 0.5, 1.0])

    def test_detects_positive_short_target_with_negative_remainder(self) -> None:
        one_step = np.asarray(
            [1.0, 1.0, 1.0, 1.0, -10.0, -10.0],
            dtype=np.float64,
        )
        n_step, multipliers = recompute_n_step_targets(
            one_step,
            gamma=1.0,
            episode_length=6,
            n_step_horizon=4,
        )
        result, _derived = replay_target_metrics(
            {
                "one_step_rewards": one_step,
                "rewards": n_step,
                "discount_multipliers": multipliers,
                "dones": np.zeros_like(one_step),
            },
            gamma=1.0,
            episode_length=6,
            n_step_horizon=4,
            self_imitation_minimum_return=0.5,
            sign_tolerance=1e-12,
        )

        self.assertEqual(result["self_imitation_active_count"], 1)
        self.assertEqual(
            result["self_imitation_active_remainder"]["negative_fraction"],
            1.0,
        )

    def test_online_replay_order_excludes_offline_prefix(self) -> None:
        replay = {
            "size": 5,
            "capacity": 10,
            "position": 5,
            "online_mask": np.asarray([False, False, True, True, True]),
            "rewards": np.arange(5, dtype=float).reshape(-1, 1),
            "one_step_rewards": np.arange(5, dtype=float).reshape(-1, 1),
            "dones": np.zeros((5, 1), dtype=float),
            "discount_multipliers": np.ones((5, 1), dtype=float),
        }

        arrays = ordered_online_replay_arrays(replay)

        np.testing.assert_array_equal(arrays["rewards"], [2.0, 3.0, 4.0])

    def test_legal_ranking_metrics_separate_critic_and_headroom(self) -> None:
        rows = []
        for step in (0, 1):
            for index, (true, predicted, gradient) in enumerate(
                (
                    (0.0, 0.0, 0.0),
                    (2_000_000.0, -1.0, -1.0),
                    (-1_000_000.0, 1.0, 1.0),
                )
            ):
                rows.append(
                    {
                        "step": step,
                        "candidate_index": index,
                        "executed_action_distinct": 1,
                        "true_cost_advantage": true,
                        "critic_advantage": predicted,
                        "gradient_score": gradient,
                    }
                )

        metrics = legal_ranking_metrics(
            rows,
            material_improvement=1_000_000.0,
        )

        self.assertEqual(metrics["states"], 2)
        self.assertEqual(metrics["headroom_state_fraction"], 1.0)
        self.assertEqual(metrics["critic_top1_accuracy"], 0.0)
        self.assertLess(metrics["critic_pairwise_accuracy"], 0.5)

    def test_ranking_branch_can_justify_design_without_authorizing_training(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        primary = config["primary_algorithm"]
        replay_per_seed = {}
        manifold_per_seed = {}
        for seed in config["training_seeds"]:
            replay_per_seed[str(seed)] = {
                "self_imitation_active_remainder": {
                    "negative_fraction": 0.01,
                },
                "final_vs_frozen_total_cost_difference": (
                    100.0 if seed == 51 else -100.0
                ),
            }
            manifold_per_seed[str(seed)] = {
                "pretrain": {
                    "remaining": {"headroom_state_fraction": 0.5},
                },
                "final": {
                    "remaining": {
                        "critic_pairwise_accuracy": 0.5,
                        "gradient_top1_accuracy": 0.5,
                    },
                },
            }

        decision = stage_f0_decision(
            config,
            replay_result={"per_seed": {primary: replay_per_seed}},
            manifold_result={"metrics": {primary: manifold_per_seed}},
            smoke=False,
        )

        self.assertTrue(decision["gate_passed"])
        self.assertEqual(decision["replay_target_branch_seeds"], [])
        self.assertEqual(
            decision["legal_action_ranking_branch_seeds"],
            [50, 51, 52],
        )
        self.assertFalse(decision["f1_training_authorized"])


if __name__ == "__main__":
    unittest.main()
