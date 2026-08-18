import copy
import json
import unittest
from pathlib import Path

import numpy as np

from evaluation.audit_ddpg_legal_action_ranker_feasibility import (
    FORMAL_HOLDOUT_SEED,
    diagnostic_crn_seed_sequence,
    fold_seed_assignments,
    legal_action_ranker_loss,
    ranking_metrics,
    smoke_config,
    stage_g1_decision,
    target_scales,
    validate_config,
)
from src.rl.networks import torch


class _LinearCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states, actions):
        del states
        return actions[:, :1] * self.weight


class _Agent:
    def __init__(self) -> None:
        self.critic = _LinearCritic()
        self.device = torch.device("cpu")


class DDPGLegalActionRankerG1Tests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "experiments/configs/"
            "patient_indexed_specimen_routing_ddpg_legal_action_ranker_g1.json"
        )
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def test_locked_config_uses_fresh_disjoint_streams(self) -> None:
        validate_config(self.config)
        used = diagnostic_crn_seed_sequence(self.config)

        self.assertEqual(len(used), 2499)
        self.assertEqual(len(used), len(set(used)))
        self.assertNotIn(FORMAL_HOLDOUT_SEED, used)
        self.assertGreater(min(used), 100_500_000)

        smoke = smoke_config(self.config)
        validate_config(smoke)
        self.assertEqual(smoke["dataset"]["decision_steps"], [0, 25, 51])
        self.assertEqual(smoke["ranker"]["epochs"], 2)
        self.assertEqual(len(diagnostic_crn_seed_sequence(smoke)), 39)

    def test_leave_one_seed_out_assignments_never_include_held_out_seed(self) -> None:
        assignments = fold_seed_assignments((60, 61, 62))

        self.assertEqual(assignments[60], (61, 62))
        self.assertEqual(assignments[61], (60, 62))
        self.assertEqual(assignments[62], (60, 61))
        for held_out, training in assignments.items():
            self.assertNotIn(held_out, training)

    def test_target_scale_is_within_state_and_has_fixed_floor(self) -> None:
        advantages = np.asarray(
            [
                [0.0, 100_000.0, -200_000.0, 300_000.0, -400_000.0],
                [0.0, 1_000_000.0, -2_000_000.0, 3_000_000.0, -4_000_000.0],
            ]
        )

        scales = target_scales(advantages, floor=1_000_000.0)

        np.testing.assert_allclose(scales, [1_000_000.0, 2_500_000.0])

    def test_ranking_metrics_reward_correct_legal_action_order(self) -> None:
        advantages = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, -1.0, 4.0, 2.0, 1.0],
            ]
        )

        correct = ranking_metrics(
            advantages,
            advantages,
            minimum_pairwise_cost_gap=0.5,
            material_improvement=1.0,
        )
        reversed_result = ranking_metrics(
            -advantages,
            advantages,
            minimum_pairwise_cost_gap=0.5,
            material_improvement=1.0,
        )

        self.assertEqual(correct["top1_accuracy"], 1.0)
        self.assertEqual(correct["pairwise_accuracy"], 1.0)
        self.assertEqual(reversed_result["top1_accuracy"], 0.0)
        self.assertEqual(reversed_result["pairwise_accuracy"], 0.0)

    def test_fixed_loss_learns_higher_q_for_lower_cost_action(self) -> None:
        agent = _Agent()
        state_tensor = torch.zeros((5, 1), dtype=torch.float32)
        action_tensor = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
        target_tensor = torch.asarray(
            [[0.0, 0.25, 0.5, 0.75, 1.0]],
            dtype=torch.float32,
        )
        raw_advantage_tensor = target_tensor * 4_000_000.0
        ranker = {
            "regression_weight": 1.0,
            "pairwise_ranking_weight": 1.0,
            "pairwise_margin": 0.1,
            "pairwise_minimum_cost_gap": 250_000.0,
        }
        initial = legal_action_ranker_loss(
            agent,
            state_tensor=state_tensor,
            action_tensor=action_tensor,
            target_tensor=target_tensor,
            raw_advantage_tensor=raw_advantage_tensor,
            ranker=ranker,
        )["total"].item()
        optimizer = torch.optim.Adam(agent.critic.parameters(), lr=0.05)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = legal_action_ranker_loss(
                agent,
                state_tensor=state_tensor,
                action_tensor=action_tensor,
                target_tensor=target_tensor,
                raw_advantage_tensor=raw_advantage_tensor,
                ranker=ranker,
            )["total"]
            loss.backward()
            optimizer.step()
        final = legal_action_ranker_loss(
            agent,
            state_tensor=state_tensor,
            action_tensor=action_tensor,
            target_tensor=target_tensor,
            raw_advantage_tensor=raw_advantage_tensor,
            ranker=ranker,
        )["total"].item()

        self.assertLess(final, initial)
        self.assertGreater(agent.critic.weight.item(), 0.0)

    def test_decision_can_authorize_only_actor_transfer_smoke(self) -> None:
        folds = {}
        for seed in (60, 61, 62):
            folds[str(seed)] = {
                "actor_unchanged": True,
                "metrics": {
                    "validation": {
                        "baseline": {"top1_accuracy": 0.20},
                        "fitted": {"top1_accuracy": 0.42},
                    }
                },
            }
        aggregate = {
            "discovery": {"fitted": {"top1_accuracy": 0.45}},
            "validation": {
                "fitted": {
                    "top1_accuracy": 0.42,
                    "pairwise_accuracy": 0.70,
                }
            },
        }
        stability = {"best_action_agreement": 0.80}

        passed = stage_g1_decision(
            self.config,
            aggregate=aggregate,
            folds=folds,
            label_stability=stability,
            smoke=False,
        )
        unstable = stage_g1_decision(
            self.config,
            aggregate=aggregate,
            folds=folds,
            label_stability={"best_action_agreement": 0.60},
            smoke=False,
        )
        smoke = stage_g1_decision(
            copy.deepcopy(self.config),
            aggregate=aggregate,
            folds=folds,
            label_stability=stability,
            smoke=True,
        )

        self.assertTrue(passed["legal_action_ranker_feasibility_passed"])
        self.assertTrue(passed["actor_transfer_smoke_design_authorized"])
        self.assertFalse(passed["online_training_authorized"])
        self.assertFalse(passed["formal_confirmation_authorized"])
        self.assertEqual(
            unstable["classification"],
            "unstable_counterfactual_labels_close_extension",
        )
        self.assertEqual(smoke["classification"], "smoke_only_no_scientific_decision")


if __name__ == "__main__":
    unittest.main()
