"""Synthetic GCN optimizer acceptance: no patient trajectory or scientific fit."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from evaluation.run_clean_critic_generalization import allocate_streams, summarize_test
from evaluation.summarize_clean_critic_generalization import error_accounting
from src.models.gcn_ddpg import GCNDDPGAgent
from src.rl.clean_critic_probe import (
    advantage, checkpoint, fit, fresh_critic, seal_predictions, target_scale, tensor_inputs,
)
from src.rl.critic_probe_contract import PublicInputs, StateLineage
from src.rl.frozen_value_probe import Budget


class CriticSmoke(unittest.TestCase):
    def setUp(self):
        config = {"algorithm": "gcn_ddpg", "seed": 7, "device": "cpu", "replay_buffer_size": 1,
                  "gcn_hidden_sizes": [8], "actor_hidden_sizes": [16], "critic_hidden_sizes": [16],
                  "env": {"num_facilities": 20, "production_lead_time": 3,
                          "action_mode": "facility_net", "include_supplier_state": True,
                          "include_central_capacity_hub": True}}
        self.agent = GCNDDPGAgent(140, 80, config)
        self.agent.actor.requires_grad_(False)
        self.public = []
        for i in range(3):
            state = np.linspace(.1, .9, 140) + i / 10
            actions = np.zeros((6, 80))
            actions[:, :20] = np.arange(6)[:, None] / 10
            self.public.append(PublicInputs(state, actions, actions[0]))
        self.support = np.ones((3, 6), bool)
        self.targets = np.tile(np.arange(6) * .01, (3, 1))
        self.settings = dict(updates=3, learning_rate=.0003, gradient_clip=5., scale_floor=1e-9, log_every=1)

    def test_fresh_initialization_repeatable_not_existing_critic(self):
        a = fresh_critic(self.agent, 11)
        b = fresh_critic(self.agent, 11)
        c = fresh_critic(self.agent, 12)
        self.assertTrue(all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items()))
        self.assertTrue(any(not torch.equal(a.state_dict()[k], v) for k, v in c.state_dict().items()))
        self.assertTrue(any(not torch.equal(a.state_dict()[k], v) for k, v in self.agent.critic.state_dict().items()))

    def test_real_gcn_fit_checkpoint_and_prediction_seal(self):
        actor = copy.deepcopy(self.agent.actor.state_dict())
        legacy = copy.deepcopy(self.agent.critic.state_dict())
        model = fresh_critic(self.agent, 11)
        budget = Budget(0, 60)
        optimizer, report = fit(model, self.agent, self.public, self.targets, self.support,
                                ["train"] * 3, self.settings, budget, lambda row: None)
        self.assertEqual(report["updates"], 3)
        self.assertTrue(report["parameter_changed"])
        self.assertEqual(budget.steps, 0)
        self.assertEqual(len(self.agent.replay_buffer), 0)
        self.assertFalse(self.agent.actor_optimizer.state)
        self.assertFalse(self.agent.critic_optimizer.state)
        self.assertTrue(all(torch.equal(actor[k], v) for k, v in self.agent.actor.state_dict().items()))
        self.assertTrue(all(torch.equal(legacy[k], v) for k, v in self.agent.critic.state_dict().items()))
        seal = seal_predictions(model, self.agent, self.public, self.support, report["target_scale"],
                                10 - self.targets, self.support)
        self.assertEqual(seal["training_selected_constant"], 5)
        self.assertTrue(np.isfinite(seal["advantage_predictions"]).all())
        self.assertFalse(seal["selection_uses_test_labels"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "full.pt"
            checkpoint(path, model, optimizer, updates=3, scale=report["target_scale"], seed=11)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved["updates"], 3)
            self.assertTrue(saved["optimizer"]["state"])
            with self.assertRaises(FileExistsError):
                checkpoint(path, model, optimizer, updates=3, scale=1, seed=11)
            restored = fresh_critic(self.agent, 100)
            restored.load_state_dict(saved["critic"], strict=True)
            nodes, actions = tensor_inputs(self.agent, self.public)
            self.assertTrue(torch.equal(advantage(restored, nodes, actions), advantage(model, nodes, actions)))

    def test_frozen_advantage_exactly_zero(self):
        model = fresh_critic(self.agent, 11)
        got = advantage(model, *tensor_inputs(self.agent, self.public))
        self.assertTrue(torch.equal(got[:, 0], torch.zeros(3)))

    def test_train_scale_is_equal_state_not_action_pooled(self):
        mask = self.support.copy()
        mask[0, 1:] = False
        expected = np.sqrt(np.mean(np.sum(self.targets ** 2 * mask, axis=1) / mask.sum(axis=1)))
        self.assertEqual(target_scale(self.targets, mask, ["train"] * 3, 1e-9), expected)
        with self.assertRaises(ValueError):
            target_scale(self.targets, mask, ["train", "train", "test"], 1e-9)

    def test_nonfinite_and_nonzero_reference_rejected(self):
        for value in (float("nan"), 1.):
            target = self.targets.copy()
            target[0, 0] = value
            with self.assertRaises(ValueError):
                target_scale(target, self.support, ["train"] * 3, 1e-9)

    def test_budget_checked_before_update(self):
        with self.assertRaises(TimeoutError):
            fit(fresh_critic(self.agent, 11), self.agent, self.public, self.targets, self.support,
                ["train"] * 3, self.settings, Budget(0, -1), lambda row: None)


class ProductionContract(unittest.TestCase):
    def test_saved_error_accounting_has_no_refit_or_reweighting(self):
        actual = error_accounting([[0., 2.], [0., 99.]], [[0., 1.], [0., 999.]],
                                  [[True, True], [True, False]], 2., [[0., .5], [0., 999.]])
        self.assertEqual(actual["normalized_mse"], .0625)
        self.assertEqual(actual["zero_advantage_predictor_mse"], .0625)
        self.assertEqual(actual["estimated_mc_label_mean_variance"], .015625)

    def test_exact_limits_and_disjoint_streams(self):
        spec = json.loads(Path("experiments/configs/clean_critic_generalization_20260930.json").read_text())
        base, streams = allocate_streams(spec)
        self.assertEqual(len(streams), 594)
        self.assertEqual(len(set(streams.values())), 594)
        self.assertEqual(min(streams.values()), base)
        self.assertEqual(spec["maximum_environment_steps"], 18 * 52 + 18 * 6 * 8 * (52 + 39 + 26 + 13))
        self.assertEqual(spec["maximum_records"], 18 * 4 * 8 * 6)
        self.assertEqual(spec["fit"]["updates"], 1000)
        train = {v for k, v in streams.items() if "/train" in k}
        test = {v for k, v in streams.items() if "/test" in k}
        self.assertFalse(train & test)

    def test_adverse_clinical_directions_retained_despite_lower_cost(self):
        states, labels, records = [], [], []
        for trajectory in range(2):
            for step in (0, 13, 26, 39):
                name = f"test{trajectory}/t{step}"
                lineage = StateLineage(name, "a", "b", "c", trajectory, step, "prospective_unseen")
                states.append({"lineage": lineage, "support": [True] * 6, "trajectory": trajectory})
                labels.append({"mean_costs": [100, 99, 98, 97, 96, 95]})
                for draw in range(8):
                    for action in range(6):
                        records.append({"state_id": name, "draw": draw, "action_index": action,
                                        "outcome": {"total_cost": 100 - action, "patients_lost": action,
                                                    "patients_completed": 10, "completion_service_level": .5,
                                                    "manufacturing_loss_rate": .1}})
        seal = {"advantage_predictions": [list(range(6))] * 8,
                "critic_actions": [5] * 8, "frozen_actions": [0] * 8,
                "mdl2_actions": [1] * 8, "constant_actions": [5] * 8}
        report = summarize_test(states, labels, seal, records)
        self.assertEqual(report["ranking"]["pairwise_accuracy"], 1)
        self.assertEqual(len(report["states"]), 8)
        for row in report["trajectories"]:
            self.assertEqual(row["mean_contrasts"]["frozen"]["total_cost"], -5)
            self.assertTrue(row["adverse_clinical_mean_vs_frozen"])
            self.assertEqual(row["mean_contrasts"]["constant"]["total_cost"], 0)


if __name__ == "__main__":
    unittest.main()
