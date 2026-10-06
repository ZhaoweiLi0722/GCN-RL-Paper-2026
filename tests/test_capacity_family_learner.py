"""Synthetic tensors/episodes only; no environment, saved models or file loading."""

import copy
import math
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.models.gcn import build_normalized_adjacency
from src.rl.capacity_family_learner import FamilyLearner, METHODS, _normal_log_prob


class FamilyLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.events = []
        self.addCleanup(patch.stopall)
        patch("torch.load", side_effect=AssertionError("saved model loading forbidden")).start()

    def learner(self, method="ddpg", *, adjacency=None, **kwargs):
        return FamilyLearner(method, adjacency=np.eye(4, dtype=np.float32) if adjacency is None else adjacency,
                             model_seed=17, sampler_seed=23, feature_dim=3, lr=3e-4,
                             before_forward=lambda name, n: self.events.append(("forward", name, n)),
                             before_optimizer=lambda name, n: self.events.append(("optimizer", name, n)), **kwargs)

    def episode(self, learner, n=4):
        states = np.arange((n + 1) * 12, dtype=np.float32).reshape(n + 1, 4, 3) / 100
        rows = []
        for i in range(n):
            action, metadata = learner.act(states[i], explore=True)
            settlement = 50. if i == n - 1 else None
            full_cost = 100. + 10 * i + (settlement or 0.)
            rows.append(dict(state=states[i], next_state=states[i + 1], action=action,
                             reward=-full_cost / 1e6, full_cost=full_cost, settlement_cost=settlement,
                             control_cost=100. + 10 * i, done=i == n - 1, metadata=metadata))
        return rows

    def equal(self, a, b):
        if isinstance(b, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(b, np.ndarray):
            self.assertEqual(a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b)
        elif isinstance(b, dict):
            self.assertEqual(set(a), set(b))
            for key in b:
                self.equal(a[key], b[key])
        elif isinstance(b, (list, tuple)):
            self.assertEqual(type(a), type(b))
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_shared_mean_initialization_and_self_only_adjacency(self):
        ring = build_normalized_adjacency(4, [(0, 1), (1, 2), (2, 3), (3, 0)]).numpy()
        anchor = self.learner()
        for method in METHODS:
            for adjacency in (ring, np.eye(4, dtype=np.float32)):
                learner = self.learner(method, adjacency=adjacency)
                for name, parameter in anchor.actor.named_parameters():
                    self.equal(dict(learner.actor.named_parameters())[name], parameter)
                for module in learner.modules.values():
                    self.equal(module.encoder.adjacency, torch.from_numpy(adjacency))
                actual = learner.parameter_counts()
                self.assertEqual(actual["actor_mean"], anchor.parameter_counts()["actor_mean"])
                self.assertEqual(actual["stochastic_log_std"], 4 if method in ("sac", "ppo") else 0)
                self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32
                                    for module in learner.modules.values() for p in module.parameters()))
        for bad in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.ones((3, 3))):
            with self.assertRaisesRegex(ValueError, "adjacency"):
                self.learner(adjacency=bad)

    def test_global_rng_and_dtype_unchanged_by_constructor_and_training(self):
        torch_rng, np_rng, py_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            learner = self.learner("sac")
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(dtype)
        learner.observe_episode(self.episode(learner))
        learner.fit_episode(updates=1, batch_size=4)
        self.equal(torch.get_rng_state(), torch_rng)
        self.equal(np.random.get_state(), np_rng)
        self.equal(random.getstate(), py_rng)

    def test_deterministic_evaluation_no_rng_optimizer_or_value_calls(self):
        for method in METHODS:
            learner = self.learner(method)
            before = learner.state_dict()
            self.events.clear()
            action, metadata = learner.act(np.ones((4, 3), dtype=np.float32), explore=False)
            second, _ = learner.act(np.ones((4, 3), dtype=np.float32), explore=False)
            self.equal(action, second)
            self.assertTrue(np.all(action >= 0) and np.all(action <= 4) and action.sum() <= 8 + 2e-6)
            self.equal(before["sampler_rng"], learner.state_dict()["sampler_rng"])
            self.equal(before["noise_rng"], learner.state_dict()["noise_rng"])
            self.equal(before["modules"], learner.state_dict()["modules"])
            self.equal(before["optimizers"], learner.state_dict()["optimizers"])
            self.assertEqual(self.events, [("forward", "actor", 1)] * 2)
            self.assertEqual(metadata["learner_config"]["critic_action"], "projected_request_hours")
            self.assertEqual(metadata["parameter_counts"], learner.parameter_counts())

    def test_sac_density_matches_box_proposal_not_projected_density(self):
        learner = self.learner("sac")
        for p in learner.actor.parameters():
            with torch.no_grad():
                p.zero_()
        with torch.no_grad():
            learner.actor.head[-1].bias.fill_(1.)
            learner.actor.log_std.fill_(-2.)
        action, metadata = learner.act(np.zeros((4, 3), dtype=np.float32), explore=True)
        latent, mean, log_std = [np.asarray(metadata[k], dtype=np.float64) for k in ("latent", "mean", "log_std")]
        proposal = 2 * (1 + np.tanh(latent))
        gaussian_lp = np.sum(-0.5 * ((latent - mean) / np.exp(log_std))**2 - log_std - 0.5 * np.log(2 * np.pi))
        expected = gaussian_lp - np.log(2 * (1 - np.tanh(latent)**2)).sum()
        self.assertAlmostEqual(metadata["log_prob"], expected, places=5)
        np.testing.assert_allclose(metadata["proposal_hours"], proposal, atol=1e-6)
        self.assertGreater(proposal.sum(), 8)
        np.testing.assert_allclose(action, proposal * (8 / proposal.sum()), atol=1e-6)
        self.assertEqual(metadata["likelihood"], "box_proposal")

    def test_ppo_preserves_latents_and_old_policy_ratio_one(self):
        learner = self.learner("ppo")
        rows = self.episode(learner)
        self.assertFalse(np.array_equal(rows[0]["metadata"]["latent"], rows[0]["action"]))
        learner.observe_episode(rows)
        self.events.clear()
        receipts = learner.fit_episode(updates=1, batch_size=64)
        self.assertAlmostEqual(receipts[0]["ratio_mean"], 1., places=5)
        self.assertEqual(learner.replay, [])
        self.assertIsNone(learner.pending)
        self.assertEqual(self.events, [("forward", "actor", 64), ("optimizer", "actor", 64),
                                       ("forward", "value", 64), ("optimizer", "value", 64)])
        with self.assertRaisesRegex(ValueError, "newly observed"):
            learner.fit_episode()
        stale = self.learner("ppo")
        stale.observe_episode(self.episode(stale))
        stale.fit_episode(updates=1, batch_size=4)
        with self.assertRaisesRegex(ValueError, "current-policy"):
            stale.observe_episode(rows)

    def test_complete_32_updates_match_forecast_and_budget_caps(self):
        expected = {"ddpg": (208, 64), "td3": (240, 80), "sac": (304, 96), "ppo": (160, 64)}
        for method in METHODS:
            with self.subTest(method=method):
                learner = self.learner(method)
                forecast = learner.accounting()
                initial = copy.deepcopy(learner.actor.state_dict())
                learner.observe_episode(self.episode(learner, 48))
                receipts = learner.fit_episode()
                self.assertEqual(len(receipts), 32)
                self.assertEqual(learner.counts["forward_calls"], forecast["total_forward_calls"])
                self.assertEqual(learner.counts["forward_examples"], forecast["forward_examples"])
                self.assertEqual(learner.counts["optimizer_steps"], forecast["optimizer_steps"])
                self.assertEqual(learner.counts["optimizer_examples"], forecast["optimizer_examples"])
                self.assertEqual((sum(learner.counts["forward_calls"].values()),
                                  sum(learner.counts["optimizer_steps"].values())), expected[method])
                self.assertLessEqual(sum(learner.counts["forward_calls"].values()), 400)
                self.assertLessEqual(sum(learner.counts["optimizer_steps"].values()), 96)
                self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in learner.actor.state_dict().items()))
                for receipt in receipts:
                    self.assertEqual(len(receipt["indices"]), 64)
                    for loss in receipt["losses"].values():
                        self.assertTrue(math.isfinite(loss["loss"]) and math.isfinite(loss["grad_norm"]))
                        self.assertTrue(loss["optimizer_completed"])
                self.assertTrue(all(1 <= event[2] <= 64 for event in self.events))

    def test_td3_delays_actor_and_target_updates(self):
        learner = self.learner("td3")
        target = copy.deepcopy(learner.modules["target_critic1"].state_dict())
        actor = copy.deepcopy(learner.actor.state_dict())
        learner.observe_episode(self.episode(learner))
        receipt = learner.fit_episode(updates=1, batch_size=4)[0]
        self.assertNotIn("actor", receipt["losses"])
        self.equal(actor, learner.actor.state_dict())
        self.equal(target, learner.modules["target_critic1"].state_dict())
        self.assertEqual(learner.accounting(updates=1, act_calls=0)["optimizer_steps"]["actor"], 1)
        learner.observe_episode(self.episode(learner))
        receipt = learner.fit_episode(updates=1, batch_size=4)[0]
        self.assertIn("actor", receipt["losses"])
        self.assertTrue(any(not torch.equal(value, target[key])
                            for key, value in learner.modules["target_critic1"].state_dict().items()))

    def test_terminal_targets_do_not_bootstrap_and_ppo_gae_uses_settlement(self):
        for method in METHODS:
            learner = self.learner(method)
            rows = self.episode(learner, 1)
            learner.observe_episode(rows)
            receipt = learner.fit_episode(updates=1, batch_size=4)[0]
            self.assertAlmostEqual(receipt["target_mean"], rows[0]["reward"], places=7)

    def test_invalid_rows_rejected_before_any_fit_forward_or_optimizer(self):
        mutations = [
            lambda rows: rows[-1].update(settlement_cost=None),
            lambda rows: rows[-1].update(settlement_cost=-1),
            lambda rows: rows[-1].update(settlement_cost=math.inf),
            lambda rows: rows[-1].update(settlement_cost=1e9),
            lambda rows: rows[0].update(settlement_cost=0),
            lambda rows: rows[-1].update(done=False),
            lambda rows: rows[0].update(done=1),
            lambda rows: rows[0].update(reward=math.nan),
            lambda rows: rows[0].update(reward=-99),
            lambda rows: rows[0].update(full_cost=-1),
            lambda rows: rows[0].update(control_cost=1e9),
            lambda rows: rows[0].update(action=np.full(4, 4, dtype=np.float32)),
            lambda rows: rows[0].update(executed_hours=np.zeros(4, dtype=np.float32)),
            lambda rows: rows[0].update(next_state=np.zeros((4, 3), dtype=np.float32)),
            lambda rows: rows[0].update(state=np.full((4, 3), np.inf, dtype=np.float32)),
        ]
        for mutate in mutations:
            learner = self.learner()
            rows = self.episode(learner)
            before = copy.deepcopy(learner.counts)
            mutate(rows)
            with self.assertRaises(ValueError):
                learner.observe_episode(rows)
            self.assertEqual(before, learner.counts)
            self.assertEqual(learner.replay, [])
        for key, value in (("latent", [99.] * 4), ("log_prob", 500.), ("value", math.nan),
                           ("policy_version", 1), ("explore", False), ("state_sha256", "bad")):
            learner = self.learner("ppo")
            rows = self.episode(learner)
            rows[0]["metadata"][key] = value
            with self.assertRaises(ValueError):
                learner.observe_episode(rows)

    def test_cost_fields_in_metadata_and_executed_alias(self):
        learner = self.learner()
        rows = self.episode(learner)
        for row in rows:
            row["executed_hours"] = row.pop("action")
            row["metadata"].update(full_cost=row.pop("full_cost"), settlement_cost=row.pop("settlement_cost"))
        learner.observe_episode(rows)
        self.assertEqual(len(learner.replay), len(rows))

    def test_replay_bounded_and_minibatches_sample_with_replacement(self):
        learner = self.learner(replay_capacity=64)
        for _ in range(2):
            learner.observe_episode(self.episode(learner, 48))
            receipt = learner.fit_episode(updates=1, batch_size=64)[0]
            self.assertLess(len(set(receipt["indices"])), 64)
        self.assertEqual(len(learner.replay), 64)
        self.assertEqual(receipt["source_rows"], 64)

    def test_exact_state_restore_rng_optimizers_pending_and_continuation(self):
        for method in METHODS:
            with self.subTest(method=method):
                learner = self.learner(method)
                learner.observe_episode(self.episode(learner))
                learner.fit_episode(updates=2, batch_size=4)
                learner.observe_episode(self.episode(learner))
                snapshot = learner.state_dict()
                restored = self.learner(method)
                restored.load_state_dict(snapshot)
                self.equal(restored.state_dict(), snapshot)
                receipts = learner.fit_episode(updates=3, batch_size=8)
                self.equal(receipts, restored.fit_episode(updates=3, batch_size=8))
                self.equal(learner.state_dict(), restored.state_dict())
                self.equal(learner.act(np.ones((4, 3), dtype=np.float32), explore=True),
                           restored.act(np.ones((4, 3), dtype=np.float32), explore=True))
                self.equal(snapshot["modules"]["actor"]["encoder.adjacency"], torch.eye(4))
                with self.assertRaisesRegex(ValueError, "unused"):
                    restored.load_state_dict(snapshot)

    def test_restore_rejects_corrupt_graph_optimizer_and_counters_atomically(self):
        learner = self.learner()
        learner.observe_episode(self.episode(learner))
        learner.fit_episode(updates=1, batch_size=4)
        snapshot = learner.state_dict()
        for field in ("adjacency", "dtype", "optimizer", "moment", "counts"):
            fresh = self.learner()
            before = fresh.state_dict()
            bad = copy.deepcopy(snapshot)
            if field == "adjacency":
                bad["modules"]["actor"]["encoder.adjacency"].fill_(0.25)
            elif field == "dtype":
                bad["modules"]["actor"]["head.0.weight"] = bad["modules"]["actor"]["head.0.weight"].double()
            elif field == "optimizer":
                bad["optimizers"]["actor"]["param_groups"][0]["lr"] = 0.1
            elif field == "moment":
                next(iter(bad["optimizers"]["actor"]["state"].values()))["exp_avg"].fill_(math.nan)
            else:
                bad["counts"]["optimizer_steps"]["critic"] = 999
            with self.assertRaises(ValueError):
                fresh.load_state_dict(bad)
            self.equal(before, fresh.state_dict())

    def test_budget_hook_runs_before_neural_dispatch_and_failure_latches(self):
        learner = self.learner()
        before = learner.state_dict()
        learner.actor.forward = Mock(side_effect=AssertionError("must not forward"))
        def deny(name, n):
            self.assertEqual(learner.counts["forward_calls"][name], 1)
            raise RuntimeError("denied forward")
        learner.before_forward = deny
        with self.assertRaisesRegex(RuntimeError, "denied forward"):
            learner.act(np.zeros((4, 3), dtype=np.float32))
        learner.actor.forward.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "latched"):
            learner.load_state_dict(before)

    def test_optimizer_hook_before_step_receipts_and_failed_twin_restore(self):
        learner = self.learner("td3")
        learner.observe_episode(self.episode(learner))
        optimizer = learner.optimizers["critic2"]
        optimizer.step = Mock(side_effect=AssertionError("second step forbidden"))
        def deny_second(name, n):
            self.assertEqual(name, "critic")
            if learner.counts["optimizer_attempts"][name] == 2:
                raise RuntimeError("denied second critic")
        learner.before_optimizer = deny_second
        with self.assertRaisesRegex(RuntimeError, "denied second critic"):
            learner.fit_episode(updates=1, batch_size=4)
        optimizer.step.assert_not_called()
        self.assertEqual(learner.counts["optimizer_attempts"]["critic"], 2)
        self.assertEqual(learner.counts["optimizer_steps"]["critic"], 1)
        restored = self.learner("td3")
        restored.load_state_dict(learner.state_dict())
        self.equal(learner.state_dict(), restored.state_dict())
        with self.assertRaisesRegex(RuntimeError, "latched"):
            restored.fit_episode()

    def test_hooks_precede_all_nested_networks_and_projected_critic_actions(self):
        learner = self.learner("sac")
        rows = self.episode(learner)
        learner.observe_episode(rows)
        last = []
        handles = []
        def admit(name, n):
            last[:] = [name, n]
        learner.before_forward = admit
        for name, module in learner.modules.items():
            category = "actor" if "actor" in name else "critic"
            def check(mod, args, category=category):
                self.assertEqual(last, [category, len(args[0])])
                last.clear()
                if category == "critic":
                    self.assertTrue((args[1] >= 0).all() and (args[1] <= 4).all())
                    self.assertTrue((args[1].sum(-1) <= 8 + 4e-6).all())
            handles.append(module.register_forward_pre_hook(check))
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        learner.fit_episode(updates=1, batch_size=4)


if __name__ == "__main__":
    unittest.main()
