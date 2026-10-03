"""Artificial tensor fixtures only; every Adam step is forbidden or a no-op."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.networks import torch
from src.rl.patient_constrained_learner import PatientConstrainedLearner, PatientLossTransition


ROOT = Path(__file__).resolve().parents[1]


def proposal():
    with (ROOT / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json").open() as stream:
        value = json.load(stream)
    with (ROOT / "experiments/configs/patient_constrained_capacity_20261003.json").open() as stream:
        value["patient_constrained"] = json.load(stream)
    return value


def row(condition=0, *, world="synthetic", done=False, losses=0, tail_losses=0, marker=0.25):
    features = [[marker, 0.5, 1.0] for _ in range(4)]
    return PatientLossTransition.from_cost(state=features, executed_hours=[2.] * 4,
        next_state=features, done=done, world_id=world, control_cost=100.,
        settlement_cost=200. if done else None, reward_divisor=100000.,
        patient_losses=losses, settlement_losses=tail_losses if done else None, condition=condition)


@unittest.skipIf(torch is None, "torch required")
class PatientConstrainedLearnerTests(unittest.TestCase):
    def setUp(self):
        # Any test forgetting its explicit no-op is stopped before an update.
        self.barrier = patch("torch.optim.Adam.step", side_effect=RuntimeError("artificial optimizer barrier"))
        self.barrier.start()
        self.addCleanup(self.barrier.stop)
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        self.p = proposal()

    def learner(self, **kwargs):
        return PatientConstrainedLearner.from_proposal(self.p, node_input_dim=3, model_seed=7,
            replay_seed=11, before_forward=kwargs.pop("before_forward", lambda *_: None),
            before_optimizer=kwargs.pop("before_optimizer", lambda *_: None), **kwargs)

    def references(self, learner):
        for index in range(12):
            for epoch in range(48):
                learner.add_reference(row(index % 3, world=f"reference-{index}", done=epoch == 47,
                    losses=epoch % 2, tail_losses=3, marker=float(index + 1)))

    def warm_fixture(self):
        learner = self.learner()
        self.references(learner)
        # This is an explicitly artificial stage boundary, not a fit or a seal from science.
        learner.completed["warmup"] = 256
        learner.counts.update(warmup_steps=256, neural_forward_module_calls=1280,
            reward_critic_optimizer_steps=256, loss_critic_optimizer_steps=256,
            critic_optimizer_steps=512, total_optimizer_steps=512, optimizer_example_presentations=32768)
        return learner

    def fork(self, arm="constrained", **kwargs):
        learner = PatientConstrainedLearner.from_snapshot(self.warm_fixture().snapshot(),
            before_forward=kwargs.get("before_forward", lambda *_: None),
            before_optimizer=kwargs.get("before_optimizer", lambda *_: None))
        learner.set_arm(arm)
        return learner

    def cohort(self, learner, *, losses=1, tail=2, condition=0):
        for epoch in range(48):
            learner.add_training(row(condition, world=f"training-{condition}", done=epoch == 47,
                                     losses=losses, tail_losses=tail))
        return losses * 48 + tail

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(type(first), type(second))
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_initialization_has_no_forward_exact_start_and_independent_critics(self):
        global_states = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        calls = []
        learner = self.learner(before_forward=lambda *args: calls.append(args))
        self.assertEqual(calls, [])
        self.assertTrue(all(value == 0 for value in learner.counts.values()))
        self.assert_nested_equal(global_states, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
        self.assert_nested_equal(learner.actor.state_dict(), learner.target_actor.state_dict())
        self.assert_nested_equal(learner.loss_critic.state_dict(), learner.target_loss_critic.state_dict())
        modules = [learner.actor, learner.critic, learner.loss_critic]
        addresses = [{parameter.data_ptr() for parameter in module.parameters()} for module in modules]
        self.assertTrue(all(not addresses[a] & addresses[b] for a in range(3) for b in range(a)))
        self.assertFalse(torch.equal(learner.critic.head[-1].weight, learner.loss_critic.head[-1].weight))
        for features in (np.zeros((4, 3), dtype=np.float32), np.full((4, 3), 37., dtype=np.float32)):
            raw, executed = learner.act(features, explore=False)
            np.testing.assert_array_equal(raw, [2.] * 4)
            np.testing.assert_array_equal(executed, [2.] * 4)
        self.assertEqual(calls, [("behavior_actor", 1), ("behavior_actor", 1)])
        self.assertEqual(learner.actor_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(learner.critic_optimizer.param_groups[0]["lr"], 3e-4)
        self.assertEqual(learner.loss_critic_optimizer.param_groups[0]["lr"], 3e-4)

    def test_action_clipping_shared_projection_and_supplied_noise(self):
        learner = self.learner()
        raw, executed = learner.act(np.zeros((4, 3), dtype=np.float32), explore=True,
                                   noise_hours=np.array([-10., 10., 10., 10.], dtype=np.float32))
        np.testing.assert_array_equal(raw, [0., 4., 4., 4.])
        np.testing.assert_allclose(executed, [0., 8. / 3, 8. / 3, 8. / 3], rtol=1e-6)
        self.assertEqual(learner.counts["neural_forward_module_calls"], 1)

    def test_loss_transition_tail_sign_serialization_and_corruption(self):
        terminal = row(done=True, losses=2, tail_losses=7)
        self.assertEqual((terminal.full_cost, terminal.reward, terminal.total_losses), (300., -.003, 9))
        self.assertEqual(PatientLossTransition(**json.loads(json.dumps(asdict(terminal)))), terminal)
        self.assertEqual(row(losses=2).total_losses, 2)
        for updates in ({"patient_losses": True}, {"patient_losses": -1}, {"patient_losses": 1.2},
                        {"settlement_losses": None}, {"total_losses": 8}, {"condition": 3}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(terminal, **updates)
        with self.assertRaises(ValueError):
            replace(row(), settlement_losses=0)
        learner = self.learner()
        before = asdict(terminal)
        learner.add_reference(terminal)
        self.assertEqual(asdict(terminal), before)
        with self.assertRaises(ValueError):
            learner.add_reference(replace(terminal, reward=0.))

    def test_reference_and_own_batches_are_condition_stratified_with_replacement(self):
        learner = self.learner()
        for condition in range(3):
            learner.add_reference(row(condition, marker=condition + 1.))
        warm = learner.sample_batch(1, warmup=True)
        self.assertEqual(warm.sources, ("reference",) * 64)
        self.assertEqual(warm.indices, (1,) * 64)
        self.assertEqual(tuple(warm.states.shape), (64, 4, 3))
        trained = self.fork()
        self.cohort(trained)
        batch = trained.sample_batch(0)
        self.assertEqual(batch.sources, ("reference",) * 32 + ("training",) * 32)
        self.assertTrue(all(trained.reference_replay[i].condition == 0 for i in batch.indices[:32]))
        self.assertTrue(all(trained.training_replay[i].condition == 0 for i in batch.indices[32:]))
        self.assertEqual(tuple(batch.states.shape), (64, 4, 3))

    def test_warmup_five_forwards_two_critic_noops_no_actor_gradients(self):
        calls, optimizers = [], []
        learner = self.learner(before_forward=lambda *args: calls.append(args))
        self.references(learner)
        before = copy.deepcopy(learner.actor.state_dict())
        def observe(name, batch):
            optimizers.append((name, batch))
            owners = [key for key, module in (("actor", learner.actor), ("reward_critic", learner.critic),
                                              ("loss_critic", learner.loss_critic))
                      if any(parameter.grad is not None for parameter in module.parameters())]
            self.assertEqual(owners, [name])
        learner.before_optimizer = observe
        with patch("torch.optim.Adam.step", return_value=None) as steps:
            receipt = learner.warmup_step(0)
        self.assertEqual(steps.call_count, 2)
        self.assertEqual([name for name, _ in calls], ["target_actor", "target_reward_critic", "target_loss_critic", "reward_critic", "loss_critic"])
        self.assertEqual(optimizers, [("reward_critic", 64), ("loss_critic", 64)])
        self.assertEqual(learner.counts["optimizer_example_presentations"], 128)
        self.assertEqual(learner.counts["actor_optimizer_steps"], 0)
        self.assertEqual(learner.completed["warmup"], 1)
        self.assertNotIn("actor_loss", receipt)
        self.assert_nested_equal(before, learner.actor.state_dict())
        self.assert_nested_equal(before, learner.target_actor.state_dict())
        self.assertTrue(all(p.grad is None for name in learner.module_names if name.startswith("target")
                            for p in getattr(learner, name).parameters()))

    def test_training_eight_forwards_three_noops_and_gradient_ownership(self):
        calls, optimizers = [], []
        learner = self.fork(before_forward=lambda *args: calls.append(args))
        losses = self.cohort(learner)
        learner.update_multiplier(0, losses, losses - 10)
        before = copy.deepcopy({name: getattr(learner, name).state_dict() for name in ("actor", "critic", "loss_critic")})
        def observe(name, batch):
            optimizers.append((name, batch))
            owners = [key for key, module in (("actor", learner.actor), ("reward_critic", learner.critic),
                                              ("loss_critic", learner.loss_critic))
                      if any(p.grad is not None for p in module.parameters())]
            self.assertEqual(owners, [name])
            self.assertLessEqual(float(torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad) for p in getattr(learner, name if name != "reward_critic" else "critic").parameters()
                if p.grad is not None]))), 10.0001)
        learner.before_optimizer = observe
        previous = dict(learner.counts)
        with patch("torch.optim.Adam.step", return_value=None) as steps:
            receipt = learner.training_step(0)
        self.assertEqual(steps.call_count, 3)
        self.assertEqual(len(calls), 8)
        self.assertEqual([name for name, _ in calls][-3:], ["actor", "actor_reward_critic", "actor_loss_critic"])
        self.assertEqual(optimizers, [("reward_critic", 64), ("loss_critic", 64), ("actor", 64)])
        self.assertEqual(learner.counts["total_optimizer_steps"] - previous["total_optimizer_steps"], 3)
        self.assertEqual(learner.counts["critic_optimizer_steps"] - previous["critic_optimizer_steps"], 2)
        self.assertEqual(learner.counts["optimizer_example_presentations"] - previous["optimizer_example_presentations"], 192)
        self.assertEqual((learner.completed["training"], learner.cohort["pending_updates"]), (1, 47))
        self.assertIn("actor_loss", receipt)
        for name, value in before.items():
            self.assert_nested_equal(value, getattr(learner, name).state_dict())

    def test_terminal_targets_include_tail_without_bootstrap_and_have_opposite_signs(self):
        learner = self.learner()
        learner.add_reference(row(done=True, losses=2, tail_losses=7))
        batch = learner.sample_batch(0, warmup=True)
        _, _, reward_targets, loss_targets = learner._dual_critic_losses(batch)
        torch.testing.assert_close(reward_targets, torch.full((64, 1), -.003))
        torch.testing.assert_close(loss_targets, torch.full((64, 1), 9.))
        self.assertEqual(learner.counts["neural_forward_module_calls"], 5)

    def test_actor_objective_uses_condition_multiplier_and_cost_only_shadow_critic(self):
        for arm, expected in (("constrained", -2. + 1.5 * 3.), ("cost_only", -2.)):
            learner = self.fork(arm)
            self.cohort(learner)
            if arm == "constrained":
                learner.multipliers[0] = 1.5
            for critic, value in ((learner.critic, 2.), (learner.loss_critic, 3.)):
                with torch.no_grad():
                    critic.head[-1].weight.zero_()
                    critic.head[-1].bias.fill_(value)
            loss = learner._constrained_actor_loss(learner.sample_batch(0))
            self.assertAlmostEqual(float(loss.detach()), expected)
            self.assertTrue(all(p.requires_grad for critic in (learner.critic, learner.loss_critic) for p in critic.parameters()))
            if arm == "cost_only":
                names = []
                learner.before_optimizer = lambda name, batch: names.append(name)
                with patch("torch.optim.Adam.step", return_value=None):
                    learner.training_step(0)
                self.assertEqual(names, ["reward_critic", "loss_critic", "actor"])
                self.assertEqual(learner.multipliers, [0., 0., 0.])

    def test_multiplier_sign_caps_and_condition_isolation(self):
        for losses, reference, expected in ((1, 40, 1.1), (100, 0, 20.), (1, 300, 0.)):
            with self.subTest(losses=losses, reference=reference):
                learner = self.fork()
                total = self.cohort(learner, losses=losses)
                receipt = learner.update_multiplier(0, total, reference)
                self.assertAlmostEqual(receipt["after"], expected)
                self.assertEqual(learner.multipliers[1:], [1, 1])
                self.assertEqual(learner.counts["scalar_multiplier_updates"], 1)
                self.assertEqual(receipt["difference"], total - reference)
                with self.assertRaises(ValueError):
                    learner.update_multiplier(0, total, reference)

    def test_schedule_requires_fork_full_warmup_settlement_and_one_multiplier(self):
        with self.assertRaises(ValueError):
            self.warm_fixture().set_arm("constrained")
        learner = self.learner()
        self.references(learner)
        with self.assertRaises(ValueError):
            learner.warmup_step(1)
        for operation in (lambda x: x.training_step(0),
                          lambda x: x.update_multiplier(0, 0, 0),
                          lambda x: x.add_training(row(done=True))):
            with self.assertRaises(ValueError):
                operation(self.fork())
        learner = self.fork()
        self.cohort(learner)
        with self.assertRaises(ValueError):
            learner.training_step(0)
        learner = self.fork("cost_only")
        total = self.cohort(learner)
        with self.assertRaises(ValueError):
            learner.update_multiplier(0, total, total)

    def test_full_snapshot_restores_three_optimizers_replays_rng_and_is_independent(self):
        learner = self.fork()
        total = self.cohort(learner)
        learner.update_multiplier(0, total, total - 5)
        learner.sample_batch(0)
        learner.exploration_rng.normal(size=3)
        learner.python_rng.random()
        local = torch.Generator().set_state(learner.torch_rng_state)
        torch.rand(3, generator=local)
        learner.torch_rng_state = local.get_state()
        # Artificial Adam moments, assigned directly with no optimizer dispatch.
        for index, name in enumerate(learner.optimizer_names):
            optimizer = getattr(learner, name + "_optimizer")
            for parameter in optimizer.param_groups[0]["params"]:
                optimizer.state[parameter] = dict(step=torch.tensor(4.), exp_avg=torch.full_like(parameter, .01 * (index + 1)),
                                                   exp_avg_sq=torch.full_like(parameter, .1 * (index + 1)))
        expected = learner.state_dict()
        restored = PatientConstrainedLearner.from_snapshot(learner.snapshot(), before_forward=lambda *_: None, before_optimizer=lambda *_: None)
        self.assert_nested_equal(expected, restored.state_dict())
        self.assertEqual(learner.sample_batch(0).indices, restored.sample_batch(0).indices)
        np.testing.assert_array_equal(learner.exploration_rng.normal(size=8), restored.exploration_rng.normal(size=8))
        self.assertEqual(learner.python_rng.random(), restored.python_rng.random())
        restored.multipliers[0] = 2.
        self.assertNotEqual(restored.multipliers, learner.multipliers)
        with torch.no_grad():
            restored.loss_critic.head[-1].bias.add_(1.)
        self.assertFalse(torch.equal(restored.loss_critic.head[-1].bias, learner.loss_critic.head[-1].bias))

    def test_identical_warmup_forks_share_start_not_storage_or_rng(self):
        seal = self.warm_fixture().snapshot()
        first = PatientConstrainedLearner.from_snapshot(seal, before_forward=lambda *_: None, before_optimizer=lambda *_: None)
        second = PatientConstrainedLearner.from_snapshot(seal, before_forward=lambda *_: None, before_optimizer=lambda *_: None)
        self.assert_nested_equal(first.state_dict(), second.state_dict())
        first.set_arm("constrained")
        second.set_arm("cost_only")
        for name in first.module_names:
            self.assert_nested_equal(getattr(first, name).state_dict(), getattr(second, name).state_dict())
        self.assertEqual(first.sample_batch(2, warmup=True).indices, second.sample_batch(2, warmup=True).indices)
        self.assertNotEqual(first.multipliers, second.multipliers)

    def test_hook_failure_is_precharged_latched_and_preserved_in_snapshot(self):
        learner = self.learner()
        self.references(learner)
        before = learner.state_dict()
        with self.assertRaisesRegex(RuntimeError, "artificial optimizer barrier"):
            learner.warmup_step(0)
        self.assertEqual(learner.counts["neural_forward_module_calls"], 5)
        self.assertEqual(learner.counts["reward_critic_optimizer_steps"], 1)
        self.assertEqual(learner.counts["critic_optimizer_steps"], 1)
        self.assertEqual(learner.counts["optimizer_example_presentations"], 64)
        self.assertEqual(learner.completed["warmup"], 0)
        counts = dict(learner.counts)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            learner.warmup_step(0)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            learner.load_state_dict(before)
        self.assertEqual(learner.counts, counts)
        restored = PatientConstrainedLearner.from_snapshot(learner.snapshot(), before_forward=lambda *_: None, before_optimizer=lambda *_: None)
        self.assertEqual(restored.failure, learner.failure)
        self.assertEqual(restored.counts, counts)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            restored.act(np.zeros((4, 3), dtype=np.float32), explore=False)

    def test_forward_and_optimizer_hook_failure_debits_precede_dispatch(self):
        def blocked(*_):
            raise RuntimeError("hook denied")
        learner = self.learner(before_forward=blocked)
        with patch.object(learner.actor, "forward", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(RuntimeError, "hook denied"):
                learner.act(np.zeros((4, 3), dtype=np.float32), explore=False)
        self.assertEqual(learner.counts["neural_forward_module_calls"], 1)
        learner = self.learner(before_optimizer=blocked)
        self.references(learner)
        with patch("torch.optim.Adam.step", side_effect=AssertionError("must not execute")) as step:
            with self.assertRaisesRegex(RuntimeError, "hook denied"):
                learner.warmup_step(0)
        self.assertEqual(step.call_count, 0)
        self.assertEqual(learner.counts["total_optimizer_steps"], 1)

    def test_invalid_restore_is_atomic_and_cannot_refund_counters(self):
        learner = self.learner()
        before = learner.state_dict()
        corrupt = copy.deepcopy(before)
        corrupt["modules"]["loss_critic"]["head.2.bias"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            learner.load_state_dict(corrupt)
        self.assert_nested_equal(learner.state_dict(), before)
        corrupt = copy.deepcopy(before)
        corrupt["counts"]["actor_optimizer_steps"] = 1
        with self.assertRaises(ValueError):
            learner.load_state_dict(corrupt)
        self.assert_nested_equal(learner.state_dict(), before)
        learner.act(np.zeros((4, 3), dtype=np.float32), explore=False)
        with self.assertRaisesRegex(ValueError, "refunded"):
            learner.load_state_dict(before)


if __name__ == "__main__":
    unittest.main()
