"""Artificial tensors only. Every real Adam.step is patched to raise.

No patient environment, saved scientific model, fit or result directory is used.
"""

import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.models.capacity_ddpg import CapacityActor, CapacityCritic
from src.rl.capacity_ddpg_learner import (
    CapacityDDPGLearner, CapacityLearnerConfig, CapacityTransition, td_targets,
)
from src.rl.networks import torch


@unittest.skipIf(torch is None, "PyTorch unavailable")
class CapacityLearnerArtificialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json"
        cls.proposal = json.loads(path.read_text())
        cls.config = CapacityLearnerConfig.from_proposal(cls.proposal, node_input_dim=3)

    def setUp(self):
        self.step_patch = patch("torch.optim.Adam.step", side_effect=RuntimeError("artificial optimizer barrier"))
        self.blocked_step = self.step_patch.start()
        self.addCleanup(self.step_patch.stop)
        self.events = []
        self.learner = self.make_learner()

    def make_learner(self, config=None, **hooks):
        return CapacityDDPGLearner(config or self.config, model_seed=17, replay_seed=19,
                                  before_forward=hooks.get("before_forward", lambda name, n: self.events.append((name, n))),
                                  before_optimizer=hooks.get("before_optimizer", lambda name, n: self.events.append((name + "_step", n))))

    def row(self, index=0, *, world="artificial-offline", done=False):
        # Explicitly synthetic values, not observations from a simulator.
        state = tuple((0.1 * site, index / 100.0, 0.25) for site in range(4))
        return CapacityTransition.from_cost(state=state, executed_hours=(0.5, 1.0, 1.5, 2.0),
                                            next_state=state, done=done, world_id=world,
                                            control_cost=100.0 + index, settlement_cost=250.0 if done else None,
                                            reward_divisor=self.config.reward_divisor)

    def fill(self, learner=None):
        learner = learner or self.learner
        for i in range(self.config.batch_size):
            learner.add_offline(self.row(i, done=i == 63))

    def mark_artificial_stage(self, mode, learner=None):
        """Synthetic scheduler state only; not evidence any fit occurred."""
        learner = learner or self.learner
        if mode != "bc":
            learner.completed["bc"] = learner.config.bc_updates
        if mode in ("offline", "online"):
            learner.completed["warmup"] = learner.config.critic_warmup_updates
        if mode == "online":
            learner.completed["offline"] = learner.config.offline_ddpg_pairs

    def assert_tree_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_proposal_and_exact_architecture(self):
        c = self.config
        self.assertEqual((c.bc_updates, c.critic_warmup_updates, c.offline_ddpg_pairs), (256, 256, 576))
        self.assertEqual((c.online_offline_rows, c.online_world_rows), (32, 32))
        self.assertEqual(c.gamma, 1)
        self.assertEqual(c.target_polyak_tau, 0.005)
        actor, critic = self.learner.actor, self.learner.critic
        self.assertEqual([(x.linear.in_features, x.linear.out_features) for x in actor.encoder.layers], [(3, 32), (32, 32)])
        self.assertEqual([(x.linear.in_features, x.linear.out_features) for x in critic.encoder.layers], [(4, 32), (32, 32)])
        self.assertEqual([(x.in_features, x.out_features) for x in actor.head if isinstance(x, torch.nn.Linear)], [(32, 32), (32, 1)])
        expected = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]], dtype=torch.float32) / 3
        torch.testing.assert_close(actor.encoder.adjacency, expected)
        self.assertTrue(set(map(id, actor.parameters())).isdisjoint(set(map(id, critic.parameters()))))
        for module in (actor, critic, self.learner.target_actor, self.learner.target_critic):
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in module.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.learner.target_actor.parameters()))
        bad = copy.deepcopy(self.proposal)
        bad["learner"]["reward_shaping"] = True
        with self.assertRaises(ValueError):
            CapacityLearnerConfig.from_proposal(bad, node_input_dim=3)

    def test_raw_shift_clip_radial_and_exploration(self):
        actor = self.learner.actor
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        x = torch.zeros((1, 4, 3), dtype=torch.float32)
        raw, legal = self.learner.act(x)
        torch.testing.assert_close(raw, torch.full((1, 4), 2.0))
        torch.testing.assert_close(legal, raw)
        raw, legal = self.learner.act(x, explore=True, noise_hours=[[9, 9, 9, -9]])
        torch.testing.assert_close(raw, torch.tensor([[4.0, 4.0, 4.0, 0.0]]))
        torch.testing.assert_close(legal, torch.tensor([[8 / 3, 8 / 3, 8 / 3, 0.0]]))
        self.assertEqual(self.learner.counts["neural_forward_module_calls"], 2)
        zeros = actor.project(torch.zeros((1, 4), dtype=torch.float32))
        self.assertEqual(float(zeros.sum()), 0)
        feasible = torch.tensor([[0.0, 0.5, 1.0, 1.5]], requires_grad=True)
        self.assertTrue(torch.equal(actor.project(feasible), feasible))
        actor.project(feasible).sum().backward()
        self.assertTrue(torch.isfinite(feasible.grad).all())

    def test_dtype_shape_and_critic_support(self):
        for bad in (torch.zeros((1, 4, 3), dtype=torch.float64), torch.zeros((4, 3)),
                    torch.full((1, 4, 3), float("nan"))):
            with self.assertRaises(ValueError):
                self.learner.actor(bad)
        x = torch.zeros((1, 4, 3))
        for hours in (torch.full((1, 4), 4.0), torch.tensor([[-1.0, 0, 0, 0]])):
            with self.assertRaises(ValueError):
                self.learner.critic(x, hours)
        captured = []
        handle = self.learner.critic.encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        self.addCleanup(handle.remove)
        hours = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        self.assertEqual(tuple(self.learner.critic(x, hours).shape), (1, 1))
        torch.testing.assert_close(captured[0][..., -1], hours / self.config.model.hours_divisor)

    def test_runner_numpy_and_passed_fixed_adjacency_boundary(self):
        adjacency = self.learner.actor.encoder.adjacency.numpy().copy()
        learner = CapacityDDPGLearner.from_proposal(
            self.proposal, node_input_dim=3, adjacency=adjacency, model_seed=17, replay_seed=19,
            before_forward=lambda *args: None, before_optimizer=lambda *args: None)
        raw, executed = learner.act(np.zeros((4, 3), dtype=np.float32), explore=True,
                                    noise_hours=np.zeros(4, dtype=np.float32))
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(executed.shape, (4,))
        self.assertLessEqual(executed.sum(), 8.00001)
        adjacency.fill(0)
        self.assertGreater(float(learner.actor.encoder.adjacency.sum()), 0)
        with self.assertRaisesRegex(ValueError, "normalized self-loop ring"):
            CapacityActor(self.config.model, adjacency=adjacency)

    def test_terminal_cost_and_no_reward_transform(self):
        final = self.row(done=True)
        self.assertEqual(final.full_cost, 350)
        self.assertEqual(final.reward, -350 / 100000)
        self.learner.add_offline(final)
        with self.assertRaisesRegex(ValueError, "complete settlement"):
            replace(final, settlement_cost=None)
        with self.assertRaisesRegex(ValueError, "only on the final"):
            replace(final, done=False)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.make_learner().add_offline(replace(final, reward=-1.0))
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.make_learner().add_offline(replace(final, executed_hours=(4, 4, 4, 4)))
        rewards = torch.tensor([[-5.0], [-7.0]], requires_grad=True)
        next_q = torch.tensor([[10.0], [99.0]], requires_grad=True)
        target = td_targets(rewards, torch.tensor([[0.0], [1.0]]), next_q, 0.5)
        torch.testing.assert_close(target, torch.tensor([[0.0], [-7.0]]))
        self.assertFalse(target.requires_grad)

    def test_three_plus_two_forwards_and_gradient_isolation_without_steps(self):
        self.fill()
        batch = self.learner.sample_batch()
        loss, targets = self.learner._critic_loss(batch)
        loss.backward()
        self.assertFalse(targets.requires_grad)
        self.assertEqual([e[0] for e in self.events], ["target_actor", "target_critic", "critic"])
        self.assertTrue(any(p.grad is not None for p in self.learner.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in self.learner.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.learner.target_critic.parameters()))
        self.learner.critic_optimizer.zero_grad(set_to_none=True)
        self.learner._actor_loss(batch).backward()
        self.assertEqual(len(self.events), 5)
        self.assertEqual([e[0] for e in self.events[-2:]], ["actor", "actor_loss_critic"])
        self.assertTrue(all(p.grad is None for p in self.learner.critic.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.learner.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.learner.critic.parameters()))
        self.assertEqual(self.blocked_step.call_count, 0)

    def test_polyak_math_warmup_only_critic(self):
        with torch.no_grad():
            for module, value in ((self.learner.actor, 4), (self.learner.critic, 6),
                                  (self.learner.target_actor, 1), (self.learner.target_critic, 2)):
                for p in module.parameters():
                    p.fill_(value)
        self.learner._polyak(actor=False)
        self.assertTrue(all(torch.all(p == 1) for p in self.learner.target_actor.parameters()))
        self.assertAlmostEqual(next(self.learner.target_critic.parameters()).flatten()[0].item(), 2.02, places=6)
        self.learner._polyak()
        self.assertAlmostEqual(next(self.learner.target_actor.parameters()).flatten()[0].item(), 1.015, places=6)

    def test_counters_before_optimizer_failures_no_refund_or_retry(self):
        for mode, method, expected_calls, optimizer in (
                ("bc", "bc_step", 1, "actor"), ("warmup", "critic_warmup_step", 3, "critic"),
                ("offline", "offline_ddpg_step", 3, "critic"), ("online", "online_ddpg_step", 3, "critic")):
            with self.subTest(mode=mode):
                learner = self.make_learner()
                self.fill(learner)
                self.mark_artificial_stage(mode, learner)
                if mode == "online":
                    learner.begin_world("artificial-world", replay_seed=3, exploration_seed=5)
                    learner.add_world(self.row(world="artificial-world"))
                previous = learner.snapshot()
                observed = []
                learner.before_optimizer = lambda name, n: observed.append((name, n, learner.counts[f"{name}_optimizer_steps"]))
                with self.assertRaisesRegex(RuntimeError, "artificial optimizer barrier"):
                    getattr(learner, method)()
                self.assertEqual(observed, [(optimizer, 64, 1)])
                self.assertEqual(learner.counts["neural_forward_module_calls"], expected_calls)
                self.assertEqual(learner.counts["optimizer_example_presentations"], 64)
                consumed = dict(learner.counts)
                with self.assertRaisesRegex(RuntimeError, "no retry"):
                    getattr(learner, method)()
                with self.assertRaisesRegex(RuntimeError, "no retry"):
                    learner.load_state_dict(previous.state_dict())
                self.assertEqual(learner.counts, consumed)
                failed = CapacityDDPGLearner.from_snapshot(learner.snapshot(), before_forward=lambda *a: None,
                                                          before_optimizer=lambda *a: None)
                with self.assertRaisesRegex(RuntimeError, "no retry"):
                    failed.bc_step()

    def test_forward_hook_and_module_failures_consume_before_dispatch(self):
        self.fill()
        seen = []
        def fail_hook(name, n):
            seen.append(self.learner.counts["neural_forward_module_calls"])
            raise RuntimeError("admission failed")
        self.learner.before_forward = fail_hook
        with patch.object(self.learner.actor, "forward") as module:
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                self.learner.bc_step()
            module.assert_not_called()
        self.assertEqual(seen, [1])
        self.assertEqual(self.learner.counts["actor_optimizer_steps"], 0)
        other = self.make_learner()
        self.fill(other)
        with patch.object(other.actor, "forward", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                other.bc_step()
        self.assertEqual(other.counts["neural_forward_module_calls"], 1)
        self.assertIsNotNone(other.failure)

    def test_optimizer_hook_failure_consumes_without_dispatch(self):
        self.fill()
        def fail(name, n):
            self.assertEqual(self.learner.counts["actor_optimizer_steps"], 1)
            raise RuntimeError("optimizer admission failed")
        self.learner.before_optimizer = fail
        with self.assertRaisesRegex(RuntimeError, "optimizer admission failed"):
            self.learner.bc_step()
        self.blocked_step.assert_not_called()
        self.assertEqual(self.learner.counts["optimizer_example_presentations"], 64)

    def test_snapshot_complete_immutable_and_forks_no_shared_state(self):
        self.fill()
        # Artificial Adam moments exercise restore without ever calling Adam.step.
        for optimizer in (self.learner.actor_optimizer, self.learner.critic_optimizer):
            for group in optimizer.param_groups:
                for p in group["params"]:
                    optimizer.state[p] = {"step": torch.tensor(2.0), "exp_avg": torch.full_like(p, 0.2),
                                          "exp_avg_sq": torch.full_like(p, 0.3)}
        seal = self.learner.snapshot()
        with self.assertRaises(FrozenInstanceError):
            seal.payload = b"changed"
        left = CapacityDDPGLearner.from_snapshot(seal, before_forward=lambda *a: None, before_optimizer=lambda *a: None)
        right = CapacityDDPGLearner.from_snapshot(seal, before_forward=lambda *a: None, before_optimizer=lambda *a: None)
        self.assert_tree_equal(left.state_dict(), right.state_dict())
        self.assert_tree_equal(left.state_dict(), self.learner.state_dict())
        for name in ("actor", "critic", "target_actor", "target_critic"):
            self.assertNotEqual(next(getattr(left, name).parameters()).data_ptr(), next(getattr(right, name).parameters()).data_ptr())
        self.assertEqual(left.sample_batch().indices, right.sample_batch().indices)
        self.assertEqual(left.python_rng.random(), right.python_rng.random())
        x = torch.zeros((1, 4, 3))
        self.assert_tree_equal(left.act(x, explore=True), right.act(x, explore=True))
        with torch.no_grad():
            next(left.actor.parameters()).add_(1)
        first = next(left.actor.parameters())
        left.actor_optimizer.state[first]["exp_avg"].add_(1)
        left.add_offline(self.row(99))
        self.assert_tree_equal(right.state_dict()["modules"], seal.state_dict()["modules"])
        self.assertEqual(len(right.offline_replay), 64)
        unpacked = seal.state_dict()
        unpacked["modules"]["actor"]["head.0.weight"].zero_()
        self.assert_tree_equal(self.learner.state_dict(), seal.state_dict())

    def test_replay_exact_mixture_world_ring_and_rng_restore(self):
        learner = self.make_learner(replace(self.config, online_replay_capacity=2))
        self.fill(learner)
        learner.begin_world("artificial-world", replay_seed=23, exploration_seed=29)
        for i in range(3):
            learner.add_world(self.row(i, world="artificial-world"))
        self.assertEqual(learner.world_position, 1)
        self.assertEqual(len(learner.world_replay), 2)
        restored = CapacityDDPGLearner.from_snapshot(learner.snapshot(), before_forward=lambda *a: None, before_optimizer=lambda *a: None)
        batch = learner.sample_batch(online=True)
        self.assertEqual(batch.sources.count("offline"), 32)
        self.assertEqual(batch.sources.count("world"), 32)
        self.assertEqual(batch.indices, restored.sample_batch(online=True).indices)
        self.assertEqual(tuple(batch.states.shape), (64, 4, 3))
        with self.assertRaisesRegex(ValueError, "current world"):
            learner.add_world(self.row(world="another-world"))

    def test_restore_rejects_counter_refund_bad_dtype_and_preserves_hooks(self):
        self.fill()
        old = self.learner.snapshot()
        self.learner.act(torch.zeros((1, 4, 3)))
        with self.assertRaisesRegex(ValueError, "refunded"):
            self.learner.load_state_dict(old.state_dict())
        current = self.learner.state_dict()
        bad = copy.deepcopy(current)
        bad["modules"]["actor"]["head.0.weight"] = bad["modules"]["actor"]["head.0.weight"].double()
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.learner.load_state_dict(bad)
        self.assert_tree_equal(self.learner.state_dict(), current)
        hook = self.learner.before_forward
        self.learner.load_state_dict(current)
        self.assertIs(self.learner.before_forward, hook)

    def test_no_global_rng_mutation_or_seed_coercion(self):
        state = torch.get_rng_state().clone()
        self.make_learner()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "native"):
            CapacityDDPGLearner(self.config, model_seed="17", replay_seed=19,
                               before_forward=lambda *a: None, before_optimizer=lambda *a: None)

    def test_snapshot_rejects_changed_graph_and_optimizer_settings_atomically(self):
        current = self.learner.state_dict()
        for field in ("graph", "optimizer", "completed"):
            bad = copy.deepcopy(current)
            if field == "graph":
                bad["modules"]["actor"]["encoder.adjacency"].zero_()
            elif field == "optimizer":
                bad["actor_optimizer"]["param_groups"][0]["lr"] = float("nan")
            else:
                bad["completed"]["bc"] = -1
            with self.assertRaises(ValueError):
                self.learner.load_state_dict(bad)
            self.assert_tree_equal(self.learner.state_dict(), current)

    def test_constructor_ignores_ambient_double_default_and_restores_it(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            learner = self.make_learner()
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertTrue(all(p.dtype == torch.float32 for p in learner.actor.parameters()))
            self.assert_tree_equal(learner.state_dict()["modules"], self.learner.state_dict()["modules"])
        finally:
            torch.set_default_dtype(original)


if __name__ == "__main__":
    unittest.main()
