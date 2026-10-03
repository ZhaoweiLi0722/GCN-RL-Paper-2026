"""Artificial fixtures and in-memory snapshots only; never fit or step Adam."""

import copy
import io
import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.models.fixed_budget_capacity import FixedBudgetCapacityActor, native_fixed_budget_hours
from src.rl.fixed_budget_capacity_learner import FixedBudgetCapacityLearner
from src.rl.patient_constrained_learner import PatientConstrainedLearner, PatientLossTransition
from src.rl.networks import torch


ROOT = Path(__file__).resolve().parents[1]


def proposal():
    with (ROOT / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json").open() as stream:
        p = json.load(stream)
    with (ROOT / "experiments/configs/patient_constrained_capacity_20261003.json").open() as stream:
        p["patient_constrained"] = json.load(stream)
    return p


def row(condition=0, index=0):
    features = [[.2, .3, .4]] * 4
    return PatientLossTransition.from_cost(state=features, executed_hours=[2.] * 4,
        next_state=features, done=False, world_id=f"artificial-{index}", control_cost=100.,
        settlement_cost=None, reward_divisor=100000., patient_losses=1, condition=condition)


@unittest.skipIf(torch is None, "torch required")
class FixedBudgetCapacityLearnerTests(unittest.TestCase):
    def setUp(self):
        for name in ("torch.optim.Adam.step", "torch.optim.SGD.step",
                     "src.rl.patient_constrained_learner.PatientConstrainedLearner.warmup_step",
                     "src.rl.patient_constrained_learner.PatientConstrainedLearner.training_step"):
            barrier = patch(name, side_effect=AssertionError("no fits or optimizer steps in adapter tests"))
            barrier.start()
            self.addCleanup(barrier.stop)
        load = torch.load
        def memory_only(source, *args, **kwargs):
            if not isinstance(source, io.BytesIO):
                raise AssertionError("research/file checkpoint loads forbidden")
            return load(source, *args, **kwargs)
        barrier = patch("torch.load", side_effect=memory_only)
        barrier.start()
        self.addCleanup(barrier.stop)
        self.p = proposal()
        self.features = np.zeros((4, 3), dtype=np.float32)

    def learner(self, *, cls=FixedBudgetCapacityLearner, before_forward=None):
        return cls.from_proposal(self.p, node_input_dim=3, model_seed=71, replay_seed=91,
            before_forward=before_forward or (lambda *_: None), before_optimizer=lambda *_: None)

    def equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(type(first), type(second))
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.equal(a, b)
        else:
            self.assertEqual(first, second)

    def restore(self, learner, before_forward=None):
        return FixedBudgetCapacityLearner.from_snapshot(learner.snapshot(),
            before_forward=before_forward or (lambda *_: None), before_optimizer=lambda *_: None)

    def test_replacement_preserves_weights_critics_rng_and_rebinds_actor_optimizer(self):
        state = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        hooks = []
        learner = self.learner(before_forward=lambda *args: hooks.append(args))
        parent = self.learner(cls=PatientConstrainedLearner)
        self.equal(state, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
        self.assertEqual(hooks, [])
        self.assertEqual(type(learner.actor), FixedBudgetCapacityActor)
        self.assertEqual(type(learner.target_actor), FixedBudgetCapacityActor)
        for name in learner.module_names:
            self.equal(getattr(learner, name).state_dict(), getattr(parent, name).state_dict())
        for name in learner.optimizer_names:
            self.equal(getattr(learner, name + "_optimizer").state_dict(), getattr(parent, name + "_optimizer").state_dict())
        self.equal(learner.torch_rng_state, parent.torch_rng_state)
        self.equal(learner.replay_rng.bit_generator.state, parent.replay_rng.bit_generator.state)
        self.equal(learner.exploration_rng.bit_generator.state, parent.exploration_rng.bit_generator.state)
        bound = {id(p) for group in learner.actor_optimizer.param_groups for p in group["params"]}
        self.assertEqual(bound, {id(p) for p in learner.actor.parameters()})
        self.assertTrue(all(not p.requires_grad for p in learner.target_actor.parameters()))
        self.assertEqual(learner.counts, parent.counts)

    def test_initial_behavior_is_native_exact_two_and_one_charged_forward_only(self):
        calls = []
        learner = self.learner(before_forward=lambda *args: calls.append(args))
        before = learner.state_dict()
        request, executed = learner.act(self.features)
        self.assertEqual(request.dtype, np.float64)
        self.assertEqual(executed.dtype, np.float64)
        np.testing.assert_array_equal(request, [2.] * 4)
        np.testing.assert_array_equal(executed, request)
        self.assertFalse(np.shares_memory(request, executed))
        self.assertEqual(calls, [("behavior_actor", 1)])
        expected_counts = dict(before["counts"], neural_forward_module_calls=1)
        self.assertEqual(learner.counts, expected_counts)
        after = learner.state_dict()
        for key in set(before) - {"counts"}:
            self.equal(before[key], after[key])

    def test_supplied_noise_is_before_map_and_does_not_consume_rng(self):
        calls = []
        learner = self.learner(before_forward=lambda *args: calls.append(args))
        noise = np.array([.25, -.25, .5, -.5], dtype=np.float64)
        rng = copy.deepcopy(learner.exploration_rng.bit_generator.state)
        expected = learner.actor.project_native(torch.zeros(4), noise=torch.as_tensor(noise, dtype=torch.float32))
        request, executed = learner.act(self.features, explore=True, noise_hours=noise)
        np.testing.assert_array_equal(request, expected)
        np.testing.assert_array_equal(executed, expected)
        self.assertFalse(np.allclose(request, 2 + noise))
        self.equal(rng, learner.exploration_rng.bit_generator.state)
        self.assertEqual(math.fsum(request), 8.)
        self.assertLessEqual(sum(request), 8.)
        self.assertEqual(calls, [("behavior_actor", 1)])
        self.assertEqual(learner.counts["total_optimizer_steps"], 0)

    def test_tensor_behavior_maps_logits_and_keeps_parent_batch_shape(self):
        calls = []
        learner = self.learner(before_forward=lambda *args: calls.append(args))
        features = torch.zeros(2, 4, 3)
        noise = torch.tensor([[.25, -.25, .5, -.5], [-4., 4., -.3, .3]])
        request, executed = learner.act(features, explore=True, noise_hours=noise)
        expected = learner.actor.project_noisy(torch.zeros(2, 4), noise)
        self.equal(request, expected)
        self.equal(executed, expected)
        self.assertEqual(request.dtype, torch.float32)
        self.assertNotEqual(request.data_ptr(), executed.data_ptr())
        self.assertEqual(calls, [("behavior_actor", 2)])

    def test_full_snapshot_restores_metadata_replay_optimizer_moments_and_all_rng(self):
        learner = self.learner()
        for c in range(3):
            learner.add_reference(row(c, c))
        learner.sample_batch(1, warmup=True)
        learner.act(self.features, explore=True)
        learner.python_rng.random()
        local = torch.Generator().set_state(learner.torch_rng_state)
        torch.rand(4, generator=local)
        learner.torch_rng_state = local.get_state()
        # Explicit synthetic optimizer state; no step or fit produced these moments.
        for index, name in enumerate(learner.optimizer_names):
            optimizer = getattr(learner, name + "_optimizer")
            for p in optimizer.param_groups[0]["params"]:
                optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.full_like(p, .01 * (index + 1)),
                                          exp_avg_sq=torch.full_like(p, .1 * (index + 1)))
        before = learner.state_dict()
        global_rng = torch.get_rng_state().clone()
        restored = self.restore(learner)
        self.equal(global_rng, torch.get_rng_state())
        self.equal(before, restored.state_dict())
        self.assertEqual(before["format"], "fixed-budget-capacity-learner-v1")
        self.assertEqual(before["actor_metadata"], learner.actor.metadata())
        self.assertEqual(type(restored.target_actor), FixedBudgetCapacityActor)
        self.assertEqual(restored.sample_batch(0, warmup=True).indices, learner.sample_batch(0, warmup=True).indices)
        for actual, expected in zip(restored.act(self.features, explore=True), learner.act(self.features, explore=True)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(restored.python_rng.random(), learner.python_rng.random())
        self.assertEqual(restored.counts["total_optimizer_steps"], 0)
        bound = {id(p) for group in restored.actor_optimizer.param_groups for p in group["params"]}
        self.assertEqual(bound, {id(p) for p in restored.actor.parameters()})
        with torch.no_grad():
            restored.actor.head[-1].bias.add_(1)
        self.assertFalse(torch.equal(restored.actor.head[-1].bias, learner.actor.head[-1].bias))

    def test_wrong_format_metadata_and_parent_validation_fail_atomically(self):
        learner = self.learner()
        learner.add_reference(row())
        before = learner.state_dict()
        variants = []
        for key, value in (("format", "patient-constrained-learner-v1"), ("actor_metadata", {}), ("unexpected", 1)):
            corrupt = copy.deepcopy(before)
            corrupt[key] = value
            variants.append(corrupt)
        corrupt = copy.deepcopy(before)
        del corrupt["actor_metadata"]
        variants.append(corrupt)
        corrupt = copy.deepcopy(before)
        corrupt["actor_metadata"]["forward_output"] = "raw_hours"
        variants.append(corrupt)
        corrupt = copy.deepcopy(before)
        corrupt["modules"]["actor"]["head.2.bias"].fill_(float("nan"))
        variants.append(corrupt)
        corrupt = copy.deepcopy(before)
        corrupt["reference_replay"] = []
        variants.append(corrupt)
        corrupt = copy.deepcopy(before)
        corrupt["optimizers"]["actor"]["param_groups"][0]["lr"] = .9
        variants.append(corrupt)
        for corrupt in variants:
            with self.assertRaises(ValueError):
                learner.load_state_dict(corrupt)
            self.equal(before, learner.state_dict())
        learner.act(self.features)
        used = learner.state_dict()
        with self.assertRaisesRegex(ValueError, "refunded"):
            learner.load_state_dict(before)
        self.equal(used, learner.state_dict())
        parent = self.learner(cls=PatientConstrainedLearner)
        with self.assertRaisesRegex(ValueError, "legacy"):
            FixedBudgetCapacityLearner.from_snapshot(parent.snapshot(), before_forward=lambda *_: None, before_optimizer=lambda *_: None)

    def test_failure_hook_precharges_and_snapshot_keeps_latch(self):
        def denied(*_):
            raise RuntimeError("artificial forward denial")
        learner = self.learner(before_forward=denied)
        with patch.object(learner.actor, "forward", side_effect=AssertionError("must not forward after denied hook")):
            with self.assertRaisesRegex(RuntimeError, "denial"):
                learner.act(self.features)
        self.assertEqual(learner.counts["neural_forward_module_calls"], 1)
        restored = self.restore(learner)
        self.assertEqual(restored.failure, learner.failure)
        self.assertEqual(restored.counts, learner.counts)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            restored.act(self.features)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            restored.load_state_dict(self.learner().state_dict())

    def test_artificial_complete_warmup_forks_still_select_arms_without_fit(self):
        learner = self.learner()
        for i in range(576):
            learner.add_reference(row(i % 3, i))
        # Artificial completed boundary for restore API coverage; not a training run.
        learner.completed["warmup"] = 256
        learner.counts.update(warmup_steps=256, neural_forward_module_calls=1280,
            reward_critic_optimizer_steps=256, loss_critic_optimizer_steps=256,
            critic_optimizer_steps=512, total_optimizer_steps=512, optimizer_example_presentations=32768)
        before = copy.deepcopy(learner.counts)
        for arm in ("constrained", "cost_only"):
            fork = self.restore(learner)
            fork.set_arm(arm)
            self.assertEqual(fork.arm, arm)
            self.assertEqual(fork.counts, before)
            self.assertEqual(fork.multipliers, [1, 1, 1] if arm == "constrained" else [0, 0, 0])
            self.assertEqual(type(fork.actor), FixedBudgetCapacityActor)
            self.assertEqual(type(fork.target_actor), FixedBudgetCapacityActor)

    def test_native_mapping_random_artificial_stress_without_models_or_updates(self):
        rng = np.random.default_rng(6381)
        vectors = rng.normal(size=(16384, 4)).astype(np.float32)
        vectors *= np.array([.25, 1., 4., 40.], dtype=np.float32)[np.arange(len(vectors)) % 4, None]
        for vector in vectors:
            hours = native_fixed_budget_hours(torch.from_numpy(vector))
            self.assertEqual(hours[-1], 8 - math.fsum(hours[:3]))
            self.assertTrue(all(.5 <= x <= 3.5 for x in hours))
            self.assertLessEqual(math.fsum(hours), 8.)
            self.assertLessEqual(sum(hours), 8.)
            self.assertAlmostEqual(math.fsum(hours), 8., delta=1e-12)


if __name__ == "__main__":
    unittest.main()
