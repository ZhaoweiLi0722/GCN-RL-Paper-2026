"""Bounded numerical unit tests, not environment or performance experiments."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.check_prospective_learner import CONFIG, fixture, kernel, update_rows
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import KernelSettings, ProspectiveDDPGKernel, deployed_request, state_digest


@unittest.skipIf(torch is None, "torch unavailable")
class ProspectiveLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads(CONFIG.read_text())
        cls.binding, cls.settings, cls.windows = fixture(cls.config)

    def learner(self, index=0, mode="online", **settings):
        return kernel(self.config, self.binding, replace(self.settings, **settings), index, mode)

    def loaded(self, **settings):
        learner = self.learner(**settings)
        learner.add_windows(self.windows[:6])
        return learner

    def test_explicit_opt_in_and_valid_settings_required(self):
        prototype = self.learner().agent
        with self.assertRaisesRegex(ValueError, "explicitly"):
            ProspectiveDDPGKernel(prototype, self.settings, mode="online")
        for values in ({"tau": float("nan")}, {"tau": 1.1}, {"actor_lr": 0.},
                       {"batch_size": True}, {"batch_size": 9}, {"max_updates": 0},
                       {"differentiate_gate_proposal": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.settings, **values)

    def test_frozen_insufficient_replay_and_cap_refuse_without_mutation(self):
        frozen = self.learner(mode="frozen")
        frozen.add_windows(self.windows)
        insufficient = self.learner()
        capped = self.loaded(max_updates=1)
        capped.update()
        for learner in (frozen, insufficient, capped):
            before = state_digest(learner.state_dict())
            with self.assertRaises(ValueError):
                learner.update()
            self.assertEqual(before, state_digest(learner.state_dict()))

    def test_real_steps_all_architectures_fixed_parts_and_detached_targets(self):
        for index in range(3):
            learner = self.learner(index)
            learner.add_windows(self.windows[:6])
            before = learner.state_dict()["modules"]
            result = learner.update()
            after = learner.state_dict()["modules"]
            self.assertGreater(result["actor_grad_norm"], 0)
            self.assertGreater(result["critic_grad_norm"], 0)
            self.assertTrue(result["bootstrap_detached"])
            self.assertTrue(result["critic_fixed_during_actor"])
            for prefix in ("actor.", "critic.", "target_actor.", "target_critic."):
                self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix)))
            for key in before:
                if key.startswith(("gate.", "target_gate.", "residual_scales")):
                    self.assertTrue(torch.equal(before[key], after[key]))
            for name, parameter in learner.agent.named_parameters():
                self.assertIsNone(parameter.grad)
                self.assertEqual(parameter.requires_grad, name.startswith(("actor.", "critic.")))

    def test_inference_and_differentiable_deployed_requests_agree(self):
        for index in range(3):
            learner = self.learner(index)
            view = learner._prepare(self.windows).current_actor
            expected = learner.agent.policy_tensors(view)[2]
            for differentiate in (True, False):
                actual = deployed_request(learner.agent, view, differentiate_gate_proposal=differentiate)
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(actual.requires_grad)

    def test_actor_gradient_matches_deployed_nonconstant_gate_finite_difference(self):
        n_actions = self.binding.replay.action_dim

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(.2))

            def forward(self, view):
                return self.value.expand(view.flat.shape[0], n_actions)

        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.))

            def forward(self, view):
                return self.weight * view.context[:, -n_actions:-n_actions + 1]

        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(4.))

            def forward(self, view):
                return torch.sigmoid(self.weight * view.context[:, -n_actions:])

        prototype = self.learner().agent
        prototype.actor, prototype.critic, prototype.gate = Actor(), Critic(), Gate()
        prototype.target_actor = copy.deepcopy(prototype.actor)
        prototype.target_critic = copy.deepcopy(prototype.critic)
        prototype.target_gate = copy.deepcopy(prototype.gate)
        for differentiate in (True, False):
            learner = ProspectiveDDPGKernel(prototype, replace(self.settings, differentiate_gate_proposal=differentiate),
                                             enabled=True, mode="online")
            learner.agent.critic.requires_grad_(False)
            batch = learner._prepare(self.windows[:1])
            loss = learner._actor_loss(batch)
            loss.backward()
            sigmoid = 1 / (1 + np.exp(-.08))
            expected = -.2 * (sigmoid + (.08 * sigmoid * (1 - sigmoid) if differentiate else 0))
            self.assertAlmostEqual(loss.item(), -2 * .02 * sigmoid, places=6)
            self.assertAlmostEqual(learner.agent.actor.value.grad.item(), expected, places=6)
            self.assertIsNone(learner.agent.gate.weight.grad)
            self.assertIsNone(learner.agent.critic.weight.grad)
            if differentiate:
                values = []
                for delta in (-.001, .001):
                    with torch.no_grad():
                        learner.agent.actor.value.fill_(.2 + delta)
                    values.append(learner._actor_loss(batch).item())
                self.assertAlmostEqual((values[1] - values[0]) / .002, expected, places=5)

    def test_polyak_zero_partial_and_full_match_independent_formula(self):
        for tau in (0., .05, 1.):
            learner = self.loaded(tau=tau)
            before = copy.deepcopy(learner.agent.state_dict())
            learner.update()
            after = learner.agent.state_dict()
            for source, target in (("actor.", "target_actor."), ("critic.", "target_critic.")):
                for key in before:
                    if key.startswith(target):
                        expected = before[key] * (1 - tau)
                        expected.add_(after[source + key[len(target):]], alpha=tau)
                        self.assertTrue(torch.equal(after[key], expected))

    def test_terminal_and_truncation_targets_use_existing_contract(self):
        learner = self.loaded()
        batch = learner._prepare([self.windows[5], self.windows[11]])
        _, next_q, targets = learner.agent.replay_forward(batch)
        self.assertFalse(next_q.requires_grad)
        self.assertFalse(targets.requires_grad)
        self.assertEqual(targets[0].item(), batch.rewards[0].item())
        self.assertTrue(torch.equal(targets[1], batch.rewards[1] + next_q[1]))

    def test_invalid_replay_is_atomic_and_ring_wrap_preserves_lineage(self):
        learner = self.loaded()
        before = state_digest(learner.state_dict())
        bad = replace(self.windows[0][0], action=(2.,) * self.binding.replay.action_dim)
        foreign = replace(self.windows[0][0], semantics=replace(self.binding.replay, gamma=.5))
        for batch in ([self.windows[6], (bad,)], [(foreign,)], [[{}]], [],
                      [[self.windows[0][0], self.windows[1][0]]]):
            with self.assertRaises((ValueError, TypeError)):
                learner.add_windows(batch)
            self.assertEqual(before, state_digest(learner.state_dict()))
        learner.add_windows(self.windows[6:])
        self.assertEqual(learner.position, 4)
        self.assertEqual([w[0].state_token for w in learner.windows],
                         [self.windows[i][0].state_token for i in (8, 9, 10, 11, 4, 5, 6, 7)])

    def test_checkpoint_restores_exact_updates_rng_adam_and_wrapped_replay(self):
        for index in range(3):
            learner = self.learner(index)
            learner.add_windows(self.windows[:6])
            update_rows(learner, 3)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "state.pt"
                learner.save(checkpoint)
                fresh = self.learner(index)
                fresh.load(checkpoint)
                self.assertEqual(state_digest(fresh.state_dict()), state_digest(learner.state_dict()))
                learner.add_windows(self.windows[6:])
                fresh.add_windows(self.windows[6:])
                self.assertEqual(update_rows(learner, 3), update_rows(fresh, 3))
                self.assertEqual(state_digest(learner.state_dict()), state_digest(fresh.state_dict()))

    def test_checkpoint_rejects_corrupt_metadata_modules_moments_replay_without_mutation(self):
        learner = self.loaded()
        learner.update()
        original = learner.state_dict()
        mutations = []
        bad = copy.deepcopy(original)
        bad["extra"] = 1
        mutations.append(bad)
        for path, value in ((["manifest", "settings", "actor_lr"], .8),
                            (["manifest", "contract", "replay", "gamma"], .5),
                            (["manifest_sha256"], "bad"), (["total_updates"], True),
                            (["position"], 0), (["rng", "bit_generator"], "foreign"),
                            (["windows", 0, 0, "semantics", "reward_scale"], .001)):
            bad = copy.deepcopy(original)
            cursor = bad
            for part in path[:-1]:
                cursor = cursor[part]
            cursor[path[-1]] = value
            mutations.append(bad)
        for group in ("actor_optimizer", "critic_optimizer"):
            for change in ("missing", "step", "shape", "negative", "nan", "lr"):
                bad = copy.deepcopy(original)
                moments = bad[group]["state"][0]
                if change == "missing":
                    del bad[group]["state"][0]
                elif change == "step":
                    moments["step"] += 1
                elif change == "shape":
                    moments["exp_avg"] = torch.zeros(100)
                elif change == "negative":
                    moments["exp_avg_sq"].fill_(-1)
                elif change == "nan":
                    moments["exp_avg"].fill_(float("nan"))
                else:
                    bad[group]["param_groups"][0]["lr"] = .8
                mutations.append(bad)
        for key in ("residual_scales", next(k for k in original["modules"] if k.startswith("gate.")),
                    next(k for k in original["modules"] if k.startswith("target_gate."))):
            bad = copy.deepcopy(original)
            bad["modules"][key] += .1
            mutations.append(bad)
        for bad in mutations:
            with self.assertRaises((ValueError, TypeError, KeyError)):
                learner.load_state_dict(bad)
            self.assertEqual(state_digest(original), state_digest(learner.state_dict()))

    def test_different_initial_prototype_and_frozen_mode_cannot_import_online_state(self):
        learner = self.loaded()
        learner.update()
        for target in (self.learner(1), self.learner(2), self.learner(mode="frozen"), self.learner(tau=.9)):
            before = state_digest(target.state_dict())
            with self.assertRaisesRegex(ValueError, "manifest"):
                target.load_state_dict(learner.state_dict())
            self.assertEqual(before, state_digest(target.state_dict()))

    def test_zero_update_and_frozen_checkpoints_cannot_disguise_changed_parameters(self):
        for mode in ("online", "frozen"):
            learner = self.learner(mode=mode)
            original = learner.state_dict()
            bad = copy.deepcopy(original)
            key = next(k for k in bad["modules"] if k.startswith("actor."))
            bad["modules"][key] += .1
            with self.assertRaisesRegex(ValueError, "initial prototype"):
                learner.load_state_dict(bad)
            self.assertEqual(state_digest(original), state_digest(learner.state_dict()))

    def test_failed_actor_step_rolls_back_already_advanced_critic_and_rng(self):
        learner = self.loaded()
        learner.update()
        before = learner.state_dict()
        with patch.object(learner.actor_optimizer, "step", side_effect=RuntimeError("invented step failure")):
            with self.assertRaisesRegex(RuntimeError, "invented"):
                learner.update()
        self.assertEqual(state_digest(before), state_digest(learner.state_dict()))
        reference = self.loaded()
        reference.load_state_dict(before)
        self.assertEqual(learner.update(), reference.update())

    def test_nonfinite_loss_or_post_step_parameter_rolls_back(self):
        learner = self.loaded()
        before = state_digest(learner.state_dict())
        with patch.object(learner, "_actor_loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                learner.update()
        self.assertEqual(before, state_digest(learner.state_dict()))

        def corrupt_parameter():
            with torch.no_grad():
                next(learner.agent.actor.parameters()).fill_(float("nan"))

        with patch.object(learner.actor_optimizer, "step", side_effect=corrupt_parameter):
            with self.assertRaises(ValueError):
                learner.update()
        self.assertEqual(before, state_digest(learner.state_dict()))

    def test_checkpoint_no_overwrite_and_checksum_corruption(self):
        learner = self.loaded()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            learner.save(path)
            saved_bytes = path.read_bytes()
            with self.assertRaises(FileExistsError):
                learner.save(path)
            self.assertEqual(path.read_bytes(), saved_bytes)
            envelope = torch.load(path, weights_only=True)
            envelope["state"]["position"] = 0
            corrupt = Path(directory) / "corrupt.pt"
            torch.save(envelope, corrupt)
            before = state_digest(learner.state_dict())
            with self.assertRaisesRegex(ValueError, "checksum"):
                learner.load(corrupt)
            self.assertEqual(before, state_digest(learner.state_dict()))

    def test_nonfinite_actor_gradient_rolls_back_before_optimizer_step(self):
        learner = self.loaded()
        before = state_digest(learner.state_dict())
        parameter = next(learner.agent.actor.parameters())
        hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaisesRegex(ValueError, "nonfinite gradient"):
                learner.update()
        finally:
            hook.remove()
        self.assertEqual(before, state_digest(learner.state_dict()))

    def test_global_rng_and_original_prototype_are_untouched(self):
        torch_before = torch.get_rng_state().clone()
        numpy_before, python_before = np.random.get_state(), random.getstate()
        prototype = self.learner().agent
        original = prototype.weights_digest()
        learner = ProspectiveDDPGKernel(prototype, self.settings, enabled=True, mode="online")
        learner.add_windows(self.windows)
        learner.update()
        self.assertEqual(original, prototype.weights_digest())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        for before, after in zip(numpy_before, np.random.get_state()):
            np.testing.assert_equal(before, after)

    def test_fixture_rejects_permission_scope_and_hash_changes(self):
        for key, value in (("environment_steps", 1), ("performance_evaluation", True),
                           ("scientific_launch_authorized", True), ("fixture_seed", 42),
                           ("count_source_sha256", "bad"), ("schema_source_sha256", "bad")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                fixture(self.config | {key: value})


if __name__ == "__main__":
    unittest.main()
