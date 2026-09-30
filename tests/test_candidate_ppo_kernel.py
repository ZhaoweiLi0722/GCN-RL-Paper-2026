"""Bounded invented-tensor PPO updates only; no patient simulator or research fit."""

import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.candidate_ppo_kernel import CandidatePPOKernel, CandidatePPOSettings
from src.rl.candidate_rollout import evaluate_candidate_policy, prepare_candidate_segment, reevaluate_candidate_decision
from src.rl.prospective_adapter import _decode_actor_state
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.validated_returns import OneStepRecord
from tests.test_candidate_policy_rollout import candidates, contract, observation, policy


def settings(**changes):
    values = dict(learning_rate=.001, clip_ratio=.2, value_loss_coef=.5, entropy_coef=.01,
                  max_grad_norm=.5, gae_lambda=.8, normalize_advantages=True,
                  epochs=2, batch_size=2, max_rollout_steps=8, max_updates=3, max_optimizer_steps=24)
    return CandidatePPOSettings(**(values | changes))


def kernel(*, mode="online", architecture="graph", message_mode="physical", dtype=None, **changes):
    prototype = policy(architecture, message_mode)
    if dtype is not None:
        prototype = prototype.to(dtype=dtype)
    return CandidatePPOKernel(prototype, contract(), settings(**changes), enabled=True, mode=mode,
                              sampling_seed=119, shuffle_seed=121)


def invented_segment(learner, *, start=0, length=3, trajectory="invented", terminal=True):
    decisions = [learner.decide(observation(i, learner.dtype), candidates(i))
                 for i in range(start, start + length)]
    end = evaluate_candidate_policy(learner.policy, observation(start + length, learner.dtype),
                                    candidates(start + length), learner.contract)
    evaluations = [d.evaluation for d in decisions] + [end]
    records = tuple(OneStepRecord(
        learner.contract.replay, "invented-source", "trajectory", trajectory, i,
        evaluations[i].candidates.state_token, evaluations[i + 1].candidates.state_token,
        evaluations[i].actor_state, decisions[i].choice.submitted_request, float(-10 - i),
        evaluations[i + 1].actor_state, terminal and i == length - 1,
        not terminal and i == length - 1) for i in range(length))
    return prepare_candidate_segment(decisions, records, learner.contract,
                                     behavior_sha256=learner.policy.snapshot_sha256(),
                                     bootstrap=None if terminal else end,
                                     gae_lambda=learner.settings.gae_lambda,
                                     max_steps=learner.settings.max_rollout_steps)


@unittest.skipIf(torch is None, "torch unavailable")
class CandidatePPOKernelTests(unittest.TestCase):
    def digest(self, learner):
        return state_digest(learner.state_dict())

    def ready(self, **changes):
        learner = kernel(**changes)
        learner.add_segment(invented_segment(learner))
        return learner

    def test_explicit_opt_in_contract_mode_precision_and_seeds(self):
        args = dict(enabled=True, mode="online", sampling_seed=0, shuffle_seed=1)
        for change in ({"enabled": False}, {"mode": "unknown"}, {"sampling_seed": True},
                       {"shuffle_seed": -1}, {"shuffle_seed": 2**63}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                CandidatePPOKernel(policy(), contract(), settings(), **(args | change))
        with self.assertRaises(ValueError):
            CandidatePPOKernel(policy().half(), contract(), settings(), **args)
        with self.assertRaises(ValueError):
            CandidatePPOKernel(policy(), None, settings(), **args)

    def test_settings_reject_invalid_and_implicit_values(self):
        for change in ({"learning_rate": 0}, {"learning_rate": True}, {"entropy_coef": -1},
                       {"value_loss_coef": float("nan")}, {"clip_ratio": 1}, {"clip_ratio": 0},
                       {"gae_lambda": 1.1}, {"max_grad_norm": 0}, {"max_grad_norm": float("inf")},
                       {"normalize_advantages": 1}, {"epochs": True}, {"batch_size": 9},
                       {"max_rollout_steps": 0}, {"max_updates": -1}, {"max_optimizer_steps": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                settings(**change)

    def test_prototype_is_cloned_and_frozen_has_no_optimizer(self):
        prototype = policy()
        original = prototype.snapshot_sha256()
        learner = CandidatePPOKernel(prototype, contract(), settings(), enabled=True, mode="frozen",
                                      sampling_seed=0, shuffle_seed=1)
        decision = learner.decide(observation(), candidates())
        self.assertEqual(decision.evaluation.behavior_sha256, original)
        self.assertIsNone(learner.optimizer)
        self.assertTrue(all(not p.requires_grad for p in learner.policy.parameters()))
        before = self.digest(learner)
        for operation in (learner.update, lambda: learner.add_segment(None)):
            with self.assertRaisesRegex(ValueError, "frozen"):
                operation()
            self.assertEqual(before, self.digest(learner))
        with torch.no_grad():
            prototype.value_head[-1].bias.add_(1)
        self.assertEqual(learner.policy.snapshot_sha256(), original)

    def test_no_global_torch_numpy_or_python_rng_consumed(self):
        cpu = torch.get_rng_state().clone()
        numpy = copy.deepcopy(np.random.get_state())
        python = random.getstate()
        learner = self.ready()
        learner.update()
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        actual = np.random.get_state()
        self.assertEqual(numpy[0], actual[0])
        np.testing.assert_array_equal(numpy[1], actual[1])
        self.assertEqual(numpy[2:], actual[2:])
        self.assertEqual(python, random.getstate())

    def test_each_epoch_visits_every_row_and_updates_all_layouts(self):
        for architecture, mode, dtype in (("graph", "physical", torch.float32),
                                          ("graph", "self_only", torch.float64),
                                          ("flat", "self_only", torch.float32)):
            learner = self.ready(architecture=architecture, message_mode=mode, dtype=dtype)
            before = learner.policy.snapshot_sha256()
            result = learner.update()
            self.assertNotEqual(before, learner.policy.snapshot_sha256())
            self.assertEqual(result["optimizer_steps"], 4)
            self.assertEqual(result["rollout_steps"], 3)
            for epoch in (0, 1):
                indices = [i for row in result["minibatches"] if row["epoch"] == epoch for i in row["indices"]]
                self.assertEqual(sorted(indices), [0, 1, 2])
            self.assertFalse(learner.pending)
            self.assertEqual(len(learner.consumed), 3)
            self.assertTrue(all(p.grad is None for p in learner.policy.parameters()))
            self.assertEqual({s["step"].item() for s in learner.optimizer.state.values()}, {4.})

    def test_frozen_and_online_share_initial_decisions(self):
        online, frozen = kernel(), kernel(mode="frozen")
        self.assertEqual(online.policy.snapshot_sha256(), frozen.policy.snapshot_sha256())
        for _ in range(8):
            self.assertEqual(online.decide(observation(), candidates()), frozen.decide(observation(), candidates()))

    def test_empty_update_and_atomic_admission_validation(self):
        learner = kernel()
        before = self.digest(learner)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            learner.update()
        self.assertEqual(before, self.digest(learner))
        segment = invented_segment(learner)
        for bad in (None, replace(segment, advantages=(100.,) * 3),
                    replace(segment, returns=(100.,) * 3), replace(segment, gae_lambda=.4)):
            before = self.digest(learner)
            with self.assertRaises((TypeError, ValueError)):
                learner.add_segment(bad)
            self.assertEqual(before, self.digest(learner))

    def test_old_probabilities_and_bootstrap_must_reproduce(self):
        for terminal in (True, False):
            learner = kernel()
            segment = invented_segment(learner, terminal=terminal)
            decisions = list(segment.decisions)
            if terminal:
                evaluation = replace(decisions[0].evaluation, value=decisions[0].evaluation.value + 1)
                decisions[0] = replace(decisions[0], evaluation=evaluation)
                bootstrap = None
            else:
                bootstrap = replace(segment.bootstrap, value=segment.bootstrap.value + 1)
            forged = prepare_candidate_segment(decisions, segment.records, learner.contract,
                                                behavior_sha256=learner.policy.snapshot_sha256(),
                                                bootstrap=bootstrap, gae_lambda=.8, max_steps=8)
            before = self.digest(learner)
            with self.assertRaisesRegex(ValueError, "reproduce"):
                learner.add_segment(forged)
            self.assertEqual(before, self.digest(learner))

    def test_duplicate_trajectory_ids_and_rollout_cap_rejected(self):
        learner = self.ready()
        before = self.digest(learner)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            learner.add_segment(learner.pending[0])
        self.assertEqual(before, self.digest(learner))
        second = invented_segment(learner, start=10, length=6, trajectory="other")
        before = self.digest(learner)
        with self.assertRaisesRegex(ValueError, "step cap"):
            learner.add_segment(second)
        self.assertEqual(before, self.digest(learner))

    def test_consumed_lineage_not_relabelled_as_new_on_policy_data(self):
        learner = self.ready()
        old = learner.pending[0]
        learner.update()
        with self.assertRaisesRegex(ValueError, "behavior-policy"):
            learner.add_segment(old)
        new_version_same_ids = invented_segment(learner)
        before = self.digest(learner)
        with self.assertRaisesRegex(ValueError, "consumed"):
            learner.add_segment(new_version_same_ids)
        self.assertEqual(before, self.digest(learner))

    def test_update_cap_and_optimizer_cap_stop_before_partial_work(self):
        learner = self.ready(max_updates=1)
        learner.update()
        before = self.digest(learner)
        with self.assertRaisesRegex(ValueError, "cap"):
            learner.update()
        self.assertEqual(before, self.digest(learner))
        capped = kernel(max_optimizer_steps=2)
        segment = invented_segment(capped)
        before = self.digest(capped)
        with self.assertRaisesRegex(ValueError, "optimizer step cap"):
            capped.add_segment(segment)
        self.assertEqual(before, self.digest(capped))
        capped.add_segment(invented_segment(capped, length=2))
        capped.update()
        second = invented_segment(capped, length=1, trajectory="second")
        before = self.digest(capped)
        with self.assertRaisesRegex(ValueError, "optimizer step cap"):
            capped.add_segment(second)
        self.assertEqual(before, self.digest(capped))

    def test_advantages_normalized_once_across_multiple_closed_segments(self):
        learner = self.ready()
        learner.add_segment(invented_segment(learner, start=10, length=2, trajectory="second", terminal=False))
        original = __import__("src.rl.candidate_ppo_kernel", fromlist=["normalize_rollout_advantages"])
        with patch("src.rl.candidate_ppo_kernel.normalize_rollout_advantages",
                   wraps=original.normalize_rollout_advantages) as normalize:
            result = learner.update()
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(normalize.call_args.args[0].numel(), 5)
        self.assertEqual(result["optimizer_steps"], 6)

    def test_gradient_norm_is_clipped_before_step(self):
        learner = self.ready(max_grad_norm=.01)
        original = torch.optim.Adam.step
        observed = []
        def checked_step(optimizer, *args, **kwargs):
            grads = [p.grad.flatten() for group in optimizer.param_groups for p in group["params"]]
            observed.append(torch.linalg.vector_norm(torch.cat(grads)).item())
            return original(optimizer, *args, **kwargs)
        with patch.object(torch.optim.Adam, "step", checked_step):
            learner.update()
        self.assertEqual(len(observed), 4)
        self.assertLessEqual(max(observed), .010001)

    def test_first_adam_step_matches_independent_loss_and_closed_form(self):
        learner = self.ready(dtype=torch.float64, epochs=1, batch_size=3,
                             normalize_advantages=False, max_grad_norm=1.e8)
        reference = copy.deepcopy(learner.policy)
        segment = learner.pending[0]
        evaluated = []
        for d in segment.decisions:
            state = torch.tensor([d.evaluation.actor_state], dtype=torch.float64)
            obs = _decode_actor_state(state, learner.contract)[0]
            evaluated.append(reevaluate_candidate_decision(reference, obs, d))
        logp, values, entropy = [torch.stack(items) for items in zip(*evaluated)]
        old = torch.tensor([d.old_log_prob for d in segment.decisions], dtype=torch.float64)
        adv = torch.tensor(segment.advantages, dtype=torch.float64)
        returns = torch.tensor(segment.returns, dtype=torch.float64)
        ratios = (logp - old).exp()
        objective = (-torch.minimum(ratios * adv, ratios.clamp(.8, 1.2) * adv).mean()
                     + .5 * ((values - returns) ** 2).mean() - .01 * entropy.mean())
        objective.backward()
        expected = [p.detach() - .001 * p.grad / (p.grad.abs() + 1.e-8) for p in reference.parameters()]
        result = learner.update()
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertAlmostEqual(result["minibatches"][0]["total_loss"], objective.item(), places=12)
        for actual, value in zip(learner.policy.parameters(), expected):
            torch.testing.assert_close(actual, value, rtol=0, atol=2.e-11)

    def test_single_class_zero_policy_gradient_still_updates_value(self):
        learner = kernel()
        original = candidates
        with patch("tests.test_candidate_ppo_kernel.candidates", side_effect=lambda i: original(i, option_requests=[])):
            segment = invented_segment(learner)
        self.assertTrue(all(len(d.evaluation.log_probs) == 1 for d in segment.decisions))
        learner.add_segment(segment)
        result = learner.update()
        self.assertTrue(all(row["entropy"] == 0 and row["clip_fraction"] == 0 for row in result["minibatches"]))

    def test_atomic_second_minibatch_failure_restores_all_and_next_update(self):
        learner, reference = self.ready(), self.ready()
        before = self.digest(learner)
        original = torch.optim.Adam.step
        calls = []
        def fail_after_step(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("invented second-step failure")
            return result
        with patch.object(torch.optim.Adam, "step", fail_after_step):
            with self.assertRaisesRegex(RuntimeError, "second-step"):
                learner.update()
        self.assertEqual(before, self.digest(learner))
        self.assertEqual(learner.update(), reference.update())
        self.assertEqual(self.digest(learner), self.digest(reference))

    def test_nonfinite_gradient_and_post_step_moments_are_atomic(self):
        learner = self.ready()
        before = self.digest(learner)
        module = __import__("src.rl.candidate_ppo_kernel", fromlist=["candidate_ppo_loss"])
        original_loss = module.candidate_ppo_loss
        def invalid_gradient(*args, **kwargs):
            result = original_loss(*args, **kwargs)
            return replace(result, total=result.total * float("nan"))
        with patch("src.rl.candidate_ppo_kernel.candidate_ppo_loss", invalid_gradient):
            with self.assertRaisesRegex(ValueError, "gradient"):
                learner.update()
        self.assertEqual(before, self.digest(learner))
        original_step = torch.optim.Adam.step
        def invalid_moments(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
            return result
        with patch.object(torch.optim.Adam, "step", invalid_moments):
            with self.assertRaises(ValueError):
                learner.update()
        self.assertEqual(before, self.digest(learner))

    def test_resume_pending_rollout_next_update_and_next_sampling_exact(self):
        learner = self.ready()
        learner.update()
        learner.add_segment(invented_segment(learner, start=20, trajectory="second", terminal=False))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.pt"
            learner.save(path)
            restored = kernel()
            restored.load(path)
            self.assertEqual(self.digest(learner), self.digest(restored))
            self.assertEqual(learner.update(), restored.update())
            for _ in range(8):
                self.assertEqual(learner.decide(observation(), candidates()),
                                 restored.decide(observation(), candidates()))
            self.assertEqual(self.digest(learner), self.digest(restored))

    def test_initial_pending_and_frozen_checkpoints_round_trip(self):
        for learner in (kernel(), self.ready(), kernel(mode="frozen"), self.ready(dtype=torch.float64)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.pt"
                learner.save(path)
                clone = copy.deepcopy(learner)
                clone.load(path)
                self.assertEqual(self.digest(learner), self.digest(clone))

    def test_state_dict_is_a_deep_copy(self):
        learner = self.ready()
        before = self.digest(learner)
        state = learner.state_dict()
        state["sampling_rng"].zero_()
        next(iter(state["policy"].values())).zero_()
        state["pending"].clear()
        state["manifest"]["settings"]["epochs"] = 100
        self.assertEqual(before, self.digest(learner))

    def test_checkpoint_policy_manifest_and_optimizer_corruption_is_atomic(self):
        learner = self.ready()
        learner.update()
        state = learner.state_dict()
        key = next(iter(state["policy"]))
        changes = [
            lambda s: s.update(extra=1),
            lambda s: s.update(manifest_sha256="0" * 64),
            lambda s: s["manifest"]["settings"].update(epochs=9),
            lambda s: s["policy"].update({key: torch.zeros(100)}),
            lambda s: s["policy"].update({key: s["policy"][key].double()}),
            lambda s: s["policy"][key].fill_(float("nan")),
            lambda s: s["optimizer"]["param_groups"][0].update(lr=1.),
            lambda s: s["optimizer"]["state"].pop(0),
            lambda s: s["optimizer"]["state"][0]["step"].add_(1),
            lambda s: s["optimizer"]["state"][0].update(exp_avg=torch.zeros(100)),
            lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
            lambda s: s["optimizer"]["state"][0]["exp_avg"].fill_(float("inf")),
        ]
        before = self.digest(learner)
        for mutate in changes:
            bad = copy.deepcopy(state)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, TypeError)):
                learner.load_state_dict(bad)
            self.assertEqual(before, self.digest(learner))

    def test_checkpoint_history_pending_and_rng_corruption_is_atomic(self):
        learner = self.ready()
        learner.update()
        learner.add_segment(invented_segment(learner, start=10, trajectory="second"))
        state = learner.state_dict()
        changes = [
            lambda s: s.update(history=[True]),
            lambda s: s.update(history=[9]),
            lambda s: s["consumed"].pop(),
            lambda s: s["consumed"].__setitem__(0, s["consumed"][1]),
            lambda s: s["consumed"].__setitem__(0, ("", "invented", 0)),
            lambda s: s.update(sampling_rng=torch.zeros(10, dtype=torch.uint8)),
            lambda s: s.update(shuffle_rng=s["shuffle_rng"].float()),
            lambda s: s["pending"][0].update(advantages=(1., 2., 3.)),
            lambda s: s["pending"][0]["decisions"][0]["evaluation"].update(value=999.),
            lambda s: s["pending"][0]["decisions"][0]["evaluation"]["candidates"].update(class_keys=()),
        ]
        before = self.digest(learner)
        for mutate in changes:
            bad = copy.deepcopy(state)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, TypeError)):
                learner.load_state_dict(bad)
            self.assertEqual(before, self.digest(learner))

    def test_checkpoint_cannot_switch_mode_precision_or_operator(self):
        original = self.ready()
        original.update()
        for target in (kernel(mode="frozen"), kernel(dtype=torch.float64), kernel(message_mode="self_only"),
                       kernel(architecture="flat", message_mode="self_only"), kernel(learning_rate=.002)):
            before = self.digest(target)
            with self.assertRaisesRegex(ValueError, "manifest"):
                target.load_state_dict(original.state_dict())
            self.assertEqual(before, self.digest(target))

    def test_zero_update_forged_policy_and_frozen_updates_rejected(self):
        learner = kernel()
        state = learner.state_dict()
        next(iter(state["policy"].values())).add_(1)
        with self.assertRaisesRegex(ValueError, "zero-update"):
            learner.load_state_dict(state)
        frozen = kernel(mode="frozen")
        state = frozen.state_dict()
        state["history"] = [1]
        with self.assertRaisesRegex(ValueError, "history"):
            frozen.load_state_dict(state)

    def test_save_refuses_overwrite_and_corrupt_envelope(self):
        learner = self.ready()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.pt"
            learner.save(path)
            before_bytes = path.read_bytes()
            with self.assertRaises(FileExistsError):
                learner.save(path)
            self.assertEqual(before_bytes, path.read_bytes())
            envelope = torch.load(path, weights_only=True)
            envelope["state"]["sampling_rng"][0] ^= 1
            bad = Path(directory) / "corrupt.pt"
            torch.save(envelope, bad)
            before = self.digest(learner)
            with self.assertRaisesRegex(ValueError, "checksum"):
                learner.load(bad)
            self.assertEqual(before, self.digest(learner))

    def test_failed_checkpoint_publication_leaves_no_partial_file(self):
        learner = self.ready()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.pt"
            with patch("src.rl.candidate_ppo_kernel.os.link", side_effect=OSError("invented disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    learner.save(path)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
