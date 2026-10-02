"""Two-step fake array world and fake Adam metadata; real fitting forbidden."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.leave_one_episode_out_baseline import leave_one_episode_out_targets
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_collection import TimeBaselineContinuation, TimeBaselineSession
from src.rl.time_baseline_ppo import TimeBaselinePPOKernel
from tests.test_dynamic_candidate_session import FakeEnv, FakeProducer, FakeReference, OPTIONS, session
from tests.test_dynamic_candidate_updates import fake_adam


MANIFEST = tuple(dict(trajectory_id=f"training/invented/{i}", environment_seed=101 + i) for i in range(4))


def kernel(method="leave_one_episode_out_time", **changes):
    base = session().learner
    kwargs = dict(enabled=True, mode="online", sampling_seed=41, shuffle_seed=43,
                  target_method=method, training_manifest=MANIFEST, training_source_id="invented-training",
                  episode_horizon=2, episodes_per_rollout=2)
    return TimeBaselinePPOKernel(base.policy, base.contract, base.settings, **(kwargs | changes))


def collector(learner, index, seed, **changes):
    producer = FakeProducer()
    kwargs = dict(enabled=True, options=OPTIONS, trajectory_id=MANIFEST[index]["trajectory_id"],
                  split="training", selection="sample", source_id="invented-training", environment_seed=seed)
    return TimeBaselineSession(FakeEnv(), producer, FakeReference(producer), learner, **(kwargs | changes))


@unittest.skipIf(torch is None, "torch unavailable")
class TimeBaselineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.serial = Path(self.temp.name), 0
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def run_fixture(self, method="leave_one_episode_out_time", budget=None):
        scope = "block60/" + method
        if budget is None:
            caps = dict(trajectory=8, clone=0, actor=4, critic=4, seconds=100)
            plan = dict(format="dynamic-candidate-budget-plan-v1", draft_sha256="invented",
                        limits=caps, phases={method: caps}, sections={scope: caps | {"phase": method}})
            self.serial += 1
            budget = DynamicCandidateBudget(self.root / f"budget{self.serial}.jsonl", plan,
                                            enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            budget.begin(scope)
        return TimeBaselineContinuation(kernel(method), budget,
            dict(episodes_per_model=4, episodes_per_rollout=2, max_updates=2, epochs=1, gae_lambda=1.),
            enabled=True, scope=scope, seeds=(101, 102, 103, 104), horizon=2, session_factory=collector)

    def fill(self, run):
        while not run.update_due:
            run.step()

    def finish(self, run):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while not run.done:
                run.update() if run.update_due else run.step()

    def test_kernel_requires_fixed_independent_whole_episodes(self):
        for changes in (dict(target_method="invented"), dict(episodes_per_rollout=1),
                        dict(training_manifest=MANIFEST[:-1]),
                        dict(training_manifest=(MANIFEST[0],) * 4)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                kernel(**changes)

    def test_targets_preserve_raw_pending_and_critic_returns(self):
        run = self.run_fixture()
        self.fill(run)
        before = state_digest(run.kernel.state_dict())
        receipt = run.kernel.target_receipt()
        result = leave_one_episode_out_targets(receipt["returns"], **{
            k: receipt[k] for k in ("trajectory_ids", "environment_seeds", "behavior_sha256s", "terminal_flags")})
        self.assertEqual(receipt["advantages"], result.advantages)
        self.assertEqual(receipt["returns"], tuple(s.returns for s in run.kernel.pending))
        self.assertEqual(state_digest(run.kernel.state_dict()), before)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            report = run.update()
        self.assertEqual(report["target_receipt"], receipt)
        self.assertEqual(run.kernel.target_history, [receipt])

    def test_collected_value_method_matches_frozen_original_update_arithmetic(self):
        run = self.run_fixture("collected_value")
        self.fill(run)
        k = run.kernel
        original = DynamicCandidatePPOKernel(k.policy, k.contract, k.settings, enabled=True,
                                            mode="online", sampling_seed=41, shuffle_seed=43)
        for segment in k.pending:
            original.add_segment(segment)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            a = run.update()
            b = original.update(before_optimizer_step=lambda owner: None, before_minibatch=lambda: None)
        self.assertEqual(a["minibatches"], b["minibatches"])
        self.assertEqual(a["optimizer_steps"], b["optimizer_steps"])
        self.assertEqual(state_digest(k.policy.state_dict()), state_digest(original.policy.state_dict()))

    def test_mock_complete_collection_update_restore_without_weight_change(self):
        for method in ("collected_value", "leave_one_episode_out_time"):
            run = self.run_fixture(method)
            original = run.kernel.policy.snapshot_sha256()
            self.finish(run)
            self.assertEqual(run.kernel.policy.snapshot_sha256(), original)
            self.assertEqual(len(run.kernel.target_history), 2)
            self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["optimizer"], 8)
            restored = self.run_fixture(method, budget=run.budget)
            restored.load_state_dict(run.state_dict())
            self.assertTrue(restored.done)
            self.assertEqual(state_digest(restored.state_dict()), state_digest(run.state_dict()))

    def test_interrupted_collection_and_pending_rollout_restore_exactly(self):
        run = self.run_fixture()
        run.step()
        restored = self.run_fixture(budget=run.budget)
        restored.load_state_dict(run.state_dict())
        self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
        self.fill(restored)
        after = self.run_fixture(budget=run.budget)
        after.load_state_dict(restored.state_dict())
        self.assertEqual(after.kernel.target_receipt(), restored.kernel.target_receipt())
        self.finish(after)
        uninterrupted = self.run_fixture()
        self.finish(uninterrupted)
        self.assertEqual(state_digest(after.kernel.state_dict()), state_digest(uninterrupted.kernel.state_dict()))

    def test_method_seed_and_target_history_tamper_rejected_atomically(self):
        run = self.run_fixture()
        self.finish(run)
        before = state_digest(run.kernel.state_dict())
        wrong_method = kernel("collected_value")
        with self.assertRaises(ValueError):
            wrong_method.load_state_dict(run.kernel.state_dict())
        for tamper in ("seed", "baseline", "history", "consumed", "reward"):
            saved = run.kernel.state_dict()
            if tamper == "seed":
                saved["manifest"]["training_manifest"][0]["environment_seed"] += 1
            elif tamper == "baseline":
                rows = saved["target_history"][0]["baselines"]
                saved["target_history"][0]["baselines"] = ((999., rows[0][1]), rows[1])
            elif tamper == "history":
                saved["target_history"].pop()
            elif tamper == "reward":
                rows = saved["target_history"][0]["raw_rewards"]
                saved["target_history"][0]["raw_rewards"] = ((999., rows[0][1]), rows[1])
            else:
                saved["consumed"][0] = ("forged", "episode", 0)
            with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                run.kernel.load_state_dict(saved)
            self.assertEqual(state_digest(run.kernel.state_dict()), before)
        control = self.run_fixture("collected_value")
        self.finish(control)
        saved = control.kernel.state_dict()
        rows = saved["target_history"][0]["advantages"]
        saved["target_history"][0]["advantages"] = ((999., rows[0][1]), rows[1])
        with self.assertRaises(ValueError):
            control.kernel.load_state_dict(saved)

    def test_wrong_split_seed_order_and_incomplete_rollout_rejected(self):
        k = kernel()
        for changes in (dict(environment_seed=999), dict(source_id="test"),
                        dict(split="training", selection="greedy")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                collector(k, 0, 101, **changes)
        with self.assertRaises(ValueError):
            k.target_receipt()
        c = collector(k, 1, 102)
        while not c.closed:
            c.step(before_step=lambda: None)
        before = state_digest(k.state_dict())
        with self.assertRaises(ValueError):
            k.add_segment(c.segment(1.))
        self.assertEqual(state_digest(k.state_dict()), before)

    def test_second_owner_failure_keeps_debits_and_forbids_retry(self):
        run = self.run_fixture()
        self.fill(run)
        before = run.kernel.policy.snapshot_sha256()
        calls = []
        def fail(optimizer, *args, **kwargs):
            calls.append(optimizer)
            if len(calls) == 2:
                raise RuntimeError("invented critic failure")
            return fake_adam(optimizer)
        with patch.object(torch.optim.Adam, "step", new=fail), self.assertRaises(RuntimeError):
            run.update()
        self.assertEqual(run.kernel.policy.snapshot_sha256(), before)
        self.assertEqual(run.kernel.target_history, [])
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["optimizer"], 2)
        self.assertTrue(run.budget.failed)
        with self.assertRaises(ValueError):
            run.update()
        with self.assertRaises(ValueError):
            kernel().load_state_dict(run.kernel.state_dict())


if __name__ == "__main__":
    unittest.main()
