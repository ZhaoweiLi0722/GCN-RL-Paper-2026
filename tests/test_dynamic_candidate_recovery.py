"""Additive recovery exercised on invented two-step worlds; no real fitting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_factory import dynamic_initializer, dynamic_template, fork_dynamic_initializer
from src.rl.dynamic_candidate_recovery_campaign import RecoveryCampaign, RecoverySequence
from src.rl.dynamic_candidate_recovery_execution import approved, approved_limits, validate_authorization
from src.rl.dynamic_candidate_recovery_plan import recovery_budget_plan, recovery_config, REMOVED
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, dynamic_stream_manifest, read_dynamic_ledger
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.dynamic_candidate_saved_qualification import restore_saved_policy
from tests.test_dynamic_candidate_campaign import (
    fixture_config, FakeBackend, FakeRecorder, fake_reader, fake_verifier,
)
from tests.test_dynamic_candidate_factory import proposal
from tests.test_dynamic_candidate_session import FakeProducer
from tests.test_dynamic_candidate_updates import fake_adam
from tests.test_dynamic_candidate_saved_qualification import synthetic_saved_fixture


class RecoveryPlanTests(unittest.TestCase):
    def test_exact_remaining_caps_and_no_scientific_change(self):
        original = proposal()
        before = copy.deepcopy(original)
        config, plan = recovery_config(original), recovery_budget_plan(original)
        self.assertEqual(original, before)
        self.assertEqual(plan["limits"], {"trajectory": 20124, "clone": 60,
                                         "actor": 768, "critic": 384, "seconds": 17400})
        self.assertEqual(config["totals"]["main_full_episodes"], 387)
        self.assertEqual(config["totals"]["build_env_calls_max"], 390)
        self.assertEqual(config["totals"]["unique_world_start_allocations"], 135)
        for key in original.keys() - {"totals", "phase_budgets"}:
            self.assertEqual(config[key], original[key])
        sequence = RecoverySequence(original, ".")
        self.assertEqual(len(sequence.jobs), 27)
        self.assertFalse(any(job.split("/")[0] in REMOVED for job in sequence.jobs))
        self.assertLess(sequence.jobs.index("bc_continuation/block62"), sequence.jobs.index("all_model_seal"))
        self.assertLess(sequence.jobs.index("all_model_seal"), sequence.jobs.index("final_evaluation/block60/own_frozen"))

    def test_missing_new_authorization_fails_before_science(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.load", side_effect=AssertionError("no model loading")) as load:
                with self.assertRaises(PermissionError):
                    approved(Path(directory).resolve())
                load.assert_not_called()

    def test_scope_specific_authorization_not_old_qualification(self):
        config = proposal()
        packet = {"scientific_execution_authorized": False, "ready_to_launch": False,
            "implementation_commit": "a" * 40, "budget_plan": recovery_budget_plan(config),
            "scientific_config": recovery_config(config)}
        packet["packet_sha256"] = digest(packet)
        auth = {"format": "dynamic-continuation-recovery-authorization-v1", "approved": True,
            "packet_sha256": packet["packet_sha256"], "implementation_commit": packet["implementation_commit"],
            "limits": approved_limits(packet), "automatic_retry": False, "remote_or_dropbox_actions": False,
            "reuse_completed_qualification_without_rescoring": True, "old_s1_remains_terminal": True,
            "user_approval": {"user": "Zhaowei", "verbatim": "invented fixture", "recorded_at_utc": "fixture"}}
        validate_authorization(auth, packet)
        for key, value in (("approved", False), ("format", "saved-qualification-explicit-authorization-v1"),
                           ("automatic_retry", True), ("old_s1_remains_terminal", False)):
            with self.subTest(key=key), self.assertRaises(PermissionError):
                validate_authorization(dict(auth, **{key: value}), packet)
        wrong = copy.deepcopy(auth)
        wrong["limits"]["optimizer_calls"] += 1
        with self.assertRaises(PermissionError):
            validate_authorization(wrong, packet)


@unittest.skipIf(torch is None, "torch unavailable")
class RecoveryCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        self.config = fixture_config()
        self.streams = dynamic_stream_manifest(self.config)
        self.loaded, self.initializers, self.qualifications = [], {}, {}
        for b in self.config["blocks"]:
            owner = dynamic_initializer(dynamic_template(FakeProducer(), self.config, self.streams, b),
                                        self.config, self.streams, b)
            # Artificial completed counter only; no fit was performed.
            owner.steps = self.config["initialization"]["actor_adam_calls_per_block"]
            self.initializers[b] = owner
            self.qualifications[str(b)] = {"passed": True, "kernel_sha256": state_digest(owner.state_dict())}
        self.budget = DynamicCandidateBudget(self.root / "launcher/budget.jsonl", recovery_budget_plan(self.config),
                                             enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        self.backend = FakeBackend(recovery_config(self.config))
        def load(b):
            if b in self.loaded:
                raise AssertionError("duplicate saved initializer load")
            self.loaded.append(b)
            return copy.deepcopy(self.initializers[b])
        self.run = RecoveryCampaign(self.root, self.config, self.streams, self.budget, self.backend,
            initializer_loader=load, qualifications=self.qualifications, enabled=True,
            recorder=FakeRecorder, reader=fake_reader, verifier=fake_verifier)

    def until(self, job):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while self.run.sequence.next_job != job:
                self.run.begin_next()
                self.run.dispatch(self.run.sequence.active)

    def test_full_chain_reuses_initializers_and_never_enters_old_phases(self):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            code = self.run.run()
        if code:
            self.fail((self.root / "launcher/failure.json").read_text())
        self.assertEqual(self.loaded, [60, 61, 62])
        self.assertEqual(len(self.backend.sessions), 42)
        self.assertEqual(len(self.run.raw_index), 42)
        self.assertEqual(len(self.run.test_index), 15)
        self.assertEqual(len(self.run.sequence.completed), 27)
        self.assertEqual(len(self.run.sequence.seals), 9)
        ledger = read_dynamic_ledger(self.budget.path)
        self.assertEqual(ledger["counts"], {"environment": 99, "optimizer": 18})
        self.assertEqual(ledger["owner_counts"]["global:clone"], 15)
        self.assertEqual(ledger["owner_counts"]["global:actor"], 12)
        self.assertEqual(ledger["owner_counts"]["global:critic"], 6)
        self.assertFalse(any(s[2] in ("qualification", "demonstration") for s in self.backend.sessions))
        self.assertFalse((self.root / "payload/initialization").exists())
        receipt = json.loads((self.root / "launcher/archive-receipt.json").read_text())
        self.assertTrue(receipt["local_archive_verified"])
        self.assertFalse(receipt["dropbox_exported"])

    def test_saved_receipt_mismatch_closes_before_any_episode(self):
        self.run.saved_qualifications["60"]["kernel_sha256"] = "b" * 64
        self.assertEqual(self.run.run(), 1)
        self.assertEqual(self.backend.sessions, [])
        self.assertEqual(self.budget.counts, {"environment": 0, "optimizer": 0})
        with self.assertRaises(ValueError):
            self.run.begin_next()

    def test_same_start_and_test_barrier(self):
        self.until("same_start_preflight")
        for b in self.config["blocks"]:
            owners = [self.run.models[f"block{b}/graph/{role}"]
                      for role in ("own_frozen", "own_ppo", "own_bc_continue")]
            self.assertEqual(len({o.policy.snapshot_sha256() for o in owners}), 1)
            self.assertEqual(owners[0].policy.snapshot_sha256(), self.initializers[b].policy.snapshot_sha256())
            self.assertFalse(owners[1].optimizers["actor"].state)
            self.assertFalse(owners[2].optimizer.state)
        with self.assertRaises(ValueError):
            self.run._session(self.run._descriptors("final_evaluation/block60/own_frozen")[0])

    def test_exact_boundary_restore_no_reloads_or_refund(self):
        self.until("same_start_preflight")
        self.run.begin_next()
        self.run.advance()
        saved = self.run.checkpoint_state()
        counts, loaded = copy.deepcopy(self.budget.counts), list(self.loaded)
        self.run.restore_checkpoint_state(saved)
        self.assertEqual(self.budget.counts, counts)
        self.assertEqual(self.loaded, loaded)
        self.assertEqual(state_digest(saved), state_digest(self.run.checkpoint_state()))
        self.run.advance()
        with self.assertRaises(ValueError):
            self.run.restore_checkpoint_state(saved)
        self.assertGreater(self.budget.counts["environment"], counts["environment"])

    def test_update_boundary_restore_keeps_live_budget(self):
        self.until("ppo_continuation/block60")
        self.run.begin_next()
        while self.run.continuation is None or not self.run.continuation.update_due:
            self.run.advance()
        saved = self.run.checkpoint_state()
        self.run.restore_checkpoint_state(saved)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            self.run.advance()
        after = copy.deepcopy(self.budget.counts)
        saved_after = self.run.checkpoint_state()
        self.run.restore_checkpoint_state(saved_after)
        self.assertEqual(after, self.budget.counts)
        self.assertEqual(state_digest(saved_after), state_digest(self.run.checkpoint_state()))

    def test_debited_update_failure_is_terminal(self):
        self.until("ppo_continuation/block60")
        self.run.begin_next()
        while self.run.continuation is None or not self.run.continuation.update_due:
            self.run.advance()
        old = self.run.checkpoint_state()
        before = copy.deepcopy(self.budget.counts)
        with patch.object(torch.optim.Adam, "step", side_effect=RuntimeError("injected operator failure")):
            with self.assertRaises(RuntimeError):
                self.run.advance()
        self.assertGreater(self.budget.counts["optimizer"], before["optimizer"])
        self.assertTrue((self.root / "launcher/failure.json").exists())
        with self.assertRaises(ValueError):
            self.run.restore_checkpoint_state(old)

    def test_no_unadmitted_scientific_constructor(self):
        with self.assertRaises(PermissionError):
            RecoveryCampaign(self.root, self.config, self.streams, self.budget, self.backend,
                initializer_loader=lambda _: None, qualifications=self.qualifications,
                enabled=True, engineering_only=False)

    def test_saved_readonly_view_forks_trainable_owners_without_refitting(self):
        cfg, streams, states, *_ = synthetic_saved_fixture()
        view = restore_saved_policy(states["block60/graph"], cfg, streams, 60)
        view.steps = states["block60/graph"]["steps"]
        before = state_digest(view.state_dict())
        forks = fork_dynamic_initializer(view, {"passed": True, "kernel_sha256": before}, cfg, streams, 60)
        self.assertEqual(state_digest(view.state_dict()), before)
        self.assertTrue(all(not p.requires_grad for p in view.policy.parameters()))
        self.assertTrue(all(p.requires_grad for p in forks["own_ppo"].policy.actor_parameters()))
        self.assertTrue(all(p.requires_grad for p in forks["own_ppo"].policy.critic_parameters()))
        self.assertTrue(all(p.requires_grad for p in forks["own_bc_continue"].policy.actor_parameters()))
        self.assertTrue(all(not p.requires_grad for p in forks["own_bc_continue"].policy.critic_parameters()))
        self.assertEqual(len({p.policy.snapshot_sha256() for p in forks.values()}), 1)
