"""Six-controller wiring on invented two-step data; all real updates forbidden."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.dynamic_candidate_factory import dynamic_initializer, dynamic_template
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, dynamic_stream_manifest, read_dynamic_ledger
from src.rl.dynamic_candidate_session import DynamicCandidateSession
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_backend import TimeBaselinePatientBackend
from src.rl.time_baseline_campaign import TimeBaselineCampaign
from src.rl.time_baseline_collection import TimeBaselineSession
from src.rl.time_baseline_factory import time_baseline_config, fork_time_baseline_initializer
from src.rl.time_baseline_plan import CONTROLLERS, TRAINING_ARMS, time_baseline_budget_plan, time_baseline_stream_manifest
from src.rl.time_baseline_ppo import TimeBaselinePPOKernel
from src.utils.research_archive import inventory, verify_archive
from tests.test_dynamic_candidate_campaign import fixture_config, FakeBackend, FakeRecorder, fake_reader
from tests.test_dynamic_candidate_factory import proposal as original_proposal
from tests.test_dynamic_candidate_session import FakeEnv, FakeProducer, FakeReference, OPTIONS
from tests.test_dynamic_candidate_updates import fake_adam


def proposed():
    path = Path(__file__).resolve().parents[1] / "specs/2026-10-02-time-baseline-comparison/proposal.json"
    return json.loads(path.read_text())


def saved_design():
    path = Path(__file__).resolve().parents[1] / "specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-proposal.json"
    return json.loads(path.read_text())["scientific_config"]


def invented_contract():
    """Only test doubles shorten counts; production proposal stays immutable."""
    p, cfg = proposed(), fixture_config()
    cfg["continuation"].update(episodes_per_rollout=2, rollouts_per_arm_per_block=1, rows_per_rollout=4)
    cfg["final_controllers"] = list(CONTROLLERS)
    cfg["evaluation"].update(total_episodes=18, controllers_per_block=6, environment_steps=36)
    cfg["totals"].update(main_full_episodes=54, main_environment_steps=126)
    cfg["time_baseline_proposal"] = copy.deepcopy(p)
    streams = time_baseline_stream_manifest(p)
    streams["namespace"] = "invented-time-baseline-no-science"
    for values in streams["environment"].values():
        values["training"] = values["training"][:2]
        values["test"] = values["test"][:1]
    plan = time_baseline_budget_plan(p)
    for name, row in plan["sections"].items():
        row.update(trajectory=0, clone=0, actor=0, critic=0)
        if name == "same_start_preflight":
            row.update(trajectory=36, clone=18)
        elif name.split("/")[0] in TRAINING_ARMS:
            row.update(trajectory=4, actor=2, critic=0 if name.startswith("bc_continue/") else 2)
        elif name.startswith("final_evaluation/"):
            row.update(trajectory=2)
    for phase, row in plan["phases"].items():
        for key in ("trajectory", "clone", "actor", "critic"):
            row[key] = sum(r[key] for r in plan["sections"].values() if r["phase"] == phase)
    for key in ("trajectory", "clone", "actor", "critic"):
        plan["limits"][key] = sum(r[key] for r in plan["phases"].values())
    return p, cfg, streams, plan


class NewFakeBackend(FakeBackend):
    def __init__(self, config, streams):
        super().__init__(config)
        self.streams = streams

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        self.sessions.append((block, seed, split, selection, trajectory))
        producer = self.producer(block)
        cls = TimeBaselineSession if type(kernel) is TimeBaselinePPOKernel else DynamicCandidateSession
        extra = {"environment_seed": seed} if cls is TimeBaselineSession else {}
        return cls(FakeEnv(), producer, FakeReference(producer), kernel, enabled=True,
                   options=OPTIONS, trajectory_id=trajectory, split=split, selection=selection,
                   source_id=self.streams["namespace"], **extra)


def fixture_verifier(root, index, config, streams):
    files, outcomes = fake_reader(root, index, config)
    expected = {(b, r, w, int(seed)) for b in config["blocks"] for r in CONTROLLERS
                for w, seed in enumerate(streams["environment"][str(b)]["test"])}
    actual = {(r["block"], r["role"], r["world_index"], r["seed"]) for r in outcomes}
    if actual != expected or len(outcomes) != len(expected):
        raise ValueError("invented six-controller evaluation matrix differs")
    return dict(engineering_fixture_only=True, outcomes=outcomes, files=files, scientific_performance_claim=False)


class ConfigTests(unittest.TestCase):
    def test_pure_config_keeps_scientific_mechanisms_and_exact_counts(self):
        original, p = saved_design(), proposed()
        originals = copy.deepcopy((original, p))
        cfg = time_baseline_config(original, p)
        self.assertEqual((original, p), originals)
        for key in ("objective", "optimizer", "model_proposal", "reference", "candidate_support", "initialization"):
            self.assertEqual(cfg[key], original[key])
        self.assertEqual(cfg["totals"]["main_environment_steps"], 27216)
        self.assertEqual(cfg["totals"]["main_optimizer_calls"], 1920)
        self.assertEqual(cfg["totals"]["fresh_episode_builds"], 522)
        self.assertEqual(cfg["evaluation"]["total_episodes"], 216)
        self.assertFalse(cfg["scientific_execution_authorized"])
        from src.rl.time_baseline_verification import _validate_plan
        worlds, _ = _validate_plan(cfg, time_baseline_stream_manifest(p))
        self.assertEqual(worlds, 12)
        original["objective"]["reward_scale"] = 1.
        with self.assertRaises(ValueError):
            time_baseline_config(original, p)

    def test_real_backend_has_no_default_admission(self):
        with tempfile.TemporaryDirectory() as path:
            p = proposed()
            backend = TimeBaselinePatientBackend(path, time_baseline_config(original_proposal(), p),
                                                time_baseline_stream_manifest(p))
            with patch("torch.load", side_effect=AssertionError("no saved model")) as load:
                with self.assertRaises(PermissionError):
                    backend.prepare(60, 0)
                with self.assertRaises(PermissionError):
                    backend.session(60, None, 0, trajectory="x", split="training", selection="sample")
                load.assert_not_called()
            self.assertEqual(backend.build_attempts, 0)
            self.assertEqual(backend.episode_builds, 0)


@unittest.skipIf(torch is None, "torch unavailable")
class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        self.p, self.cfg, self.streams, self.plan = invented_contract()
        for name, value in (("time_baseline_config", self.cfg), ("time_baseline_budget_plan", self.plan),
                            ("time_baseline_stream_manifest", self.streams)):
            mock = patch("src.rl.time_baseline_campaign." + name, return_value=copy.deepcopy(value))
            mock.start()
            self.addCleanup(mock.stop)
        self.initializers, self.qualifications, self.loaded = {}, {}, []
        old_streams = dynamic_stream_manifest(self.cfg)
        for b in self.cfg["blocks"]:
            owner = dynamic_initializer(dynamic_template(FakeProducer(), self.cfg, old_streams, b),
                                        self.cfg, old_streams, b)
            owner.steps = self.cfg["initialization"]["actor_adam_calls_per_block"]
            self.initializers[b] = owner
            self.qualifications[str(b)] = dict(passed=True, kernel_sha256=state_digest(owner.state_dict()))
        self.budget = DynamicCandidateBudget(self.root / "launcher/budget.jsonl", self.plan,
                                             enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        self.backend = NewFakeBackend(self.cfg, self.streams)
        def load(b):
            if b in self.loaded:
                raise ValueError("no duplicate initializer load")
            self.loaded.append(b)
            return copy.deepcopy(self.initializers[b])
        self.run = TimeBaselineCampaign(self.root, self.cfg, self.p, self.streams, self.budget,
            self.backend, enabled=True, initializer_loader=load, qualifications=self.qualifications,
            recorder=FakeRecorder, reader=fake_reader, verifier=fixture_verifier)

    def until(self, job):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while self.run.sequence.next_job != job:
                self.run.begin_next()
                self.run.dispatch(self.run.sequence.active)

    def test_full_fake_chain_exact_nonrefundable_counts_and_archive(self):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            code = self.run.run()
        if code:
            self.fail((self.root / "launcher/failure.json").read_text())
        self.assertEqual(self.loaded, [60, 61, 62])
        self.assertEqual(len(self.run.sequence.completed), 33)
        self.assertEqual(len(self.run.sequence.seals), 12)
        self.assertEqual(len(self.backend.sessions), 54)
        self.assertEqual(len(self.run.raw_index), 54)
        self.assertEqual(len(self.run.test_index), 18)
        ledger = read_dynamic_ledger(self.budget.path)
        self.assertEqual(ledger["counts"], dict(environment=126, optimizer=30))
        for key, value in dict(trajectory=108, clone=18, actor=18, critic=12).items():
            self.assertEqual(ledger["owner_counts"]["global:" + key], value)
        for b in self.cfg["blocks"]:
            for role in CONTROLLERS[:4]:
                owner = self.run.models[f"block{b}/graph/{role}"]
                self.assertEqual(owner.policy.snapshot_sha256(), self.initializers[b].policy.snapshot_sha256())
        self.assertEqual(inventory(self.root / "payload"), self.run.payload_seal)
        verify_archive(self.root / "archives/completed-payload.tar.gz", self.run.payload_seal)
        self.assertTrue((self.root / "launcher/completed.json").exists())
        self.assertFalse(any(s[2] in ("demonstration", "qualification") for s in self.backend.sessions))
        targets = json.loads((self.root / "payload/independent-target-verification.json").read_text())
        self.assertEqual(len(targets["rollouts"]), 6)
        self.assertTrue(targets["unchanged_critic_targets_verified"])
        self.assertTrue(targets["entire_trajectory_exclusion_verified"])

    def test_forks_are_identical_private_and_preflight_does_not_advance_training(self):
        self.until("same_start_preflight")
        starts = {k: state_digest(v.state_dict()) for k, v in self.run.models.items()}
        for b in self.cfg["blocks"]:
            owners = [self.run.models[f"block{b}/graph/{r}"] for r in CONTROLLERS[:4]]
            self.assertEqual(len({o.policy.snapshot_sha256() for o in owners}), 1)
            self.assertEqual(len({id(o.policy) for o in owners}), 4)
            for role in TRAINING_ARMS:
                key = f"block{b}/graph/{role}"
                self.assertNotEqual(state_digest(self.run.templates[key].sampling_rng.get_state()),
                                    state_digest(self.run.models[key].sampling_rng.get_state()))
        self.until("current_ppo/block60")
        self.assertEqual(starts, {k: state_digest(v.state_dict()) for k, v in self.run.models.items()})

    def test_saved_receipt_failure_stops_before_episodes(self):
        self.run.saved_qualifications["60"]["kernel_sha256"] = "a" * 64
        self.assertEqual(self.run.run(), 1)
        self.assertEqual(self.backend.sessions, [])
        self.assertEqual(self.budget.counts, dict(environment=0, optimizer=0))

    def test_exact_collection_and_update_restore_no_reload_or_refund(self):
        self.until("time_baseline_ppo/block60")
        self.run.begin_next()
        self.run.advance()
        state = self.run.checkpoint_state()
        counts, builds = copy.deepcopy(self.budget.counts), len(self.backend.sessions)
        self.run.restore_checkpoint_state(state)
        self.assertEqual(state_digest(state), state_digest(self.run.checkpoint_state()))
        self.assertEqual(counts, self.budget.counts)
        self.assertEqual(builds, len(self.backend.sessions))
        self.assertEqual(self.loaded, [60, 61, 62])
        while not self.run.continuation.update_due:
            self.run.advance()
        state = self.run.checkpoint_state()
        self.run.restore_checkpoint_state(state)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            self.run.advance()
        state = self.run.checkpoint_state()
        self.run.restore_checkpoint_state(state)
        self.assertEqual(state_digest(state), state_digest(self.run.checkpoint_state()))

    def test_debited_failure_terminal_not_an_unused_budget_retry(self):
        self.until("current_ppo/block60")
        self.run.begin_next()
        while self.run.continuation is None or not self.run.continuation.update_due:
            self.run.advance()
        state = self.run.checkpoint_state()
        count = self.budget.counts["optimizer"]
        with patch.object(torch.optim.Adam, "step", side_effect=RuntimeError("invented failure")):
            with self.assertRaises(RuntimeError):
                self.run.advance()
        self.assertEqual(self.budget.counts["optimizer"], count + 1)
        self.assertTrue((self.root / "launcher/failure.json").exists())
        with self.assertRaises(ValueError):
            self.run.restore_checkpoint_state(state)

    def test_no_test_before_all_seals_and_no_unadmitted_constructor(self):
        self.until("same_start_preflight")
        with self.assertRaises(ValueError):
            self.run._session(self.run._descriptors("final_evaluation/block60/current_ppo")[0])
        with self.assertRaises(PermissionError):
            TimeBaselineCampaign(self.root, self.cfg, self.p, self.streams, self.budget, self.backend,
                initializer_loader=lambda _: None, qualifications=self.qualifications, enabled=True,
                engineering_only=False)

    def test_factory_rejects_unqualified_or_mutated_saved_payload(self):
        for receipt in (dict(passed=False), dict(passed=True, kernel_sha256="a" * 64)):
            with self.assertRaises(ValueError):
                fork_time_baseline_initializer(self.initializers[60], receipt, self.cfg, self.streams, 60)
