"""Full serial wiring with invented arrays and fake optimizer metadata only."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl import cohort_public, patient_replay_collector
from src.rl.candidate_imitation import ImitationSettings
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.cohort_backend import CohortPatientBackend
from src.rl.cohort_campaign import CohortCampaign
from src.rl.cohort_objective_plan import cohort_budget_plan, cohort_stream_manifest
from src.rl.cohort_public import CohortPrefixSession
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.utils.research_archive import inventory, verify_archive
from tests.test_cohort_public import ArrayLayout, fixture, kernel
from tests.test_dynamic_candidate_campaign import fixture_config
from tests.test_dynamic_candidate_session import FakeReference, OPTIONS, fake_info
from tests.test_dynamic_candidate_updates import fake_adam
from tests.test_time_baseline_campaign import proposed as base_proposal


def proposed():
    return json.loads((Path(__file__).resolve().parents[1] /
        "specs/2026-10-02-terminal-obligation/proposal.json").read_text())


def invented():
    base, p, cfg = base_proposal(), proposed(), fixture_config()
    streams, plan = cohort_stream_manifest(base, p), cohort_budget_plan(base, p)
    p.update(enrollment_steps=2, accounting_steps=3, patient_resolution_steps=2)
    cfg["cohort_proposal"] = p
    cfg["final_controllers"] = p["evaluation_controllers"]
    cfg["continuation"].update(episodes_per_rollout=2, rollouts_per_arm_per_block=1, rows_per_rollout=4)
    cfg["evaluation"].update(total_episodes=18, controllers_per_block=6, environment_steps=90)
    cfg["totals"].update(main_full_episodes=54, main_environment_steps=330)
    for row in streams["environment"].values():
        row["training"], row["test"] = row["training"][:2], row["test"][:1]
    for name, row in plan["sections"].items():
        row.update(trajectory=0, clone=0, actor=0, critic=0)
        if name == "same_start_preflight":
            row.update(trajectory=90, clone=60)
        elif name.split("/")[0] in p["training_arms"]:
            row.update(trajectory=10, actor=2, critic=0 if name.startswith("bc_continue/") else 2)
        elif name.startswith("final_evaluation/"):
            row.update(trajectory=5)
    for phase, row in plan["phases"].items():
        for k in ("trajectory", "clone", "actor", "critic"):
            row[k] = sum(s[k] for s in plan["sections"].values() if s["phase"] == phase)
    for k in ("trajectory", "clone", "actor", "critic"):
        plan["limits"][k] = sum(s[k] for s in plan["phases"].values())
    return base, p, cfg, streams, plan


class Backend:
    engineering_fixture = True

    def __init__(self, cfg, streams):
        self.config, self.streams = cfg, streams
        self.layouts, self.sessions, self.parities = {}, [], []

    def prepare(self, block, seed):
        self.layouts[block] = fixture()[3]
        return dict(invented=True)

    def producer(self, block):
        return self.layouts[block]

    def session(self, block, learner, seed, *, trajectory, split, selection):
        self.sessions.append((block, seed, split, trajectory))
        _, _, env, producer = fixture()
        env._episode_seed, env.cumulative_enrolled = seed, 0.
        return CohortPrefixSession(env, producer, FakeReference(producer), learner, enabled=True,
            options=OPTIONS, trajectory_id=trajectory, split=split, selection=selection,
            source_id=self.streams["namespace"], environment_seed=seed)

    def parity_environment(self, block, seed):
        env = fixture()[0]
        env._episode_seed, env.cumulative_enrolled = seed, 0.
        self.parities.append(block)
        return env

    def state_dict(self):
        return dict(layouts=sorted(self.layouts), sessions=copy.deepcopy(self.sessions), parities=list(self.parities))

    def assert_restore_compatible(self, saved):
        if saved != self.state_dict():
            raise ValueError("invented backend cannot rewind")


class Recorder:
    def __init__(self, root, relative, prefix, config, **metadata):
        self.path = Path(root) / relative / "invented.json"
        self.root, self.metadata = Path(root), metadata
        self.prefix = type("PrefixOwner", (), {})()
        self.prefix.session = prefix
        self.count, self.closed, self.failed = 0, False, False

    def record_prefix(self, session, event):
        self.prefix.session = session

    def finish_prefix(self, session):
        return dict(invented_prefix=session.index)

    def record_tail(self, collection, event):
        self.count += 1

    def finish(self, collection):
        write_json_once(self.path, self.metadata | dict(split=collection.split, steps=collection.index))
        self.closed = True
        return {"invented": file_record(self.root, self.path)}

    def snapshot(self):
        return dict(tail_rows=self.count, prefix_rows=self.prefix.session.index, closed=self.closed)

    def assert_prefix(self, saved):
        if self.snapshot() != saved:
            raise ValueError("persisted fake rows cannot rewind")

    def close_partial(self):
        self.failed = True


def reader(root, indexes, config):
    return [r["invented"] for r in indexes], [json.loads((Path(root) / r["invented"]["path"]).read_text()) for r in indexes]


def verifier(root, indexes, config, streams):
    files, outcomes = reader(root, indexes, config)
    expected = {(b, r, w, int(s)) for b in config["blocks"] for r in config["final_controllers"]
                for w, s in enumerate(streams["environment"][str(b)]["test"])}
    actual = {(r["block"], r["role"], r["world_index"], r["seed"]) for r in outcomes}
    if actual != expected or len(outcomes) != len(expected):
        raise ValueError("invented evaluation inventory differs")
    return dict(engineering_fixture_only=True, files=files, outcomes=outcomes, scientific_performance_claim=False)


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
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for owner, method in ((PatientConditionCapacityEnv, "__init__"), (PatientConditionCapacityEnv, "step"),
                              (torch.optim.Adam, "step"), (torch.optim.SGD, "step")):
            guard = patch.object(owner, method, side_effect=AssertionError("real science forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        for module in (cohort_public, patient_replay_collector):
            guard = patch.object(module, "PatientConditionCapacityEnv", ArrayLayout)
            guard.start()
            self.addCleanup(guard.stop)
        original_step = ArrayLayout.step
        def step(env, action):
            if not getattr(env, "_cohort_closed", False):
                action = action.copy()
                action.setflags(write=False)
                return original_step(env, action)
            env.actions.append(action.copy())
            env.t += 1
            env.raw[-1] = env.t / env.config.episode_horizon
            info = fake_info(action)
            return env.observation(), -info["cost"], env.t == 5, info
        guard = patch.object(ArrayLayout, "step", new=step)
        guard.start()
        self.addCleanup(guard.stop)
        self.base, self.p, self.cfg, self.streams, self.plan = invented()
        for module, name, value in (
                ("cohort_campaign", "cohort_config", self.cfg),
                ("cohort_campaign", "cohort_budget_plan", self.plan),
                ("cohort_campaign", "cohort_stream_manifest", self.streams),
                ("cohort_sequence", "cohort_budget_plan", self.plan)):
            guard = patch("src.rl." + module + "." + name, return_value=copy.deepcopy(value))
            guard.start()
            self.addCleanup(guard.stop)
        self.initializers, qualifications, self.loads = {}, {}, []
        producer = fixture()[3]
        for block in self.cfg["blocks"]:
            owner = DynamicCandidateImitationKernel(kernel(producer, "frozen").policy, producer.contract,
                ImitationSettings(.001, .5, 2, 2, 1, "demonstration"), enabled=True,
                sampling_seed=7, shuffle_seed=9)
            owner.steps = self.cfg["initialization"]["actor_adam_calls_per_block"]
            self.initializers[block] = owner
            qualifications[str(block)] = dict(passed=True, kernel_sha256=state_digest(owner.state_dict()))
        def loader(b):
            if b in self.loads:
                raise ValueError("duplicate initializer load")
            self.loads.append(b)
            return copy.deepcopy(self.initializers[b])
        self.backend = Backend(self.cfg, self.streams)
        self.budget = DynamicCandidateBudget(self.root / "launcher/budget.jsonl", self.plan, enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        self.run = CohortCampaign(self.root, self.cfg, self.base, self.p, self.streams, self.budget,
            self.backend, initializer_loader=loader, qualifications=qualifications, enabled=True,
            recorder=Recorder, reader=reader, verifier=verifier,
            target_verifier=lambda *args: dict(engineering_fixture_only=True))

    def until(self, target):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while self.run.sequence.next_job != target:
                self.run.begin_next()
                self.run.dispatch(self.run.sequence.active)

    def test_full_serial_fake_matrix_counts_parity_clones_and_archive(self):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            result = self.run.run()
        if result:
            self.fail((self.root / "launcher/failure.json").read_text())
        self.assertEqual(len(self.run.sequence.completed), 33)
        self.assertEqual(len(self.run.raw_index), 54)
        self.assertEqual(len(self.run.test_index), 18)
        self.assertEqual(len(self.backend.sessions), 54)
        self.assertEqual(self.backend.parities, [60, 61, 62])
        self.assertEqual(self.loads, [60, 61, 62])
        self.assertEqual(read_dynamic_ledger(self.budget.path)["counts"], dict(environment=330, optimizer=30))
        self.assertEqual(len(self.run.sequence.seals), 12)
        self.assertEqual(inventory(self.root / "payload"), self.run.payload_seal)
        verify_archive(self.root / "archives/completed-payload.tar.gz", self.run.payload_seal)
        for b in self.cfg["blocks"]:
            for role in self.cfg["final_controllers"][:4]:
                self.assertEqual(self.initializers[b].policy.snapshot_sha256(),
                    self.run.models[f"block{b}/graph/{role}"].policy.snapshot_sha256())

    def test_owned_restore_prefix_tail_and_update_no_rebuild_or_refund(self):
        self.until("cohort_ppo/block60")
        self.run.begin_next()
        for _ in range(10):
            self.run.advance()
            saved = self.run.checkpoint_state()
            counts, builds = copy.deepcopy(self.budget.counts), len(self.backend.sessions)
            self.run.restore_checkpoint_state(saved)
            self.assertEqual(state_digest(saved), state_digest(self.run.checkpoint_state()))
            self.assertEqual(counts, self.budget.counts)
            self.assertEqual(builds, len(self.backend.sessions))
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            self.run.advance()

    def test_no_test_before_seals_and_failure_no_retry(self):
        self.until("same_start_preflight")
        with self.assertRaises(ValueError):
            self.run._session(self.run._descriptors("final_evaluation/block60/cohort_ppo")[0])
        self.until("window_ppo/block60")
        self.run.begin_next()
        while self.run.continuation is None or not self.run.continuation.update_due:
            self.run.advance()
        saved, before = self.run.checkpoint_state(), self.budget.counts["optimizer"]
        with patch.object(torch.optim.Adam, "step", side_effect=RuntimeError("invented update failure")):
            with self.assertRaises(RuntimeError):
                self.run.advance()
        self.assertEqual(self.budget.counts["optimizer"], before + 1)
        self.assertTrue((self.root / "launcher/failure.json").exists())
        with self.assertRaises(ValueError):
            self.run.restore_checkpoint_state(saved)

    def test_real_backend_default_denies_before_any_load_or_build(self):
        backend = CohortPatientBackend(self.root, self.cfg, self.streams)
        for call in (lambda: backend.prepare(60, 0),
                     lambda: backend.session(60, None, 0, trajectory="x", split="test", selection="greedy"),
                     lambda: backend.parity_environment(60, 0)):
            with self.assertRaises(PermissionError):
                call()
        self.assertEqual(backend.episode_builds, 0)
        self.assertEqual(backend.parity_blocks, [])
