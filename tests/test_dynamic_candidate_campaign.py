"""Entire fixed dispatcher on fake arrays and metadata-only optimizer doubles.

The injected raw-reader fixture is intentionally not the patient-identity
verifier. That verifier has its own raw scalar/identity tests. Here we exercise
phase ownership, sampling/update wiring, immutable test barriers and archiving.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.candidate_patient_session import save_envelope, load_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import EpisodeRecorder, write_json_once, normalized
from src.rl.dynamic_candidate_campaign import DynamicCandidateCampaign, ROLES
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, dynamic_budget_plan, dynamic_stream_manifest, read_dynamic_ledger
from src.rl.dynamic_candidate_session import DynamicCandidateSession
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.utils.research_archive import inventory, verify_archive
from tests.test_dynamic_candidate_factory import proposal
from tests.test_dynamic_candidate_session import FakeEnv, FakeProducer, FakeReference, OPTIONS
from tests.test_dynamic_candidate_updates import fake_adam


def fixture_config():
    cfg = proposal()
    cfg["model_proposal"].update(encoder_width=3, head_width=5, initial_reference_bias=10.)
    cfg["objective"].update(horizon=2, num_facilities=2, action_width=8, raw_state_width=9,
                            reward_scale=.01, transfer_scale=4.)
    cfg["candidate_support"] = {"options": OPTIONS, "max_original_requests": 4}
    cfg["initialization"].update(demonstration_episodes_per_block=1, batch_size=2, actor_adam_calls_per_block=1)
    cfg["qualification"]["fresh_worlds_per_block"] = 1
    cfg["continuation"].update(episodes_per_arm_per_block=2, episodes_per_rollout=1,
                               rollouts_per_arm_per_block=2, epochs=1, batch_size=2, rows_per_rollout=2)
    cfg["continuation"]["ppo"].update(actor_adam_calls_per_block=2, critic_adam_calls_per_block=2)
    cfg["continuation"]["bc_continue"]["actor_adam_calls_per_block"] = 2
    cfg["evaluation"].update(fresh_paired_worlds_per_block=1, total_worlds=3, total_episodes=15,
                             environment_steps=30, environment_steps_per_controller_per_block=2)
    cfg["evaluation"]["bootstrap"].update(draws=10, worlds_per_sampled_block=1)
    # All resource partitions are deliberately tiny and still exact.
    allocations = {
        "runtime_input_binding": (0, 0, 0, 0), "prototype_preflight": (6, 3, 0, 0),
        "initialization_collection": (6, 0, 0, 0), "initialization_fit": (0, 0, 3, 0),
        "qualification": (12, 0, 0, 0), "same_start_preflight": (30, 15, 0, 0),
        "ppo_continuation": (12, 0, 6, 6), "bc_continuation": (12, 0, 6, 0),
        "all_model_seal": (0, 0, 0, 0), "final_evaluation": (30, 0, 0, 0),
        "saved_data_verification_analysis": (0, 0, 0, 0),
        "local_archive_verification": (0, 0, 0, 0), "supervisor_dispatch_terminal_closure": (0, 0, 0, 0)}
    for row in cfg["phase_budgets"]:
        trajectory, clone, actor, critic = allocations[row["id"]]
        row.update(trajectory_steps=trajectory, mandatory_clone_steps=clone,
                   actor_adam_calls=actor, critic_adam_calls=critic)
    owners = cfg["owner_budgets"]
    owners["initialization_fit_each_block"].update(actor_adam_calls=1)
    owners["ppo_each_block"].update(environment_steps=4, actor_adam_calls=2, critic_adam_calls=2)
    owners["bc_each_block"].update(environment_steps=4, actor_adam_calls=2)
    owners["evaluation_each_controller_each_block"]["environment_steps"] = 2
    cfg["totals"].update(main_trajectory_steps=108, mandatory_restore_clone_steps=18,
                         main_environment_steps=126, main_actor_adam_calls=15, main_critic_adam_calls=6,
                         main_optimizer_calls=21, main_full_episodes=54, unique_world_start_allocations=21)
    count = 0
    for b in cfg["blocks"]:
        ranges = cfg["rng_proposal"]["block_ordinal_ranges"][str(b)]
        for role, n in (("prototype_preflight", 1), ("fork_preflight", 1), ("demonstration", 1),
                        ("qualification", 1), ("training", 2), ("test", 1)):
            ranges[role] = [count, count + n - 1]
            count += n
    cfg["rng_proposal"]["namespace"] = "invented-dynamic-dispatch-no-science-v1"
    return cfg


class FakeBackend:
    engineering_fixture = True

    def __init__(self, config):
        self.config, self.contexts, self.sessions = config, {}, []

    def prepare(self, block, seed):
        if block in self.contexts:
            raise AssertionError("duplicate preparation")
        self.contexts[block] = FakeProducer()
        return {"invented_fixture": True, "block": block, "seed": seed}

    def producer(self, block):
        return self.contexts[block]

    def state_dict(self):
        return {"prepared_blocks": sorted(self.contexts), "episode_builds": len(self.sessions)}

    def assert_restore_compatible(self, saved):
        if (saved["prepared_blocks"] != sorted(self.contexts)
                or saved["episode_builds"] > len(self.sessions)):
            raise ValueError("fixture construction ownership changed")

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        self.sessions.append((block, seed, split, selection, trajectory))
        producer = self.producer(block)
        return DynamicCandidateSession(FakeEnv(), producer, FakeReference(producer), kernel, enabled=True,
            options=OPTIONS, trajectory_id=trajectory, split=split, selection=selection,
            source_id="invented-dynamic-dispatch-no-science-v1")


class FakeRecorder(EpisodeRecorder):
    """Persist the fake records, without pretending they are patient registries."""
    def finish(self):
        if self.failed or self.closed or not self.session.closed or self.count != 2:
            raise ValueError("incomplete fixture")
        self.handle.close()
        write_json_once(self.directory / "final_state.json", normalized(self.session.env.state_dict()))
        self.session.save(self.directory / "collector.pt")
        self.closed = True
        return {key: file_record(self.root, self.directory / name) for key, name in
                (("header", "header.json"), ("events", "events.jsonl"), ("final_state", "final_state.json"))}


def fake_reader(root, indexes, config):
    files, outcomes = [], []
    for entry in indexes:
        for record in entry.values():
            if file_record(root, record["path"]) != record:
                raise ValueError("fake raw hash mismatch")
            files.append(record)
        h = json.loads((root / entry["header"]["path"]).read_text())
        rows = [json.loads(s) for s in (root / entry["events"]["path"]).read_text().splitlines()]
        if len(rows) != config["objective"]["horizon"]:
            raise ValueError("fake raw row count")
        outcomes.append({k: h[k] for k in ("block", "role", "world_index", "seed")} | {
            "cost": sum(r["event"]["info"]["cost"] for r in rows),
            "losses": sum(sum(r["event"]["info"]["patients_lost"]) for r in rows),
            "completions": sum(sum(r["event"]["info"]["patients_completed"]) for r in rows),
            "terminal_active": rows[-1]["event"]["info"]["identity_active_count"]})
    return files, outcomes


def fake_verifier(root, indexes, config, streams):
    files, outcomes = fake_reader(root, indexes, config)
    expected = {(b, role, w, seed) for b in config["blocks"] for role in ROLES
                for w, seed in enumerate(streams["environment"][str(b)]["test"])}
    actual = {(r["block"], r["role"], r["world_index"], r["seed"]) for r in outcomes}
    if len(outcomes) != len(expected) or actual != expected:
        raise ValueError("fake evaluation schedule mismatch")
    return {"engineering_fixture_only": True, "outcomes": outcomes, "files": files,
            "scientific_performance_claim": False}


@unittest.skipIf(torch is None, "torch unavailable")
class DynamicCandidateCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def build(self, *, reader=fake_reader, bias=10.):
        self.serial += 1
        root = self.root / f"case{self.serial}"
        cfg = fixture_config()
        cfg["model_proposal"]["initial_reference_bias"] = bias
        streams = dynamic_stream_manifest(cfg)
        budget = DynamicCandidateBudget(root / "launcher/budget.jsonl", dynamic_budget_plan(cfg),
                                        enabled=True, clock=lambda: 0.)
        self.addCleanup(budget.close)
        backend = FakeBackend(cfg)
        return DynamicCandidateCampaign(root, cfg, streams, budget, backend, enabled=True,
            recorder=FakeRecorder, reader=reader, verifier=fake_verifier)

    def until(self, campaign, job):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while campaign.sequence.next_job != job:
                campaign.begin_next()
                campaign.dispatch(campaign.sequence.active)

    def test_complete_serial_dispatch_exact_counts_and_local_archive(self):
        run = self.build()
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            code = run.run()
        if code:
            self.fail((run.root / "launcher/failure.json").read_text())
        self.assertEqual(len(run.sequence.completed), 33)
        self.assertEqual(len(run.test_index), 15)
        self.assertEqual(len(run.raw_index), 54)
        self.assertEqual(len(run.backend.sessions), 54)
        self.assertEqual(len(run.sequence.seals), 9)
        checked = read_dynamic_ledger(run.budget.path)
        self.assertEqual(checked["counts"], {"environment": 126, "optimizer": 21})
        self.assertEqual(checked["owner_counts"]["global:clone"], 18)
        self.assertEqual(checked["owner_counts"]["global:actor"], 15)
        self.assertEqual(checked["owner_counts"]["global:critic"], 6)
        self.assertEqual(inventory(run.root / "payload"), run.payload_seal)
        verify_archive(run.root / "archives/completed-payload.tar.gz", run.payload_seal)
        receipt = json.loads((run.root / "launcher/archive-receipt.json").read_text())
        self.assertFalse(receipt["dropbox_exported"])
        self.assertFalse(receipt["cloud_sync_verified"])
        self.assertFalse(receipt["howard_access_verified"])
        for b in run.config["blocks"]:
            initial = load_envelope(run.root / f"payload/models/block{b}/graph/own_ppo/initial.pt")
            final = load_envelope(run.root / f"payload/models/block{b}/graph/own_ppo/final.pt")
            self.assertEqual(state_digest(initial["policy"]), state_digest(final["policy"]))

    def test_no_real_backend_or_scientific_mode_admission(self):
        run = self.build()
        args = (run.root, run.config, run.streams, run.budget, run.backend)
        with self.assertRaises(ValueError):
            DynamicCandidateCampaign(*args)
        with self.assertRaises(PermissionError):
            DynamicCandidateCampaign(*args, enabled=True, engineering_only=False)
        run.backend.engineering_fixture = False
        with self.assertRaises(ValueError):
            DynamicCandidateCampaign(*args, enabled=True)

    def test_qualification_failure_preserved_before_any_continuation_or_test(self):
        def adverse(root, index, cfg):
            files, rows = fake_reader(root, index, cfg)
            for row in rows:
                if row["role"] == "initializer_greedy":
                    row["losses"] += 1
            return files, rows
        run = self.build(reader=adverse)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            self.assertEqual(run.run(), 1)
        self.assertEqual(run.failure["error_type"], "ValueError")
        self.assertIn("qualification failed", run.failure["message"])
        self.assertFalse(any(s[2] in ("training", "test") for s in run.backend.sessions))
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["optimizer"], 3)
        self.assertTrue((run.root / "payload/qualification.json").exists())
        self.assertTrue((run.root / "launcher/failure-state.pt").exists())
        with self.assertRaises(ValueError):
            run.begin_next()

    def test_test_data_forbidden_until_all_nine_model_files_sealed(self):
        run = self.build()
        self.until(run, "all_model_seal")
        self.assertFalse(any(s[2] == "test" for s in run.backend.sessions))
        descriptor = run._descriptors("final_evaluation/block60/own_frozen")[0]
        with self.assertRaises(ValueError):
            run._session(descriptor)
        run.begin_next()
        run.advance()
        model = run.root / next(iter(run.model_paths.values()))
        with model.open("ab") as handle:
            handle.write(b"invented-corruption")
        with self.assertRaises(ValueError):
            run.begin_next()
        self.assertFalse(any(s[2] == "test" for s in run.backend.sessions))
        self.assertIsNotNone(run.failure)

    def test_collector_boundary_restore_keeps_live_ledger_and_raw_prefix(self):
        run = self.build()
        self.until(run, "prototype_preflight")
        run.begin_next()
        run.advance()
        saved = run.checkpoint_state()
        before = run.budget.snapshot()
        run.restore_checkpoint_state(saved)
        self.assertEqual(run.budget.snapshot(), before)
        self.assertEqual(run.live_session.index, 1)
        run.advance()
        with self.assertRaises(ValueError):
            run.restore_checkpoint_state(saved)
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["environment"], 3)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            self.assertEqual(run.run(), 0)

    def test_training_scope_and_update_restore_keep_correct_model_owner(self):
        run = self.build()
        self.until(run, "ppo_continuation/block60")
        run.begin_next()
        run.advance()
        saved = run.checkpoint_state()
        builds = len(run.backend.sessions)
        run.restore_checkpoint_state(saved)
        self.assertEqual(len(run.backend.sessions), builds)
        self.assertEqual(run.continuation.scope, "ppo_continuation/block60")
        self.assertIs(run.models["block60/graph/own_ppo"], run.continuation.kernel)
        self.assertIs(run.recorder.session, run.continuation.active)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            run.advance()
            run.advance()
            at_update = run.checkpoint_state()
            spent = copy.deepcopy(run.budget.owner_counts)
            builds = len(run.backend.sessions)
            run.restore_checkpoint_state(at_update)
            self.assertEqual(len(run.backend.sessions), builds)
            self.assertEqual(run.budget.owner_counts, spent)
            self.assertIs(run.models["block60/graph/own_ppo"], run.continuation.kernel)
            self.assertEqual(run.run(), 0)

    def test_debited_collection_failure_is_terminal_and_partial_bytes_survive(self):
        run = self.build()
        self.until(run, "prototype_preflight")
        run.begin_next()
        saved = run.checkpoint_state()
        with patch.object(FakeEnv, "step", side_effect=RuntimeError("invented transition fault")):
            self.assertEqual(run.run(), 1)
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["environment"], 1)
        self.assertTrue((run.root / "launcher/failure-state.pt").exists())
        events = list((run.root / "payload/episodes").rglob("events.jsonl"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].read_text(), "")
        with self.assertRaises(ValueError):
            run.restore_checkpoint_state(saved)

    def test_partial_optimizer_failure_records_debit_without_publishing_model(self):
        run = self.build()
        self.until(run, "ppo_continuation/block60")
        run.begin_next()
        run.advance()
        run.advance()
        key = "block60/graph/own_ppo"
        before = run.models[key].policy.snapshot_sha256()
        calls = []
        def fail_second(owner, *args, **kwargs):
            calls.append(owner)
            if len(calls) == 2:
                raise KeyboardInterrupt("invented optimizer interruption")
            return fake_adam(owner, *args, **kwargs)
        with patch.object(torch.optim.Adam, "step", new=fail_second):
            self.assertEqual(run.run(), 1)
        self.assertEqual(run.models[key].policy.snapshot_sha256(), before)
        ledger = read_dynamic_ledger(run.budget.path)
        self.assertEqual(ledger["owner_counts"]["section/ppo_continuation/block60:actor"], 1)
        self.assertEqual(ledger["owner_counts"]["section/ppo_continuation/block60:critic"], 1)
        failure = json.loads((run.root / "launcher/failure.json").read_text())
        self.assertEqual(failure["error"]["error_type"], "KeyboardInterrupt")
        self.assertFalse(failure["retry_permitted"])

    def test_direct_incremental_failure_latches_without_run_wrapper(self):
        run = self.build()
        self.until(run, "prototype_preflight")
        run.begin_next()
        with patch.object(FakeEnv, "step", side_effect=RuntimeError("invented direct failure")):
            with self.assertRaises(RuntimeError):
                run.advance()
        self.assertIsNotNone(run.failure)
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["environment"], 1)
        with self.assertRaises(ValueError):
            run.advance()

    def test_raw_corruption_is_not_archived_as_verified(self):
        run = self.build()
        self.until(run, "saved_data_verification_analysis")
        record = run.test_index[0]["events"]
        with (run.root / record["path"]).open("ab") as handle:
            handle.write(b"invented corruption")
        run.begin_next()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            run.advance()
        self.assertIsNotNone(run.failure)
        self.assertFalse((run.root / "archives/completed-payload.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
