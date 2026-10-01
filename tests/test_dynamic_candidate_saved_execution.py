"""Permission, fake-loader, irreversible budget and saved-only execution tests."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from src.rl import dynamic_candidate_saved_execution as execution
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.candidate_pilot_watchdog import supervise
from src.rl.prospective_ddpg_kernel import state_digest


class SavedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("optimizer forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def budget(self, seconds=900):
        clock = Mock(return_value=100.)
        budget = execution.ScoringBudget(self.root / "debits.jsonl", started=100., seconds=seconds, clock=clock)
        self.addCleanup(budget.close)
        return budget, clock

    def test_missing_approval_never_claims_or_loads(self):
        with patch.object(execution, "committed_json", side_effect=PermissionError("not approved")), \
                patch.object(torch, "load") as load, patch.object(execution, "supervise") as launch:
            with self.assertRaises(PermissionError):
                execution.launch(self.root)
        load.assert_not_called()
        launch.assert_not_called()
        self.assertFalse(list(self.root.iterdir()))

    def approved_pair(self):
        packet = {"scientific_execution_authorized": False, "ready_to_launch": False,
                  "implementation_commit": "a" * 40, "proposal": {"sha256": "b" * 64},
                  "limits": {"attempts": 1, "checkpoint_loads": 4, "saved_state_scorings": 624}}
        packet["packet_sha256"] = digest(packet)
        authorization = {"format": "saved-qualification-explicit-authorization-v1", "approved": True,
            "packet_sha256": packet["packet_sha256"], "implementation_commit": packet["implementation_commit"],
            "proposal_sha256": packet["proposal"]["sha256"], "limits": packet["limits"],
            "automatic_retry": False, "training_after_qualification": False,
            "user_approval": {"user": "Zhaowei", "verbatim": "SYNTHETIC TEST APPROVAL ONLY", "recorded_at_utc": "fixture"}}
        return packet, authorization

    def test_authorization_is_separate_and_exact_not_a_frozen_flag(self):
        packet, authorization = self.approved_pair()
        execution.validate_authorization(authorization, packet)
        for field, value in (("approved", False), ("packet_sha256", "c" * 64),
                             ("automatic_retry", True), ("training_after_qualification", True),
                             ("limits", {}), ("user_approval", {})):
            invalid = dict(authorization, **{field: value})
            with self.subTest(field=field), self.assertRaises(PermissionError):
                execution.validate_authorization(invalid, packet)
        packet["limits"]["attempts"] = 2
        with self.assertRaises(ValueError):
            execution.validate_authorization(authorization, packet)

    def test_624_scoring_and_four_load_caps_have_durable_nonrefundable_chain(self):
        budget, _ = self.budget()
        for _ in range(4):
            budget.debit("checkpoint_load")
        for _ in range(624):
            budget.debit("scoring")
        for operation in ("scoring", "checkpoint_load", "optimizer", "environment"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                budget.debit(operation)
        rows = [json.loads(line) for line in (self.root / "debits.jsonl").read_text().splitlines()]
        previous = "0" * 64
        for sequence, row in enumerate(rows):
            sha = row.pop("sha256")
            self.assertEqual(row["sequence"], sequence)
            self.assertEqual(row["previous"], previous)
            self.assertEqual(digest(row), sha)
            previous = sha
        self.assertEqual(len(rows), 628)
        self.assertEqual(budget.counts, budget.limits)
        with self.assertRaises(FileExistsError):
            execution.ScoringBudget(self.root / "debits.jsonl", started=100., seconds=900)

    def test_deadline_before_load_prevents_parser(self):
        budget, clock = self.budget()
        clock.return_value = 1000.
        loader = Mock()
        with self.assertRaises(TimeoutError):
            execution.load_envelope(self.root / "absent.pt", "a" * 64, budget, loader=loader)
        loader.assert_not_called()
        self.assertEqual(budget.counts["checkpoint_load"], 0)

    def fake_file(self):
        path = self.root / "invented.pt"
        path.write_bytes(b"not-a-research-checkpoint")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_fake_loader_uses_exact_hashed_bytes_cpu_weights_only_after_debit(self):
        path, sha = self.fake_file()
        budget, _ = self.budget()
        state = {"invented": torch.tensor([1., 2.])}
        def loader(handle, **kwargs):
            self.assertEqual(budget.counts["checkpoint_load"], 1)
            self.assertEqual(handle.read(), b"not-a-research-checkpoint")
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            return {"state": state, "sha256": state_digest(state)}
        with patch.object(torch, "load", side_effect=AssertionError("real parser forbidden")):
            loaded = execution.load_envelope(path, sha, budget, loader=loader)
        self.assertEqual(state_digest(loaded), state_digest(state))

    def test_wrong_hash_prevents_debit_and_parser(self):
        path, _ = self.fake_file()
        budget, _ = self.budget()
        loader = Mock()
        with self.assertRaises(ValueError):
            execution.load_envelope(path, "0" * 64, budget, loader=loader)
        loader.assert_not_called()
        self.assertEqual(budget.counts["checkpoint_load"], 0)

    def test_bad_envelope_and_parser_failure_consume_load_without_retry(self):
        path, sha = self.fake_file()
        budget, _ = self.budget()
        bad = Mock(return_value={"state": {}, "sha256": "0" * 64})
        with self.assertRaises(ValueError):
            execution.load_envelope(path, sha, budget, loader=bad)
        bad.assert_called_once()
        failure = Mock(side_effect=RuntimeError("invented parser failure"))
        with self.assertRaises(RuntimeError):
            execution.load_envelope(path, sha, budget, loader=failure)
        failure.assert_called_once()
        self.assertEqual(budget.counts["checkpoint_load"], 2)

    def test_forward_is_not_allowed_when_debit_hits_exact_deadline(self):
        budget, clock = self.budget()
        clock.side_effect = [999., 999., 1000.]
        with self.assertRaises(TimeoutError):
            budget.debit("scoring")
        self.assertEqual(budget.counts["scoring"], 1)
        self.assertEqual(len((self.root / "debits.jsonl").read_text().splitlines()), 1)

    def test_owned_outer_watchdog_stops_and_reaps_stalled_fake_child(self):
        import sys
        result = supervise([sys.executable, "-c", "import time; time.sleep(20)"], cwd=self.root,
            stdout_path=self.root / "out", stderr_path=self.root / "err", ledger_path=self.root / "unused",
            report_path=self.root / "supervisor.json", maximum_seconds=.15, poll_seconds=.02,
            termination_grace_seconds=.1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.assertIsNotNone(result["exit_code"])
        self.assertFalse(result["automatic_retry"])

    def test_completion_readback_reconciles_chain_paths_and_zero_call_claims(self):
        budget, _ = self.budget()
        for kind, count in (("checkpoint_load", 4), ("scoring", 624)):
            for _ in range(count):
                budget.debit(kind)
        packet = {"packet_sha256": "a" * 64, "input_files": {}}
        result = {"counts": budget.counts, "last_debit_sha256": budget.previous,
            "packet_sha256": packet["packet_sha256"], "input_files_before": {}, "input_files_after": {},
            "source_unchanged": True, "automatic_training_authorized": False,
            "environment_calls": 0, "optimizer_updates": 0, "new_rollouts": 0, "final_test_episodes": 0,
            "blocks": {}}
        for block in ("60", "61", "62"):
            result["blocks"][block] = {"paths": {}}
            for role in ("r4", "initializer_greedy"):
                path = {"rows": 104, "passed": False}
                result["blocks"][block]["paths"][role] = path
                write_json_once(self.root / f"paths/block{block}-{role}.json", path)
        write_json_once(self.root / "qualification.json", result)
        self.assertEqual(execution.verify_saved_output(self.root, packet, 100.), result)
        with (self.root / "debits.jsonl").open("a") as handle:
            handle.write('{}\n')
        with self.assertRaises((KeyError, ValueError)):
            execution.verify_saved_output(self.root, packet, 100.)

    def test_prepare_binds_without_loading_and_remains_unapproved(self):
        proposal = {"limits": {"attempts": 1}}
        def git(_root, *args):
            return execution.BRANCH if args == ("branch", "--show-current") else "" if args == ("status", "--porcelain") else "a" * 40
        with patch.object(execution, "git", side_effect=git), \
                patch.object(execution, "committed_json", return_value=proposal), \
                patch.object(execution, "input_bindings", return_value=({}, {"scientific_config": {}, "streams": {}}, [])), \
                patch.object(execution, "source_locks", return_value={}), \
                patch.object(execution, "runtime_record", return_value={}), \
                patch.object(execution, "file_record", return_value={"sha256": "b" * 64}), \
                patch.object(torch, "load") as load:
            packet = execution.freeze(self.root)
        load.assert_not_called()
        self.assertFalse(packet["scientific_execution_authorized"])
        self.assertFalse(packet["ready_to_launch"])
        sha = packet.pop("packet_sha256")
        self.assertEqual(digest(packet), sha)

    def test_fake_saved_run_uses_four_inputs_and_stops_after_exact_scoring(self):
        source = self.root / "source"
        source.mkdir()
        output = self.root / "output"
        output.mkdir()
        proposal = {"source_root": "source", "new_output_root": "output", "inputs": {}, "limits": {"blocks": [60, 61, 62]}}
        for name in (execution.BOUNDARY, *(f"payload/initialization/block{b}/graph/final.pt" for b in (60, 61, 62))):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            proposal["inputs"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        states = {}
        loader = Mock(side_effect=lambda handle, **kwargs: {"state": states, "sha256": state_digest(states)})
        packet = {"packet_sha256": "a" * 64, "input_files": {}}
        original = {"scientific_config": {}, "streams": {}}
        budget, _ = self.budget()
        calls = []
        def score(owners, examples, outcomes, config, *, before_forward, on_path):
            self.assertEqual(set(owners), {60, 61, 62})
            for block in (60, 61, 62):
                for role in ("r4", "initializer_greedy"):
                    for _ in range(104):
                        before_forward()
                        calls.append("invented score")
                    on_path(block, role, {"passed": True})
            return {"passed": True, "automatic_training_authorized": False}
        with patch.object(execution, "verify_bindings", return_value=(original, [])), \
                patch("src.rl.dynamic_candidate_saved_qualification.restore_saved_policy", return_value="fake view") as restore, \
                patch("src.rl.dynamic_candidate_saved_qualification.validate_saved_boundary", return_value={}) as validate, \
                patch("src.rl.dynamic_candidate_saved_qualification.score_saved_qualification", side_effect=score), \
                patch.object(torch, "load", side_effect=AssertionError("real parser forbidden")):
            result = execution.run_saved(self.root, packet, proposal, budget, loader=loader)
        self.assertEqual(loader.call_count, 4)
        self.assertEqual(restore.call_count, 3)
        validate.assert_called_once()
        self.assertEqual(len(calls), 624)
        self.assertEqual(len(list((output / "paths").glob("*.json"))), 6)
        self.assertTrue(result["source_unchanged"])
        self.assertEqual(result["environment_calls"], 0)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertFalse(result["automatic_training_authorized"])

    def test_child_preserves_first_failure_without_reinvocation(self):
        packet, authorization = self.approved_pair()
        proposal = {"new_output_root": "output"}
        output = self.root / "output"
        output.mkdir()
        import os
        from src.utils.research_clock import CLOCK_ID, shared_monotonic
        write_json_once(output / "claim.json", {"pid": os.getppid(), "head": "fake-head",
            "packet_sha256": packet["packet_sha256"], "authorization_sha256": digest(authorization),
            "clock_id": CLOCK_ID, "started": shared_monotonic()})
        with patch.object(execution, "approved", return_value=(packet, proposal, authorization)), \
                patch.object(execution, "git", return_value="fake-head"), \
                patch.object(execution, "run_saved", side_effect=RuntimeError("invented first error")) as run:
            self.assertEqual(execution.child(self.root), 1)
            run.assert_called_once()
        failure = json.loads((output / "failure.json").read_text())
        self.assertIn("invented first error", failure["error"])
        self.assertFalse(failure["automatic_retry"])
        self.assertFalse((output / "qualification.json").exists())


if __name__ == "__main__":
    unittest.main()
