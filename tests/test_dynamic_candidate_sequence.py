"""Full phase-cursor fixture; artifact bytes are fake and no numerical work runs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.rl.dynamic_candidate_resources import dynamic_budget_plan
from src.rl.dynamic_candidate_sequence import DynamicPilotSequence, dynamic_jobs, dynamic_required_models


class DynamicSequenceTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.draft = json.loads((root / "specs/2026-10-01-adaptive-paper-delivery/pilot-budget-draft.json").read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.receipt = self.root / "fake.json"
        self.receipt.write_text('{"synthetic_fixture": true}')
        self.sequence = self.new()

    def new(self):
        return DynamicPilotSequence(self.draft, self.root, enabled=True)

    def advance_to_barrier(self):
        while self.sequence.next_job != "all_model_seal":
            self.sequence.begin(self.sequence.next_job)
            self.sequence.finish([self.receipt])
        self.sequence.begin("all_model_seal")

    def fake_models(self):
        result = {}
        for index, key in enumerate(dynamic_required_models(self.draft)):
            path = self.root / f"model{index}.fake"
            path.write_bytes(b"not a checkpoint; no inference")
            result[key] = path
        return result

    def test_manifest_exhausts_budget_in_phase_order(self):
        jobs = dynamic_jobs(self.draft)
        self.assertEqual(len(jobs), 33)
        self.assertEqual(set(jobs), set(dynamic_budget_plan(self.draft)["sections"]))
        self.assertLess(jobs.index("qualification"), jobs.index("same_start_preflight"))
        self.assertLess(jobs.index("ppo_continuation/block62"), jobs.index("bc_continuation/block60"))
        self.assertLess(jobs.index("bc_continuation/block62"), jobs.index("all_model_seal"))
        self.assertEqual(len(dynamic_required_models(self.draft)), 9)

    def test_full_cursor_mock_chain_and_every_boundary_restore(self):
        for job in self.sequence.jobs:
            self.sequence.begin(job)
            restored = self.new()
            restored.load_state_dict(self.sequence.state_dict())
            self.assertEqual(restored.state_dict(), self.sequence.state_dict())
            if job == "all_model_seal":
                self.sequence.seal_models(self.fake_models())
            self.sequence.finish([self.receipt])
            restored.load_state_dict(self.sequence.state_dict())
            self.assertEqual(restored.state_dict(), self.sequence.state_dict())
        self.assertIsNone(self.sequence.next_job)
        self.assertEqual(len(self.sequence.completed), 33)

    def test_test_entry_before_all_models_forbidden(self):
        with self.assertRaises(ValueError):
            self.sequence.begin("final_evaluation/block60/own_frozen")
        self.advance_to_barrier()
        with self.assertRaises(ValueError):
            self.sequence.finish([self.receipt])
        paths = self.fake_models()
        with self.assertRaises(ValueError):
            self.sequence.seal_models(dict(list(paths.items())[:-1]))
        self.sequence.seal_models(paths)
        restored = self.new()
        restored.load_state_dict(self.sequence.state_dict())
        self.assertEqual(restored.seals, self.sequence.seals)
        self.sequence.finish([self.receipt])
        self.sequence.begin(self.sequence.next_job)

    def test_sealed_mutation_or_path_alias_refused(self):
        self.advance_to_barrier()
        paths = self.fake_models()
        alias = {key: next(iter(paths.values())) for key in paths}
        with self.assertRaises(ValueError):
            self.sequence.seal_models(alias)
        self.sequence.seal_models(paths)
        self.sequence.finish([self.receipt])
        next(iter(paths.values())).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.sequence.begin(self.sequence.next_job)
        with self.assertRaises(ValueError):
            self.new().load_state_dict(self.sequence.state_dict())

    def test_failure_latches_and_cannot_restore_to_retry(self):
        before = self.sequence.state_dict()
        self.sequence.begin(self.sequence.next_job)
        self.sequence.fail("fake terminal error")
        with self.assertRaises(ValueError):
            self.sequence.load_state_dict(before)
        restored = self.new()
        restored.load_state_dict(self.sequence.state_dict())
        self.assertIsNone(restored.next_job)
        with self.assertRaises(ValueError):
            restored.begin("runtime_input_binding")
        with self.assertRaises(ValueError):
            restored.finish([self.receipt])

    def test_no_opt_in_duplicate_scope_or_evidence(self):
        with self.assertRaises(ValueError):
            DynamicPilotSequence(self.draft, self.root)
        self.sequence.begin(self.sequence.next_job)
        with self.assertRaises(ValueError):
            self.sequence.begin(self.sequence.next_job)
        with self.assertRaises(ValueError):
            self.sequence.finish([self.receipt, self.receipt])
        self.sequence.finish([self.receipt])
        with self.assertRaises(ValueError):
            self.sequence.begin("runtime_input_binding")

    def test_tampered_snapshot_or_completed_artifact_rejected_atomically(self):
        self.sequence.begin(self.sequence.next_job)
        self.sequence.finish([self.receipt])
        saved = self.sequence.state_dict()
        target = self.new()
        before = target.state_dict()
        bad = copy.deepcopy(saved)
        bad["completed"][0]["job"] = "qualification"
        with self.assertRaises(ValueError):
            target.load_state_dict(bad)
        self.assertEqual(target.state_dict(), before)
        self.receipt.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            target.load_state_dict(saved)
        self.assertEqual(target.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
