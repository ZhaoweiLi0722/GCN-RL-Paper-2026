"""Remaining-only branch dispatch and immutable import on authored fixtures."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.paired_cohort_recovery_data import import_saved_data
from src.rl.paired_cohort_recovery_training import collect_remaining
from src.rl.paired_cohort_resources import budget_plan, stream_manifest, runtime_streams
from src.rl.paired_cohort_training import branch_verifier
from src.rl.prospective_patient_session import encode_arrays
from tests.test_paired_cohort_training import RawBranchFixture
from tests.test_paired_cohort_verification import make_context, branch, config


class RemainingCollectionTests(unittest.TestCase):
    def setUp(self):
        for cls in (torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW):
            guard = patch.object(cls, "step", side_effect=AssertionError("real update forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cfg = config()
        self.streams = runtime_streams(stream_manifest(self.cfg, {str(b): str(b) for b in self.cfg["blocks"]}))
        for field, function in (("context_seed", lambda b, c: self.streams["environment"][str(b)]["context"][c]),
                                ("future_seed", lambda b, c, t, r: self.streams["conditional_future"][f"block{b}/cohort{c}/after{t}"][r])):
            guard = patch("tests.test_paired_cohort_verification." + field, function)
            guard.start()
            self.addCleanup(guard.stop)
        self.context = make_context()
        self.budget = DynamicCandidateBudget(self.root / "ledger.jsonl", budget_plan(self.cfg), enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        self.budget.begin("paired_branches/block60")
        self.calls, self.created = [], []
        self.row = dict(block=60, cohort=0, after_prefix_steps=4, replication=0, candidate_index=1,
            future_seed=str(self.streams["conditional_future"]["block60/cohort0/after4"][0]),
            environment_calls=59, branch_id="branches/block60/cohort0/after4/rep0/class1")

    def factory(self, template, producer, reference, context, **kw):
        kw["before_clone"]()
        self.created.append(kw["branch_id"])
        _, payload = branch(context, kw["candidate_index"], kw["replication"])
        payload["manifest"]["branch_id"] = kw["branch_id"]
        payload["receipt"]["manifest"] = copy.deepcopy(payload["manifest"])
        return RawBranchFixture(context, payload)

    def collect(self, rows):
        return collect_remaining(self.root, self.cfg, self.streams, 60, {(0, 4): self.context},
            None, None, None, self.budget, remaining=rows, admit=self.calls.append,
            verifier=branch_verifier(self.cfg, self.streams), branch_factory=self.factory)

    def test_only_declared_unfinished_branch_runs_and_full_label_is_saved(self):
        index = self.collect([self.row])
        self.assertEqual(self.created, [self.row["branch_id"]])
        self.assertEqual(self.budget.counts, dict(environment=59, optimizer=0))
        self.assertEqual(index[0]["result"]["steps"], 59)
        self.assertFalse((self.root / "payload/branches/block60/cohort0/after4/rep0/class0").exists())

    def test_wrong_future_seed_fails_before_clone_or_debit(self):
        row = dict(self.row, future_seed="1")
        with self.assertRaisesRegex(ValueError, "seed changed"):
            self.collect([row])
        self.assertEqual(self.budget.counts["environment"], 0)
        self.assertFalse(self.created)

    def test_duplicate_branch_failure_does_not_refund_first_branch(self):
        with self.assertRaisesRegex(ValueError, "seed changed"):
            self.collect([self.row, self.row])
        self.assertEqual(self.budget.counts["environment"], 59)
        self.assertEqual(len(self.created), 1)
        self.assertTrue((self.root / "launcher/branch-failure-block60.json").exists())

    def test_complete_import_preserves_bytes_without_replay(self):
        indexes = self.collect([self.row])
        source = self.root / "payload/contexts/block60/cohort0/after4.pt"
        save_envelope(source, encode_arrays(self.context))
        boundary = self.root / "launcher/imported-boundary.json"
        write_json_once(boundary, dict(index=indexes[0]))
        # The fixture workspace is the source root's parent, like the native layout.
        workspace = self.root.parent
        manifest = dict(old_run=self.root.name, manifest_sha256="fixture", old_environment_charge=60,
            contexts=[dict(key=[60, 0, 4], file=file_record(workspace, source))],
            completed=[dict(boundary=file_record(workspace, boundary), files=indexes[0]["files"])])
        target = self.root / "new-output"
        contexts, imported = import_saved_data(workspace, target, manifest, admit=self.calls.append, budget=self.budget)
        self.assertEqual(set(contexts), {(60, 0, 4)})
        self.assertEqual(imported, indexes)
        self.assertEqual(self.budget.counts["environment"], 59)
        for record in imported[0]["files"].values():
            self.assertEqual(file_record(target, record["path"]), record)


if __name__ == "__main__":
    unittest.main()
