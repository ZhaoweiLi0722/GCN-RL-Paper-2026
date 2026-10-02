"""Artificial files and serial callbacks; no scientific factories or updates."""

import copy
from pathlib import Path
import tempfile
import unittest

from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.paired_cohort_execution import write_json_once
from src.rl.paired_cohort_recovery_resources import budget_plan
from src.rl.paired_cohort_recovery_sequence import PairedCohortRecoverySequence, dispatch_serial
from tests.test_paired_cohort_resources import config
from tests.test_paired_cohort_recovery_resources import recovery


class RecoverySequenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.sequence = PairedCohortRecoverySequence(config(), recovery(), self.root, enabled=True)
        self.budget = DynamicCandidateBudget(self.root / "budget.jsonl", budget_plan(config(), recovery()),
                                             enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)

    def perform(self, job, sequence, budget):
        if job == "seal":
            paths = {}
            for name in sequence.models:
                path = self.root / "fake-models" / (name + ".json")
                write_json_once(path, dict(invented=name))
                paths[name] = path
            sequence.seal_models(paths)
        if job.startswith("evaluation/"):
            self.assertEqual(len(sequence.seals), 12)
        path = self.root / "evidence" / (job + ".json")
        write_json_once(path, dict(invented=job))
        return [path]

    def test_entire_chain_exact_order_seals_and_restore(self):
        initial, admits, saved = self.sequence.state_dict(), [], []
        def admit(job):
            self.assertIsNone(self.budget.active)
            admits.append(job)
        final = dispatch_serial(self.sequence, self.budget, self.perform,
            lambda event, state, budget: saved.append(copy.deepcopy(state)), admit=admit)
        self.assertEqual(admits, self.sequence.jobs)
        self.assertEqual(len(admits), 32)
        self.assertLess(admits.index("paired_branches/block62"), admits.index("paired_actor/block60"))
        self.assertLess(admits.index("bc_actor/block62"), admits.index("seal"))
        restored = PairedCohortRecoverySequence(config(), recovery(), self.root, enabled=True)
        restored.load_state_dict(final)
        self.assertEqual(restored.state_dict(), final)
        with self.assertRaises(ValueError):
            self.sequence.load_state_dict(initial)

    def test_test_barrier_and_terminal_failure_cannot_resume(self):
        initial = self.sequence.state_dict()
        with self.assertRaises(ValueError):
            self.sequence.begin("evaluation/block60/paired_cost")
        def fail(*args):
            raise RuntimeError("invented terminal failure")
        events = []
        with self.assertRaises(RuntimeError):
            dispatch_serial(self.sequence, self.budget, fail, lambda e, *args: events.append(e), admit=lambda _: None)
        self.assertEqual(events, ["begin", "failed"])
        with self.assertRaises(ValueError):
            self.sequence.load_state_dict(initial)
        with self.assertRaises(ValueError):
            dispatch_serial(self.sequence, self.budget, self.perform, lambda *args: None, admit=lambda _: None)

    def test_active_or_durable_budget_cannot_be_rewound(self):
        self.sequence.begin("binding")
        state = self.sequence.state_dict()
        state["active"] = None
        with self.assertRaises(ValueError):
            self.sequence.load_state_dict(state)
        self.budget.begin("binding")
        self.budget.finish()
        with self.assertRaises(ValueError):
            dispatch_serial(self.sequence, self.budget, self.perform, lambda *args: None, admit=lambda _: None)

    def test_evidence_and_proposal_are_bound(self):
        self.sequence.begin("binding")
        evidence = self.perform("binding", self.sequence, self.budget)
        self.sequence.finish(evidence)
        saved = self.sequence.state_dict()
        changed = copy.deepcopy(saved)
        changed["recovery_proposal_sha256"] = "0" * 64
        target = PairedCohortRecoverySequence(config(), recovery(), self.root, enabled=True)
        with self.assertRaises(ValueError):
            target.load_state_dict(changed)
        evidence[0].write_text("changed fake metadata")
        with self.assertRaises(ValueError):
            target.load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
