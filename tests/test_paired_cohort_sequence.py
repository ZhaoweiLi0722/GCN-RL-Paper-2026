"""Fake file-phase callbacks only: no scientific factory or optimizer."""

import copy
from pathlib import Path
import tempfile
import unittest

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.paired_cohort_resources import budget_plan
from src.rl.paired_cohort_sequence import PairedCohortSequence, dispatch_serial
from tests.test_paired_cohort_resources import config


class PairedCohortSequenceTests(unittest.TestCase):
    def test_whole_serial_order_seals_before_evaluation_and_exact_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sequence = PairedCohortSequence(config(), root, enabled=True)
            budget = DynamicCandidateBudget(root / "budget.jsonl", budget_plan(config()), enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            visits, saved, admits = [], [], []
            def perform(job, current, ledger):
                visits.append(job)
                if job == "seal":
                    paths = {}
                    for name in current.models:
                        path = root / "models" / (name + ".json")
                        write_json_once(path, {"invented_model_metadata_only": name})
                        paths[name] = path
                    current.seal_models(paths)
                if job.startswith("evaluation/"):
                    self.assertEqual(len(current.seals), 12)
                    self.assertLess(visits.index("seal"), visits.index(job))
                path = root / "evidence" / (job + ".json")
                write_json_once(path, dict(invented_phase=job))
                return [path]
            def persist(event, state, budget_state):
                saved.append((event, copy.deepcopy(state), budget_state))
            result = dispatch_serial(sequence, budget, perform, persist, admit=admits.append)
            self.assertEqual(visits, sequence.jobs)
            self.assertEqual(visits, admits)
            self.assertEqual(len([v for v in visits if v.startswith("evaluation/")]), 18)
            self.assertLess(visits.index("paired_branches/block62"), visits.index("paired_actor/block60"))
            self.assertLess(visits.index("paired_actor/block62"), visits.index("bc_actor/block60"))
            self.assertIsNone(sequence.next_job)
            restored = PairedCohortSequence(config(), root, enabled=True)
            restored.load_state_dict(result)
            self.assertEqual(restored.state_dict(), result)
            with self.assertRaises(ValueError):
                sequence.load_state_dict(saved[0][1])

    def test_no_test_before_all_models_and_no_terminal_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sequence = PairedCohortSequence(config(), root, enabled=True)
            budget = DynamicCandidateBudget(root / "budget.jsonl", budget_plan(config()), enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            initial = sequence.state_dict()
            with self.assertRaises(ValueError):
                sequence.begin("evaluation/block60/paired_cost")
            def fail(*args):
                raise OSError("invented write failure")
            events = []
            with self.assertRaises(OSError):
                dispatch_serial(sequence, budget, fail, lambda e, *args: events.append(e), admit=lambda _: None)
            self.assertEqual(events, ["begin", "failed"])
            self.assertIsNone(sequence.next_job)
            with self.assertRaises(ValueError):
                sequence.load_state_dict(initial)
            with self.assertRaises(ValueError):
                dispatch_serial(sequence, budget, fail, lambda *args: None, admit=lambda _: None)

    def test_durable_budget_ahead_of_checkpoint_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sequence = PairedCohortSequence(config(), root, enabled=True)
            budget = DynamicCandidateBudget(root / "budget.jsonl", budget_plan(config()), enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            budget.begin("binding")
            budget.finish()
            with self.assertRaises(ValueError):
                dispatch_serial(sequence, budget, lambda *args: None, lambda *args: None, admit=lambda _: None)


if __name__ == "__main__":
    unittest.main()
