"""Recovery admission and byte-preserving import, with no scientific loading."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl import capacity_planner_tail_recovery1_execution as entry
from src.utils.research_archive import inventory


class ExecutionTests(unittest.TestCase):
    def test_import_preserves_old_files_and_moves_only_partial_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/entry.original.RUN/"payload"
            partial = "raw/"+entry.PARTIAL+".jsonl.gz"
            contents = {partial:b"partial bytes", "raw/completed.jsonl.gz":b"complete bytes",
                "training-tails/reference.pkl.gz":b"saved tails", "models/final.pt":b"frozen weights",
                "progress.jsonl":b"old progress\n", "failure-state.pkl.gz":b"not copied or loaded"}
            for name, data in contents.items():
                path = old/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            before = inventory(old.parent)
            packet = dict(prior_files=before, contract=dict(old_counts={"value_optimizer_steps":11520},
                retained_interrupted_reservation={"planner_total_model_epochs":dict(reserved=768,dispatched=411)}),
                seed_reuse="intentional_remaining_only_not_independent_confirmation")
            target = root/entry.RUN/"payload"
            with patch.object(entry.pickle, "load", side_effect=AssertionError("no unpickle during import")):
                prefix = entry.import_payload(root, target, packet)
            self.assertEqual(prefix.read_bytes(), contents[partial])
            self.assertFalse((target/partial).exists())
            self.assertFalse((target/"failure-state.pkl.gz").exists())
            self.assertEqual((target/"training-tails/reference.pkl.gz").read_bytes(), b"saved tails")
            self.assertEqual(inventory(old.parent), before)
            receipt = json.loads((target/"recovery-inputs.json").read_text())
            self.assertTrue(receipt["scientific_predictor_unchanged"])
            self.assertEqual(receipt["old_counts"]["value_optimizer_steps"], 11520)

    def test_missing_authority_prevents_child_load_and_process_launch(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(entry, "approved", side_effect=PermissionError("not approved")), \
                patch.object(entry.pickle, "load", side_effect=AssertionError("no model/state load")), \
                patch.object(entry.subprocess, "Popen", side_effect=AssertionError("no process")):
            for fn in (entry.child, entry.launch):
                with self.subTest(fn=fn.__name__), self.assertRaises(PermissionError):
                    fn(Path(tmp))
            self.assertFalse((Path(tmp)/entry.RUN).exists())

    def test_unapproved_intent_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(entry.base, "_clean"), patch.object(entry.base, "_sealed_body"), \
                patch.object(entry.base, "file_record", return_value=dict(sha256="hash")), \
                patch.object(entry.base, "committed_json", side_effect=[{},
                    dict(scope=entry.RECOVERY_SCOPE, scope_approved=False),
                    dict(scientific_execution_authorized=False)]), \
                patch.object(entry, "old_evidence", side_effect=AssertionError("no repeated inventory")):
            with self.assertRaises(PermissionError):
                entry.freeze(Path(tmp))
