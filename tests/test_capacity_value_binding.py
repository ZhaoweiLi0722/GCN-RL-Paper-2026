"""Metadata-only authority guards; never create a scientific backend."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.rl import capacity_value_execution as entry
from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_value_resources import numeric_contract

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT/entry.PROPOSAL).read_text())


class ValueBindingTests(unittest.TestCase):
    def test_unapproved_numerical_package_cannot_freeze_or_inspect_scientific_inputs(self):
        with patch.object(entry.base, "_clean"), \
                patch.object(entry.base, "committed_json", side_effect=[STUDY, dict(scope_approved=False)]), \
                patch.object(entry.base, "file_record", return_value=dict(sha256="artificial")), \
                patch.object(entry, "source_locks", side_effect=AssertionError("must reject first")), \
                patch.object(entry, "build_runner", side_effect=AssertionError("scientific backend forbidden")):
            with self.assertRaisesRegex(PermissionError, "exact complete"):
                entry.freeze(ROOT)

    def test_derived_authority_binds_counts_and_rejects_packet_mutation(self):
        packet = dict(format="capacity-value-mpc-frozen-v1", intent=dict(scope_approved=True,
                      scope=entry.SCOPE, user_literal="artificial fixture, not actual approval"),
                      seed_audit=dict(passed=True), implementation_commit="artificial",
                      config=dict(value_mpc_study=STUDY), contract=numeric_contract(STUDY))
        packet["packet_sha256"] = digest(packet)
        auth = entry.authorization(packet)
        self.assertTrue(auth["scientific_execution_authorized"])
        self.assertEqual(auth["attempts"], 1)
        self.assertFalse(auth["automatic_retry"])
        self.assertFalse(auth["automatic_follow_on"])
        for key in ("contract", "intent"):
            bad = copy.deepcopy(packet)
            bad[key] = {}
            with self.assertRaises((ValueError, PermissionError)):
                entry.authorization(bad)

    def test_schedule_change_cannot_keep_the_old_budget(self):
        for section, key in (("design", "initial_worlds_per_block"),
                             ("value", "updates_per_world"), ("budget", "worlds")):
            bad = copy.deepcopy(STUDY)
            bad[section][key] += 1
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                numeric_contract(bad)


if __name__ == "__main__":
    unittest.main()
