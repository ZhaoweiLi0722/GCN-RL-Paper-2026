"""Invented JSON arithmetic only; no simulator, tensor, checkpoint or optimizer."""

import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "reports/2026-10-02-terminal-obligation/analyze.py"
SPEC = importlib.util.spec_from_file_location("terminal_obligation", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture():
    patients = {}
    for pid, status, epoch, age in (("new", "waiting", 4, 0), ("old", "waiting", 1, 3),
                                   ("work", "in_production", 2, 2), ("done", "delivered", 0, 3),
                                   ("loss", "lost", 0, 2)):
        patients[pid] = dict(patient_id=pid, specimen_id=pid, status=status,
                             enrollment_epoch=epoch, age=age, material_facility=0,
                             manufacturing_facility=0, collection_facility=0,
                             risk_type=1, risk_multiplier=1., health_index=.5, deterioration_epoch=3.)
    state = dict(patients=patients, patient_queues=[["new", "old"]],
                 in_production_patients=[[[], ["work"]]], specimen_transits=[], product_return_transits=[],
                 scalars=dict(t=5, cumulative_enrolled=5, cumulative_lost=1, cumulative_served=1))
    outcome = dict(steps=5, final_state_sha256=AUDIT.json_hash(state), terminal_compartments=AUDIT.identity_counts(state),
                   terminal_active=3, components={"patient_loss_cost": 10., "purchase": 2.}, cost=12.,
                   block=60, role="own_frozen", world_index=0, seed=123, initial_state_sha256="same",
                   arrivals=[[1], [1], [1], [1], [1]])
    return state, outcome


class TerminalObligationTests(unittest.TestCase):
    def test_new_arrival_not_processed_and_conservation(self):
        s, o = fixture()
        r = AUDIT.summarize_state(s, o, 5)
        self.assertEqual(r["final_step_new_waiting"], 1)
        self.assertEqual(r["older_active"], 2)
        self.assertEqual(r["active_age_histogram"], {"0": 1, "3": 1, "2": 1})
        self.assertEqual(r["enrolled"], r["active"] + r["lost"] + r["delivered"])

    def test_identical_state_comparison(self):
        s, o = fixture()
        r = AUDIT.summarize_state(s, o, 5)
        d = AUDIT.compare(r, copy.deepcopy(r))
        self.assertTrue(d["same_full_final_state"])
        self.assertTrue(d["same_patient_registry"])
        self.assertTrue(all(v == 0 for v in d["differences"].values()))

    def test_hash_mismatch(self):
        s, o = fixture()
        s["patients"]["old"]["age"] += 1
        with self.assertRaisesRegex(ValueError, "binding"):
            AUDIT.summarize_state(s, o, 5)

    def test_duplicate_compartment(self):
        s, o = fixture()
        s["patient_queues"][0].append("old")
        o["final_state_sha256"] = AUDIT.json_hash(s)
        with self.assertRaises(ValueError):
            AUDIT.summarize_state(s, o, 5)

    def test_terminal_summary_not_trusted(self):
        s, o = fixture()
        o["terminal_active"] = 2
        with self.assertRaises(ValueError):
            AUDIT.summarize_state(s, o, 5)

    def test_invalid_cost_or_horizon(self):
        for key, value in (("cost", float("nan")), ("cost", 11.), ("steps", 4)):
            s, o = fixture()
            o[key] = value
            with self.assertRaises(ValueError):
                AUDIT.summarize_state(s, o, 5)

    def test_final_enrollment_order_guard(self):
        s, o = fixture()
        s["patients"]["new"]["age"] = 1
        o["final_state_sha256"] = AUDIT.json_hash(s)
        with self.assertRaisesRegex(ValueError, "unexpected processing"):
            AUDIT.summarize_state(s, o, 5)

    def test_no_individual_pairing_when_enrollment_attributes_differ(self):
        s, o = fixture()
        right = AUDIT.summarize_state(s, o, 5)
        s["patients"]["old"]["risk_type"] = 2
        o["final_state_sha256"] = AUDIT.json_hash(s)
        left = AUDIT.summarize_state(s, o, 5)
        d = AUDIT.compare(left, right)
        self.assertFalse(d["same_enrollment_attributes"])
        self.assertFalse(d["same_patient_registry"])
        self.assertEqual(d["differences"]["active"], 0)

    def test_wrong_start_rejected(self):
        s, o = fixture()
        r = AUDIT.summarize_state(s, o, 5)
        other = dict(r, initial_state_sha256="different")
        with self.assertRaises(ValueError):
            AUDIT.compare(r, other)


if __name__ == "__main__":
    unittest.main()
