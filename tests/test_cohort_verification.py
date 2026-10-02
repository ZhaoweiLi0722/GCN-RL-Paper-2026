"""Invented JSON receipts only. Does not run any patient engine."""

import copy
import unittest

from src.rl.candidate_pilot_verification import OPERATING, PATIENT, json_hash
from src.rl.cohort_verification import verify_cohort_tail
from tests.test_terminal_obligation_saved import fixture as patient_fixture


def fixture():
    start, _ = patient_fixture()
    end = copy.deepcopy(start)
    for pid in ("new", "old", "work"):
        end["patients"][pid]["status"] = "delivered"
    end.update(patient_queues=[[]], in_production_patients=[[[], []]])
    end["scalars"].update(t=8, cumulative_served=4)
    end["arrays"] = dict(reagents=[3.5], bioreactors=[[2., 0.]],
                         specimen_transfer_pipeline=[], reagent_transfer_pipeline=[[0.]],
                         capacity_transfer_pipeline=[[0.]])
    rows = []
    for i in range(1, 4):
        info = dict.fromkeys(OPERATING + PATIENT, 0.)
        info.update(reagent_holding_cost=2., base_cost=2., cost=2., demand=[0.],
                    patients_lost=[0.], patients_completed=[3. if i == 1 else 0.],
                    identity_active_count=0., waiting_patients=[0.])
        rows.append(dict(index=i, accounting_done=i == 3, action=[0., 0., 0., -1.],
                         info=info, raw_reward=-2., cost=2., active=0, resolution_step=1))
    return start, end, rows


def verify(start, end, rows):
    return verify_cohort_tail(start, end, rows,
        dict(prefix_final_sha256=json_hash(start), final_sha256=json_hash(end)),
        enrollment_steps=5, patient_resolution_steps=2, accounting_steps=3, num_facilities=1)


class CohortVerificationTests(unittest.TestCase):
    def test_primitive_costs_and_patients_recounted_and_stock_retained(self):
        report = verify(*fixture())
        self.assertEqual(report["tail_cost"], 6.)
        self.assertEqual(report["tail_patient_cost"], 2.)
        self.assertEqual(report["tail_post_resolution_cost"], 4.)
        self.assertEqual(report["tail_completions"], 3)
        self.assertEqual(report["retained_reagents"], [3.5])
        self.assertFalse(report["terminal_stock_valued"])

    def test_wrong_hash_or_short_tail_rejected(self):
        start, end, rows = fixture()
        with self.assertRaises(ValueError):
            verify_cohort_tail(start, end, rows, dict(prefix_final_sha256="0" * 64, final_sha256=json_hash(end)),
                enrollment_steps=5, patient_resolution_steps=2, accounting_steps=3, num_facilities=1)
        with self.assertRaises(ValueError):
            verify(start, end, rows[:-1])

    def test_wrong_raw_primitive_patient_or_termination_rejected(self):
        for mutate in (
                lambda rows: rows[0]["info"].update(base_cost=4.),
                lambda rows: rows[0]["info"].update(patients_lost=[1.]),
                lambda rows: rows[0]["info"].update(demand=[1.]),
                lambda rows: rows[1].update(action=[0., 0., 0., 0.]),
                lambda rows: rows[0].update(accounting_done=True),
                lambda rows: rows[0].update(resolution_step=0),
                lambda rows: rows[0].update(active=1)):
            start, end, rows = fixture()
            mutate(rows)
            with self.assertRaises(ValueError):
                verify(start, end, rows)

    def test_no_risk_substitution_or_unsettled_pipeline(self):
        start, end, rows = fixture()
        end["patients"]["new"]["health_index"] = .9
        with self.assertRaises(ValueError):
            verify(start, end, rows)
        start, end, rows = fixture()
        end["arrays"]["reagent_transfer_pipeline"] = [[.5]]
        with self.assertRaises(ValueError):
            verify(start, end, rows)


if __name__ == "__main__":
    unittest.main()
