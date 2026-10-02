"""Invented raw records with continuous stocks and nonzero sunk patient counts."""

import copy
import unittest

from src.rl.candidate_pilot_verification import json_hash
from src.rl.conservative_cohort_verification import branch_reader, assemble_round
from tests.test_conservative_cohort_plan import config
from tests.test_paired_cohort_verification import make_context, branch, refresh, context_seed, future_seed


def current_branch(cohort=0, start=4, k=0, rep=0):
    c = make_context(cohort=cohort, start=start)
    c["source_id"] = config()["rng_namespace"]
    del c["context_sha256"]
    c["context_sha256"] = json_hash(c)
    _, p = branch(c, k=k, replication=rep)
    p["manifest"].update(format="conservative-cohort-branch-v1", round=cohort // 2,
        continuation_sha256="a" * 64, continuation="frozen_round_start_greedy_then_common_tail")
    for row in p["rows"]:
        if row["action_source"] == "fixed_r4":
            row["action_source"] = "frozen_round_start_policy"
    refresh(p)
    return c, p


def manifest():
    return dict(environment={"60": {"context": [context_seed(60, c) for c in range(4)]}},
        conditional_future={f"block60/cohort{c}/after{t}": [future_seed(60, c, t, r) for r in range(4)]
                            for c in range(4) for t in (4, 20, 36)})


def read(c, p):
    return branch_reader(config(), manifest(), 60, c["cohort"] // 2, "a" * 64)(
        header=dict(context=c, manifest=p["manifest"], initial_state=p["initial_state"]),
        rows=p["rows"], states={k: p[k] for k in ("prefix_final", "final_state")}, receipt=p["receipt"])


class ReaderTests(unittest.TestCase):
    def test_raw_total_retains_precision_and_subtracts_sunk_counts(self):
        c, p = current_branch(rep=3)
        result = read(c, p)
        self.assertEqual(result["remaining_raw_cost"], 1e9 + 58 + .375)
        self.assertEqual(result["remaining_losses"], 1)
        self.assertEqual(result["remaining_completions"], 2)
        self.assertFalse(result["independent_policy_replay_performed"])

    def test_reject_old_continuation_even_with_resealed_hashes(self):
        c, p = current_branch()
        p["rows"][1]["action_source"] = "fixed_r4"
        refresh(p)
        with self.assertRaises(ValueError):
            read(c, p)

    def test_complete_round_and_extra_branch_rejection(self):
        contexts, indexes = {}, []
        for cohort in (0, 1):
            for t in (4, 20, 36):
                for r in range(4):
                    for k in range(3):
                        c, p = current_branch(cohort, t, k, r)
                        contexts[cohort, t] = c
                        indexes.append(dict(result=read(c, p)))
        data = assemble_round(contexts, indexes, config(), manifest(), 60, 0, "a" * 64)
        self.assertEqual(len(data["states"]), 6)
        self.assertEqual(len(data["states"][0]["raw_costs"]), 4)
        self.assertEqual(len(data["states"][0]["raw_costs"][0]), 3)
        with self.assertRaises(ValueError):
            assemble_round(contexts, indexes + [indexes[0]], config(), manifest(), 60, 0, "a" * 64)

    def test_cost_and_patient_corruption_is_detected(self):
        c, p = current_branch()
        p["rows"][0]["info"]["cost"] += .25
        refresh(p)
        with self.assertRaises(ValueError):
            read(c, p)


if __name__ == "__main__":
    unittest.main()
