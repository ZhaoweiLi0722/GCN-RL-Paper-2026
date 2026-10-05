"""Pure schedule and exact nested-work arithmetic; no simulation or updates."""

import copy
import json
from pathlib import Path
import unittest

from src.rl.capacity_policy_tail_design import (
    TRAIN_ROLES, candidate_index, counts, learner_config, numeric_contract, schedule,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "experiments/configs/capacity_policy_tail_20261005.json").read_text())


class DesignTests(unittest.TestCase):
    def test_exact_inner_planning_work_not_just_clone_count(self):
        c = numeric_contract(STUDY)["limits"]
        self.assertEqual(c["paired_endpoint_starts"], 720)
        self.assertEqual(c["continuation_planning_decisions"], 12300)
        self.assertEqual(c["continuation_planning_epochs"], 4723200)
        self.assertEqual(c["total_predictor_epochs"], 14729040)
        self.assertEqual(c["forward_calls"], 36190)
        self.assertEqual(c["total_filter_transitions"], 5988000)
        self.assertEqual(c["tail_rows_per_policy"], 23400)
        self.assertFalse(STUDY["scientific_execution_authorized"])

    def test_mock_whole_schedule_seals_before_first_test(self):
        sources, updates, seals, barrier, evaluations = set(), {}, set(), False, 0
        for row in schedule(STUDY):
            event = row["event"]
            if event == "collect_reference":
                self.assertFalse(barrier)
                w = row["world"]
                sources.add((w["block"], w["index"]))
                self.assertEqual(len(row["roots"]), 2)
            elif event == "fit":
                w = row["world"]
                self.assertIn((w["block"], w["index"]), sources)
                key = w["block"], row["role"]
                updates[key] = updates.get(key, 0) + row["updates"]
            elif event == "seal":
                key = row["block"], row["role"]
                self.assertEqual(updates[key], 768)
                seals.add(key)
            elif event == "all_models_sealed":
                self.assertEqual(len(seals), 15)
                barrier = True
            elif event == "evaluate":
                self.assertTrue(barrier)
                self.assertEqual(row["updates"], 0)
                evaluations += 1
        self.assertEqual(len(sources), 120)
        self.assertEqual(evaluations, 360)
        self.assertEqual(sum(updates.values()), 11520)

    def test_roles_bind_same_ancestor_but_distinct_target_policy(self):
        configs = [learner_config(STUDY, r, "a" * 64) for r in TRAIN_ROLES]
        self.assertEqual([c["tail_policy"] for c in configs], ["adaptive", "frozen_mpc", "frozen_mpc"])
        self.assertIsNone(configs[0]["continuation_sha256"])
        self.assertEqual(configs[1]["continuation_sha256"], configs[2]["continuation_sha256"])
        self.assertEqual(configs[1]["tail_schema"], "capacity-policy-tail-record-v1")
        for i in range(24):
            self.assertEqual(candidate_index(i, i), i % 16)
            self.assertEqual(candidate_index(i, i + 24), (i + 8) % 16)

    def test_contract_changes_rejected_not_silently_rebudgeted(self):
        for key, value in counts().items():
            if isinstance(value, int):
                bad = copy.deepcopy(STUDY)
                bad["budget"][key] += 1
                with self.subTest(key=key), self.assertRaises(ValueError):
                    numeric_contract(bad)
        bad = copy.deepcopy(STUDY)
        bad["budget"]["seconds"]["frozen_evaluation"] = 14400
        with self.assertRaises(ValueError):
            numeric_contract(bad)


if __name__ == "__main__":
    unittest.main()
