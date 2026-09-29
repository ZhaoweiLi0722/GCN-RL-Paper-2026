"""Read-only receipt checks and invented numeric mutations; no simulator/learner."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.audit_reward_objective_bridge import (
    DEFAULT_CONFIG, audit, check_case, read_json,
)


def fixture():
    config = {"expected_steps_per_case": 3, "archived_return_steps": 2,
              "top_level_cost_components": ["base_cost", "patient_loss_cost", "expiry_cost", "urgency_cost"]}
    proposed = {"gamma": 1., "return_steps": 1}
    semantics = {"gamma": .5, "reward_scale": .1, "reward_kind": "absolute_environment",
                 "reward_definition_id": "invented-negative-cost", "bootstrap_on_truncation": True}
    events = []
    for i, cost in enumerate((10., 20., 30.)):
        record = {"semantics": dict(semantics), "origin": "trajectory", "step_index": i,
                  "trajectory_id": "invented-episode", "source_id": "invented-source",
                  "state": [i], "next_state": [i + 1], "state_token": str(i),
                  "next_state_token": str(i + 1), "raw_reward": -cost,
                  "terminated": i == 2, "truncated": False}
        execution = {"cost": cost, "base_cost": cost - 2., "patient_loss_cost": 2.,
                     "expiry_cost": 0., "urgency_cost": 0., "specimen_transfer_cost": 3.,
                     "identity_active_count": 0.}
        events.append({"receipt": {"record": record, "execution": execution}})
    case = {"case": "invented", "events": events,
            "specification": {"horizon_end": "terminal", "episode_horizon": 3},
            "final": {"steps": 3, "pending_count": 0, "emitted_lengths": [2, 2, 1]}}
    return case, config, proposed


class RewardObjectiveBridgeTests(unittest.TestCase):
    def test_cost_sum_scale_and_discount_are_independent(self):
        result = check_case(*fixture())
        self.assertEqual(result["raw_cost_sum"], 60.)
        self.assertEqual(result["top_level_components_sum"], 60.)
        self.assertEqual(result["scaled_undiscounted_segment_reward"], -6.)
        self.assertEqual(result["discounted_segment_cost"], 27.5)
        self.assertEqual(result["discount_weighting_difference"], 32.5)
        self.assertEqual(result["last_step_weight"], .25)

    def test_transfer_subcomponent_is_not_counted_twice(self):
        case, config, proposed = fixture()
        config["top_level_cost_components"].append("specimen_transfer_cost")
        with self.assertRaisesRegex(ValueError, "components"):
            check_case(case, config, proposed)

    def test_component_or_reward_corruption_is_detected(self):
        for section, field in (("execution", "cost"), ("execution", "base_cost"),
                               ("record", "raw_reward")):
            case, config, proposed = fixture()
            case["events"][1]["receipt"][section][field] += 1.
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_case(case, config, proposed)

    def test_reward_cannot_be_scaled_twice_or_turned_anchor_relative(self):
        for reward in (-2., 0., 20.):
            case, config, proposed = fixture()
            case["events"][1]["receipt"]["record"]["raw_reward"] = reward
            with self.subTest(reward=reward), self.assertRaisesRegex(ValueError, "negative incurred"):
                check_case(case, config, proposed)

    def test_mixed_definition_scale_and_discount_fail(self):
        for field, value in (("reward_definition_id", "another-objective"),
                             ("reward_scale", .2), ("gamma", 1.)):
            case, config, proposed = fixture()
            case["events"][1]["receipt"]["record"]["semantics"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mixed"):
                check_case(case, config, proposed)

    def test_disconnected_lineage_is_rejected(self):
        for field, value in (("state", [99]), ("state_token", "another-state"),
                             ("source_id", "another-source"), ("trajectory_id", "another-episode")):
            case, config, proposed = fixture()
            case["events"][1]["receipt"]["record"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "disconnected"):
                check_case(case, config, proposed)

    def test_duplicate_or_missing_step_is_rejected(self):
        case, config, proposed = fixture()
        case["events"][1] = copy.deepcopy(case["events"][0])
        with self.assertRaisesRegex(ValueError, "index"):
            check_case(case, config, proposed)
        case["events"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            check_case(case, config, proposed)

    def test_nonfinite_string_or_flag_numbers_fail(self):
        for value in (True, "10", float("nan"), float("inf")):
            case, config, proposed = fixture()
            case["events"][0]["receipt"]["execution"]["cost"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "numeric"):
                check_case(case, config, proposed)

    def test_boundary_flags_cannot_be_inferred_or_crossed(self):
        for value in (1, True):
            case, config, proposed = fixture()
            case["events"][0]["receipt"]["record"]["terminated"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_case(case, config, proposed)

    def test_short_tails_and_actual_n_step_bootstrap(self):
        result = check_case(*fixture())
        windows = result["recomputed_windows"]
        self.assertEqual([r["scaled_return"] for r in windows], [-2., -3.5, -3.])
        self.assertEqual([r["bootstrap_discount"] for r in windows], [.25, 0., 0.])
        case, config, proposed = fixture()
        case["final"]["emitted_lengths"].pop()
        with self.assertRaisesRegex(ValueError, "tail"):
            check_case(case, config, proposed)

    def test_truncation_retains_bootstrap_and_is_not_terminal(self):
        case, config, proposed = fixture()
        case["specification"]["horizon_end"] = "truncation"
        case["events"][-1]["receipt"]["record"].update(terminated=False, truncated=True)
        result = check_case(case, config, proposed)
        self.assertEqual([r["bootstrap_discount"] for r in result["recomputed_windows"]], [.25, .25, .5])
        self.assertFalse(result["terminal_mask"])
        self.assertIn("recorded_segment_has_bootstrap_continuation_not_complete_return", result["objective_differences"])

    def test_zero_active_patients_and_correct_gamma_do_not_certify_closure(self):
        case, config, proposed = fixture()
        for event in case["events"]:
            event["receipt"]["record"]["semantics"]["gamma"] = 1.
        config["archived_return_steps"] = 1
        case["final"]["emitted_lengths"] = [1, 1, 1]
        result = check_case(case, config, proposed)
        self.assertEqual(result["objective_differences"], [])
        self.assertEqual(result["active_patients_at_segment_end"], 0.)
        self.assertFalse(result["complete_liability_settlement_verified"])
        self.assertFalse(result["learned_q_targets_independently_recomputed"])

    def test_duplicate_json_keys_and_nonfinite_constants_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invented.json"
            for text in ('{"cost": 1, "cost": 2}', '{"cost": NaN}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_json(path)

    def test_real_archive_is_read_only_and_never_claims_performance(self):
        result = audit(read_json(DEFAULT_CONFIG))
        self.assertTrue(result["accounting_audit_passed"])
        self.assertEqual(result["unique_case_count"], 6)
        self.assertEqual(result["receipt_count"], 36)
        self.assertEqual(result["resumed_duplicates_checked_not_extra_replicates"], 6)
        self.assertEqual(result["source_locks_verified"], 34)
        self.assertEqual(result["environment_queries"], 0)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertFalse(result["scientific_launch_authorized"])
        self.assertFalse(result["performance_or_reward_improvement_claimed"])

    def test_hash_mismatch_fails_before_using_evidence(self):
        config = read_json(DEFAULT_CONFIG)
        config["inputs"]["receipts"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit(config)


if __name__ == "__main__":
    unittest.main()
