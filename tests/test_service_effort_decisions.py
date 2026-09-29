"""Decision and settlement checks on synthetic finite service-work fixtures."""

import copy
import json
from pathlib import Path
import unittest

from evaluation.check_service_effort_decisions import (
    CONFIG, ServiceDecisionTree, compute, evaluate_method, settle, validate,
)
from evaluation.audit_service_effort_decisions import verify_step
from src.baselines.service_effort_control import plan_public_mpc, predict_step
from src.env.service_effort_mechanics import ServiceEffortConfig, ServiceEffortMechanics


def fixture():
    return json.loads(CONFIG.read_text())


class DecisionFixtureTest(unittest.TestCase):
    def test_public_predictor_matches_executed_model(self):
        f = fixture()
        config = validate(f)
        for response in ((1, 1), (0.5, 1.5), (1.5, 0.5)):
            model = ServiceEffortMechanics(config, f["jobs"], [response] * 6)
            for name in ("balanced", "left", "right", "balanced", "idle", "idle"):
                predicted, cost = predict_step(model.observation(), f["actions"][name], response, config)
                actual, receipt = model.step(f["actions"][name])
                self.assertEqual(predicted, actual)
                self.assertAlmostEqual(cost, receipt.labor_cost + receipt.change_cost)

    def test_response_tape_not_visible_before_actual_feedback(self):
        f = fixture()
        model = ServiceDecisionTree(f, f["families"][1])
        left, right = (0, ()), (1, ())
        self.assertEqual(model.information_key(left), model.information_key(right))
        for _ in range(2):
            left = model.transition(left, "balanced")[1]
            right = model.transition(right, "balanced")[1]
            self.assertEqual(model.information_key(left), model.information_key(right))
        left = model.transition(left, "balanced")[1]
        right = model.transition(right, "balanced")[1]
        self.assertNotEqual(model.information_key(left), model.information_key(right))

    def test_branching_does_not_mutate_parent(self):
        f = fixture()
        tree = ServiceDecisionTree(f, f["families"][1])
        root = (0, ())
        before = tree.nodes[root][0].observation()
        tree.transition(root, "left")
        tree.transition(root, "right")
        self.assertEqual(tree.nodes[root][0].observation(), before)
        self.assertEqual(tree.transition(root, "left"), tree.transition(root, "left"))

    def test_terminal_executes_prepaid_effort_then_closes_remaining_work(self):
        f = fixture()
        model = ServiceEffortMechanics(validate(f), f["jobs"], [(1, 1)] * 6)
        for _ in range(5):
            model.step((0.5, 0.0))
        before = model.observation()
        cost, ledger = settle(model, f)
        self.assertEqual(ledger["drain_rows"][0]["receipt"]["applied_hours"], (0.5, 0))
        self.assertEqual(model.observation().pending_hours, ((0.0, 0.0),))
        self.assertEqual(ledger["remaining_work_after_external_service"], [0, 0])
        self.assertAlmostEqual(ledger["external_service_cost"], f["terminal_work_cost"] * sum(model.observation().remaining_work))
        self.assertAlmostEqual(cost, sum(r["cost"] for r in ledger["drain_rows"]) + ledger["external_service_cost"])
        self.assertEqual(before.epoch, 5)

    def test_mpc_rejects_environment_instead_of_public_observation(self):
        f = fixture()
        model = ServiceEffortMechanics(validate(f), f["jobs"], [(1, 1)] * 6)
        with self.assertRaises(TypeError):
            plan_public_mpc(model, f["prior"], model.config, f["actions"], 1, f["holding_weights"], f["terminal_work_cost"])

    def test_invalid_scientific_scope_and_job_sizes(self):
        for field, value in (("role", "patient_experiment"), ("horizon", 6), ("jobs", [[2.0], [1.0]]), ("terminal_work_cost", -1)):
            f = fixture()
            f[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(f)

    def test_known_nominal_model_needs_no_parameter_adaptation(self):
        f = fixture()
        family = f["families"][0]
        world = family["worlds"][0]
        frozen, _ = evaluate_method(f, family, world, "fixed_model_mpc")
        adaptive, _ = evaluate_method(f, family, world, "identification_mpc")
        self.assertEqual(frozen["total_cost"], adaptive["total_cost"])
        self.assertEqual(frozen["decisions"], adaptive["decisions"])

    def test_independent_audit_rejects_changed_cost_or_response(self):
        f = fixture()
        family = f["families"][0]
        world = family["worlds"][0]
        _, rows = evaluate_method(f, family, world, "fixed_balanced")
        for row in rows:
            verify_step(row, world, f)
        bad = copy.deepcopy(rows[1])
        bad["cost"] += 1
        with self.assertRaises(AssertionError):
            verify_step(bad, world, f)
        bad = copy.deepcopy(rows[1])
        bad["receipt"]["delivered_work"] = (9, 9)
        with self.assertRaises(AssertionError):
            verify_step(bad, world, f)

    def test_full_fixture_ordering_and_cardinality(self):
        summary, episodes, rows, transitions = compute(fixture())
        self.assertEqual(len(episodes), 15)
        self.assertEqual(len(rows), 75)
        self.assertEqual(len(transitions), 4092)
        self.assertFalse(summary["online_policy_training"])
        self.assertFalse(summary["clinical_model_validated"])
        for name, comparison in summary["comparisons"].items():
            d = comparison["finite_grid_diagnostic"]
            self.assertLessEqual(d["clairvoyant_cost"], d["nonanticipative_cost"] + 1e-10)
            self.assertLessEqual(d["nonanticipative_cost"], d["best_open_loop_cost"] + 1e-10)
            for value in comparison["expected_synthetic_cost"].values():
                self.assertGreaterEqual(value, d["nonanticipative_cost"] - 1e-10)
        nominal = summary["comparisons"]["no_change"]
        self.assertAlmostEqual(nominal["expected_synthetic_cost"]["fixed_model_mpc"], nominal["finite_grid_diagnostic"]["nonanticipative_cost"])


if __name__ == "__main__":
    unittest.main()
