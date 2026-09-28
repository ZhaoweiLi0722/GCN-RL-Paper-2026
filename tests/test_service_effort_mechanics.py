"""No patient simulation or scientific performance estimates in these tests."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

from src.baselines.public_service_response import PublicServiceResponseEstimator
from src.env.service_effort_mechanics import ServiceEffortConfig, ServiceEffortMechanics
from evaluation.check_service_effort_mechanics import check_fixture


def config(**kwargs):
    values = dict(site_hour_caps=(1.0, 1.0), shared_hour_budget=1.0,
                  commitment_lead_steps=1, hourly_cost=2.0, quadratic_cost=0.5, switching_cost=0.25)
    values.update(kwargs)
    return ServiceEffortConfig(**values)


def env(**kwargs):
    return ServiceEffortMechanics(config(**kwargs), [[1.0, 1.0], [1.0, 1.0]], [(1.0, 1.0)] * 6)


class ServiceEffortTest(unittest.TestCase):
    def test_fractional_work_without_fractional_completion(self):
        model = env()
        obs, receipt = model.step((0.25, 0.0))
        self.assertEqual(obs.remaining_work, (2.0, 2.0))
        self.assertEqual(receipt.applied_hours, (0.0, 0.0))
        for step in range(1, 5):
            obs, receipt = model.step((0.25, 0.0))
            self.assertEqual(obs.remaining_work[0], 2 - step * 0.25)
            self.assertEqual(obs.completed_jobs[0], int(step == 4))
        model.audit_conservation()

    def test_subgrid_actions_change_work_not_completed_count(self):
        outcomes = []
        for amount in (0.25, 0.251):
            model = env()
            model.step((amount, 0.0))
            outcomes.append(model.step((0.0, 0.0))[0])
        self.assertEqual(outcomes[0].completed_jobs, (0, 0))
        self.assertEqual(outcomes[0].completed_jobs, outcomes[1].completed_jobs)
        self.assertNotEqual(outcomes[0].remaining_work, outcomes[1].remaining_work)

    def test_two_step_delay(self):
        model = env(commitment_lead_steps=2)
        self.assertEqual(model.step((0.5, 0.0))[1].delivered_work, (0.0, 0.0))
        self.assertEqual(model.step((0.0, 0.0))[1].delivered_work, (0.0, 0.0))
        self.assertEqual(model.step((0.0, 0.0))[1].delivered_work, (0.5, 0.0))

    def test_effectiveness_at_service_time(self):
        model = ServiceEffortMechanics(config(), [[1.0], [1.0]], [(1.0, 1.0), (0.5, 1.0)])
        model.step((0.5, 0.0))
        self.assertEqual(model.step((0.0, 0.0))[1].delivered_work, (0.25, 0.0))

    def test_cost_paid_upfront_and_idle_effort_not_free(self):
        model = ServiceEffortMechanics(config(), [[], []], [(1.0, 1.0)] * 2)
        obs, receipt = model.step((0.5, 0.5))
        self.assertEqual(receipt.labor_cost, 2.25)
        self.assertEqual(receipt.change_cost, 0.25)
        self.assertEqual(obs.pending_hours, ((0.5, 0.5),))
        obs, receipt = model.step((0.0, 0.0))
        self.assertEqual(receipt.delivered_work, (0.0, 0.0))
        self.assertEqual(receipt.observation_kind, ("backlog_limited", "backlog_limited"))
        self.assertEqual(model.total_labor_cost, 2.25)

    def test_invalid_action_without_mutation(self):
        for action in [(0.6, 0.6), (1.1, 0), (-0.1, 0), (float("nan"), 0), (float("inf"), 0), (0.1,)]:
            with self.subTest(action=action):
                model = env()
                before = model.observation()
                with self.assertRaises(ValueError):
                    model.step(action)
                self.assertEqual(model.observation(), before)
                self.assertEqual(model.total_labor_cost, 0)

    def test_invalid_config(self):
        for kwargs in [dict(commitment_lead_steps=0), dict(commitment_lead_steps=1.5), dict(shared_hour_budget=3), dict(hourly_cost=float("nan")), dict(site_hour_caps=()), dict(quadratic_cost=-1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config(**kwargs)

    def test_invalid_response_tape(self):
        for tape in [[(0.0, 1.0)], [(float("nan"), 1.0)], [(1.0,)], []]:
            with self.subTest(tape=tape), self.assertRaises(ValueError):
                ServiceEffortMechanics(config(), [[1.0], [1.0]], tape)

    def test_public_observation_hides_future_and_response(self):
        first = ServiceEffortMechanics(config(), [[2.0], [2.0]], [(1, 1), (1, 1), (1, 1)])
        second = ServiceEffortMechanics(config(), [[2.0], [2.0]], [(1, 1), (1, 1), (0.5, 1)])
        self.assertEqual(first.observation(), second.observation())
        for _ in range(2):
            self.assertEqual(first.step((0.5, 0)), second.step((0.5, 0)))
        self.assertEqual(first.observation(), second.observation())
        self.assertNotEqual(first.step((0, 0))[0], second.step((0, 0))[0])
        self.assertEqual(set(asdict(first.observation())), {"epoch", "remaining_work", "unfinished_jobs", "completed_jobs", "pending_hours", "previous_commitment"})

    def test_estimator_skips_censored_measurements(self):
        model = ServiceEffortMechanics(config(), [[10], [0.1]], [(0.5, 1)] * 3)
        estimator = PublicServiceResponseEstimator((1, 1), alpha=1)
        for _ in range(2):
            _, receipt = model.step((0.5, 0.5))
            estimator.observe(receipt)
        self.assertEqual(estimator.estimate, (0.5, 1))
        self.assertEqual(estimator.samples, [1, 0])
        before = estimator.estimate
        with self.assertRaises(ValueError):
            estimator.observe(receipt)
        self.assertEqual(estimator.estimate, before)
        with self.assertRaises(TypeError):
            estimator.observe(model)

    def test_estimator_rejects_inconsistent_flag_transactionally(self):
        _, receipt = env().step((0.5, 0.5))
        estimator = PublicServiceResponseEstimator((1, 1), 1)
        with self.assertRaises(ValueError):
            estimator.observe(replace(receipt, observation_kind=("uncensored", "no_effort")))
        self.assertEqual(estimator.last_epoch, -1)
        self.assertEqual(estimator.samples, [0, 0])

    def test_terminal_ledger_keeps_obligations(self):
        model = env()
        model.step((0.5, 0.5))
        ledger = model.terminal_ledger()
        self.assertFalse(ledger["settled"])
        self.assertFalse(ledger["performance_comparison_allowed"])
        self.assertEqual(ledger["pending_prepaid_hours"], ((0.5, 0.5),))
        self.assertEqual(ledger["unfinished_jobs"], (2, 2))
        self.assertGreater(ledger["labor_cost"], 0)

    def test_input_aliases_and_tape_exhaustion(self):
        jobs, tape = [[1.0], [1.0]], [[1.0, 1.0]] * 2
        model = ServiceEffortMechanics(config(), jobs, tape)
        jobs[0][0] = 99
        tape[0][0] = 99
        old = model.observation()
        model.step((0.5, 0))
        _, receipt = model.step((0, 0))
        self.assertEqual(old.remaining_work, (1, 1))
        self.assertEqual(receipt.delivered_work, (0.5, 0))
        with self.assertRaises(ValueError):
            model.step((0, 0))

    def test_fixed_config_cases(self):
        fixture = json.loads(Path("experiments/configs/service_effort_mechanics_20260928.json").read_text())
        for case in fixture["cases"]:
            model = ServiceEffortMechanics(ServiceEffortConfig(**fixture["config"]), case["jobs"], case["response_tape"])
            estimator = PublicServiceResponseEstimator(fixture["prior"], fixture["estimator_alpha"])
            for action in fixture["commitments"]:
                _, receipt = model.step(action)
                estimator.observe(receipt)
            model.audit_conservation()
            if case["name"] == "backlog_limited":
                self.assertEqual(estimator.samples, [0, 0, 0])
                self.assertEqual(estimator.estimate, (1, 1, 1))
            if case["name"] == "persistent_shift":
                self.assertEqual(estimator.estimate, (0.5, 1.25, 1))

    def test_checker_classification_and_information_timing(self):
        fixture = json.loads(Path("experiments/configs/service_effort_mechanics_20260928.json").read_text())
        rows, summary = check_fixture(fixture)
        self.assertEqual(len(rows), 24)
        self.assertEqual(summary["first_distinguishable_decision_epoch"], 4)
        self.assertFalse(summary["research_performance_evidence"])
        self.assertFalse(summary["training_performed"])
        fixture["role"] = "scientific_experiment"
        with self.assertRaises(ValueError):
            check_fixture(fixture)

    def test_invalid_estimator_parameters(self):
        for prior, alpha in [([], 1), ([float("nan")], 1), ([1], 0), ([1], 2), ([1], float("nan"))]:
            with self.subTest(prior=prior, alpha=alpha), self.assertRaises(ValueError):
                PublicServiceResponseEstimator(prior, alpha)


if __name__ == "__main__":
    unittest.main()
