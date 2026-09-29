"""Mechanics and information-contract tests, not outcome-selection tests."""

import copy
from dataclasses import asdict, replace
import json
import math
import unittest

from evaluation.audit_service_queue_boundary import verify_closure, verify_step
from evaluation.check_service_queue_boundary import CONFIG, QueueTree, evaluate, make_config, response_at, validate
from src.baselines.public_service_response import PublicServiceResponseEstimator
from src.baselines.service_queue_control import plan_queue_mpc
from src.env.service_queue_network import BookedJob, advance, available_work, close_episode, closed, initial_observation


def fixture():
    return json.loads(CONFIG.read_text())


class QueueMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture()
        self.config = make_config(self.fixture, "booked_flow", "shared_bottleneck")

    def test_invalid_bookings_capacity_and_durations(self):
        job = self.config.jobs[0]
        for kwargs in ({"downstream_slots": 0}, {"holding_cost": math.nan},
                       {"jobs": (job, job)}, {"max_settlement_steps": 0},
                       {"jobs": (replace(job, downstream_steps=0),)},
                       {"jobs": (replace(job, site=3),)},
                       {"jobs": (replace(job, release=-1),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs)

    def test_action_validation_and_immutable_input(self):
        initial = initial_observation(self.config)
        snapshot = asdict(initial)
        for action in ((1, 1), (-1, 0), (float("nan"), 0), (0,)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                advance(initial, action, (1, 1), self.config)
        advance(initial, (1, 0), (1, 1), self.config)
        self.assertEqual(asdict(initial), snapshot)
        with self.assertRaises(TypeError):
            advance(snapshot, (0, 0), (1, 1), self.config)

    def test_effort_is_delayed_and_work_is_conserved(self):
        state = initial_observation(self.config)
        total_work, delivered = sum(state.remaining_work), 0.0
        committed, applied = 0.0, 0.0
        for step, action in enumerate(((1, 0), (0, 1), (0.5, 0.5), (0, 0), (0, 0))):
            state, receipt, _ = advance(state, action, (0.5, 1.5), self.config)
            delivered += sum(receipt.delivered_work)
            committed += sum(receipt.committed_hours)
            applied += sum(receipt.applied_hours)
            self.assertAlmostEqual(sum(state.remaining_work) + delivered, total_work)
            self.assertAlmostEqual(committed, applied + sum(sum(r) for r in state.pending_hours))
            if step == 0:
                self.assertEqual(receipt.applied_hours, (0, 0))
                self.assertEqual(receipt.delivered_work, (0, 0))

    def test_arrival_time_visible_before_action_not_charged_early(self):
        state = initial_observation(self.config)
        state, _, cost0 = advance(state, (0, 0), (1, 1), self.config)
        state, _, cost1 = advance(state, (0, 0), (1, 1), self.config)
        self.assertEqual(state.epoch, 2)
        self.assertEqual(available_work(state, self.config), (3, 3))
        self.assertEqual(cost0["holding"], 4)
        self.assertEqual(cost1["holding"], 4)
        state, _, cost2 = advance(state, (0, 0), (1, 1), self.config)
        self.assertEqual(available_work(state, self.config), (4, 4))
        self.assertEqual(cost2["holding"], 6)

    def test_whole_jobs_mandatory_duration_and_shared_capacity(self):
        cfg = replace(self.config, jobs=(BookedJob("a", 0, 0, 1, 2), BookedJob("b", 1, 0, 1, 2)))
        state = initial_observation(cfg)
        state, _, _ = advance(state, (0.5, 0.5), (1, 1), cfg)
        state, _, _ = advance(state, (0.5, 0.5), (1, 1), cfg)
        self.assertEqual(state.stages, ("support", "support"))
        state, _, _ = advance(state, (0, 0), (1, 1), cfg)
        self.assertEqual(state.stages, ("waiting", "waiting"))
        self.assertEqual(state.ready_epochs, (3, 3))
        state, _, cost = advance(state, (0, 0), (9, 9), cfg)
        self.assertEqual(cost["downstream_started"], ["a"])
        self.assertEqual(state.stages, ("active", "waiting"))
        self.assertEqual(state.downstream_remaining, (1, 0))
        state, _, cost = advance(state, (0, 0), (9, 9), cfg)
        self.assertEqual(cost["downstream_started"], [])
        self.assertEqual(state.stages, ("done", "waiting"))
        state, _, _ = advance(state, (0, 0), (9, 9), cfg)
        self.assertEqual(state.stages, ("done", "active"))

    def test_public_history_cannot_see_future_response(self):
        family = self.fixture["families"][1]
        tree = QueueTree(self.fixture, "booked_flow__shared_bottleneck__persistent_change", self.config, family)
        left, right = (0, ()), (1, ())
        self.assertEqual(tree.information_key(left), tree.information_key(right))
        for _ in range(2):
            left = tree.transition(left, "balanced")[1]
            right = tree.transition(right, "balanced")[1]
            self.assertEqual(tree.information_key(left), tree.information_key(right))
        left = tree.transition(left, "balanced")[1]
        right = tree.transition(right, "balanced")[1]
        self.assertNotEqual(tree.information_key(left), tree.information_key(right))
        self.assertEqual(tree.nodes[(0, ())][0], initial_observation(self.config))

    def test_censored_and_no_effort_receipts_do_not_identify_response(self):
        cfg = replace(self.config, jobs=(BookedJob("a", 0, 0, 0.25, 2),))
        state = initial_observation(cfg)
        estimator = PublicServiceResponseEstimator((1, 1), 1)
        state, receipt, _ = advance(state, (1, 0), (0.5, 1.5), cfg)
        estimator.observe(receipt)
        state, receipt, _ = advance(state, (0, 0), (0.5, 1.5), cfg)
        estimator.observe(receipt)
        self.assertEqual(receipt.observation_kind, ("backlog_limited", "no_effort"))
        self.assertEqual(estimator.samples, [0, 0])

    def test_terminal_closure_keeps_all_jobs_and_prepaid_effort(self):
        state = initial_observation(self.config)
        for _ in range(5):
            state, _, _ = advance(state, (1, 0), (0.5, 1.5), self.config)
        before = state
        cost, final, rows = close_episode(state, self.config, lambda epoch: (0.5, 1.5), record=True)
        self.assertEqual(rows[0]["receipt"]["applied_hours"], (1, 0))
        self.assertEqual(rows[0]["receipt"]["committed_hours"], (0, 0))
        self.assertTrue(closed(final))
        self.assertEqual(len(final.stages), 8)
        self.assertEqual(sum(final.remaining_work), 0)
        self.assertEqual(state, before)
        self.assertAlmostEqual(cost, sum(r["cost"]["total"] for r in rows))

    def test_closure_cannot_silently_truncate(self):
        cfg = replace(self.config, max_settlement_steps=1)
        with self.assertRaises(RuntimeError):
            close_episode(initial_observation(cfg), cfg, lambda epoch: (1, 1))

    def test_closure_exact_bound_is_allowed(self):
        cfg = replace(self.config, jobs=(BookedJob("a", 0, 0, 1, 1),), max_settlement_steps=1)
        state = replace(initial_observation(cfg), stages=("active",), remaining_work=(0,), ready_epochs=(0,), downstream_remaining=(1,))
        _, final, _ = close_episode(state, cfg, lambda epoch: (1, 1))
        self.assertTrue(closed(final))

    def test_mpc_matches_one_decision_enumeration_and_counts_queries(self):
        state = initial_observation(self.config)
        actual = plan_queue_mpc(state, (1, 1), self.config, self.fixture["actions"], 1)
        choices = []
        for name, action in self.fixture["actions"].items():
            after, _, cost = advance(state, action, (1, 1), self.config)
            terminal, _, _ = close_episode(after, self.config, lambda epoch: (1, 1))
            choices.append((cost["total"] + terminal, name))
        value, name = min(choices)
        self.assertEqual(actual[0], name)
        self.assertAlmostEqual(actual[1], value)
        self.assertGreater(actual[2], 4)
        with self.assertRaises(TypeError):
            plan_queue_mpc(asdict(state), (1, 1), self.config, self.fixture["actions"], 1)

    def test_smoke_rollout_passes_independent_rows_and_terminal_audit(self):
        cell = "booked_flow__shared_bottleneck__persistent_change"
        world = self.fixture["families"][1]["worlds"][0]
        episode, rows = evaluate(self.fixture, cell, self.config, world, "fixed_balanced")
        for row in rows:
            verify_step(row, self.fixture, cell, world)
        verify_closure(episode["settlement"], rows[-1]["after"], self.fixture, cell, world)

    def test_independent_auditor_rejects_premature_completion_and_bad_cost(self):
        cell = "booked_flow__shared_bottleneck__no_change"
        world = self.fixture["families"][0]["worlds"][0]
        _, rows = evaluate(self.fixture, cell, self.config, world, "fixed_balanced")
        bad = copy.deepcopy(rows[0])
        bad["cost"]["total"] += 1
        with self.assertRaises(AssertionError):
            verify_step(bad, self.fixture, cell, world)
        bad = copy.deepcopy(rows[0])
        bad["after"]["stages"] = ["done"] * 8
        with self.assertRaises(AssertionError):
            verify_step(bad, self.fixture, cell, world)

    def test_locked_factorial_rejects_scope_changes(self):
        validate(self.fixture)
        for field, value in (("horizon", 6), ("role", "patient_confirmation"), ("release_patterns", {"batch": [0]})):
            f = fixture()
            f[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(f)


if __name__ == "__main__":
    unittest.main()
