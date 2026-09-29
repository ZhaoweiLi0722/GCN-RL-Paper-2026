"""Scalar parity, full-tail, information and deterministic forecast checks."""

from dataclasses import asdict
import json
import unittest

import numpy as np

from evaluation.check_completion_feedback_mechanics import CONFIG, make_config
from src.baselines.completion_rollout_control import action_grid, named_seed, plan_rollout, reservation_action, zero_is_dominant
from src.env.completion_feedback_batch import STAGES, forecast
from src.env.completion_feedback_queue import advance, booked_backlog_action, closed, initial_observation


def scalar_forecast(state, prefix, rates, noise, cfg, boundary):
    cost, trace = 0.0, []
    for offset, draw in enumerate(noise):
        if closed(state):
            break
        if state.epoch == boundary:
            action = (0, 0)
        elif offset < len(prefix) and state.epoch < boundary:
            action = prefix[offset]
        else:
            action = booked_backlog_action(state, cfg)
        state, _, charges = advance(state, action, rates, draw, cfg)
        cost += charges["total"]
        trace.append((state, charges["total"]))
    if not closed(state):
        raise RuntimeError("unpaid scalar tail")
    return cost, trace


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(CONFIG.read_text())

    def test_grid_is_exact_shared_simplex(self):
        grid = action_grid(4)
        self.assertEqual(len(grid), 15)
        self.assertIn((0.5, 0.5), grid)
        self.assertTrue(all(min(a) >= 0 and sum(a) <= 1 for a in grid))
        self.assertEqual(grid[0], (0, 0))
        with self.assertRaises(ValueError):
            action_grid(0)

    def test_batched_scalar_cost_parity_across_actions_sites_and_capacity(self):
        prefixes = [((0, 0), (1, 0)), ((0.25, 0.75), (0.5, 0.5)), ((0.5, 0.5), (0, 1))]
        noise = np.random.Generator(np.random.PCG64(73)).random((3, 288, 2))
        for slots, rates in ((1, (1,1)), (4, (0.5,1.5)), (1, (1.5,0.5))):
            cfg = make_config(self.fixture, slots)
            state = initial_observation(cfg)
            result = forecast(state, prefixes, rates, noise, cfg, 32)
            for i, prefix in enumerate(prefixes):
                for j, tape in enumerate(noise):
                    expected, _ = scalar_forecast(state, prefix, rates, tape, cfg, 32)
                    self.assertAlmostEqual(result["sample_costs"][i,j], expected)

    def test_every_batched_public_state_matches_scalar_state(self):
        cfg = make_config(self.fixture, 1)
        state = initial_observation(cfg)
        noise = np.random.Generator(np.random.PCG64(17)).random((1, 288, 2))
        prefix = ((0.75, 0.25), (0, 1))
        result = forecast(state, [prefix], (0.5, 1.5), noise, cfg, 32, trace=True)
        _, expected = scalar_forecast(state, prefix, (0.5, 1.5), noise[0], cfg, 32)
        self.assertEqual(len(result["trace"]), len(expected))
        for batched, (scalar, cost) in zip(result["trace"], expected):
            self.assertEqual(batched["epoch"], scalar.epoch)
            self.assertEqual([STAGES[i] for i in batched["stages"][0]], list(scalar.stages))
            for key in ("ready_epochs", "downstream_remaining", "pending_hours", "previous_request"):
                self.assertEqual(batched[key][0].tolist(), list(getattr(scalar, key)))
            self.assertAlmostEqual(batched["cost"][0], cost)

    def test_forecast_full_closure_crosses_early_decision_boundary(self):
        cfg = make_config(self.fixture, 1)
        noise = np.random.Generator(np.random.PCG64(13)).random((2, 257, 2))
        state = initial_observation(cfg)
        prefix = ((1,0), (1,0))
        actual = forecast(state, [prefix], (1,1), noise, cfg, 1)
        for j in range(2):
            expected, trace = scalar_forecast(state, prefix, (1,1), noise[j], cfg, 1)
            self.assertGreater(len(trace), 24)
            self.assertAlmostEqual(actual["sample_costs"][0,j], expected)

    def test_forecast_unsettled_path_fails_not_free_terminal(self):
        cfg = make_config(self.fixture, 1)
        with self.assertRaises(RuntimeError):
            forecast(initial_observation(cfg), [((0,0),)], (1,1), np.full((1,2,2), 0.99), cfg, 32)

    def test_forecast_inputs_unchanged(self):
        cfg = make_config(self.fixture, 1)
        state = initial_observation(cfg)
        before = asdict(state)
        noise = np.random.Generator(np.random.PCG64(19)).random((2,288,2))
        copy = noise.copy()
        prefixes = np.array([[[0.5,0.5]]])
        forecast(state, prefixes, (1,1), noise, cfg, 32)
        self.assertEqual(asdict(state), before)
        np.testing.assert_array_equal(noise, copy)
        np.testing.assert_array_equal(prefixes, [[[0.5,0.5]]])

    def test_invalid_forecast_inputs(self):
        cfg = make_config(self.fixture, 1)
        s = initial_observation(cfg)
        for prefix, rates, noise in (([[[1,1]]],(1,1),np.zeros((1,288,2))),
                                      ([[[0,0]]],(0,1),np.zeros((1,288,2))),
                                      ([[[0,0]]],(1,1),np.ones((1,288,2)))):
            with self.assertRaises(ValueError):
                forecast(s, prefix, rates, noise, cfg, 32)

    def test_stream_separation_and_budget_prefix(self):
        actual = named_seed("test", "actual", 0)
        planning = named_seed("test", "planning", 0)
        self.assertNotEqual(actual, planning)
        self.assertNotEqual(actual, named_seed("test", "actual", 1))
        small = np.random.Generator(np.random.PCG64(planning)).random((16,288,2))
        large = np.random.Generator(np.random.PCG64(planning)).random((64,288,2))
        np.testing.assert_array_equal(small, large[:16])

    def test_same_information_and_estimate_produces_identical_plans(self):
        cfg = make_config(self.fixture, 1)
        state, grid = initial_observation(cfg), action_grid(4)
        first = plan_rollout(state, (1,1), cfg, grid, 4, 1, 71, 32)
        self.assertEqual(first, plan_rollout(state, (1,1), cfg, grid, 4, 1, 71, 32))
        self.assertEqual(first[1]["candidates"], 15)
        self.assertIn(first[0], grid)
        self.assertGreater(first[1]["transition_queries"], 0)

    def test_two_request_prefix_is_not_one_step_immediate_cost(self):
        cfg = make_config(self.fixture, 1)
        action, detail = plan_rollout(initial_observation(cfg), (1,1), cfg, action_grid(2), 4, 2, 19, 32)
        self.assertEqual(detail["candidates"], 36)
        self.assertEqual(len(detail["selected_prefix"]), 2)
        self.assertGreater(min(detail["candidate_means"]), 10)
        self.assertEqual(action, detail["selected_prefix"][0])

    def test_reservation_rule_responds_to_pending_completion(self):
        cfg = make_config(self.fixture, 1)
        state, grid = initial_observation(cfg), action_grid(4)
        self.assertEqual(reservation_action(state, (1,1), cfg, grid), (0.5,0.5))
        state, _, _ = advance(state, (0.5,0.5), (1,1), (0.99,0.99), cfg)
        self.assertEqual(reservation_action(state, (1,1), cfg, grid), (0.25,0.25))

    def test_no_current_or_next_support_has_dominant_zero(self):
        f = dict(self.fixture, booked_releases=[8])
        cfg = make_config(f, 1)
        state = initial_observation(cfg)
        self.assertTrue(zero_is_dominant(state,cfg))
        action, detail = plan_rollout(state,(1,1),cfg,action_grid(4),64,2,0,32)
        self.assertEqual(action,(0,0))
        self.assertEqual(detail["transition_queries"],0)


if __name__ == "__main__":
    unittest.main()
