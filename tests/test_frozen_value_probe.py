import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.frozen_value_probe import (
    Budget, behavioral_digest, collect_draw, discovery_choice, recorded_step, result_from_steps, streams, write_json,
)
from src.rl.experiment import COST_COMPONENT_METRICS


class FakeEnv:
    def __init__(self):
        self.t = 0
        self.rng = np.random.default_rng(1)

    def observation(self):
        return np.array([self.t / 52], dtype=np.float32)

    def step(self, action):
        self.t += 1
        info = {key: 0.0 for key in COST_COMPONENT_METRICS}
        info.update(cost=10.0, base_cost=10.0, reagent_purchase_cost=10.0)
        info.update(patients_lost=np.zeros(1), patients_completed=np.ones(1),
                    completion_service_level=1., patient_ineligibility_during_manufacturing_rate=0.,
                    specimen_route_count=0, specimen_route_events=[])
        return self.observation(), -10., self.t == 52, info

    def state_dict(self):
        return {"scalars": {"t": self.t}, "rng_state": self.rng.bit_generator.state,
                "arrays": {"reagents": [1], "bioreactors": [[1]]}, "patient_queues": [[]],
                "in_production_patients": [[[]]], "specimen_transits": [],
                "product_return_transits": [], "finished_product_buckets": []}

    def assert_identity_conservation(self):
        return {"active_count": 0, "terminal_count": 0}


class FakePolicy:
    def act(self, state, env=None):
        return np.zeros(1)


class FrozenValueProbeTests(unittest.TestCase):
    def test_stream_count_and_separation(self):
        spec = json.loads(Path("experiments/configs/replacement_fixed_window_pilot_20260929.json").read_text())
        values = streams(spec)
        self.assertEqual(len(values), 195)
        self.assertEqual(len(set(values.values())), 195)
        self.assertGreater(min(values.values()), 2 ** 64)
        self.assertNotIn(int(spec["rng_namespace"]) + spec["engineering_seed_offset"], values.values())

    def test_budget_stops_before_step(self):
        env, budget = FakeEnv(), Budget(1, 100)
        recorded_step(env, np.zeros(1), budget)
        with self.assertRaises(RuntimeError):
            recorded_step(env, np.zeros(1), budget)
        self.assertEqual(env.t, 1)

    def test_wall_limit(self):
        with patch("src.rl.frozen_value_probe.time.monotonic", return_value=0):
            budget = Budget(10, 1)
        with patch("src.rl.frozen_value_probe.time.monotonic", return_value=2):
            with self.assertRaises(TimeoutError):
                budget.consume()
        self.assertEqual(budget.steps, 0)

    def test_terminal_and_no_post_done(self):
        env, budget = FakeEnv(), Budget(100, 100)
        env.t = 51
        row = recorded_step(env, np.zeros(1), budget)
        self.assertTrue(row["objective_terminal"])
        self.assertEqual(row["scaled_reward"], -1e-8)
        with self.assertRaises(ValueError):
            recorded_step(env, np.zeros(1), budget)
        self.assertEqual(budget.steps, 1)

    def test_truncated_is_not_value(self):
        row = recorded_step(FakeEnv(), np.zeros(1), Budget(1, 100))
        with self.assertRaises(ValueError):
            result_from_steps([row], None)

    def test_invalid_cost_rejected(self):
        env = FakeEnv()
        original = env.step
        def corrupt(action):
            obs, reward, done, info = original(action)
            info["reagent_purchase_cost"] = float("nan")
            return obs, reward, done, info
        env.step = corrupt
        with self.assertRaises(ValueError):
            recorded_step(env, np.zeros(1), Budget(1, 100))

    def test_identity_ignores_only_reporting_counter(self):
        state = {"scalars": {"t": 1, "cumulative_blocked_specimen_requests": 5}, "rng_state": {"state": 1}}
        info = {"cost": 4., "specimen_route_events": []}
        other = copy.deepcopy(state)
        other["scalars"]["cumulative_blocked_specimen_requests"] = 99
        self.assertEqual(behavioral_digest(state, info), behavioral_digest(other, info))
        other["rng_state"]["state"] = 2
        self.assertNotEqual(behavioral_digest(state, info), behavioral_digest(other, info))
        self.assertEqual(state["scalars"]["cumulative_blocked_specimen_requests"], 5)

    def test_cost_and_routes_part_of_identity(self):
        state = {"scalars": {}}
        a = {"cost": 4., "specimen_route_events": []}
        self.assertNotEqual(behavioral_digest(state, a), behavioral_digest(state, dict(a, cost=5.)))
        self.assertNotEqual(behavioral_digest(state, a), behavioral_digest(state, dict(a, specimen_route_events=[1])))

    def test_discovery_tie_and_no_safety_filter(self):
        rows = [{"action_index": i, "outcome": {"total_cost": c, "patients_lost": 100 if i == 1 else 0}}
                for i, c in enumerate([4, 3, 3, 5, 6, 7]) for _ in range(8)]
        selected = discovery_choice(rows)
        self.assertEqual(selected["selected_action"], 1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text())["value"], 1)

    def test_actual_aliases_share_tail_with_exact_budget(self):
        env, budget = FakeEnv(), Budget(7, 100)
        env.t = 50
        choices = [{"index": i, "request": np.array([i / 10]), "label": str(i),
                    "map_reachable_certified": i == 0} for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            rows = collect_draw(env, FakePolicy(), choices, 9, budget, Path(directory), "fixture")
        self.assertEqual(len(rows), 6)
        self.assertEqual(budget.steps, 7)
        self.assertEqual(env.t, 50)
        self.assertIsNone(rows[0]["tail_from"])
        self.assertTrue(all(r["tail_from"] == rows[0]["trace"] for r in rows[1:]))
        self.assertEqual({r["outcome"]["total_cost"] for r in rows}, {20.0})
        self.assertEqual({r["outcome"]["step_count"] for r in rows}, {2})

    def test_independent_interval_zero_and_uncertainty(self):
        from evaluation.verify_replacement_fixed_window_pilot import interval
        self.assertEqual(interval([0.] * 8)["conditional_descriptive_t7_interval"], [0., 0.])
        result = interval(list(range(8)))
        self.assertEqual(result["mean"], 3.5)
        self.assertGreater(result["se"], 0)


if __name__ == "__main__":
    unittest.main()
