"""Exact prospective accounting and artificial forecasts; no scientific calls."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC, forecast_tail
from src.rl.capacity_planner_tail_design import capture_epochs, numeric_contract, schedule, worlds


ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "experiments/configs/capacity_planner_tail_20261004.json").read_text())


class FakeForecast:
    def __init__(self, epoch=8):
        self.epoch = epoch
        self.hours = []

    def adaptive(self):
        return np.ones(4)

    def terminal_value(self):
        return float(64 - self.epoch)

    def step(self, hours, base):
        self.hours.append(tuple(hours))
        self.epoch += 1
        return 1. + float(np.sum(hours))


def features(model):
    return np.full((4, 31), model.epoch / 64, dtype=np.float32)


class DesignTests(unittest.TestCase):
    def test_complete_fake_dispatch_order_seals_before_tests(self):
        completed, sources, updates, seals, barrier = [], set(), {}, set(), False
        for task in schedule(STUDY):
            event = task["event"]
            completed.append(event)
            if event == "collect_reference":
                self.assertFalse(barrier)
                w = task["world"]
                sources.add((w["block"], w["index"]))
            elif event == "fit":
                w = task["world"]
                self.assertIn((w["block"],w["index"]),sources)
                key = w["block"],task["role"]
                updates[key] = updates.get(key,0) + task["updates"]
            elif event == "seal":
                key = task["block"],task["role"]
                self.assertEqual(updates[key],task["new_updates"])
                seals.add(key)
            elif event == "all_models_sealed":
                self.assertEqual(len(seals),15)
                barrier=True
            elif event == "evaluate":
                self.assertTrue(barrier)
                self.assertEqual(task["updates"],0)
        self.assertEqual(completed.count("collect_reference"),120)
        self.assertEqual(completed.count("fit"),360)
        self.assertEqual(completed.count("evaluate"),360)
        self.assertEqual(sum(updates.values()),11520)

    def test_exact_counts_and_no_run_authority(self):
        counts = numeric_contract(STUDY)
        self.assertFalse(STUDY["scientific_execution_authorized"])
        self.assertEqual(counts["limits"]["total_predictor_epochs"], 10327680)
        self.assertEqual(counts["limits"]["forward_calls"], 30000)
        self.assertEqual(counts["limits"]["value_optimizer_steps"], 11520)
        self.assertEqual(sum(counts["tail_rows_per_world_index"]), 74880)
        self.assertEqual(counts["phase_counts"]["frozen_evaluation"]["optimizer_steps"], 0)
        for key in ("forward_calls", "planning_epochs", "training_tail_epochs", "native_steps"):
            bad = copy.deepcopy(STUDY)
            bad["budget"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                numeric_contract(bad)

    def test_balanced_capture_epochs_and_nonoverlapping_new_worlds(self):
        self.assertEqual(sorted(e for i in range(24) for e in capture_epochs(i)), list(range(48)))
        rows = [w for b in range(5) for p in ("reference", "evaluation") for w in worlds(STUDY, p, b)]
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({r["seed"] for r in rows}), 180)
        for i in (True, -1, 24, 1.5):
            with self.subTest(index=i), self.assertRaises(ValueError):
                capture_epochs(i)

    def test_tail_is_copy_charged_before_work_and_settled(self):
        model, calls = FakeForecast(), []
        result = forecast_tail(model, None, before_clone=lambda p:calls.append(("clone", p)),
            before_step=lambda p:calls.append(("step", p)), feature_adapter=features)
        self.assertEqual(model.epoch, 8)
        self.assertEqual(model.hours, [])
        self.assertEqual(len(calls), 57)
        self.assertEqual(calls[0][0], "clone")
        self.assertEqual(result["features"].shape, (57, 4, 31))
        self.assertEqual(result["heuristics"][-1], 0)
        self.assertEqual(result["epochs"], list(range(8, 65)))
        np.testing.assert_array_equal(result["costs"][:40], np.full(40, 5.))
        np.testing.assert_array_equal(result["costs"][40:], np.ones(16))
        with self.assertRaises(ValueError):
            forecast_tail(model, None, before_clone=None, before_step=None, feature_adapter=features)

    def controller(self, horizon, capture=()):
        # Skip all real controller/filter constructors in this arithmetic fixture.
        control = object.__new__(CapacityPlannerTailMPC)
        control.planning_horizon, control.capture_epochs = horizon, capture
        control.value, control.scientific, control.base_control = None, True, None
        control.proposal = dict(id_mpc=dict(response_quantiles=[.1,.5,.9], summary_weights=[.25,.5,.25]))
        control.filter, control.lifecycle = None, None
        control.observe = lambda *args, **kwargs: None
        control.candidates = lambda view: [(np.zeros(4), False)] * 16
        control.forecast_factory = lambda view, *args: FakeForecast(view.common.epoch)
        control.feature_adapter = features
        control.before_tail_clone = lambda p: None
        control.before_tail_step = lambda p: None
        return control

    def test_H16_doubles_queries_and_tail_capture_keeps_H8_action(self):
        view = SimpleNamespace(common=SimpleNamespace(epoch=0))
        records = []
        plain, captured, long = self.controller(8), self.controller(8, (0,)), self.controller(16)
        a = plain.act(view, before_query=records.append)
        self.assertEqual(len(records), 384)
        b = captured.act(view, before_query=lambda p: None)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(plain.last_plan, captured.last_plan)
        self.assertEqual(len(captured.last_training_tails), 48)
        records.clear()
        long.act(view, before_query=records.append)
        self.assertEqual(len(records), 768)
        self.assertEqual(long.last_plan["planning_horizon"], 16)


if __name__ == "__main__":
    unittest.main()
