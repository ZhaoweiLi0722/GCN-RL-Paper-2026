"""Fake public-model/controller contracts; no native environment or model fits."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from src.baselines.capacity_policy_tail_mpc import collect_policy_tail_pair


def public(epoch):
    return SimpleNamespace(common=SimpleNamespace(epoch=epoch))


class Forecast:
    def __init__(self, view, proposal, interval_filter, lifecycle, quantile):
        self.epoch = view.common.epoch
        self.quantile = quantile
        self.hours = []

    def public_view(self):
        return public(self.epoch)

    def adaptive(self):
        return np.ones(4)

    def terminal_value(self):
        return 64. - self.epoch

    def step(self, hours, base):
        self.hours.append(tuple(hours))
        self.epoch += 1
        return 1. + np.sum(hours) + self.quantile / 1000


class Controller:
    def __init__(self, proposal=None, **kwargs):
        self.proposal = proposal or dict(id_mpc=dict(response_quantiles=[.1, .5, .9]))
        self.base_control = lambda *a, **k: np.zeros(16)
        self.filter = self.lifecycle = None
        self.epoch, self.recorded, self.last_plan = 0, -1, None

    def state_dict(self):
        return dict(filter=dict(epoch=self.epoch),
                    lifecycle=dict(epoch=self.epoch, recorded_epoch=self.recorded))

    def load_state_dict(self, s):
        assert s["capture_epochs"] == ()
        self.epoch = s["filter"]["epoch"]
        self.recorded = s["lifecycle"]["recorded_epoch"]

    def candidates(self, view):
        return [(np.full(4, i / 10), i >= 8) for i in range(16)]

    def record_operation(self, view, action):
        assert self.epoch == view.common.epoch and self.recorded < self.epoch
        self.recorded = self.epoch

    def observe(self, view, *, before_filter):
        assert view.common.epoch == self.epoch + 1
        self.epoch += 1
        before_filter(dict(hypothesis_transitions=100))

    def act(self, view, *, before_query, before_filter, role):
        assert self.epoch == view.common.epoch and self.recorded < self.epoch
        before_query(dict(model_epochs=384))
        self.last_plan = dict(epoch=self.epoch, chosen=1)
        return np.full(4, 2.)


def features(model):
    return np.full((4, 31), model.epoch, dtype=np.float32)


class CollectorTests(unittest.TestCase):
    def test_actual_controller_public_history_interface_on_empty_artificial_records(self):
        from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
        from tests.test_capacity_completion_control import PROPOSAL, view, no_operations
        proposal = copy.deepcopy(PROPOSAL)
        proposal["synthetic_system"]["demand_rates"] = [0.] * 4
        class ZeroValue:
            def residuals(self, x):
                return np.zeros(len(x))
        reference = CapacityPlannerTailMPC(proposal, base_control=no_operations, scientific=False)
        for epoch in range(40):
            reference.observe(view(epoch))
            if epoch < 39:
                reference.record_operation(view(epoch), no_operations(None, None))
        initial = copy.deepcopy(reference.state_dict())
        out = collect_policy_tail_pair(view(39), reference, 0, value=ZeroValue(),
            continuation_sha256="a" * 64, before_clone=lambda p: None,
            before_prefix=lambda p: None, before_tail=lambda p: None,
            before_query=lambda p: None, before_filter=lambda p: None,
            before_plan=lambda p: None, after_plan=lambda p: None, after_filter=lambda p: None,
            scientific=False)
        self.assertEqual(reference.state_dict(), initial)
        for records in out.values():
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r["epochs"] == list(range(47, 65)) for r in records))
        self.assertTrue(all(r["plans"][0] is not None for r in out["frozen_mpc"]))

    def collect(self, epoch=0, candidate=1, **changes):
        reference = Controller()
        reference.epoch, reference.recorded = epoch, epoch - 1
        events = []
        kwargs = dict(value=object(), continuation_sha256="a" * 64,
            forecast_factory=Forecast, controller_factory=Controller,
            feature_adapter=features)
        for event in ("clone", "prefix", "tail", "query", "filter"):
            kwargs["before_" + event] = lambda p, event=event: events.append((event, p))
        for event in ("before_plan", "after_plan", "after_filter"):
            kwargs[event] = lambda p, event=event: events.append((event, p))
        kwargs.update(changes)
        before = copy.deepcopy(reference.state_dict())
        output = collect_policy_tail_pair(public(epoch), reference, candidate, **kwargs)
        self.assertEqual(before, reference.state_dict())
        return output, events

    def test_matched_endpoints_frozen_policy_and_rule_diverge_without_native_calls(self):
        out, events = self.collect()
        self.assertEqual(set(out), {"adaptive", "frozen_mpc"})
        self.assertEqual(sum(e == "clone" for e, _ in events), 9)
        self.assertEqual(sum(e == "prefix" for e, _ in events), 24)
        self.assertEqual(sum(e == "tail" for e, _ in events), 336)
        self.assertEqual(sum(e == "query" for e, _ in events), 120)
        self.assertEqual(sum(e == "filter" for e, _ in events), 192)
        for a, p in zip(out["adaptive"], out["frozen_mpc"]):
            np.testing.assert_array_equal(a["features"][0], p["features"][0])
            self.assertEqual(a["prefix_cost"], p["prefix_cost"])
            self.assertEqual(a["epochs"], list(range(8, 65)))
            self.assertEqual(p["continuation_sha256"], "a" * 64)
            self.assertIsNone(a["continuation_sha256"])
            np.testing.assert_array_equal(a["actions"][:40], np.ones((40, 4)))
            np.testing.assert_array_equal(p["actions"][:40], np.full((40, 4), 2.))
            np.testing.assert_array_equal(p["actions"][40:], np.zeros((16, 4)))
            self.assertEqual(p["heuristics"][-1], 0.)
            self.assertEqual(p["costs"].dtype, np.float64)
            self.assertFalse(np.shares_memory(a["features"], p["features"]))

    def test_late_tail_has_no_planning_and_no_new_support_commitments(self):
        out, events = self.collect(epoch=47)
        self.assertFalse(any(e == "query" for e, _ in events))
        for role, records in out.items():
            for r in records:
                self.assertEqual(r["epochs"], list(range(55, 65)))
                np.testing.assert_array_equal(r["actions"], np.zeros((9, 4)))
                self.assertTrue(all(p is None for p in r["plans"]))

    def test_admission_rejection_precedes_construction(self):
        calls = []
        def denied(payload):
            raise PermissionError("budget exhausted")
        with self.assertRaises(PermissionError):
            self.collect(before_clone=denied, forecast_factory=lambda *a: calls.append(a))
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            self.collect(before_query=None)
        with self.assertRaises(ValueError):
            self.collect(continuation_sha256="not-a-hash")

    def test_callback_failure_does_not_modify_reference(self):
        saved = []
        with self.assertRaises(PermissionError):
            self.collect(before_query=lambda p: (_ for _ in ()).throw(PermissionError("charged")),
                         on_failure=saved.append)
        self.assertEqual(saved[0]["phase"], "frozen_mpc")
        self.assertEqual(saved[0]["controller"]["filter"]["epoch"], 8)
        self.assertFalse(saved[0]["automatic_resume"])
        self.assertTrue(saved[0]["primitive_may_be_partially_applied"])

    def test_prescribed_candidate_and_prefix_switch(self):
        out, _ = self.collect(candidate=9)
        # Two initial .9h/site steps then six adaptive 1h/site steps.
        expected = 2 * (1 + 3.6) + 6 * 5
        self.assertAlmostEqual(out["adaptive"][0]["prefix_cost"], expected + 8 * .0001)
        self.assertTrue(all(r["candidate"] == 9 for rows in out.values() for r in rows))
        for invalid in (-1, 16, True, 1.5):
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                self.collect(candidate=invalid)


if __name__ == "__main__":
    unittest.main()
