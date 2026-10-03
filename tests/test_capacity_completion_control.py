"""Artificial public records only: no native host, fit, neural or random tape."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.capacity_completion_control import (
    CapacityCompletionControl, CompletionIntervalFilter, NODE_SUMMARY_FIELDS,
    PublicPatientForecast, _condition_prefix, project_hours, proximal_hours,
)
from src.env.patient_support_public import (
    PublicPatientRecord, PublicSupportOperations, PublicSupportServiceEvent,
)
from src.rl.public_support_collector import PublicSupportInput
from src.rl.public_support_input import PublicSupportControlInput


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = json.loads((ROOT / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json").read_text())
ZERO = (0.,) * 4
NO_BUY = ZERO * 3 + (-1.,) * 4


def patient(pid, site=0, *, status="waiting", ready=False, survival=1., age=0, specimen_age=0):
    return PublicPatientRecord(pid, 0, status, site, site if status != "in_transit" else None,
                               site if status == "in_production" else None, age, specimen_age, survival, ready)


def view(epoch=0, patients=(), *, order=None, eligible=None, completed=None,
         ordinary=(4.,) * 4, applied=ZERO, previous=ZERO, pending=(ZERO, ZERO),
         reactors=None, reagents=(8.,) * 4, orders=(ZERO, ZERO),
         reagent_transfers=(ZERO,), capacity_transfers=(ZERO,), arrivals=ZERO):
    if order is None:
        order = tuple(tuple(p.patient_id for p in patients if p.status == "waiting" and p.material_site == i) for i in range(4))
    eligible = eligible or ((),) * 4
    completed = completed or ((),) * 4
    event = None if epoch == 0 else PublicSupportServiceEvent(epoch - 1, epoch, eligible, completed, ordinary, applied)
    history = tuple(((0.,) * 6,) * 7 + ((previous[i], applied[i], ordinary[i], len(eligible[i]), len(completed[i]), 1.),)
                    if epoch else ((0.,) * 6,) * 8 for i in range(4))
    reactors = reactors if reactors is not None else ((8., 0., 0., 0., 0.),) * 4
    ops = PublicSupportOperations(epoch, ("S0", "S1", "S2", "S3"), tuple(patients), order,
                                  reagents, reactors, reagent_transfers, capacity_transfers,
                                  orders, (1.,) * 4, (1.5,) * 4, event, current_arrivals=arrivals)
    by_id = {p.patient_id: p for p in patients}
    common = PublicSupportInput(epoch, ops.site_ids, (), history, pending,
                                tuple(sum(by_id[pid].support_complete for pid in q) for q in order))
    return PublicSupportControlInput(common, ops)


def no_operations(v, proposal, tail=False):
    return np.asarray(NO_BUY)


class IntervalTests(unittest.TestCase):
    def test_point_equality_backlog_censoring_and_total_staff(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        f.update(view(patients=(patient("a"),)))
        f.update(view(1, (patient("a", ready=True),), eligible=(("a",), (), (), ()), completed=(("a",), (), (), ())))
        np.testing.assert_allclose(f.weights[0], [0, 0, 1/3, 1/3, 1/3])
        self.assertNotIn("a", f.intervals)
        self.assertEqual(f.reset_events, [])
        self.assertEqual(f.node_summaries().shape, (4, 9))
        self.assertEqual(f.summary_fields, NODE_SUMMARY_FIELDS)
        g = CompletionIntervalFilter(PROPOSAL, scientific=False)
        g.update(view(patients=(patient("a"),)))
        g.update(view(1, (patient("a", ready=True),), ordinary=(2.,)*4, applied=(2.,)*4,
                      eligible=(("a",), (), (), ()), completed=(("a",), (), (), ())))
        np.testing.assert_array_equal(f.weights, g.weights)

    def test_zero_completions_retains_positive_efficiency_and_partial_id_work(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        f.update(view(patients=(patient("a"), patient("b"))))
        f.update(view(1, (patient("a"), patient("b")), eligible=(("a", "b"), (), (), ())))
        np.testing.assert_allclose(f.weights[0], [.5, .5, 0, 0, 0])
        self.assertEqual(f.intervals["a"]["bounds_by_response"][:2], ((2., 2.), (1., 1.)))
        self.assertEqual(f.intervals["b"]["bounds_by_response"][0], (4., 4.))
        self.assertEqual(f.residual_means["a"], 1.5)

    def test_priority_swap_does_not_reset_work_or_use_fifo_head_only(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        f.update(view(patients=(patient("a"), patient("b"))))
        f.update(view(1, (patient("a"), patient("b")), eligible=(("a", "b"), (), (), ())))
        f.update(view(2, (patient("a"), patient("b")), order=(("b", "a"), (), (), ()),
                      eligible=(("b", "a"), (), (), ())))
        self.assertLess(f.residual_means["a"], 4)
        self.assertLess(f.residual_means["b"], 4)
        self.assertFalse(f.reset_events)

    def test_transfer_carries_source_hull_and_zero_hours_do_not_reset_to_four(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        f.update(view(patients=(patient("a"),)))
        f.update(view(1, (patient("a"),), eligible=(("a",), (), (), ())))
        f.update(view(2, (patient("a", status="in_transit"),)))
        self.assertEqual(f.intervals["a"]["site"], 0)
        self.assertEqual(f.intervals["a"]["bounds_by_response"], ((1., 2.),)*5)
        f.update(view(3, (patient("a", site=1),), ordinary=ZERO, eligible=((), ("a",), (), ())))
        self.assertEqual(f.intervals["a"]["site"], 1)
        self.assertTrue(all(pair == (1., 2.) for pair in f.intervals["a"]["bounds_by_response"]))

    def test_prefix_interval_likelihood_and_hulls(self):
        prior = {"a": (1., 3.), "b": (2., 4.), "c": (4., 4.)}
        weight, boxes = _condition_prefix(prior, ("a", "b", "c"), ("a",), 2.)
        self.assertAlmostEqual(weight, .5)
        self.assertEqual(boxes["a"], (0., 0.))
        self.assertEqual(boxes["b"], (1., 4.))
        self.assertEqual(boxes["c"], (4., 4.))
        self.assertEqual(_condition_prefix({"a": (4., 4.)}, ("a",), (), 4.)[0], 0)
        self.assertEqual(_condition_prefix({"a": (4., 4.)}, ("a",), ("a",), 4.)[0], 1)

    def test_impossible_receipt_resets_logs_and_never_reopens_retired(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        f.update(view(patients=(patient("a"), patient("b"))))
        f.update(view(1, (patient("a"), patient("b", status="lost")), ordinary=(20.,)*4,
                      eligible=(("a", "b"), (), (), ())))
        np.testing.assert_allclose(f.weights[0], [.2]*5)
        self.assertEqual(f.reset_events[0]["patient_ids"], ["a"])
        self.assertEqual(f.intervals["a"]["bounds_by_response"], ((0., 4.),)*5)
        self.assertNotIn("b", f.intervals)
        self.assertEqual(f.node_summaries()[0, -1], 1)
        with self.assertRaisesRegex(ValueError, "regained work"):
            f.update(view(2, (patient("a"), patient("b"))))

    def test_each_pair_traversal_charged_before_work_and_json_restore(self):
        f = CompletionIntervalFilter(PROPOSAL)
        v0 = view(patients=(patient("a"),))
        v1 = view(1, (patient("a"),), eligible=(("a",), (), (), ()))
        f.update(v0)
        charged = []
        original = _condition_prefix
        calls = []
        def traverse(*args):
            self.assertEqual(len(charged), len(calls) + 1)
            calls.append(1)
            return original(*args)
        with patch("src.baselines.capacity_completion_control._condition_prefix", side_effect=traverse):
            f.update(v1, before_filter=charged.append)
        self.assertEqual(len(calls), 100)
        self.assertEqual(len(charged), 100)
        self.assertEqual(sum(e["hypothesis_transitions"] for e in charged), 100)
        self.assertEqual({e["old_response_index"] for e in charged}, set(range(5)))
        self.assertEqual({e["new_response_index"] for e in charged}, set(range(5)))
        restored = CompletionIntervalFilter(PROPOSAL)
        restored.load_state_dict(json.loads(json.dumps(f.state_dict())))
        np.testing.assert_array_equal(restored.update(v1), f.node_summaries())
        self.assertEqual(restored.intervals, f.intervals)
        v2 = view(2, (patient("a", ready=True),), eligible=(("a",), (), (), ()), completed=(("a",), (), (), ()))
        f.update(v2, before_filter=lambda _: None)
        restored.update(v2, before_filter=lambda _: None)
        np.testing.assert_array_equal(f.weights, restored.weights)

    def test_missing_or_failed_admission_prevents_hypothesis_work(self):
        for callback in (None, lambda event: (_ for _ in ()).throw(RuntimeError("budget"))):
            f = CompletionIntervalFilter(PROPOSAL)
            f.update(view())
            with patch("src.baselines.capacity_completion_control._condition_prefix", side_effect=AssertionError("dispatched")):
                with self.assertRaises((ValueError, RuntimeError)):
                    f.update(view(1), before_filter=callback)

    def test_no_skipped_boundaries_no_mutable_exports_invalid_restore(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            f.update(view(2))
        f.update(view(patients=(patient("a"),)))
        w, b = f.weights, f.intervals
        w[:] = 0
        b["a"]["site"] = 3
        self.assertEqual(f.intervals["a"]["site"], 0)
        self.assertAlmostEqual(f.weights.sum(), 4)
        state = f.state_dict()
        state["weights"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            f.load_state_dict(state)


class AdaptiveTests(unittest.TestCase):
    def test_solver_exact_iterations_and_feasible_joint_l1_solution(self):
        import src.baselines.capacity_completion_control as module
        original = np.clip
        calls = []
        def clip(*a, **kw):
            calls.append(1)
            return original(*a, **kw)
        with patch.object(module.np, "clip", side_effect=clip):
            h = proximal_hours([100, 20, 3, 0], [3, 2, 1, 1], [.5, 1, 1.25, 1.5], [2]*4, PROPOSAL)
        self.assertGreaterEqual(len(calls), 20*33)
        self.assertTrue(np.all(h >= 0) and np.all(h <= 4))
        self.assertLessEqual(h.sum(), 8 + 1e-12)
        self.assertGreater(h[0], h[3])
        np.testing.assert_allclose(project_hours([10]*4, PROPOSAL), [2]*4)

    def test_no_backlog_decreases_previous_and_no_rollout_called(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        with patch.object(PublicPatientForecast, "step", side_effect=AssertionError("rule ran model")):
            h = c.act(view(), role="adaptive_rule")
        self.assertTrue(np.all(h >= 0))  # Known future mean arrivals still matter.
        np.testing.assert_array_equal(proximal_hours([0]*4, [1]*4, [1]*4, [0]*4, PROPOSAL), ZERO)
        h = proximal_hours([0]*4, [1]*4, [1]*4, [2]*4, PROPOSAL)
        self.assertTrue(np.all(h < 2))

    def test_delayed_staff_reduces_deficit_and_public_demand_matters(self):
        patients = tuple(patient(str(i)) for i in range(5))
        v = view(patients=patients)
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        c.observe(v)
        first = c.adaptive(v)
        staffed = replace(v, common=replace(v.common, pending_hours=((4.,)*4, (4.,)*4)))
        second = c.adaptive(staffed)
        self.assertGreater(first[0], second[0])


class ForecastTests(unittest.TestCase):
    def control_and_model(self, v, q=.5):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        c.observe(v)
        return c, PublicPatientForecast(v, PROPOSAL, c.filter, c.lifecycle, q)

    def test_individual_five_stage_manufacture_reagent_and_reactor_conservation(self):
        v = view(patients=(patient("a", ready=True),))
        _, model = self.control_and_model(v)
        initial_reagents = model.reagents.sum()
        for step in range(5):
            model.step(ZERO, no_operations)
            p = model.patients["a"]
            self.assertEqual(p.stage, max(0, 4-step))
            self.assertEqual(p.record["status"], "in_production" if step < 4 else "delivered")
            public = model.public_view()
            self.assertEqual(np.asarray(public.operations.bioreactors).sum() + model.capacity_transfers.sum(), 32)
        started = sum(p.record["manufacturing_site"] is not None for p in model.patients.values())
        self.assertEqual(model.reagents.sum(), initial_reagents - started)
        self.assertGreater(model.total_components["ordinary_labor"], 0)

    def test_loss_expiry_urgency_and_terminal_values_are_not_hidden_work(self):
        v = view(patients=(patient("old", survival=.9, specimen_age=5), patient("ill", survival=.76)),
                 reagents=ZERO, reactors=((0.,)*5,)*4)
        _, model = self.control_and_model(v, q=.1)
        model.step(ZERO, no_operations)
        self.assertEqual(model.patients["old"].record["status"], "lost")
        self.assertEqual(model.patients["ill"].record["status"], "lost")
        self.assertEqual(model.total_components["patient_loss"], 100000)
        self.assertEqual(model.total_components["expiry"], 40000)
        self.assertGreaterEqual(model.terminal_value(), 0)
        self.assertNotIn("lost_work", model.total_components)

    def test_resource_pipeline_lead_procurement_and_no_double_labor_charge(self):
        _, model = self.control_and_model(view(reagents=ZERO, arrivals=ZERO))
        def buy(v, proposal, tail=False):
            return np.asarray(ZERO * 3 + (1., -1., -1., -1.)) if v.common.epoch == 0 else np.asarray(NO_BUY)
        model.step((2.,)*4, buy)
        self.assertEqual(model.reagents[0], 0)
        self.assertEqual(model.orders[0, 0], 6)
        self.assertEqual(model.pending[1, 0], 2)
        model.step(ZERO, buy)
        self.assertEqual(model.reagents[0], 6)
        model.step(ZERO, buy)
        self.assertEqual(model.total_components["flexible_labor"], 880)
        self.assertEqual(model.terminal_value(), sum(50000*(1-p.record["survival"]) for p in model.patients.values()
                                                   if p.record["status"] not in {"lost", "delivered"}))

    def test_delayed_resource_transfers_are_conserved_and_costed(self):
        _, model = self.control_and_model(view())
        def transfer(v, proposal, tail=False):
            return np.asarray(ZERO + (-1., 1., 0., 0.) + (-1., 1., 0., 0.) + (-1.,)*4)
        model.step(ZERO, transfer)
        self.assertEqual(model.reagents[0], 2)
        self.assertEqual(model.reagents[1], 8)
        self.assertEqual(model.reagent_transfers[0, 1], 6)
        self.assertEqual(model.capacity_transfers[0, 1], 4)
        self.assertEqual(model.reagents.sum() + model.reagent_transfers.sum(), 32)
        self.assertEqual(model.idle.sum() + model.capacity_transfers.sum(), 32)
        self.assertEqual(model.total_components["reagent_transfer"], 1200)
        self.assertEqual(model.total_components["bioreactor_transfer"], 2000)

    def test_midpoint_intervals_survive_nonbinary_hour_arithmetic(self):
        _, model = self.control_and_model(view(patients=(patient("a"),)), q=.1)
        model.pending[:] = 4 / 3
        model.step((4/3,)*4, no_operations)
        lo, hi = model.patients["a"].interval
        self.assertLessEqual(lo, model.patients["a"].work)
        self.assertGreaterEqual(hi, model.patients["a"].work)
        model.step((4/3,)*4, no_operations)
        self.assertTrue(model.patients["a"].record["support_complete"])

    def test_public_record_consistency_and_restore_do_not_guess(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        c.observe(view(patients=(patient("a"),)))
        with self.assertRaisesRegex(ValueError, "disappeared"):
            c.filter.update(view(1))
        missing = view()
        missing = replace(missing, operations=replace(missing.operations, current_arrivals=()))
        with self.assertRaisesRegex(ValueError, "current_arrivals"):
            CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False).act(missing)

    def test_mean_arrivals_are_indivisible_identity_bound_not_future_actuals(self):
        _, model = self.control_and_model(view(arrivals=(3., 0., 0., 0.)))
        model.step(ZERO, no_operations)
        self.assertEqual(len(model.patients), 3)
        model.step(ZERO, no_operations)
        self.assertEqual(len(model.patients), 7)
        model.step(ZERO, no_operations)
        self.assertEqual(len(model.patients), 15)
        self.assertTrue(all(isinstance(pid, str) for pid in model.patients))

    def test_transfers_carry_partial_work_and_do_not_serve_in_transit(self):
        v = view(patients=(patient("a"),))
        _, model = self.control_and_model(v)
        model.patients["a"].work = 1.5
        model.patients["a"].interval = (1., 2.)
        def route(v, proposal, tail=False):
            return np.asarray((-.25, .25, 0., 0.) + ZERO * 2 + (-1.,)*4) if v.common.epoch == 0 else np.asarray(NO_BUY)
        model.step(ZERO, route)
        self.assertEqual(model.patients["a"].record["status"], "in_transit")
        self.assertEqual(model.patients["a"].work, 1.5)
        self.assertEqual(model.patients["a"].destination, 1)
        model.step(ZERO, route)
        self.assertEqual(model.patients["a"].record["manufacturing_site"], 1)
        self.assertEqual(model.total_components["specimen_transfer"], 600)

    def test_causal_production_and_transit_bookkeeping_restores(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        v0 = view(patients=(patient("a"),))
        c.observe(v0)
        c.record_operation(v0, (-.25, .25, 0., 0.) + ZERO * 2 + (-1.,)*4)
        c.observe(view(1, (patient("a", status="in_transit"),)))
        restored = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        restored.load_state_dict(json.loads(json.dumps(c.state_dict())))
        self.assertEqual(restored.lifecycle.destinations["a"], 1)
        reactors = [list(r) for r in ((8., 0., 0., 0., 0.),)*4]
        reactors[1] = [7., 0., 0., 0., 1.]
        v2 = view(2, (patient("a", site=1, status="in_production", ready=True),), reactors=reactors,
                  eligible=((), ("a",), (), ()), completed=((), ("a",), (), ()))
        restored.observe(v2)
        self.assertEqual(restored.lifecycle.stages["a"], 4)
        reactors[1] = [7., 0., 0., 1., 0.]
        restored.observe(view(3, (patient("a", site=1, status="in_production", ready=True),), reactors=reactors))
        self.assertEqual(restored.lifecycle.stages["a"], 3)

    def test_missing_transit_destination_and_unaccounted_stage_are_errors(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        c.observe(view(patients=(patient("a"),)))
        with self.assertRaisesRegex(ValueError, "destination"):
            c.observe(view(1, (patient("a", status="in_transit"),)))
        bad = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        with self.assertRaisesRegex(ValueError, "stage unidentifiable"):
            bad.observe(view(patients=(patient("a", status="in_production", ready=True),)))


class PlannerTests(unittest.TestCase):
    def test_sixteen_three_eight_exact_admission_and_native_public_adapter(self):
        from src.rl.capacity_native import common_operation
        c = CapacityCompletionControl(PROPOSAL, base_control=common_operation)
        v = view()
        admissions, calls = [], []
        original = PublicPatientForecast.step
        def step(model, *args):
            self.assertEqual(len(admissions), len(calls) + 1)
            calls.append(model.epoch)
            return original(model, *args)
        with patch.object(PublicPatientForecast, "step", step):
            h = c.act(v, role="id_mpc", before_query=admissions.append)
        self.assertEqual(len(admissions), 384)
        self.assertEqual(len(calls), 384)
        self.assertEqual(len(c.last_plan["scores"]), 16)
        self.assertEqual(len(c.candidates(v)), 16)
        self.assertEqual([x[1] for x in c.candidates(v)], [False]*8+[True]*8)
        self.assertEqual({x["quantile"] for x in admissions}, {.1, .5, .9})
        self.assertEqual(c.last_plan["model_epochs"], 384)
        self.assertLessEqual(h.sum(), 8)
        self.assertEqual(c.filter._epoch, 0)
        self.assertFalse(c.filter.reset_events)

    def test_first_admission_failure_prevents_model_construction(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations)
        with patch("src.baselines.capacity_completion_control.PublicPatientForecast", side_effect=AssertionError("constructed")):
            with self.assertRaisesRegex(RuntimeError, "cap"):
                c.act(view(), role="id_mpc", before_query=lambda _: (_ for _ in ()).throw(RuntimeError("cap")))

    def test_ties_candidate_order_and_feedback_after_two(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        feedback = []
        original = PublicPatientForecast.adaptive
        def adaptive(model):
            feedback.append(model.epoch)
            return original(model)
        with patch.object(PublicPatientForecast, "adaptive", adaptive), patch.object(PublicPatientForecast, "terminal_value", return_value=0.), patch.object(PublicPatientForecast, "step", autospec=True) as mocked:
            def step(model, *args):
                model.epoch += 1
                return 0.
            mocked.side_effect = step
            h = c.act(view(), role="id_mpc")
        np.testing.assert_array_equal(h, ZERO)
        self.assertEqual(c.last_plan["chosen"], 0)
        self.assertEqual(len(feedback), 8*3*6)
        self.assertTrue(all(t >= 2 for t in feedback))

    def test_closure_stops_new_arrivals_commitments_and_tail_queries(self):
        c = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        for t in range(48):
            c.observe(view(t))
        queries = []
        c.act(view(47), role="id_mpc", before_query=queries.append)
        self.assertEqual(len(queries), 384)
        c.act(view(48), role="id_mpc", before_query=queries.append)
        self.assertEqual(len(queries), 384)
        model = PublicPatientForecast(view(48), PROPOSAL, c.filter, c.lifecycle, .5)
        model.step((4.,)*4, no_operations)
        self.assertEqual(len(model.patients), 0)
        self.assertEqual(model.total_components["flexible_labor"], 0)


if __name__ == "__main__":
    unittest.main()
