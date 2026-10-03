"""ARTIFICIAL public-view regressions; no native host, neural work or science."""

import copy
from dataclasses import asdict
import math
import unittest

import numpy as np

from src.baselines.capacity_completion_control_recovery1 import (
    CapacityCompletionControl, PublicPatientForecast,
)
from src.env.patient_support_public import validate_public_operations
from src.rl.capacity_native import common_operation
from tests.test_capacity_completion_control import (
    PROPOSAL, ZERO, no_operations, patient, view,
)


FRACTIONAL = (4 / 3,) * 4
HORIZON = PROPOSAL["id_mpc"]["predictive_horizon"]


class ForecastRecoveryTests(unittest.TestCase):
    def control_and_model(self, public, quantile=.1):
        control = CapacityCompletionControl(
            PROPOSAL, base_control=common_operation, scientific=False,
        )
        control.observe(public)
        model = PublicPatientForecast(public, PROPOSAL, control.filter, control.lifecycle, quantile)
        return control, model

    def assert_valid_forecast(self, model):
        public = model.public_view()
        self.assertEqual(validate_public_operations(public.operations), public.operations)
        records = public.operations.patients
        self.assertEqual(len(records), len({p.patient_id for p in records}))
        self.assertEqual({p.patient_id for p in records}, set(model.patients))
        occupied = np.zeros((4, 5))
        for pid, p in model.patients.items():
            with self.subTest(epoch=model.epoch, patient=pid):
                lo, hi = p.interval
                self.assertTrue(all(math.isfinite(x) for x in (lo, hi, p.work)))
                self.assertGreaterEqual(lo, 0.)
                self.assertLessEqual(lo, p.work)
                self.assertLessEqual(p.work, hi)
                self.assertLessEqual(hi, 4.)
                self.assertEqual(p.work, (lo + hi) / 2)
                if p.record["support_complete"]:
                    self.assertEqual((lo, hi, p.work), (0., 0., 0.))
                elif p.record["status"] not in {"lost", "delivered"}:
                    self.assertGreater(p.work, 0.)
                if p.record["status"] == "in_production":
                    self.assertIn(p.stage, range(1, 5))
                    self.assertTrue(p.record["support_complete"])
                    occupied[p.record["manufacturing_site"], p.stage] += 1
        np.testing.assert_array_equal(np.asarray(public.operations.bioreactors)[:, 1:], occupied[:, 1:])
        for stock in (model.reagents, model.idle, model.orders, model.reagent_transfers,
                      model.capacity_transfers, model.pending):
            self.assertTrue(np.isfinite(stock).all())
            self.assertTrue((stock >= 0).all())
        return public

    def test_tiny_positive_head_remains_uncompleted_after_exact_capacity_prefix(self):
        public = view(patients=tuple(patient(pid) for pid in ("a", "b", "tiny", "later")),
                      pending=((4., 0., 0., 0.), ZERO))
        _, model = self.control_and_model(public, .5)
        # Artificial planner belief, following the existing partial-work fixture.
        residuals = (4., 4., 8.881784197001252e-16, 2.666666666666667)
        for pid, work in zip(("a", "b", "tiny", "later"), residuals):
            model.patients[pid].work = work
            model.patients[pid].interval = (work, work)
        for offset in range(HORIZON):
            with self.subTest(offset=offset):
                model.step(FRACTIONAL, no_operations)
                self.assert_valid_forecast(model)
                if offset == 0:
                    self.assertEqual(model.last_service.completed_ids[0], ("a", "b"))
                    tiny = model.patients["tiny"]
                    self.assertFalse(tiny.record["support_complete"])
                    self.assertEqual(tiny.work, residuals[2])
                    self.assertEqual(tiny.interval, (residuals[2], residuals[2]))
                    self.assertEqual(model.patients["later"].work, residuals[3])
                elif offset == 1:
                    self.assertEqual(model.last_service.completed_ids[0], ("tiny", "later"))
        self.assertTrue(all(model.patients[pid].record["status"] == "delivered"
                            for pid in ("a", "b", "tiny", "later")))

    def test_fractional_service_carries_work_through_entire_original_horizon(self):
        public = view(patients=(patient("a"), patient("b")),
                      pending=(FRACTIONAL, FRACTIONAL))
        _, model = self.control_and_model(public)
        for offset in range(HORIZON):
            with self.subTest(offset=offset):
                model.step(FRACTIONAL, no_operations)
                self.assert_valid_forecast(model)
                if offset == 0:
                    self.assertAlmostEqual(model.patients["a"].work, 4 / 3)
                    self.assertEqual(model.patients["b"].work, 4.)
                    self.assertEqual(model.last_service.completed_ids[0], ())
                elif offset == 1:
                    self.assertEqual(model.patients["a"].work, 0.)
                    self.assertAlmostEqual(model.patients["b"].work, 8 / 3)
                    self.assertEqual(model.last_service.completed_ids[0], ("a",))
        self.assertEqual(model.epoch, HORIZON)
        self.assertEqual(model.patients["a"].record["status"], "delivered")
        self.assertEqual(model.patients["b"].record["status"], "delivered")

    def test_production_public_adapter_conserves_horizon_resources_and_identities(self):
        records = tuple(patient(f"S{site}-{j}", site=site)
                        for site, count in enumerate((5, 2, 0, 1)) for j in range(count))
        records += (patient("expiring", specimen_age=5, survival=.99),
                    patient("frail", site=2, survival=.76))
        public = view(patients=records, pending=(FRACTIONAL, FRACTIONAL),
                      arrivals=(1., 2., 0., 1.), reagents=(12., 2., 8., 8.),
                      reagent_transfers=((0., 1., 0., 0.),),
                      capacity_transfers=((0., 0., 1., 0.),),
                      orders=((0., 0., 0., 1.), (0., 0., 1., 0.)))
        proposal_before = copy.deepcopy(PROPOSAL)
        for quantile in PROPOSAL["id_mpc"]["response_quantiles"]:
            with self.subTest(quantile=quantile):
                control, model = self.control_and_model(public, quantile)
                filter_before = control.filter.state_dict()
                lifecycle_before = control.lifecycle.state_dict()
                initial_reagents = model.reagents.sum() + model.orders.sum() + model.reagent_transfers.sum()
                initial_reactors = model.idle.sum() + model.capacity_transfers.sum()
                booked = model.pending.sum(axis=0).copy()
                matured = np.zeros(4)
                purchased = 0.
                initial_ids = set(model.patients)
                expected_arrivals = np.zeros(4)
                completed_ids, started_ids = set(), set()
                cumulative_work = 0.
                action_calls = []

                def operation(v, proposal, tail=False):
                    action = common_operation(v, proposal, tail=tail)
                    action_calls.append((v, action.copy()))
                    return action

                for offset in range(HORIZON):
                    with self.subTest(offset=offset):
                        before = copy.deepcopy(model.patients)
                        hours = np.asarray((4., 4 / 3, 4 / 3, 4 / 3) if offset % 2 else FRACTIONAL)
                        cost = model.step(hours, operation)
                        after_public = self.assert_valid_forecast(model)
                        self.assertTrue(math.isfinite(cost))
                        self.assertEqual(len(action_calls), offset + 1)
                        operation_view, action = action_calls[-1]
                        self.assertEqual(operation_view.common.epoch, offset)
                        event = model.last_service
                        self.assertEqual((event.epoch, event.known_at), (offset, offset + 1))
                        purchased += float(np.sum((action[12:] + 1) / 2
                                                 * PROPOSAL["synthetic_system"]["max_reagent_replenishment"]
                                                 * np.asarray(operation_view.operations.supplier_available)))
                        booked += hours
                        matured += event.applied_hours
                        np.testing.assert_allclose(booked, matured + model.pending.sum(axis=0), rtol=0, atol=1e-12)
                        self.assertTrue(set(before).issubset(model.patients))
                        for site, (eligible, completed) in enumerate(zip(event.eligible_order, event.completed_ids)):
                            self.assertEqual(completed, eligible[:len(completed)])
                            self.assertTrue(completed_ids.isdisjoint(completed))
                            completed_ids.update(completed)
                            used = math.fsum(before[pid].work - model.patients[pid].work for pid in eligible)
                            capacity = model.eta[site] * (event.ordinary_hours[site] + event.applied_hours[site])
                            expected_used = min(capacity, math.fsum(before[pid].work for pid in eligible))
                            self.assertGreaterEqual(used, 0.)
                            self.assertAlmostEqual(used, expected_used, places=12)
                            cumulative_work += used
                        eligible_ids = {pid for q in event.eligible_order for pid in q}
                        for pid, old in before.items():
                            current = model.patients[pid]
                            if pid not in eligible_ids:
                                self.assertEqual(current.work, old.work)
                            if old.record["status"] in {"lost", "delivered"}:
                                self.assertEqual(asdict(current), asdict(old))
                            if current.record["manufacturing_site"] is not None:
                                started_ids.add(pid)
                        self.assertAlmostEqual(
                            model.reagents.sum() + model.orders.sum() + model.reagent_transfers.sum() + len(started_ids),
                            initial_reagents + purchased, places=10,
                        )
                        self.assertAlmostEqual(
                            np.asarray(after_public.operations.bioreactors).sum() + model.capacity_transfers.sum(),
                            initial_reactors, places=12,
                        )
                        expected_arrivals += public.operations.current_arrivals if offset == 0 else np.asarray(PROPOSAL["synthetic_system"]["demand_rates"])
                        new_ids = set(model.patients) - initial_ids
                        counts = [sum(model.patients[pid].record["collection_site"] == site for pid in new_ids)
                                  for site in range(4)]
                        np.testing.assert_array_equal(counts, np.floor(expected_arrivals))
                        np.testing.assert_allclose(model.arrival_carry, expected_arrivals - counts, rtol=0, atol=1e-12)
                        self.assertAlmostEqual(
                            cumulative_work + math.fsum(p.work for p in model.patients.values()),
                            4 * len(model.patients), places=10,
                        )
                        self.assertEqual(control.filter.state_dict(), filter_before)
                        self.assertEqual(control.lifecycle.state_dict(), lifecycle_before)
                self.assertTrue(started_ids)
                self.assertTrue(any(p.record["status"] == "lost" for p in model.patients.values()))
                self.assertTrue(any(p.record["status"] == "delivered" for p in model.patients.values()))
        self.assertEqual(PROPOSAL, proposal_before)

    def test_existing_forecast_is_independent_of_later_live_filter_receipts(self):
        records = tuple(patient(f"S{site}-{j}", site=site) for site in range(4) for j in range(2))
        public = view(patients=records, pending=(FRACTIONAL, FRACTIONAL))
        control, model = self.control_and_model(public, .5)
        _, reference = self.control_and_model(public, .5)
        original_filter = control.filter.state_dict()
        control.observe(view(1, records, eligible=public.operations.waiting_order))
        updated_filter = control.filter.state_dict()
        self.assertNotEqual(updated_filter, original_filter)
        self.assertNotEqual(control.filter.intervals["S0-0"]["bounds_by_response"][0], (4., 4.))
        for offset in range(HORIZON):
            with self.subTest(offset=offset):
                np.testing.assert_array_equal(model.adaptive(), reference.adaptive())
                self.assertEqual(model.step(FRACTIONAL, common_operation),
                                 reference.step(FRACTIONAL, common_operation))
                self.assertEqual(model.public_view(), reference.public_view())
                self.assertEqual({pid: asdict(p) for pid, p in model.patients.items()},
                                 {pid: asdict(p) for pid, p in reference.patients.items()})
                self.assert_valid_forecast(model)
                self.assertEqual(control.filter.state_dict(), updated_filter)

    def test_repeated_real_mpc_decisions_do_not_mutate_live_filter_or_view(self):
        public = view(patients=tuple(patient(f"S{site}-{j}", site=site)
                                    for site in range(4) for j in range(2)),
                      pending=(FRACTIONAL, FRACTIONAL), arrivals=(1., 0., 2., 1.))
        control, _ = self.control_and_model(public)
        filter_before = control.filter.state_dict()
        lifecycle_before = control.lifecycle.state_dict()
        view_before = copy.deepcopy(public)
        proposal_before = copy.deepcopy(PROPOSAL)
        queries = []
        first = control.act(public, role="id_mpc", before_query=queries.append)
        first_plan = copy.deepcopy(control.last_plan)
        first_queries = copy.deepcopy(queries)
        self.assertEqual(control.filter.state_dict(), filter_before)
        self.assertEqual(control.lifecycle.state_dict(), lifecycle_before)
        queries.clear()
        second = control.act(public, role="id_mpc", before_query=queries.append)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(control.last_plan, first_plan)
        self.assertEqual(queries, first_queries)
        self.assertEqual(len(queries), 16 * 3 * HORIZON)
        self.assertEqual({(q["candidate"], q["quantile"], q["horizon"]) for q in queries},
                         {(c, q, h) for c in range(16)
                          for q in PROPOSAL["id_mpc"]["response_quantiles"] for h in range(HORIZON)})
        self.assertEqual(len(first_plan["scores"]), 16)
        self.assertTrue(np.isfinite(first_plan["scores"]).all())
        self.assertEqual(first_plan["model_epochs"], 16 * 3 * HORIZON)
        self.assertTrue((first >= 0).all() and (first <= 4).all())
        self.assertLessEqual(first.sum(), 8 + 1e-12)
        self.assertEqual(control.filter.state_dict(), filter_before)
        self.assertEqual(control.lifecycle.state_dict(), lifecycle_before)
        self.assertEqual(public, view_before)
        self.assertEqual(PROPOSAL, proposal_before)


if __name__ == "__main__":
    unittest.main()
