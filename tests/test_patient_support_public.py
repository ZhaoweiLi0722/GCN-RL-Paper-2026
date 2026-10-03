"""Fabricated patient/resource records; native construction and stepping patched."""

import copy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.env.patient_condition import PatientStatus
from src.env.patient_support_capacity import PatientSupportAction
from src.env.patient_support_public import (
    PublicPatientSupportCapacityEnv, PublicSupportOperations, PublicSupportServiceEvent,
    validate_public_operations,
)
from src.rl.public_support_collector import CAPACITY_CONTROL_ROLES
from src.rl.public_support_input import PublicSupportControlInput
from tests import test_patient_support_capacity as fixtures


def fake_step(env, action):
    production = np.minimum.reduce((np.array([len(q) for q in env.patient_queues], dtype=float),
                                    env.reagents, env.bioreactors[:, 0])).copy()
    started = env._start_production(production)
    for p in env.patient_registry.values():
        p.age += 1
    for queue in env.patient_queues:
        for p in queue:
            p.specimen_age += 1
    env.reagents -= production
    env.bioreactors[:, 0] -= production
    env.fake_rng.append(sum(len(row) for row in started))
    env.t += 1
    return np.zeros(1), -7.0, False, {"cost": 7.0, "base_cost": 2.0, "production": production.copy()}


class PublicPatientViewTests(unittest.TestCase):
    def setUp(self):
        for name, method in (("__init__", lambda *args, **kwargs: self.fail("no real patient env")),
                             ("state_dict", fixtures.host_snapshot), ("load_state_dict", fixtures.host_restore),
                             ("observation", lambda self: np.array([2.0, 3.0])), ("step", fake_step)):
            p = patch.object(PatientConditionCapacityEnv, name, method)
            p.start()
            self.addCleanup(p.stop)

    def env(self, ordinary=(0.5, 0.0)):
        # Reuse fabricated v1 host metadata; no source constructor is called.
        prior = fixtures.NativeAdapterMockTests.fake_env(None, ordinary)
        env = object.__new__(PublicPatientSupportCapacityEnv)
        env.__dict__.update(prior.__dict__)
        env._last_public_service = env._staged_public_service = None
        for p in env.patient_registry.values():
            p.enrollment_epoch = 0
            p.collection_facility = p.material_facility
            p.specimen_age, p.survival = 0, 1.0
        env.bioreactors = np.array([[2.0, 0], [2.0, 0]])
        env.reagent_transfer_pipeline = np.zeros((1, 2))
        env.capacity_transfer_pipeline = np.zeros((1, 2))
        env.reagent_purchase_pipeline = np.zeros((0, 2))
        env.supplier_available = np.ones(2)
        env.demand_forecast = np.array([1.0, 1.0])
        env.demand = np.array([0.0, 0.0])
        return env

    def action(self):
        return PatientSupportAction((0.0,) * 8, (0.0, 0.0))

    def test_public_input_all_six_roles_same_and_no_latent_fields(self):
        views = [PublicSupportControlInput.capture(self.env()) for _ in CAPACITY_CONTROL_ROLES]
        self.assertTrue(all(x == views[0] for x in views))
        encoded = json.dumps(asdict(views[0]))
        for forbidden in ("health_index", "deterioration_epoch", "response_tape", "remaining_work"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(views[0].operations.waiting_order, (("p", "q"), ()))
        self.assertIsNone(views[0].operations.last_service)

    def test_partial_work_hidden_but_completion_identity_visible_next_epoch(self):
        env = self.env()
        before = PublicSupportControlInput.capture(env)
        env.step(self.action())
        first = PublicSupportControlInput.capture(env)
        self.assertEqual(first.operations.last_service.eligible_order, (("p", "q"), ()))
        self.assertEqual(first.operations.last_service.completed_ids, ((), ()))
        self.assertEqual(first.operations.last_service.known_at, 1)
        self.assertEqual(before.common.epoch, 0)
        env.step(self.action())
        second = PublicSupportControlInput.capture(env)
        self.assertEqual(second.operations.last_service.completed_ids, (("p",), ()))
        self.assertTrue(second.operations.patients[0].support_complete)
        self.assertEqual(second.operations.patients[0].status, "in_production")
        self.assertEqual(second.operations.waiting_order, (("q",), ()))

    def test_post_route_service_order_only_published_after_step(self):
        env = self.env((0.25, 0.5))
        before = PublicSupportControlInput.capture(env)
        def route_then_step(env, action):
            patient = env.patient_queues[0].pop(0)
            patient.material_facility = 1
            env.patient_queues[1].append(patient)
            return fake_step(env, action)
        with patch.object(PatientConditionCapacityEnv, "step", route_then_step):
            env.step(self.action())
        after = PublicSupportControlInput.capture(env)
        self.assertEqual(before.operations.waiting_order, (("p", "q"), ()))
        self.assertEqual(after.operations.last_service.eligible_order, (("q",), ("p",)))
        self.assertEqual(after.operations.patients[0].material_site, 1)

    def test_snapshot_preserves_event_and_failure_restores_boundary(self):
        env = self.env()
        env.step(self.action())
        snapshot = env.state_dict()
        expected = PublicSupportControlInput.capture(env)
        clone = self.env()
        clone.load_state_dict(snapshot)
        self.assertEqual(PublicSupportControlInput.capture(clone), expected)
        def fail(env, action):
            fake_step(env, action)
            raise RuntimeError("fake recording failure")
        with patch.object(PatientConditionCapacityEnv, "step", fail):
            with self.assertRaises(RuntimeError):
                clone.step(self.action())
        self.assertEqual(PublicSupportControlInput.capture(clone), expected)
        self.assertEqual(clone.fake_rng, env.fake_rng)

    def test_restore_rejects_future_event_without_leaving_changed_host(self):
        env = self.env()
        env.step(self.action())
        before = PublicSupportControlInput.capture(env)
        damaged = env.state_dict()
        damaged["last_public_service"]["epoch"] = 4
        damaged["last_public_service"]["known_at"] = 5
        with self.assertRaises(ValueError):
            env.load_state_dict(damaged)
        self.assertEqual(PublicSupportControlInput.capture(env), before)

    def test_public_input_fails_at_mid_step_and_on_count_identity_disagreement(self):
        env = self.env()
        env._pending_support_request = (0, 0)
        with self.assertRaises(RuntimeError):
            env.public_operations()
        env._pending_support_request = None
        env.step(self.action())
        ev = env._last_public_service
        env._last_public_service = replace(ev, completed_ids=(("p",), ()))
        with self.assertRaises(ValueError):
            PublicSupportControlInput.capture(env)

    def test_resource_records_and_patient_fields_have_no_mutable_aliases(self):
        env = self.env()
        view = env.public_operations()
        env.reagents[0] = 99
        env.patient_registry["p"].survival = 0.5
        self.assertEqual(view.reagents[0], 2.0)
        self.assertEqual(view.patients[0].survival, 1.0)
        source = [["p", "q"], []]
        another = replace(view, waiting_order=source)
        source[0].pop()
        self.assertEqual(another.waiting_order, (("p", "q"), ()))

    def test_public_validation_rejects_missing_patient_and_service_order_violation(self):
        view = self.env().public_operations()
        with self.assertRaises(ValueError):
            validate_public_operations(replace(view, patients=view.patients[:1]))
        with self.assertRaises(ValueError):
            PublicSupportServiceEvent(0, 1, (("p", "q"), ()), (("q",), ()), (1, 0), (0, 0))
        with self.assertRaises(ValueError):
            validate_public_operations(replace(view, last_service=PublicSupportServiceEvent(
                0, 1, (("p",), ()), ((), ()), (1, 0), (0, 0))))


if __name__ == "__main__":
    unittest.main()
