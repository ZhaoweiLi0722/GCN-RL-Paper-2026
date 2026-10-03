"""Native binding contracts on fabricated records; never construct a patient env."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_support_capacity import PatientSupportAction
from src.env.patient_support_public import PublicPatientSupportCapacityEnv
from src.rl.capacity_native import (CapacityPilotEnv, CapacityWorldTape, common_operation,
                                    keyed_seed, native_configs, resource_totals)
from src.rl.public_support_input import PublicSupportControlInput
from tests import test_patient_support_public as fixtures


def proposal():
    return json.loads((Path(__file__).resolve().parents[1] /
        "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json").read_text())


class NativeBindingTests(unittest.TestCase):
    def test_config_exact_scientific_values_without_construction(self):
        with patch.object(PublicPatientSupportCapacityEnv, "__init__", side_effect=AssertionError("no native construction")):
            env, support = native_configs(proposal())
        self.assertEqual(env.base.num_facilities, 4)
        self.assertEqual(env.base.episode_horizon, 64)
        self.assertTrue(env.enable_specimen_routing)
        self.assertEqual(env.base.reagent_lead_time_probabilities, [0, 1])
        self.assertEqual(env.weight_patient_lost, 50000)
        self.assertEqual(support.ordinary_hours, (4, 4, 4, 4))
        self.assertEqual(support.effort.shared_hour_budget, 8)
        self.assertFalse(env.base.enable_overtime_control)

    def test_keyed_seeds_exclude_arm_and_separate_purposes(self):
        world = dict(phase="teacher", block=0, condition=1, replicate=2, seed=62600012)
        p = proposal()
        self.assertEqual(keyed_seed(p, world, "demand"), keyed_seed(p, {**world, "role": "online"}, "demand"))
        self.assertNotEqual(keyed_seed(p, world, "demand"), keyed_seed(p, world, "exploration"))
        self.assertNotEqual(keyed_seed(p, world, "demand"), keyed_seed(p, {**world, "phase": "evaluation"}, "demand"))

    def test_tape_hash_covers_private_attributes_and_not_controller_inputs(self):
        tape = CapacityWorldTape({"seed": 1}, ((1, 0), (0, 0)), ((1., 1.), (1., 1.)), ((0.2, 5.0),), None)
        self.assertEqual(tape.digest(), copy.deepcopy(tape).digest())
        self.assertNotEqual(tape.digest(), replace(tape, patient_attributes=((0.3, 5.),)).digest())

    def test_native_constructor_debit_happens_before_error(self):
        calls = []
        tape = CapacityWorldTape({"seed": 1}, (), ((1.,) * 4,) * 64, (), None)
        with patch.object(PublicPatientSupportCapacityEnv, "__init__", side_effect=RuntimeError("fake failure")):
            with self.assertRaisesRegex(RuntimeError, "fake failure"):
                CapacityPilotEnv(proposal(), tape, calls.append)
        self.assertEqual(calls, ["construction"])

    def test_reset_budget_cannot_repeat_even_if_fake_reset_raises(self):
        host = object.__new__(CapacityPilotEnv)
        calls = []
        host._reset_consumed, host._before_native = False, calls.append
        with patch.object(PublicPatientSupportCapacityEnv, "reset", side_effect=RuntimeError("fake reset")):
            with self.assertRaisesRegex(RuntimeError, "fake reset"):
                host.reset()
            with self.assertRaisesRegex(RuntimeError, "additional reset"):
                host.reset()
        self.assertEqual(calls, ["construction_reset"])

    def test_resource_clipping_ends_attempt_without_counter_refund(self):
        host = object.__new__(CapacityPilotEnv)
        host.reagents, host.bioreactors = np.array([2., 2.]), np.array([[2., 0.], [2., 0.]])
        for field in ("reagent_transfer_pipeline", "reagent_purchase_pipeline", "capacity_transfer_pipeline"):
            setattr(host, field, np.zeros((1, 2)))
        calls = []
        host._before_native, host._failed = calls.append, False
        from types import SimpleNamespace
        host.support = SimpleNamespace(_delivered=(0.,0.))
        host._cumulative_purchases = host._cumulative_consumption = 0.
        def bad_step(h, action):
            h.reagents[0] = 1.
            return None, -1., False, {"replenishment": [0, 0], "production": [0, 0]}
        with patch.object(PublicPatientSupportCapacityEnv, "step", bad_step):
            with self.assertRaisesRegex(ValueError, "clipping"):
                host.step(None)
            with self.assertRaisesRegex(RuntimeError, "cannot retry"):
                host.step(None)
        self.assertEqual(calls, ["step"])
        self.assertEqual(resource_totals(host), (3., 4.))


class PublicCommonOperationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PublicPatientViewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.env = self.fixture.env()
        self.p = proposal()
        # Explicit two-site artificial input, not a proposed scientific variant.
        s = self.p["synthetic_system"]
        for key in ("max_reagent_replenishment", "demand_rates"):
            s[key] = s[key][:2]
        s["transport_resource_information_edges"] = [[0, 1]]

    def test_common_rule_only_reads_typed_public_arrays(self):
        view = PublicSupportControlInput.capture(self.env)
        action = common_operation(view, self.p)
        self.assertEqual(action.shape, (8,))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((abs(action) <= 1).all())
        with self.assertRaises(ValueError):
            common_operation(replace(view, operations=replace(view.operations, current_arrivals=())), self.p)

    def test_tail_purchases_only_missing_live_reagents_and_never_transfers(self):
        view = PublicSupportControlInput.capture(self.env)
        v = replace(view.operations, epoch=48, reagents=(0., 0.), reagent_orders=((1., 0.),))
        action = common_operation(replace(view, operations=v), self.p, tail=True)
        np.testing.assert_array_equal(action[:6], np.zeros(6))
        np.testing.assert_allclose(action[6:], [2/6-1, -1])
        action = common_operation(replace(view, operations=replace(v, epoch=61)), self.p, tail=True)
        np.testing.assert_array_equal(action[6:], [-1, -1])


if __name__ == "__main__":
    unittest.main()
