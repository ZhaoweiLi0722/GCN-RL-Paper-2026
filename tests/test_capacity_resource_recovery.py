"""Saved-public-boundary and artificial inventory regressions; no fit/native run."""

import copy
import gzip
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.capacity_completion_control_recovery1 import (
    CapacityCompletionControl as OldControl, PublicPatientForecast as OldForecast,
)
from src.baselines.capacity_completion_control_recovery2 import (
    CapacityCompletionControl, PublicPatientForecast, whole_patient_starts,
)
from src.env.patient_support_public import (
    PublicPatientRecord, PublicSupportOperations, PublicSupportServiceEvent,
)
from src.rl.capacity_native import common_operation
from src.rl.public_support_collector import PublicSupportInput
from src.rl.public_support_input import PublicSupportControlInput
from tests.test_capacity_completion_control import PROPOSAL, no_operations, patient, view
from tests import test_capacity_forecast_recovery as horizon_contracts


def saved_boundary():
    root = Path(__file__).resolve().parents[1]
    proposal = json.loads((root / 'specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json').read_text())
    with gzip.open(root / 'tests/fixtures/capacity_resource_failure_20261003.json.gz', 'rt') as handle:
        fixture = json.load(handle)
    state = fixture['controller']
    state['filter']['proposal'] = proposal
    def tuples(value):
        if isinstance(value, list):
            return tuple(tuples(x) for x in value)
        if isinstance(value, dict):
            return {key: tuples(x) for key, x in value.items()}
        return value
    data = tuples(state['filter']['last_view'])
    op = data['operations']
    op['patients'] = tuple(PublicPatientRecord(**p) for p in op['patients'])
    op['last_service'] = PublicSupportServiceEvent(**op['last_service'])
    return proposal, state, PublicSupportControlInput(
        PublicSupportInput(**data['common']), PublicSupportOperations(**op))


class ResourceRecoveryTests(unittest.TestCase):
    def test_saved_failure_is_resource_overdraft_then_negative_zero_request_transfer(self):
        proposal, state, public = saved_boundary()
        control = OldControl(proposal, base_control=common_operation, scientific=False)
        control.load_state_dict(state)
        invalid_stocks = []
        class TraceForecast(OldForecast):
            def _resource_transfer(self, stock, request, pipeline):
                if (stock < 0).any():
                    invalid_stocks.append((self.epoch, stock.copy(), request.copy()))
                return super()._resource_transfer(stock, request, pipeline)
        model = TraceForecast(public, proposal, control.filter, control.lifecycle, .5)
        initial, switch = control.candidates(public)[0]
        self.assertFalse(switch)
        for _ in range(5):
            model.step(initial, common_operation)
        self.assertEqual(model.epoch, 52)
        self.assertEqual(len(invalid_stocks), 1)
        self.assertEqual(invalid_stocks[0][0], 51)
        self.assertEqual(invalid_stocks[0][1][3], -math.ulp(1.) / 2)
        np.testing.assert_array_equal(invalid_stocks[0][2], np.zeros(4))
        self.assertEqual(model.reagent_transfers[0, 0], -math.ulp(1.) / 2)
        with self.assertRaisesRegex(ValueError, 'invalid public resource value'):
            model.step(initial, common_operation)

    def test_saved_boundary_completes_original_384_forecasts_without_live_mutation(self):
        proposal, state, public = saved_boundary()
        control = CapacityCompletionControl(proposal, base_control=common_operation, scientific=False)
        original = copy.deepcopy(state)
        control.load_recovery1_boundary(state)
        before = copy.deepcopy(control.filter.state_dict())
        calls = []
        hours = control.act(public, role='id_mpc', before_query=calls.append)
        self.assertEqual(len(calls), 384)
        self.assertEqual(state, original)
        self.assertEqual(before, control.filter.state_dict())
        self.assertTrue(np.isfinite(control.last_plan['scores']).all())
        self.assertTrue((hours >= 0).all())
        self.assertLessEqual(hours.sum(), 8.)
        restored = CapacityCompletionControl(proposal, base_control=common_operation, scientific=False)
        restored.load_state_dict(control.state_dict())
        self.assertEqual(restored.state_dict(), control.state_dict())

    def test_integer_start_never_borrows_fractional_resource(self):
        below = math.nextafter(1., 0.)
        self.assertEqual(math.floor(below + 1e-12), 1)
        for resource in (0., below, 1., math.nextafter(1., math.inf), 2.5):
            for reagents, idle in ((resource, 5.), (5., resource)):
                starts = whole_patient_starts(5, reagents, idle)
                self.assertEqual(starts, math.floor(min(reagents, idle)))
                self.assertGreaterEqual(reagents - starts, 0.)
                self.assertGreaterEqual(idle - starts, 0.)
        for bad in (-math.ulp(1.), math.nan, math.inf):
            with self.assertRaises(ValueError):
                whole_patient_starts(1, bad, 1.)

    def make_model(self, reagents, idle):
        public = view(patients=(patient('ready'),), reagents=(reagents, 8., 8., 8.),
                      reactors=((idle, 0., 0., 0., 0.), (8., 0., 0., 0., 0.),
                                (8., 0., 0., 0., 0.), (8., 0., 0., 0., 0.)))
        control = CapacityCompletionControl(PROPOSAL, base_control=no_operations, scientific=False)
        control.observe(public)
        model = PublicPatientForecast(public, PROPOSAL, control.filter, control.lifecycle, .5)
        model.patients['ready'].record['support_complete'] = True
        model.patients['ready'].interval = (0., 0.)
        model.patients['ready'].work = 0.
        return model

    def test_underfunded_patient_stays_whole_and_resources_conserved(self):
        below = math.nextafter(1., 0.)
        for reagents, idle in ((below, 2.), (2., below), (1., 1.)):
            model = self.make_model(reagents, idle)
            model.step(np.zeros(4), no_operations)
            starts = int(reagents >= 1 and idle >= 1)
            self.assertEqual(model.patients['ready'].record['status'],
                             'in_production' if starts else 'waiting')
            self.assertEqual(model.reagents[0] + starts, reagents)
            self.assertEqual(model.idle[0] + starts, idle)
            model.public_view()

    def test_transfer_rejects_invalid_input_before_any_mutation(self):
        model = self.make_model(1., 1.)
        stock = np.array([1., 1., 1., -math.ulp(1.) / 2])
        pipeline = np.zeros((1, 4))
        before = stock.copy()
        with self.assertRaisesRegex(ValueError, 'before transfer'):
            model._resource_transfer(stock, np.zeros(4), pipeline)
        np.testing.assert_array_equal(stock, before)
        np.testing.assert_array_equal(pipeline, np.zeros((1, 4)))

    def test_valid_fractional_transfer_and_diagnostic_field(self):
        model = self.make_model(1., 1.)
        stock, pipe = np.array([.3, 0., 0., 0.]), np.zeros((1, 4))
        amount = model._resource_transfer(stock, np.array([-.2, .2, 0., 0.]), pipe)
        self.assertAlmostEqual(amount, .2)
        self.assertAlmostEqual(stock.sum() + pipe.sum(), .3)
        self.assertTrue((stock >= 0).all())
        model.reagent_transfers[0, 0] = -math.ulp(1.) / 2
        with self.assertRaisesRegex(ValueError, r'forecast resource reagent_transfers\(0, 0\).*epoch 0'):
            model.public_view()


class ResourceHorizonTests(horizon_contracts.ForecastRecoveryTests):
    """Reuse the existing horizon/conservation checks against the new binding."""

    def setUp(self):
        binding = patch.multiple(horizon_contracts,
                                 CapacityCompletionControl=CapacityCompletionControl,
                                 PublicPatientForecast=PublicPatientForecast)
        binding.start()
        self.addCleanup(binding.stop)


if __name__ == '__main__':
    unittest.main()
