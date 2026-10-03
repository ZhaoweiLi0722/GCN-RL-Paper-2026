"""Saved-failure arithmetic, checked against exact rational prefix constraints."""

from fractions import Fraction
import copy
import json
import math
from pathlib import Path
import unittest

from src.baselines.capacity_completion_control import _condition_prefix as legacy
from src.baselines.capacity_completion_control_recovery1 import _condition_prefix
from src.baselines.capacity_completion_control_recovery1 import CapacityCompletionControl
from src.env.patient_support_public import PublicPatientRecord, PublicSupportOperations, PublicSupportServiceEvent
from src.rl.capacity_native import common_operation
from src.rl.public_support_collector import PublicSupportInput
from src.rl.public_support_input import PublicSupportControlInput


class PrecisionTests(unittest.TestCase):
    def test_saved_public_boundary_completes_same_384_queries_without_native_or_fit(self):
        root = Path(__file__).resolve().parents[1]
        proposal = json.loads((root/'specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json').read_text())
        state = json.loads((root/'tests/fixtures/capacity_forecast_failure_20261003.json').read_text())['controller']
        state['filter']['proposal'] = proposal
        state['format'] = 'capacity-completion-control-recovery1'
        def tuples(value):
            if isinstance(value, list): return tuple(tuples(x) for x in value)
            if isinstance(value, dict): return {k: tuples(v) for k,v in value.items()}
            return value
        v = tuples(state['filter']['last_view'])
        op = v['operations']
        op['patients'] = tuple(PublicPatientRecord(**x) for x in op['patients'])
        op['last_service'] = PublicSupportServiceEvent(**op['last_service'])
        public = PublicSupportControlInput(PublicSupportInput(**v['common']), PublicSupportOperations(**op))
        c = CapacityCompletionControl(proposal, base_control=common_operation, scientific=False)
        c.load_state_dict(state)
        before = copy.deepcopy(c.filter.state_dict())
        calls = []
        hours = c.act(public, role='id_mpc', before_query=calls.append)
        self.assertEqual(len(calls), 384)
        self.assertLessEqual(sum(hours), 8.)
        self.assertTrue(all(math.isfinite(x) for x in c.last_plan['scores']))
        self.assertEqual(c.filter.state_dict(), before)

    def test_saved_failure_positive_sub_ulp_head_is_not_zero(self):
        tiny = 8.881784197001252e-16
        prior = dict(a=(4., 4.), b=(4., 4.), c=(tiny, tiny),
                     d=(2.666666666666667, 2.666666666666667))
        order = tuple(prior)
        self.assertEqual(legacy(prior, order, order[:2], 8.)[0], 0.)
        weight, posterior = _condition_prefix(prior, order, order[:2], 8.)
        self.assertEqual(weight, 1.)
        self.assertEqual(posterior['c'], (tiny, tiny))
        self.assertEqual(posterior['d'], prior['d'])
        self.assertEqual(_condition_prefix(prior, order, order[:3], 8.)[0], 0.)

    def test_point_prefixes_match_exact_binary_rational_oracle(self):
        cases = [([4., 4., math.ulp(8.)/2, 2.666666666666667], 8.),
                 ([.1, .2, .3], .3), ([4., 4., 4.], 8.),
                 ([0., .5, .75, 1.25], 0.), ([.125, .25, .5], .875)]
        for values, capacity in cases:
            order = tuple(map(str, range(len(values))))
            prior = {pid: (value, value) for pid, value in zip(order, values)}
            cap = Fraction(capacity)
            for k in range(len(values)+1):
                done = sum(map(Fraction, values[:k]), Fraction(0))
                feasible = done <= cap and (k == len(values) or done + Fraction(values[k]) > cap)
                weight, posterior = _condition_prefix(prior, order, order[:k], capacity)
                self.assertEqual(weight > 0, feasible, (values, capacity, k))
                if feasible and k < len(values):
                    expected = float(done + Fraction(values[k]) - cap)
                    self.assertEqual(posterior[order[k]], (expected, expected))

    def test_interval_hull_unchanged_away_from_roundoff(self):
        prior = dict(a=(1., 3.), b=(2., 4.), c=(4., 4.))
        args = (prior, tuple(prior), ('a',), 2.)
        self.assertEqual(_condition_prefix(*args), legacy(*args))

    def test_strict_uncompleted_boundary_and_nonprefix_rejection(self):
        prior = dict(a=(4., 4.), b=(1., 1.))
        self.assertEqual(_condition_prefix(prior, tuple(prior), (), 4.)[0], 0.)
        self.assertEqual(_condition_prefix(prior, tuple(prior), ('a',), 4.)[0], 1.)
        with self.assertRaisesRegex(ValueError, 'service prefix'):
            _condition_prefix(prior, tuple(prior), ('b',), 4.)


if __name__ == '__main__':
    unittest.main()
