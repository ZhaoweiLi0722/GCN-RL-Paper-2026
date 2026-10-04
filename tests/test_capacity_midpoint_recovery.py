"""Zero-update boundary arithmetic regressions; no patient episodes."""
import math
import unittest
from fractions import Fraction

from src.baselines.capacity_forecast_recovery3 import midpoint_residual
from src.baselines.capacity_completion_control_recovery1 import _condition_prefix


class MidpointRecoveryTests(unittest.TestCase):
    def test_adjacent_positive_midpoint_is_not_completed(self):
        bounds = [(2., math.nextafter(2., math.inf))]
        residual, complete = midpoint_residual(bounds, 2.)
        self.assertFalse(complete)
        self.assertGreater(residual, 0)
        weight, _ = _condition_prefix({"a": bounds[0], "b": (4., 4.)}, ["a", "b"], [], 2.)
        self.assertGreater(weight, 0)

    def test_exact_point_equality_is_completed(self):
        self.assertEqual(midpoint_residual([(2., 2.)], 2.), (0., True))

    def test_prefix_cancellation_matches_exact_dyadics(self):
        values = [0., .5, 1., 2., 4., 8.]
        for lo in values:
            for hi in [lo, math.nextafter(lo, math.inf)]:
                for capacity in values:
                    intervals = [(1., 1.), (lo, hi)]
                    exact = sum((Fraction(a)+Fraction(b))/2 for a,b in intervals)-Fraction(capacity)
                    residual, complete = midpoint_residual(intervals, capacity)
                    self.assertEqual(complete, exact <= 0)
                    self.assertEqual(math.copysign(1, residual) if residual else 0,
                                     math.copysign(1, float(exact)) if float(exact) else 0)

    def test_ordinary_decisions_unchanged(self):
        self.assertEqual(midpoint_residual([(4., 4.)], 2.), (2., False))
        self.assertEqual(midpoint_residual([(4., 4.), (4., 4.)], 9.), (-1., True))
        self.assertEqual(midpoint_residual([], 0.), (0., True))

    def test_invalid_intervals_rejected(self):
        for intervals, capacity in [([(2., 1.)], 2.), ([(float("nan"), 2.)], 2.), ([(1., 2.)], -1.)]:
            with self.assertRaises(ValueError):
                midpoint_residual(intervals, capacity)


if __name__ == "__main__":
    unittest.main()
