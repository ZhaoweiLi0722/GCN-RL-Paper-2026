"""Artificial arithmetic only; no policy forward, optimizer or patient call."""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "reports/2026-10-01-p2-training-diagnostic/diagnose.py"
SPEC = importlib.util.spec_from_file_location("p2_training_diagnostic", PATH)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class TrainingDiagnosticTests(unittest.TestCase):
    def test_bound_covers_tanh_features_and_is_tight_in_limit(self):
        w = np.array([.2, -.3, .1])
        bound = D.score_bound(w)["residual_pairwise_span_upper_bound"]
        for x, y in (([-1, 0, .3], [.5, .5, 1]), (np.sign(w), -np.sign(w))):
            self.assertLessEqual(abs(float(w @ (np.array(x) - y))), bound)
        self.assertAlmostEqual(float(w @ (2*np.sign(w))), bound)

    def test_common_bias_cancels(self):
        w, x, y = np.array([.1, .2]), np.array([1., -1.]), np.array([-.5, 1.])
        for bias in (-100., 0., 100.):
            self.assertAlmostEqual((w@x+bias)-(w@y+bias), w@(x-y))

    def test_certificate_is_strict_and_not_a_converse(self):
        self.assertTrue(D.score_bound([0.])["reference_certified_for_all_k_ge_2"])
        for w in ([math.log(9)/2], [2.]):
            self.assertFalse(D.score_bound(w)["reference_certified_for_all_k_ge_2"])
        self.assertTrue(D.score_bound([2.])["singleton_has_no_choice"])
        self.assertAlmostEqual(D.score_bound([0.])["prior_margin_k6"], math.log(45))

    def test_invalid_mass_and_nonfinite_rejected(self):
        for mass in (0, .5, 1, float("nan")):
            with self.assertRaises(ValueError):
                D.score_bound([0.], mass)
        for weights in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                D.score_bound(weights)

    def test_scalar_value_range(self):
        self.assertEqual(D.value_range([.2, -.3], [-.5]), (-1., 0.))
        with self.assertRaises(ValueError):
            D.value_range([1.], [1., 2.])

    def test_unavoidable_mse_bound(self):
        report = D.target_range_readback([-2., -.5, 1.], (-1., 0.))
        self.assertEqual(report["outside_count"], 2)
        self.assertAlmostEqual(report["mse_lower_bound_from_output_range"], 2/3)
        self.assertAlmostEqual(report["outside_fraction"], 2/3)
        with self.assertRaises(ValueError):
            D.target_range_readback([0.], (1., -1.))

    def test_returns_do_not_cross_episode_boundaries(self):
        first, a = D.reverse_returns([-1., -2.], [.5, .25])
        second, _ = D.reverse_returns([-10., -20.], [0., 0.])
        np.testing.assert_array_equal(first, [-3., -2.])
        np.testing.assert_array_equal(a, [-3.5, -2.25])
        np.testing.assert_array_equal(second, [-30., -20.])
        with self.assertRaises(ValueError):
            D.reverse_returns([[-1., -2.], [-10., -20.]], [[0., 0.], [0., 0.]])
        with self.assertRaises(ValueError):
            D.reverse_returns([-1., -2.], [0.])

    def test_normalization_is_global_not_per_episode(self):
        a = np.array([[0., 1.], [10., 11.]])
        result = D.normalize(a)
        self.assertAlmostEqual(float(result.mean()), 0)
        self.assertAlmostEqual(float(result.std()), 1, places=7)
        self.assertLess(float(result[0].mean()), -.9)
        np.testing.assert_array_equal(D.normalize([1., 1.]), [0., 0.])

    def test_position_variation_decomposition(self):
        self.assertAlmostEqual(D.position_share([[0., 1.], [0., 1.]]), 1)
        self.assertEqual(D.position_share([[0., 0.], [1., 1.]]), 0)
        self.assertIsNone(D.position_share([[0., 0.], [0., 0.]]))
        with self.assertRaises(ValueError):
            D.position_share([0., 1.])

    def test_reference_logit_signal_matches_scalar_difference(self):
        p = .9
        entropy = -p*math.log(p)-(1-p)*math.log1p(-p)
        for chosen in (False, True):
            a = .7
            ret, ent = D.reference_logit_signal([a], [chosen], [p], [entropy])
            def objective(z):
                q = math.exp(z)/(1+math.exp(z))
                h = -q*math.log(q)-(1-q)*math.log1p(-q)
                ratio = q/p if chosen else (1-q)/(1-p)
                return a*ratio+.01*h
            z, eps = math.log(9), 1e-5
            derivative = (objective(z+eps)-objective(z-eps))/(2*eps)
            self.assertAlmostEqual(float(ret[0]+ent[0]), derivative, places=8)
        self.assertLess(ent[0], 0)

    def test_signal_rejects_bad_probability_or_shape(self):
        for p in (0., -1., 1.1, float("nan")):
            with self.assertRaises(ValueError):
                D.reference_logit_signal([1.], [True], [p], [.3])
        with self.assertRaises(ValueError):
            D.reference_logit_signal([1., 2.], [True], [.9], [.3])


if __name__ == "__main__":
    unittest.main()
