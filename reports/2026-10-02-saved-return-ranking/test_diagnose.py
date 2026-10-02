"""Artificial scalar fixtures only; no fitting or patient simulation."""
import math
import unittest
from diagnose import centered, corr, logit_signals, normalize, returns_to_go, summary


class ScalarDiagnosticTest(unittest.TestCase):
    def test_terminal_returns(self):
        self.assertEqual(returns_to_go([-1, -2, -3]), [-6, -5, -3])
        self.assertEqual(returns_to_go([]), [])

    def test_rollout_normalization(self):
        a = normalize([1, 2, 3])
        self.assertAlmostEqual(sum(a), 0)
        self.assertAlmostEqual(summary(a)["sd"], 1, places=7)
        self.assertEqual(normalize([5, 5]), [0, 0])

    def test_centering_is_group_specific(self):
        rows = [{"g": 0, "x": 2}, {"g": 0, "x": 4}, {"g": 1, "x": 100}]
        self.assertEqual(centered(rows, "x", ("g",)), [-1, 1, 0])

    def test_correlation_constants(self):
        self.assertIsNone(corr([1, 1], [2, 3]))
        self.assertAlmostEqual(corr([1, 2, 3], [-2, -4, -6]), -1)

    def test_logit_gradient_signs_and_sums(self):
        pg, eg, entropy = logit_signals([.8, .2], 0, 2)
        self.assertAlmostEqual(pg[0], .4)
        self.assertAlmostEqual(sum(pg), 0)
        self.assertAlmostEqual(sum(eg), 0)
        self.assertLess(eg[0], 0)
        self.assertGreater(entropy, 0)

    def test_entropy_gradient_finite_difference(self):
        logits = [.4, -.2, 1.1]
        def objective(z):
            exps = [math.exp(x) for x in z]
            p = [x/sum(exps) for x in exps]
            return .01*(-sum(x*math.log(x) for x in p)), p
        _, p = objective(logits)
        _, eg, _ = logit_signals(p, 1, .3)
        for i in range(3):
            a, b = list(logits), list(logits)
            a[i] += 1e-5
            b[i] -= 1e-5
            self.assertAlmostEqual((objective(a)[0]-objective(b)[0])/2e-5, eg[i], places=9)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            summary([float("nan")])


if __name__ == "__main__":
    unittest.main()
