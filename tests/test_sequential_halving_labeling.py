"""Tests for CRN + sequential-halving counterfactual labelling."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.sequential_halving_labeling import (
    agreement_between,
    sequential_halving_best_arm,
    uniform_allocation_best_arm,
)

N_ARMS = 11
TRUE_BEST = 4
TRUE_MEAN = np.array([float((i - TRUE_BEST) ** 2) for i in range(N_ARMS)])


def _evaluator(trial: int, noise: float = 12.0):
    """Smooth cost curve plus a shared per-world shock (exact CRN)."""

    def evaluate(arm, world):
        shared = np.random.default_rng((world * 1000003) ^ (trial * 7919)).normal(0, noise)
        idio = np.random.default_rng(
            (world * 1000003) ^ (arm * 104729) ^ (trial * 7919)
        ).normal(0, 1.5)
        return TRUE_MEAN[arm] + shared + idio
    return evaluate


class BudgetAccountingTest(unittest.TestCase):
    def test_uniform_budget_is_arms_times_worlds(self) -> None:
        out = uniform_allocation_best_arm(range(N_ARMS), _evaluator(0), list(range(6)))
        self.assertEqual(out["simulator_calls"], N_ARMS * 6)

    def test_sequential_halving_respects_the_budget(self) -> None:
        budget = N_ARMS * 8
        out = sequential_halving_best_arm(
            range(N_ARMS), _evaluator(0), list(range(64)), budget=budget
        )
        self.assertLessEqual(out["simulator_calls"], budget)

    def test_survivors_halve_each_round(self) -> None:
        out = sequential_halving_best_arm(
            range(N_ARMS), _evaluator(0), list(range(64)), budget=N_ARMS * 8
        )
        counts = [r["arms"] for r in out["rounds"]]
        self.assertEqual(counts[0], N_ARMS)
        for earlier, later in zip(counts, counts[1:]):
            self.assertLessEqual(later, max(1, earlier // 2))

    def test_best_arm_gets_the_most_worlds(self) -> None:
        out = sequential_halving_best_arm(
            range(N_ARMS), _evaluator(0), list(range(64)), budget=N_ARMS * 8
        )
        # The point of the method: budget concentrates on the contenders.
        self.assertGreater(out["worlds_on_best"], N_ARMS * 8 // N_ARMS)


class AccuracyTest(unittest.TestCase):
    def _hit_rates(self, worlds: int, trials: int = 200):
        uniform_hits = halving_hits = 0
        for trial in range(trials):
            evaluate = _evaluator(trial)
            base = list(range(trial * 100, trial * 100 + worlds))
            uniform = uniform_allocation_best_arm(range(N_ARMS), evaluate, base)
            halving = sequential_halving_best_arm(
                range(N_ARMS),
                evaluate,
                list(range(trial * 100, trial * 100 + worlds * 4)),
                budget=uniform["simulator_calls"],
            )
            uniform_hits += uniform["best_arm"] == TRUE_BEST
            halving_hits += halving["best_arm"] == TRUE_BEST
        return uniform_hits / trials, halving_hits / trials

    def test_beats_uniform_allocation_at_equal_budget(self) -> None:
        uniform, halving = self._hit_rates(worlds=4)
        self.assertGreater(halving, uniform + 0.05)

    def test_both_improve_with_budget(self) -> None:
        small_u, small_h = self._hit_rates(worlds=4)
        large_u, large_h = self._hit_rates(worlds=16)
        self.assertGreater(large_u, small_u)
        self.assertGreater(large_h, small_h)

    def test_noise_free_problem_is_solved_by_both(self) -> None:
        evaluate = _evaluator(0, noise=0.0)

        def clean(arm, world):
            return float(TRUE_MEAN[arm])

        self.assertEqual(
            uniform_allocation_best_arm(range(N_ARMS), clean, [1, 2])["best_arm"],
            TRUE_BEST,
        )
        self.assertEqual(
            sequential_halving_best_arm(range(N_ARMS), clean, list(range(16)))["best_arm"],
            TRUE_BEST,
        )


class AgreementTest(unittest.TestCase):
    def test_disjoint_groups_agree_on_an_easy_problem(self) -> None:
        def clean(arm, world):
            return float(TRUE_MEAN[arm])

        out = agreement_between(
            lambda group: uniform_allocation_best_arm(range(N_ARMS), clean, group)["best_arm"],
            [[0, 1], [2, 3], [4, 5]],
        )
        self.assertEqual(out["agreement"], 1.0)
        self.assertTrue(out["unanimous"])

    def test_requires_two_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "two seed groups"):
            agreement_between(lambda g: 0, [[1]])


class GuardTest(unittest.TestCase):
    def test_rejects_single_arm(self) -> None:
        with self.assertRaisesRegex(ValueError, "two arms"):
            sequential_halving_best_arm([0], _evaluator(0), [1, 2])

    def test_rejects_empty_world_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "one world seed"):
            sequential_halving_best_arm(range(4), _evaluator(0), [])


if __name__ == "__main__":
    unittest.main()
