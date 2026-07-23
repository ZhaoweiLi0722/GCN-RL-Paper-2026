"""Fast tests for the algorithm-verification harness.

These do not train agents to convergence (that is the job of
``evaluation.verify_algorithms``); they check the LQR task, the Riccati solver,
and that the harness wiring runs end-to-end on a tiny budget.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.verification.lqr_env import (
    lqr_gain,
    make_double_integrator,
    solve_discrete_are,
)


class DiscreteARETests(unittest.TestCase):
    def test_solution_satisfies_riccati_equation(self) -> None:
        env = make_double_integrator()
        A, B, Q, R = env.A, env.B, env.Q, env.R
        P = solve_discrete_are(A, B, Q, R)
        residual = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) - P
        self.assertLess(np.max(np.abs(residual)), 1e-6)

    def test_scalar_case_matches_closed_form(self) -> None:
        # Scalar system a=b=q=r=1: p = q + a^2 p - (abp)^2/(r+b^2 p).
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])
        P = solve_discrete_are(A, B, Q, R)
        p = float(P[0, 0])
        # Closed form of the fixed point: p = (1 + sqrt(5)) / 2 (golden ratio).
        self.assertAlmostEqual(p, (1.0 + np.sqrt(5.0)) / 2.0, places=5)


class LinearQuadraticEnvTests(unittest.TestCase):
    def test_reset_is_deterministic_by_seed(self) -> None:
        env = make_double_integrator()
        s1 = env.reset(seed=7)
        s2 = env.reset(seed=7)
        np.testing.assert_allclose(s1, s2)

    def test_step_contract(self) -> None:
        env = make_double_integrator()
        env.reset(seed=0)
        next_state, reward, done, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
        self.assertEqual(next_state.shape, (env.state_dim,))
        self.assertIsInstance(reward, float)
        self.assertIn("cost", info)
        self.assertFalse(done)

    def test_action_shape_validation(self) -> None:
        env = make_double_integrator()
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step(np.zeros(env.action_dim + 1, dtype=np.float32))

    def test_lqr_policy_beats_random(self) -> None:
        env = make_double_integrator()
        gain = env.optimal_gain()

        def rollout(action_fn) -> float:
            total = 0.0
            for seed in range(20):
                state = env.reset(seed=seed)
                done = False
                while not done:
                    state, _r, done, info = env.step(action_fn(state))
                    total += info["cost"]
            return total

        rng = np.random.default_rng(0)
        random_cost = rollout(lambda s: rng.uniform(-1.0, 1.0, env.action_dim))
        lqr_cost = rollout(lambda s: env.optimal_normalized_action(s, gain))
        # The LQR policy should be dramatically cheaper on an unstable system.
        self.assertLess(lqr_cost, 0.2 * random_cost)


class HarnessWiringTest(unittest.TestCase):
    def test_verify_algorithm_runs_end_to_end(self) -> None:
        try:
            from src.rl.networks import torch  # noqa: F401
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")

        from evaluation.verify_algorithms import verify_algorithm

        # Tiny budget: we only assert the harness produces a finite score field,
        # not that the agent has converged.
        result = verify_algorithm(
            "flat_ddpg",
            train_steps=200,
            eval_episodes=3,
            seed=0,
            random_cost=100.0,
            lqr_cost=10.0,
        )
        self.assertEqual(result["algorithm"], "flat_ddpg")
        self.assertIn("normalized_score", result)
        self.assertIn(result["status"].split(":")[0], {"PASS", "FAIL", "ERROR"})


if __name__ == "__main__":
    unittest.main()
