"""Tests for the per-episode train-time domain-randomization hook.

The hook (``enable_train_randomization``) lets the *training* env resample the
stressed parameter each ``reset`` from a range, while the *eval* env keeps the
fixed config value. This enables clean train-on-range / test-OOD splits for the
robustness experiments (Phase 9) without leaking randomization into evaluation.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv
from src.rl.experiment import train_off_policy_agent


def _base_config() -> CapacityPlanningConfig:
    return CapacityPlanningConfig(
        num_facilities=2,
        production_lead_time=2,
        episode_horizon=4,
        supplier_disruption_rate=0.2,
        include_demand_forecast_state=True,
        demand_forecast_error=0.1,
    )


class _NoopAgent:
    algorithm = "noop"

    def __init__(self, env: CapacityPlanningEnv):
        self.env = env

    def reset(self):
        return None

    def select_action(self, state, explore=False, env=None):
        del state, explore
        return (env or self.env).noop_action()

    def observe(self, *args, **kwargs):
        return None

    def update(self):
        return {}

    def save(self, path):
        return None


class TrainRandomizationTest(unittest.TestCase):
    def test_disabled_by_default_keeps_config_values(self):
        env = CapacityPlanningEnv(_base_config(), seed=0)
        for ep in range(8):
            env.reset(seed=100 + ep)
            self.assertTrue(np.allclose(env.supplier_disruption_rate, 0.2))
            self.assertEqual(env.demand_forecast_error, 0.1)

    def test_disruption_range_covers_range_and_stays_within(self):
        env = CapacityPlanningEnv(_base_config(), seed=0)
        lo, hi = 0.0, 0.4
        env.enable_train_randomization(disruption_range=(lo, hi))
        sampled = []
        for ep in range(200):
            env.reset(seed=1000 + ep)
            rate = float(env.supplier_disruption_rate[0])
            # A single scalar is broadcast across all facilities each reset.
            self.assertTrue(np.allclose(env.supplier_disruption_rate, rate))
            self.assertGreaterEqual(rate, lo)
            self.assertLessEqual(rate, hi)
            sampled.append(rate)
        sampled = np.asarray(sampled)
        # Coverage: samples span most of the range, not stuck at one value.
        self.assertLess(sampled.min(), lo + 0.1)
        self.assertGreater(sampled.max(), hi - 0.1)
        self.assertGreater(sampled.std(), 0.05)

    def test_forecast_error_range_covers_range_and_stays_within(self):
        env = CapacityPlanningEnv(_base_config(), seed=0)
        lo, hi = 0.0, 0.4
        env.enable_train_randomization(forecast_error_range=(lo, hi))
        sampled = []
        for ep in range(200):
            env.reset(seed=2000 + ep)
            err = float(env.demand_forecast_error)
            self.assertGreaterEqual(err, lo)
            self.assertLessEqual(err, hi)
            sampled.append(err)
        sampled = np.asarray(sampled)
        self.assertLess(sampled.min(), lo + 0.1)
        self.assertGreater(sampled.max(), hi - 0.1)
        self.assertGreater(sampled.std(), 0.05)

    def test_demand_rate_multiplier_range_varies_true_rates_but_keeps_prior_estimates(self):
        config = CapacityPlanningConfig(
            num_facilities=2,
            production_lead_time=2,
            episode_horizon=4,
            demand_rates=(10.0, 20.0),
            demand_rate_estimates=(8.0, 18.0),
        )
        env = CapacityPlanningEnv(config, seed=0)
        lo, hi = 0.8, 1.5
        env.enable_train_randomization(demand_rate_multiplier_range=(lo, hi))

        sampled = []
        for ep in range(200):
            env.reset(seed=2500 + ep)
            multiplier = float(env.demand_rates[0] / config.demand_rates[0])
            self.assertGreaterEqual(multiplier, lo)
            self.assertLessEqual(multiplier, hi)
            np.testing.assert_allclose(env.demand_rate_estimates, np.asarray(config.demand_rate_estimates))
            sampled.append(multiplier)

        sampled = np.asarray(sampled)
        self.assertLess(sampled.min(), lo + 0.15)
        self.assertGreater(sampled.max(), hi - 0.15)
        self.assertGreater(sampled.std(), 0.08)

    def test_eval_env_is_fixed_while_train_env_varies(self):
        """Same config: an unrandomized eval env stays fixed; a randomized twin varies."""
        eval_env = CapacityPlanningEnv(_base_config(), seed=0)
        train_env = CapacityPlanningEnv(_base_config(), seed=0)
        train_env.enable_train_randomization(
            disruption_range=(0.0, 0.5),
            forecast_error_range=(0.0, 0.5),
            demand_rate_multiplier_range=(0.8, 1.5),
        )
        train_rates = set()
        train_demand_rates = set()
        for ep in range(50):
            eval_env.reset(seed=3000 + ep)
            train_env.reset(seed=3000 + ep)
            self.assertTrue(np.allclose(eval_env.supplier_disruption_rate, 0.2))
            self.assertEqual(eval_env.demand_forecast_error, 0.1)
            np.testing.assert_allclose(eval_env.demand_rates, np.asarray(_base_config().demand_rates))
            train_rates.add(round(float(train_env.supplier_disruption_rate[0]), 4))
            train_demand_rates.add(round(float(train_env.demand_rates[0]), 4))
        # The train env explored many distinct disruption regimes.
        self.assertGreater(len(train_rates), 20)
        self.assertGreater(len(train_demand_rates), 20)

    def test_training_loop_applies_configured_randomization(self):
        env = CapacityPlanningEnv(_base_config(), seed=0)
        rows = train_off_policy_agent(
            _NoopAgent(env),
            env,
            {
                "algorithm": "noop",
                "seed": 4000,
                "num_episodes": 3,
                "max_steps_per_episode": 1,
                "checkpoint_interval": 999,
                "train_randomization": {
                    "enabled": True,
                    "disruption_range": [0.0, 0.5],
                    "forecast_error_range": [0.0, 0.5],
                    "demand_rate_multiplier_range": [0.8, 1.5],
                },
            },
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["train_randomization_disruption_range"], "0|0.5")
        self.assertEqual(rows[0]["train_randomization_forecast_error_range"], "0|0.5")
        self.assertEqual(rows[0]["train_randomization_demand_rate_multiplier_range"], "0.8|1.5")

    def test_reproducible_given_episode_seed(self):
        env = CapacityPlanningEnv(_base_config(), seed=0)
        env.enable_train_randomization(disruption_range=(0.0, 0.4))
        env.reset(seed=42)
        first = float(env.supplier_disruption_rate[0])
        env.reset(seed=42)
        second = float(env.supplier_disruption_rate[0])
        self.assertEqual(first, second)

    def test_invalid_range_rejected(self):
        env = CapacityPlanningEnv(_base_config(), seed=0)
        with self.assertRaises(ValueError):
            env.enable_train_randomization(disruption_range=(0.4, 0.1))
        with self.assertRaises(ValueError):
            env.enable_train_randomization(forecast_error_range=(-0.1, 0.3))
        with self.assertRaises(ValueError):
            env.enable_train_randomization(demand_rate_multiplier_range=(1.2, 0.8))


if __name__ == "__main__":
    unittest.main()
