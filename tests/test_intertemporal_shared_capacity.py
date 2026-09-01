"""Contracts for the intertemporal shared-capacity development channel."""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.heuristics import (
    GraphSmoothedIntertemporalForecastPolicy,
    IntertemporalForecastOvertimePolicy,
)
from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)
from src.rl.preprocessing import facility_state_width


def _config(**overrides) -> CapacityPlanningConfig:
    values = dict(
        num_facilities=2,
        action_mode="facility_net",
        production_lead_time=3,
        episode_horizon=20,
        demand_rates=(0.0, 0.0),
        initial_specimens=(20.0, 20.0),
        initial_reagents=(100.0, 100.0),
        initial_idle_bioreactors=(4.0, 4.0),
        max_idle_bioreactors=(20.0, 20.0),
        max_overtime_fraction=0.5,
        enable_overtime_control=True,
        enable_intertemporal_overtime_commitment=True,
        overtime_commitment_lead_time=2,
        overtime_commitment_persistence=0.5,
        overtime_shared_budget_fraction=0.5,
        weight_overtime_activation=1_000.0,
    )
    values.update(overrides)
    return CapacityPlanningConfig(**values)


def _action(env: CapacityPlanningEnv, fractions: tuple[float, ...]) -> np.ndarray:
    n = env.config.num_facilities
    action = env.noop_action()
    action[4 * n : 5 * n] = 2.0 * np.asarray(fractions, dtype=float) - 1.0
    return action


class ScheduledReferralWaveTest(unittest.TestCase):
    def test_wave_timing_repeat_and_forward_forecast(self) -> None:
        config = CapacityPlanningConfig(
            num_facilities=4,
            action_mode="facility_net",
            episode_horizon=20,
            demand_rates=(1.0, 1.0, 1.0, 1.0),
            initial_specimens=(0.0, 0.0, 0.0, 0.0),
            initial_reagents=(20.0, 20.0, 20.0, 20.0),
            initial_idle_bioreactors=(2.0, 2.0, 2.0, 2.0),
            max_specimens=(20.0, 20.0, 20.0, 20.0),
            max_reagents=(40.0, 40.0, 40.0, 40.0),
            max_idle_bioreactors=(5.0, 5.0, 5.0, 5.0),
            max_reagent_replenishment=(20.0, 20.0, 20.0, 20.0),
            include_demand_forecast_state=True,
            demand_forecast_horizon=3,
            enable_scheduled_referral_waves=True,
            scheduled_referral_clusters=((0, 1), (2, 3)),
            scheduled_referral_start_step=2,
            scheduled_referral_block_length=2,
            scheduled_referral_peak_multiplier=2.0,
            scheduled_referral_repeat=True,
        )
        env = CapacityPlanningEnv(config, seed=0)

        np.testing.assert_array_equal(
            env._scheduled_referral_multiplier_at(0), np.ones(4)
        )
        np.testing.assert_array_equal(
            env._scheduled_referral_multiplier_at(2), (2.0, 2.0, 1.0, 1.0)
        )
        np.testing.assert_array_equal(
            env._scheduled_referral_multiplier_at(4), (1.0, 1.0, 2.0, 2.0)
        )
        np.testing.assert_array_equal(
            env._scheduled_referral_multiplier_at(6), (2.0, 2.0, 1.0, 1.0)
        )
        # At t=0 the three-step forward forecast covers t=1,2,3.
        np.testing.assert_allclose(env.demand_forecast, (5.0, 5.0, 3.0, 3.0))

    def test_nonrepeating_schedule_returns_to_baseline(self) -> None:
        config = CapacityPlanningConfig(
            num_facilities=2,
            action_mode="facility_net",
            demand_rates=(1.0, 1.0),
            initial_specimens=(0.0, 0.0),
            initial_reagents=(10.0, 10.0),
            initial_idle_bioreactors=(1.0, 1.0),
            max_idle_bioreactors=(2.0, 2.0),
            enable_scheduled_referral_waves=True,
            scheduled_referral_clusters=((0,), (1,)),
            scheduled_referral_block_length=1,
            scheduled_referral_peak_multiplier=3.0,
            scheduled_referral_repeat=False,
        )
        env = CapacityPlanningEnv(config, seed=0)
        np.testing.assert_array_equal(
            env._scheduled_referral_multiplier_at(3), np.ones(2)
        )

    def test_invalid_schedule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(
                CapacityPlanningConfig(
                    num_facilities=2,
                    enable_scheduled_referral_waves=True,
                    scheduled_referral_clusters=((0, 2),),
                )
            )


class IntertemporalCommitmentTest(unittest.TestCase):
    def test_shared_budget_projection_is_continuous_and_bounded(self) -> None:
        env = CapacityPlanningEnv(_config(), seed=0)
        fraction, requested = env._decode_overtime(_action(env, (1.0, 1.0)))
        projected_fraction, projected = env._project_overtime_commitment(
            fraction, requested
        )
        self.assertAlmostEqual(float(projected.sum()), 2.0)
        np.testing.assert_allclose(projected, (1.0, 1.0))
        np.testing.assert_allclose(projected_fraction, (0.5, 0.5))

        _, slightly_lower = env._project_overtime_commitment(
            *_decode(env, (1.0, 0.99))
        )
        self.assertLess(float(np.max(np.abs(projected - slightly_lower))), 0.02)

    def test_request_is_delayed_then_persistent(self) -> None:
        env = CapacityPlanningEnv(_config(), seed=0)
        full = _action(env, (1.0, 1.0))
        zero = _action(env, (0.0, 0.0))

        _, _, _, first = env.step(full)
        np.testing.assert_array_equal(first["overtime_active_capacity"], (0.0, 0.0))
        np.testing.assert_array_equal(first["overtime_production"], (0.0, 0.0))
        self.assertAlmostEqual(first["overtime_activation_cost"], 2_000.0)

        _, _, _, second = env.step(zero)
        np.testing.assert_array_equal(second["overtime_active_capacity"], (0.0, 0.0))

        _, _, _, third = env.step(zero)
        np.testing.assert_allclose(third["overtime_active_capacity"], (0.5, 0.5))
        np.testing.assert_allclose(third["overtime_production"], (0.5, 0.5))

        _, _, _, fourth = env.step(zero)
        np.testing.assert_allclose(fourth["overtime_active_capacity"], (0.25, 0.25))

    def test_observation_width_mirror(self) -> None:
        config = _config(enable_overtime_fatigue=True)
        env = CapacityPlanningEnv(config, seed=0)
        env_dict = {
            "num_facilities": config.num_facilities,
            "production_lead_time": config.production_lead_time,
            "enable_overtime_control": True,
            "enable_overtime_fatigue": True,
            "enable_intertemporal_overtime_commitment": True,
            "overtime_commitment_lead_time": 2,
        }
        self.assertEqual(facility_state_width(env_dict), env.features_per_facility)

    def test_forecast_comparators_respect_budget(self) -> None:
        env = CapacityPlanningEnv(_config(), seed=0)
        env.specimens[:] = 0.0
        env.demand_forecast = np.asarray((10.0, 1.0))

        forecast = IntertemporalForecastOvertimePolicy().select_action(
            env.observation(), env=env
        )
        _, forecast_surge = env._decode_overtime(forecast)
        self.assertAlmostEqual(float(forecast_surge.sum()), 2.0, places=6)
        self.assertGreater(forecast_surge[0], forecast_surge[1])

        graph = GraphSmoothedIntertemporalForecastPolicy().select_action(
            env.observation(), env=env
        )
        _, graph_surge = env._decode_overtime(graph)
        self.assertAlmostEqual(float(graph_surge.sum()), 2.0, places=6)
        np.testing.assert_allclose(graph_surge[0], graph_surge[1])

    def test_patient_snapshot_roundtrip(self) -> None:
        config = PatientEnvConfig(
            base=_config(
                initial_specimens=(0.0, 0.0),
                demand_rates=(2.0, 2.0),
                episode_horizon=15,
            )
        )
        env = PatientConditionCapacityEnv(config, seed=7)
        env.step(_action(env, (1.0, 0.0)))
        env.step(_action(env, (0.0, 1.0)))
        state = env.state_dict()
        self.assertTrue(state["enable_intertemporal_overtime_commitment"])
        self.assertIn("overtime_active_capacity", state["arrays"])
        self.assertIn("overtime_commitment_pipeline", state["arrays"])

        restored = PatientConditionCapacityEnv(config, seed=99)
        restored.load_state_dict(state)
        out_a = env.step(_action(env, (0.4, 0.6)))
        out_b = restored.step(_action(restored, (0.4, 0.6)))
        np.testing.assert_array_equal(out_a[0], out_b[0])
        self.assertEqual(out_a[1], out_b[1])
        np.testing.assert_array_equal(
            restored.overtime_active_capacity, env.overtime_active_capacity
        )

    def test_invalid_commitment_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(
                _config(enable_overtime_control=False)
            )
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(_config(overtime_commitment_lead_time=0))
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(_config(overtime_commitment_persistence=1.0))
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(_config(overtime_shared_budget_fraction=1.1))


def _decode(
    env: CapacityPlanningEnv, fractions: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    return env._decode_overtime(_action(env, fractions))


if __name__ == "__main__":
    unittest.main()
