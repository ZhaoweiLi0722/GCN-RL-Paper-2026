import unittest

import numpy as np

from evaluation.run_full_benchmark import load_benchmark_plan, select_scenarios
from src.rl.config import load_config
from src.rl.experiment import build_env


REGIONAL_DRIFT = (
    "experiments/configs/20_clinic_patient_condition_geo_regional_drift.json"
)
ABRUPT_SHIFT = (
    "experiments/configs/20_clinic_patient_condition_geo_abrupt_regime_shift.json"
)
COMPOUND_STRESS = (
    "experiments/configs/20_clinic_patient_condition_geo_compound_regional_stress.json"
)


class DemandRegimeScenarioTest(unittest.TestCase):
    def test_all_regime_scenarios_build_as_integrated_patient_geo_environments(self):
        for path in (REGIONAL_DRIFT, ABRUPT_SHIFT, COMPOUND_STRESS):
            with self.subTest(path=path):
                config = load_config(path)
                env = build_env({"env": config}, seed=51)
                state = env.reset(seed=51)

                self.assertEqual(config["env_type"], "patient_condition")
                self.assertEqual(config["num_facilities"], 20)
                self.assertEqual(config["demand_forecast_source"], "prior_estimate")
                self.assertTrue(config["include_demand_history_state"])
                self.assertEqual(config["demand_history_window"], 12)
                self.assertEqual(len(config["clinic_coordinates"]), 20)
                self.assertEqual(state.shape, (env.observation_size,))
                self.assertEqual(env.clinic_distance_matrix.shape, (20, 20))

    def test_prior_forecast_does_not_reveal_abrupt_active_regime(self):
        config = load_config(ABRUPT_SHIFT)
        env = build_env({"env": config}, seed=52)
        prior = np.asarray(config["demand_rate_estimates"], dtype=float)
        initial = np.asarray(config["demand_regime_initial_multipliers"], dtype=float)

        np.testing.assert_allclose(
            env.demand_forecast,
            prior * int(config["demand_forecast_horizon"]),
        )
        np.testing.assert_allclose(
            env._effective_demand_rates(),
            prior * initial,
        )
        self.assertFalse(
            np.allclose(
                env.demand_forecast / int(config["demand_forecast_horizon"]),
                env._effective_demand_rates(),
            )
        )

    def test_abrupt_scenario_flips_regional_hotspots_at_week_26(self):
        config = load_config(ABRUPT_SHIFT)
        env = build_env({"env": config}, seed=53)
        action = env.noop_action()
        initial = np.asarray(config["demand_regime_initial_multipliers"], dtype=float)
        final = np.asarray(config["demand_regime_final_multipliers"], dtype=float)

        for _ in range(25):
            env.step(action)
        np.testing.assert_allclose(env.demand_regime_multiplier, initial)

        state, _reward, done, _info = env.step(action)

        self.assertFalse(done)
        self.assertEqual(state.shape, (env.observation_size,))
        np.testing.assert_allclose(env.demand_regime_multiplier, final)

    def test_compound_scenario_is_reproducible_under_common_random_numbers(self):
        config = load_config(COMPOUND_STRESS)
        first = build_env({"env": config}, seed=54)
        second = build_env({"env": config}, seed=54)

        for _ in range(5):
            np.testing.assert_allclose(first.observation(), second.observation())
            first_state, first_reward, first_done, _ = first.step(first.noop_action())
            second_state, second_reward, second_done, _ = second.step(second.noop_action())
            np.testing.assert_allclose(first_state, second_state)
            self.assertEqual(first_reward, second_reward)
            self.assertEqual(first_done, second_done)

    def test_regime_scenarios_are_registered_for_matched_benchmark_runs(self):
        plan = load_benchmark_plan(
            "experiments/configs/residual_policy_benchmark.json"
        )
        names = (
            "patient_condition_geo_regional_drift",
            "patient_condition_geo_abrupt_regime_shift",
            "patient_condition_geo_compound_regional_stress",
        )

        selected = select_scenarios(plan, names)

        self.assertEqual(tuple(item["name"] for item in selected), names)


if __name__ == "__main__":
    unittest.main()
