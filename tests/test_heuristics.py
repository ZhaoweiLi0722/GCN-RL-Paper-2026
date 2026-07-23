"""Tests for deterministic heuristic benchmark policies."""

from __future__ import annotations

from dataclasses import asdict, replace
import unittest

import numpy as np

from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from src.baselines.heuristics import (
    available_heuristics,
    facility_net_action_from_state,
    ForecastMyopicPolicy,
    heuristic_settings_for_policy,
    IsolatedPolicy,
    MeanDemandLookahead1Policy,
    MeanDemandLookahead2Policy,
    MyopicPolicy,
    PatientPriorityMyopicPolicy,
    ShieldedMeanDemandLookahead2Policy,
    ShieldedPatientPriorityMyopicPolicy,
)
from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.rl.config import load_config
from src.rl.experiment import build_env


class HeuristicPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=4), seed=2)
        self.state = self.env.reset(seed=2)

    def test_all_heuristics_emit_valid_facility_net_actions(self) -> None:
        for policy_cls in (
            MyopicPolicy,
            IsolatedPolicy,
            MeanDemandLookahead1Policy,
            MeanDemandLookahead2Policy,
            ForecastMyopicPolicy,
        ):
            with self.subTest(policy=policy_cls.__name__):
                policy = policy_cls()
                action = policy.select_action(self.state, env=self.env)
                self.assertEqual(action.shape, (self.env.action_size,))
                self.assertTrue(np.all(action >= -1.0))
                self.assertTrue(np.all(action <= 1.0))

    def test_iso_disables_transfer_actions(self) -> None:
        action = IsolatedPolicy().select_action(self.state, env=self.env)
        n = self.env.config.num_facilities
        np.testing.assert_allclose(action[: 3 * n], np.zeros(3 * n, dtype=np.float32))

    def test_state_based_helper_matches_live_policy(self) -> None:
        live_action = MeanDemandLookahead2Policy().select_action(self.state, env=self.env)
        state_action = facility_net_action_from_state(
            self.state,
            asdict(self.env.config),
            settings=heuristic_settings_for_policy("mdl2"),
        )

        np.testing.assert_allclose(state_action, live_action, atol=1e-6)

    def test_mean_demand_heuristic_uses_prior_estimates_when_truth_drifts(self) -> None:
        estimated = (12.0,) * 20
        config = replace(
            make_20_clinic_config(episode_horizon=2, supplier_disruption_rate=0.0),
            demand_rates=(4.0,) * 20,
            demand_rate_estimates=estimated,
            initial_reagents=(0.0,) * 20,
        )
        env = CapacityPlanningEnv(config, seed=31)
        state = env.reset(seed=31)

        live_action = MeanDemandLookahead2Policy().select_action(state, env=env)
        state_action = facility_net_action_from_state(
            state,
            asdict(env.config),
            settings=heuristic_settings_for_policy("mdl2"),
        )
        true_rate_action = facility_net_action_from_state(
            state,
            {**asdict(env.config), "demand_rate_estimates": asdict(env.config)["demand_rates"]},
            settings=heuristic_settings_for_policy("mdl2"),
        )
        n = env.config.num_facilities

        np.testing.assert_allclose(state_action, live_action, atol=1e-6)
        self.assertGreater(
            float(live_action[3 * n : 4 * n].mean()),
            float(true_rate_action[3 * n : 4 * n].mean()),
        )

    def test_forecast_myopic_policy_uses_demand_forecast_state(self) -> None:
        config = replace(
            make_20_clinic_config(episode_horizon=2),
            include_demand_forecast_state=True,
            demand_forecast_horizon=2,
            demand_forecast_error=0.0,
        )
        env = CapacityPlanningEnv(config, seed=13)
        state = env.reset(seed=13)

        live_action = ForecastMyopicPolicy().select_action(state, env=env)
        state_action = facility_net_action_from_state(
            state,
            asdict(env.config),
            settings=heuristic_settings_for_policy("fmyo"),
        )
        myopic_action = MyopicPolicy().select_action(state, env=env)
        n = env.config.num_facilities

        np.testing.assert_allclose(state_action, live_action, atol=1e-6)
        self.assertGreaterEqual(
            float(live_action[3 * n : 4 * n].mean()),
            float(myopic_action[3 * n : 4 * n].mean()),
        )

    def test_patient_priority_helper_matches_live_policy_with_pipeline_state(self) -> None:
        env_config = load_config("experiments/configs/20_clinic_patient_condition_geo.json")
        config = load_config("configs/gcn_residual_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=14)
        state = env.reset(seed=14)
        policy = PatientPriorityMyopicPolicy()
        for _ in range(6):
            action = policy.select_action(state, env=env)
            state, _reward, done, _info = env.step(action)
            if done:
                break

        live_action = policy.select_action(state, env=env)
        state_action = facility_net_action_from_state(
            state,
            env_config,
            settings=heuristic_settings_for_policy("pmyo"),
        )

        np.testing.assert_allclose(state_action, live_action, atol=1e-6)

    def test_shielded_pmyo_is_registered_and_can_fall_back_to_anchor(self) -> None:
        self.assertIn("pmyo_shield", available_heuristics())
        env_config = load_config("experiments/configs/20_clinic_patient_condition_geo.json")
        config = load_config("configs/gcn_residual_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=21)
        state = env.reset(seed=21)

        anchor_action = PatientPriorityMyopicPolicy().select_action(state, env=env)
        shield_action = ShieldedPatientPriorityMyopicPolicy(
            config={"shield_lookahead": 0},
        ).select_action(state, env=env)

        np.testing.assert_allclose(shield_action, anchor_action, atol=1e-6)

    def test_shielded_mdl2_is_registered_and_can_fall_back_to_anchor(self) -> None:
        self.assertIn("mdl2_shield", available_heuristics())
        env_config = load_config("experiments/configs/20_clinic_patient_condition_geo.json")
        config = load_config("configs/gcn_residual_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=22)
        state = env.reset(seed=22)

        anchor_action = MeanDemandLookahead2Policy().select_action(state, env=env)
        shield_action = ShieldedMeanDemandLookahead2Policy(
            config={"shield_lookahead": 0},
        ).select_action(state, env=env)

        np.testing.assert_allclose(shield_action, anchor_action, atol=1e-6)

    def test_shielded_pmyo_emits_valid_patient_geo_action(self) -> None:
        env_config = load_config("experiments/configs/20_clinic_patient_condition_geo.json")
        config = load_config("configs/gcn_residual_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=22)
        state = env.reset(seed=22)
        policy = ShieldedPatientPriorityMyopicPolicy(
            config={
                "shield_lookahead": 1,
                "shield_epsilons": [0.005],
                "candidate_groups": ["replenishment_patient_risk_pressure"],
            },
        )

        action = policy.select_action(state, env=env)

        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_formal_evaluation_summarizes_heuristic_rows(self) -> None:
        policy = MyopicPolicy()
        rows = evaluate_agent(
            policy,
            self.env,
            algorithm="myo",
            seed=11,
            replications=2,
            max_steps=3,
        )
        summary = summarize_rows(rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["algorithm"], "myo")
        self.assertEqual(summary["replications"], 2)
        self.assertIn("total_cost_mean", summary)


if __name__ == "__main__":
    unittest.main()
