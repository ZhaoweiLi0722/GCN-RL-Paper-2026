"""Tests for deterministic heuristic benchmark policies."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

import numpy as np

import src.baselines.heuristics as heuristics
from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from src.baselines.heuristics import (
    _balance_shortage_surplus,
    available_heuristics,
    facility_net_action_from_state,
    ForecastMeanDemandLookahead2Policy,
    ForecastMyopicPolicy,
    heuristic_settings_for_policy,
    IsolatedPolicy,
    MeanDemandLookahead1Policy,
    MeanDemandLookahead2Policy,
    MyopicPolicy,
    PatientPriorityMyopicPolicy,
    RollingMeanDemandLookahead2Policy,
    ShieldedMeanDemandLookahead2Policy,
    ShieldedPatientPriorityMyopicPolicy,
    shield_rollout_metrics,
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
            ForecastMeanDemandLookahead2Policy,
            RollingMeanDemandLookahead2Policy,
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

    def test_facility_edges_are_cached_without_changing_actions(self) -> None:
        env_config = asdict(self.env.config)
        env_config["action_mode"] = "facility_net"
        env_config["clinic_coordinates"] = [
            (33.70 + 0.01 * index, -84.50 + 0.01 * index)
            for index in range(self.env.config.num_facilities)
        ]
        env_config["geographic_neighbor_k"] = 3
        env_config["specimen_edges"] = None
        env_config["capacity_edges"] = None
        env_config["resource_edges"] = None
        settings = heuristic_settings_for_policy("mdl2")
        heuristics._cached_facility_edge_sets.cache_clear()

        with patch.object(
            heuristics,
            "geographic_knn_edges",
            wraps=heuristics.geographic_knn_edges,
        ) as geographic_edges:
            expected = facility_net_action_from_state(
                self.state,
                env_config,
                settings=settings,
            )
            for _ in range(250):
                actual = facility_net_action_from_state(
                    self.state,
                    env_config,
                    settings=settings,
                )
                np.testing.assert_array_equal(actual, expected)

        self.assertEqual(geographic_edges.call_count, 1)
        cache_info = heuristics._cached_facility_edge_sets.cache_info()
        self.assertEqual(cache_info.misses, 1)
        self.assertGreaterEqual(cache_info.hits, 250)
        heuristics._cached_facility_edge_sets.cache_clear()

    def test_balance_shortage_surplus_orders_ties_by_facility_index(self) -> None:
        net = _balance_shortage_surplus(
            shortage=np.array([0.0, 1.0, 1.0]),
            surplus=np.array([1.0, 0.0, 0.0]),
            edges=((0, 1), (0, 2)),
            max_abs=1.0,
        )

        np.testing.assert_allclose(net, np.array([-1.0, 1.0, 0.0]))

    def test_balance_shortage_surplus_rejects_invalid_arrays(self) -> None:
        cases = (
            (
                np.array([[1.0, 0.0]]),
                np.array([[0.0, 1.0]]),
                1.0,
            ),
            (np.array([1.0, 0.0]), np.array([0.0]), 1.0),
            (np.array([np.nan, 0.0]), np.array([0.0, 1.0]), 1.0),
            (np.array([1.0, 0.0]), np.array([0.0, np.inf]), 1.0),
            (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.inf),
        )
        for shortage, surplus, max_abs in cases:
            with self.subTest(shortage=shortage, surplus=surplus, max_abs=max_abs):
                with self.assertRaises(ValueError):
                    _balance_shortage_surplus(
                        shortage=shortage,
                        surplus=surplus,
                        edges=((0, 1),),
                        max_abs=max_abs,
                    )

    def test_balance_shortage_surplus_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(61000000)
        edges = tuple((i, j) for i in range(8) for j in range(i + 1, 8))
        for _ in range(250):
            shortage = rng.uniform(0.0, 20.0, size=8)
            surplus = rng.uniform(0.0, 20.0, size=8)
            expected = self._numpy_balance_reference(
                shortage=shortage,
                surplus=surplus,
                edges=edges,
                max_abs=5.0,
            )

            actual = _balance_shortage_surplus(
                shortage=shortage,
                surplus=surplus,
                edges=edges,
                max_abs=5.0,
            )

            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_balance_shortage_surplus_sustains_repeated_calls(self) -> None:
        shortage = np.linspace(0.0, 20.0, num=20)
        surplus = shortage[::-1].copy()
        edges = tuple((i, (i + 1) % 20) for i in range(20))

        for _ in range(20_000):
            net = _balance_shortage_surplus(
                shortage=shortage,
                surplus=surplus,
                edges=edges,
                max_abs=5.0,
            )

        self.assertTrue(np.all(np.isfinite(net)))
        self.assertLessEqual(float(np.max(np.abs(net))), 5.0)

    @staticmethod
    def _numpy_balance_reference(
        *,
        shortage: np.ndarray,
        surplus: np.ndarray,
        edges: tuple[tuple[int, int], ...],
        max_abs: float,
    ) -> np.ndarray:
        net = np.zeros_like(shortage, dtype=float)
        shortage_remaining = np.asarray(shortage, dtype=float).copy()
        surplus_remaining = np.asarray(surplus, dtype=float).copy()
        adjacency: dict[int, set[int]] = {}
        for i, j in edges:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)

        for receiver in np.argsort(-shortage_remaining):
            if shortage_remaining[receiver] <= 1e-8:
                continue
            donors = sorted(
                adjacency.get(int(receiver), ()),
                key=lambda node: surplus_remaining[node],
                reverse=True,
            )
            for donor in donors:
                if shortage_remaining[receiver] <= 1e-8:
                    break
                if surplus_remaining[donor] <= 1e-8:
                    continue
                flow = min(
                    shortage_remaining[receiver],
                    surplus_remaining[donor],
                    max_abs,
                )
                if flow <= 1e-8:
                    continue
                net[receiver] += flow
                net[donor] -= flow
                shortage_remaining[receiver] -= flow
                surplus_remaining[donor] -= flow

        return np.clip(net, -max_abs, max_abs)

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

    def test_forecast_mdl2_is_registered_and_matches_two_period_fmyo(self) -> None:
        config = replace(
            make_20_clinic_config(episode_horizon=2),
            include_demand_forecast_state=True,
            demand_forecast_horizon=2,
            demand_forecast_error=0.0,
        )
        env = CapacityPlanningEnv(config, seed=17)
        state = env.reset(seed=17)

        forecast_mdl2 = ForecastMeanDemandLookahead2Policy().select_action(
            state,
            env=env,
        )
        forecast_myo = ForecastMyopicPolicy().select_action(state, env=env)
        state_action = facility_net_action_from_state(
            state,
            asdict(env.config),
            settings=heuristic_settings_for_policy("fmdl2"),
        )

        self.assertIn("fmdl2", available_heuristics())
        np.testing.assert_allclose(forecast_mdl2, forecast_myo, atol=1e-6)
        np.testing.assert_allclose(forecast_mdl2, state_action, atol=1e-6)

    def test_rolling_mdl2_matches_state_helper(self) -> None:
        env_config = load_config(
            "experiments/configs/20_clinic_patient_condition_geo_demand_drift.json"
        )
        env_config["demand_history_window"] = 12
        config = load_config("configs/gcn_residual_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=18)
        state = env.reset(seed=18)
        anchor = MeanDemandLookahead2Policy()
        for _ in range(5):
            state, _reward, done, _info = env.step(
                anchor.select_action(state, env=env)
            )
            if done:
                break

        live_action = RollingMeanDemandLookahead2Policy().select_action(
            state,
            env=env,
        )
        state_action = facility_net_action_from_state(
            state,
            env_config,
            settings=heuristic_settings_for_policy("rmdl2"),
        )

        self.assertIn("rmdl2", available_heuristics())
        np.testing.assert_allclose(live_action, state_action, atol=1e-6)
        self.assertAlmostEqual(
            heuristic_settings_for_policy("rmdl2").demand_history_weight,
            0.05,
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

    def test_shield_rollout_seed_is_independent_of_live_rng_state(self) -> None:
        first = copy.deepcopy(self.env)
        second = copy.deepcopy(self.env)
        second.rng.random(20)
        anchor = MeanDemandLookahead2Policy()
        action = anchor.select_action(self.state, env=self.env)

        first_metrics = shield_rollout_metrics(
            first,
            anchor,
            action,
            horizon=2,
            rollout_seed=12345,
        )
        second_metrics = shield_rollout_metrics(
            second,
            anchor,
            action,
            horizon=2,
            rollout_seed=12345,
        )

        self.assertEqual(first_metrics, second_metrics)

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
