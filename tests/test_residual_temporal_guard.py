import unittest

import numpy as np

from src.rl.residual_temporal_guard import ResidualTemporalGuard


class ResidualTemporalGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 4
        self.env_config = {
            "num_facilities": self.n,
            "production_lead_time": 3,
            "include_supplier_state": True,
            "include_demand_forecast_state": True,
            "include_transfer_pipeline_state": True,
            "include_demand_history_state": True,
        }
        self.features_per_facility = 14

    def state_with_pending_reagent(
        self,
        facility: int,
        amount: float,
    ) -> np.ndarray:
        state = np.zeros(
            self.n * self.features_per_facility,
            dtype=np.float32,
        )
        rows = state.reshape(self.n, self.features_per_facility)
        pipeline_start = 3 + 3 + 1 + 1
        rows[facility, pipeline_start + 1] = float(amount)
        return state

    def action_with_reagent_flow(
        self,
        values,
    ) -> np.ndarray:
        action = np.zeros(4 * self.n, dtype=np.float32)
        action[self.n : 2 * self.n] = np.asarray(
            values,
            dtype=np.float32,
        )
        return action

    def test_disabled_guard_is_identity(self) -> None:
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=self.env_config,
            settings={"enabled": False},
        )
        action = self.action_with_reagent_flow((-0.5, -0.5, 0.5, 0.5))
        governed = guard.apply(action, np.zeros(self.n * 14))
        np.testing.assert_allclose(governed, action)

    def test_cooldown_suppresses_one_following_decision(self) -> None:
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=self.env_config,
            settings={
                "enabled": True,
                "cooldown_steps": {"reagent_transfer": 1},
            },
        )
        state = np.zeros(self.n * self.features_per_facility)
        action = self.action_with_reagent_flow((-0.5, -0.5, 0.5, 0.5))

        first = guard.apply(action, state)
        second = guard.apply(action, state)
        third = guard.apply(action, state)

        np.testing.assert_allclose(first, action)
        np.testing.assert_allclose(second, 0.0)
        np.testing.assert_allclose(third, action)

    def test_pipeline_guard_blocks_pending_receiver_and_rebalances(self) -> None:
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=self.env_config,
            settings={
                "enabled": True,
                "pipeline_guard": {
                    "enabled": True,
                    "groups": ["reagent_transfer"],
                    "pending_thresholds": {
                        "reagent_transfer": 1.0,
                    },
                },
            },
        )
        state = self.state_with_pending_reagent(2, 5.0)
        action = self.action_with_reagent_flow((-0.5, -0.5, 0.5, 0.5))

        governed = guard.apply(action, state)
        reagent = governed[self.n : 2 * self.n]

        self.assertAlmostEqual(float(reagent.sum()), 0.0, places=6)
        self.assertAlmostEqual(float(reagent[2]), 0.0, places=6)
        np.testing.assert_allclose(
            reagent,
            np.asarray((-0.25, -0.25, 0.0, 0.5)),
            atol=1e-6,
        )

    def test_step_and_episode_budgets_scale_without_breaking_balance(self) -> None:
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=self.env_config,
            settings={
                "enabled": True,
                "per_step_l1_limits": {
                    "reagent_transfer": 1.0,
                },
                "episode_l1_budgets": {
                    "reagent_transfer": 1.5,
                },
            },
        )
        state = np.zeros(self.n * self.features_per_facility)
        action = self.action_with_reagent_flow((-0.5, -0.5, 0.5, 0.5))

        first = guard.apply(action, state)[self.n : 2 * self.n]
        second = guard.apply(action, state)[self.n : 2 * self.n]
        third = guard.apply(action, state)[self.n : 2 * self.n]

        self.assertAlmostEqual(float(np.abs(first).sum()), 1.0, places=6)
        self.assertAlmostEqual(float(np.abs(second).sum()), 0.5, places=6)
        self.assertAlmostEqual(float(np.abs(third).sum()), 0.0, places=6)
        self.assertAlmostEqual(float(first.sum()), 0.0, places=6)
        self.assertAlmostEqual(float(second.sum()), 0.0, places=6)

    def test_snapshot_restore_isolates_counterfactual_guard_state(self) -> None:
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=self.env_config,
            settings={
                "enabled": True,
                "cooldown_steps": {"reagent_transfer": 2},
                "episode_l1_budgets": {
                    "reagent_transfer": 3.0,
                },
            },
        )
        state = np.zeros(self.n * self.features_per_facility)
        action = self.action_with_reagent_flow(
            (-0.5, -0.5, 0.5, 0.5)
        )
        guard.apply(action, state)
        snapshot = guard.snapshot()
        expected = guard.summary()

        guard.apply(action, state)
        guard.apply(np.zeros_like(action), state)
        guard.restore(snapshot)

        self.assertEqual(guard.summary(), expected)
        snapshot["cooldowns"]["reagent_transfer"][0] = 99
        self.assertNotEqual(
            int(guard.cooldowns["reagent_transfer"][0]),
            99,
        )

    def test_pipeline_guard_requires_pipeline_observation(self) -> None:
        config = dict(self.env_config)
        config["include_transfer_pipeline_state"] = False
        with self.assertRaisesRegex(
            ValueError,
            "include_transfer_pipeline_state",
        ):
            ResidualTemporalGuard(
                num_facilities=self.n,
                action_dim=4 * self.n,
                env_config=config,
                settings={
                    "enabled": True,
                    "pipeline_guard": {"enabled": True},
                },
            )

    def test_observable_shift_gate_uses_history_without_scenario_label(self) -> None:
        config = dict(self.env_config)
        config.update(
            {
                "include_time_state": True,
                "demand_rate_estimates": [10.0] * self.n,
            }
        )
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=config,
            settings={
                "enabled": True,
                "observable_shift_gate": {
                    "enabled": True,
                    "groups": ["reagent_transfer"],
                    "absolute_rate_deviation_threshold": 0.25,
                    "absolute_trend_ratio_threshold": 0.10,
                    "min_facilities": 3,
                    "min_time_fraction": 0.20,
                },
            },
        )
        action = self.action_with_reagent_flow((-0.5, -0.5, 0.5, 0.5))
        state = np.zeros(
            self.n * self.features_per_facility + 1,
            dtype=np.float32,
        )
        rows = state[:-1].reshape(
            self.n,
            self.features_per_facility,
        )
        history_start = 3 + 3 + 1 + 1 + 3
        rows[:, history_start] = 10.0
        state[-1] = 0.3

        inactive = guard.apply(action, state)
        np.testing.assert_allclose(inactive, 0.0)
        self.assertFalse(
            guard.last_info["observable_shift"]["active"]
        )

        guard.reset()
        rows[:3, history_start] = 14.0
        active = guard.apply(action, state)
        np.testing.assert_allclose(active, action)
        self.assertTrue(
            guard.last_info["observable_shift"]["active"]
        )

    def test_observable_shift_gate_requires_history(self) -> None:
        config = dict(self.env_config)
        config["include_demand_history_state"] = False
        with self.assertRaisesRegex(
            ValueError,
            "include_demand_history_state",
        ):
            ResidualTemporalGuard(
                num_facilities=self.n,
                action_dim=4 * self.n,
                env_config=config,
                settings={
                    "enabled": True,
                    "observable_shift_gate": {"enabled": True},
                },
            )

    def test_graph_poisson_shift_gate_detects_coherent_neighbors(self) -> None:
        config = dict(self.env_config)
        config.update(
            {
                "include_time_state": True,
                "demand_rate_estimates": [10.0] * self.n,
                "demand_history_window": 12,
                "episode_horizon": 52,
            }
        )
        guard = ResidualTemporalGuard(
            num_facilities=self.n,
            action_dim=4 * self.n,
            env_config=config,
            settings={
                "enabled": True,
                "observable_shift_gate": {
                    "enabled": True,
                    "groups": ["reagent_transfer"],
                    "detection_mode": "graph_poisson_z",
                    "z_score_threshold": 1.5,
                    "smoothing_steps": 1,
                    "min_facilities": 2,
                    "min_time_fraction": 0.15,
                },
            },
        )
        action = self.action_with_reagent_flow((-0.5, -0.5, 0.5, 0.5))
        state = np.zeros(
            self.n * self.features_per_facility + 1,
            dtype=np.float32,
        )
        rows = state[:-1].reshape(
            self.n,
            self.features_per_facility,
        )
        history_start = 3 + 3 + 1 + 1 + 3
        rows[:, history_start] = 10.0
        rows[:3, history_start] = 14.0
        state[-1] = 0.3

        governed = guard.apply(action, state)

        np.testing.assert_allclose(governed, action)
        self.assertTrue(
            guard.last_info["observable_shift"]["active"]
        )
        self.assertGreater(
            guard.last_info["observable_shift"][
                "max_abs_smoothed_z"
            ],
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
