"""Contract tests for continuous overtime control (spec 2026-08-29).

Covers the action/observation contract, cost-on-commitment convexity, the
borrowed-capacity fleet-conservation invariant, fatigue, dormant-throttle
rejection, and snapshot round-trips.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from src.baselines.heuristics import (
    HeuristicSettings,
    MeanDemandLookahead2Policy,
    facility_net_action_from_state,
)
from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)
from src.rl.preprocessing import facility_state_width


def _base_config(**overrides) -> CapacityPlanningConfig:
    defaults = dict(
        num_facilities=2,
        action_mode="facility_net",
        enable_overtime_control=True,
        initial_specimens=(50.0, 50.0),
        initial_reagents=(150.0, 150.0),
        initial_idle_bioreactors=(4.0, 4.0),
        max_idle_bioreactors=(20.0, 20.0),
        max_overtime_fraction=0.5,
        demand_rates=(1.0, 1.0),
        episode_horizon=30,
    )
    defaults.update(overrides)
    return CapacityPlanningConfig(**defaults)


def _overtime_action(env: CapacityPlanningEnv, u: float) -> np.ndarray:
    n = env.config.num_facilities
    action = env.noop_action()
    action[4 * n : 5 * n] = 2.0 * u - 1.0
    return action


class FacilityStateWidthMirrorTest(unittest.TestCase):
    """`facility_state_width` must mirror the env's per-facility layout.

    The from-state heuristic path guards with `size < base_width` and then
    reshapes, so a helper that *under*-reports the width passes the guard and
    silently misaligns every facility after the first. Lock the mirror across
    flag combinations so the overtime block cannot drift out of it again.
    """

    def _env_config_dict(self, config: CapacityPlanningConfig) -> dict:
        return {
            "num_facilities": config.num_facilities,
            "production_lead_time": config.production_lead_time,
            "include_supplier_state": config.include_supplier_state,
            "include_demand_forecast_state": config.include_demand_forecast_state,
            "include_transfer_pipeline_state": config.include_transfer_pipeline_state,
            "include_demand_history_state": config.include_demand_history_state,
            "include_demand_sequence_state": config.include_demand_sequence_state,
            "demand_sequence_length": config.demand_sequence_length,
            "enable_overtime_control": config.enable_overtime_control,
            "enable_overtime_fatigue": config.enable_overtime_fatigue,
        }

    def test_width_matches_env_across_flag_combinations(self) -> None:
        for overtime in (False, True):
            for fatigue in (False, True):
                if fatigue and not overtime:
                    continue
                for extras in ({}, {"include_supplier_state": True,
                                    "include_demand_forecast_state": True,
                                    "include_transfer_pipeline_state": True,
                                    "include_demand_history_state": True}):
                    with self.subTest(overtime=overtime, fatigue=fatigue, extras=bool(extras)):
                        config = _base_config(
                            enable_overtime_control=overtime,
                            enable_overtime_fatigue=fatigue,
                            **extras,
                        )
                        env = CapacityPlanningEnv(config, seed=0)
                        self.assertEqual(
                            facility_state_width(self._env_config_dict(config)),
                            env.features_per_facility,
                        )

    def test_from_state_anchor_matches_live_env_anchor(self) -> None:
        config = _base_config()
        env = CapacityPlanningEnv(config, seed=0)
        policy = MeanDemandLookahead2Policy()
        live = policy.select_action(env.observation(), env=env)
        from_state = facility_net_action_from_state(
            env.observation(),
            self._env_config_dict(config)
            | {
                "max_reagent_replenishment": list(env.max_reagent_replenishment),
                "max_specimen_transfer": config.max_specimen_transfer,
                "max_bioreactor_transfer": config.max_bioreactor_transfer,
                "max_reagent_transfer": config.max_reagent_transfer,
            },
            settings=policy.settings,
        )
        n = config.num_facilities
        # The from-state path reconstructs the 4n base blocks; the live policy
        # appends the overtime block on top of the same base action.
        np.testing.assert_allclose(live[: 4 * n], from_state[: 4 * n], atol=1e-6)


class OvertimeContractTest(unittest.TestCase):
    def test_action_and_observation_contract(self) -> None:
        env = CapacityPlanningEnv(_base_config(), seed=0)
        n = env.config.num_facilities
        self.assertEqual(env.action_size, 5 * n)
        off = CapacityPlanningEnv(
            _base_config(enable_overtime_control=False), seed=0
        )
        self.assertEqual(env.features_per_facility, off.features_per_facility + 3)
        self.assertEqual(env.observation().shape[0], env.observation_size)
        node_features = env.graph_observation()["node_features"]
        self.assertGreaterEqual(node_features.shape[1], 3)

    def test_noop_and_raw_mapping(self) -> None:
        env = CapacityPlanningEnv(_base_config(), seed=0)
        n = env.config.num_facilities
        noop = env.noop_action()
        fraction, surge = env._decode_overtime(noop)
        np.testing.assert_array_equal(fraction, np.zeros(n))
        np.testing.assert_array_equal(surge, np.zeros(n))
        full = np.zeros(env.action_size)
        full[4 * n :] = 1.0
        fraction, surge = env._decode_overtime(full)
        np.testing.assert_array_equal(fraction, np.ones(n))
        # base_capacity is initial_idle_bioreactors (4), NOT max (20).
        np.testing.assert_allclose(surge, 0.5 * np.array([4.0, 4.0]))

    def test_cost_on_commitment_and_convexity(self) -> None:
        config = _base_config(initial_specimens=(0.0, 0.0), demand_rates=(0.0, 0.0))
        previous_cost = None
        for u in np.linspace(0.0, 1.0, 11):
            env = CapacityPlanningEnv(config, seed=0)
            _, _, _, info = env.step(_overtime_action(env, float(u)))
            # No specimens -> zero production, yet the committed surge is paid.
            self.assertEqual(float(info["production"].sum()), 0.0)
            surge = info["overtime_surge"]
            expected = float(
                np.sum(
                    config.weight_overtime_linear * surge
                    + config.weight_overtime_quadratic * surge**2
                )
            )
            self.assertAlmostEqual(info["overtime_cost"], expected, places=6)
            if u > 0.0:
                self.assertGreater(info["overtime_cost"], previous_cost)
            previous_cost = info["overtime_cost"]

    def test_decomposition_additivity(self) -> None:
        env = CapacityPlanningEnv(_base_config(), seed=0)
        _, reward, _, info = env.step(_overtime_action(env, 1.0))
        component_keys = [
            "reagent_purchase_cost",
            "reagent_holding_cost",
            "reagent_shortage_cost",
            "bioreactor_holding_cost",
            "bioreactor_shortage_cost",
            "specimen_transfer_cost",
            "capacity_transfer_cost",
            "reagent_transfer_cost",
            "overtime_cost",
        ]
        self.assertAlmostEqual(
            info["cost"], sum(float(info[key]) for key in component_keys), places=6
        )
        self.assertEqual(reward, -info["cost"])

    def test_fleet_conservation_base_env(self) -> None:
        env = CapacityPlanningEnv(_base_config(), seed=0)
        fleet = float(env.bioreactors.sum())
        binding_seen = False
        for _ in range(env.config.episode_horizon):
            _, _, done, info = env.step(_overtime_action(env, 1.0))
            if float(info["overtime_production"].sum()) > 0.0:
                binding_seen = True
            current = float(env.bioreactors.sum()) - float(
                env.overtime_outstanding.sum()
            )
            self.assertAlmostEqual(current, fleet, places=6)
            self.assertTrue(np.all(env.overtime_outstanding >= 0.0))
            if done:
                break
        self.assertTrue(binding_seen, "fixture never made overtime bind")
        # Quiet tail: no further overtime lets the outstanding counter drain.
        for _ in range(env.config.production_lead_time + 1):
            env.step(_overtime_action(env, 0.0))
        self.assertAlmostEqual(float(env.overtime_outstanding.sum()), 0.0, places=6)

    def test_fleet_conservation_patient_env(self) -> None:
        config = PatientEnvConfig(
            base=_base_config(
                initial_specimens=(0.0, 0.0),
                demand_rates=(6.0, 6.0),
                episode_horizon=25,
            )
        )
        env = PatientConditionCapacityEnv(config, seed=1)
        env.reset(seed=1)
        fleet = float(env.bioreactors.sum())
        binding_seen = False
        done = False
        while not done:
            _, _, done, info = env.step(_overtime_action(env, 1.0))
            if float(info["overtime_production"].sum()) > 0.0:
                binding_seen = True
            current = float(env.bioreactors.sum()) - float(
                env.overtime_outstanding.sum()
            )
            self.assertAlmostEqual(current, fleet, places=6)
            self.assertTrue(np.all(env.overtime_outstanding >= 0.0))
        self.assertTrue(binding_seen, "patient fixture never made overtime bind")

    def test_surge_is_transient(self) -> None:
        # Full overtime with nothing to produce must not change capacity state.
        config = _base_config(initial_specimens=(0.0, 0.0), demand_rates=(0.0, 0.0))
        env = CapacityPlanningEnv(config, seed=0)
        idle_before = env.bioreactors[:, 0].copy()
        env.step(_overtime_action(env, 1.0))
        np.testing.assert_array_equal(env.bioreactors[:, 0], idle_before)
        np.testing.assert_array_equal(env.overtime_outstanding, np.zeros(2))

    def test_fatigue_recursion(self) -> None:
        config = _base_config(
            enable_overtime_fatigue=True,
            overtime_fatigue_decay=0.5,
            overtime_fatigue_cost_scale=1.0,
        )
        env = CapacityPlanningEnv(config, seed=0)
        env.step(_overtime_action(env, 1.0))
        np.testing.assert_allclose(env.overtime_fatigue, np.full(2, 1.0))
        env.step(_overtime_action(env, 1.0))
        np.testing.assert_allclose(env.overtime_fatigue, np.full(2, 1.5))
        # Fatigue raises the marginal linear cost on the next commitment.
        env_fresh = CapacityPlanningEnv(config, seed=0)
        _, _, _, info_fresh = env_fresh.step(_overtime_action(env_fresh, 1.0))
        _, _, _, info_tired = env.step(_overtime_action(env, 1.0))
        self.assertGreater(info_tired["overtime_cost"], info_fresh["overtime_cost"])

    def test_dormant_throttle_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_production_throttle"):
            CapacityPlanningEnv(_base_config(enable_production_throttle=True), seed=0)

    def test_overtime_requires_facility_net(self) -> None:
        with self.assertRaisesRegex(ValueError, "facility_net"):
            CapacityPlanningEnv(
                _base_config(action_mode="edge_transfer"), seed=0
            )

    def test_validation_bounds(self) -> None:
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(_base_config(max_overtime_fraction=1.5), seed=0)
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(_base_config(weight_overtime_linear=0.0), seed=0)
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(_base_config(weight_overtime_quadratic=-1.0), seed=0)
        with self.assertRaises(ValueError):
            CapacityPlanningEnv(
                _base_config(
                    enable_overtime_control=False, enable_overtime_fatigue=True
                ),
                seed=0,
            )

    def test_patient_snapshot_roundtrip(self) -> None:
        config = PatientEnvConfig(
            base=_base_config(demand_rates=(4.0, 4.0), episode_horizon=25)
        )
        env = PatientConditionCapacityEnv(config, seed=3)
        env.reset(seed=3)
        for _ in range(6):
            env.step(_overtime_action(env, 0.7))
        snapshot = env.state_dict()
        self.assertTrue(snapshot["enable_overtime_control"])
        for name in (
            "previous_overtime_fraction",
            "overtime_outstanding",
            "overtime_fatigue",
        ):
            self.assertIn(name, snapshot["arrays"])
        restored = PatientConditionCapacityEnv(config, seed=99)
        restored.reset(seed=99)
        restored.load_state_dict(snapshot)
        np.testing.assert_array_equal(
            restored.previous_overtime_fraction, env.previous_overtime_fraction
        )
        np.testing.assert_array_equal(
            restored.overtime_outstanding, env.overtime_outstanding
        )
        out_a = env.step(_overtime_action(env, 0.4))
        out_b = restored.step(_overtime_action(restored, 0.4))
        self.assertEqual(out_a[1], out_b[1])


if __name__ == "__main__":
    unittest.main()
