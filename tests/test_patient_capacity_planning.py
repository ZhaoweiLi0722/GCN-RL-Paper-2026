"""Tests for the patient-condition environment layer (Phase 4, task group 3)."""

from __future__ import annotations

import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)


def _small_base(**overrides) -> CapacityPlanningConfig:
    params = dict(
        num_facilities=2,
        production_lead_time=2,
        episode_horizon=20,
        demand_rates=(3.0, 3.0),
        initial_specimens=(0.0, 0.0),
        initial_reagents=(100.0, 100.0),
        initial_idle_bioreactors=(10.0, 10.0),
        max_specimens=(100.0, 100.0),
        max_reagents=(200.0, 200.0),
        max_idle_bioreactors=(20.0, 20.0),
        max_reagent_replenishment=(100.0, 100.0),
        action_mode="facility_net",
        include_supplier_state=True,
        supplier_disruption_rate=0.0,
    )
    params.update(overrides)
    return CapacityPlanningConfig(**params)


def _env(base=None, **cfg) -> PatientConditionCapacityEnv:
    return PatientConditionCapacityEnv(
        PatientEnvConfig(base=base or _small_base(), **cfg), seed=0
    )


class PatientEnvBasicsTests(unittest.TestCase):
    def test_builds_and_steps(self) -> None:
        env = _env()
        obs = env.reset(seed=0)
        self.assertEqual(obs.shape, (env.observation_size,))
        next_obs, reward, done, info = env.step(env.noop_action())
        self.assertEqual(next_obs.shape, (env.observation_size,))
        self.assertIsInstance(reward, float)
        for key in ("patients_lost", "material_wasted", "eligibility_rate", "waiting_patients"):
            self.assertIn(key, info)

    def test_observation_width_is_fixed(self) -> None:
        env = _env()
        env.reset(seed=1)
        widths = set()
        for _ in range(10):
            obs, *_ = env.step(env.noop_action())
            widths.add(obs.shape[0])
        self.assertEqual(widths, {env.observation_size})

    def test_default_20_clinic_env_builds_and_steps(self) -> None:
        env = PatientConditionCapacityEnv(seed=0)  # default 20-clinic config
        obs = env.reset(seed=0)
        self.assertEqual(obs.shape, (env.observation_size,))
        _, reward, _, info = env.step(env.noop_action())
        self.assertIsInstance(reward, float)
        self.assertEqual(info["waiting_patients"].shape, (20,))
        self.assertEqual(info["risk_type_counts"].shape[0], 20)

    def test_rejects_non_facility_net_action_mode(self) -> None:
        with self.assertRaises(ValueError):
            _env(base=_small_base(action_mode="edge_transfer"))


class PatientEnvDynamicsTests(unittest.TestCase):
    def test_zero_capacity_loses_patients(self) -> None:
        env = _env(
            base=_small_base(
                initial_reagents=(0.0, 0.0),
                initial_idle_bioreactors=(0.0, 0.0),
                max_reagent_replenishment=(0.0, 0.0),
            )
        )
        env.reset(seed=0)
        total_lost = 0.0
        for _ in range(env.config.episode_horizon):
            _, _, done, info = env.step(env.noop_action())
            total_lost += float(info["patients_lost"].sum())
            if done:
                break
        self.assertGreater(total_lost, 0.0)
        self.assertLess(info["eligibility_rate"], 1.0)

    def test_ample_capacity_serves_most_patients(self) -> None:
        env = _env()  # 10 idle bioreactors, 100 reagents, demand 3/clinic
        env.reset(seed=0)
        info = {}
        for _ in range(env.config.episode_horizon):
            _, _, done, info = env.step(env.noop_action())
            if done:
                break
        self.assertGreater(info["eligibility_rate"], 0.5)

    def test_deterministic_under_seed(self) -> None:
        env1, env2 = _env(), _env()
        env1.reset(seed=7)
        env2.reset(seed=7)
        action = env1.noop_action()
        for _ in range(15):
            _, r1, _, _ = env1.step(action)
            _, r2, _, _ = env2.step(action)
            self.assertEqual(r1, r2)

    def test_specimen_transfer_action_is_ignored(self) -> None:
        # Specimens are identity-bound (autologous): the specimen-transfer action
        # components (first n entries) must not affect dynamics.
        env_a, env_b = _env(), _env()
        env_a.reset(seed=3)
        env_b.reset(seed=3)
        n = env_a.config.num_facilities
        act_a = env_a.noop_action()
        act_b = act_a.copy()
        act_b[:n] = 1.0  # request large specimen transfers
        for _ in range(10):
            _, ra, _, _ = env_a.step(act_a)
            _, rb, _, _ = env_b.step(act_b)
            self.assertEqual(ra, rb)

    def test_patient_risk_counts_match_waiting_queue(self) -> None:
        from src.env.patient_condition import PatientConditionConfig

        env = _env(
            patient=PatientConditionConfig(
                risk_type_probabilities=(0.0, 1.0),
                risk_decay_multipliers=(1.0, 1.5),
            )
        )
        env.reset(seed=4)
        _obs, _reward, _done, info = env.step(env.noop_action())

        risk_counts = info["risk_type_counts"]
        self.assertEqual(risk_counts.shape, (2, 2))
        np.testing.assert_allclose(risk_counts[:, 0], np.zeros(2))
        np.testing.assert_allclose(risk_counts.sum(axis=1), info["waiting_patients"])


if __name__ == "__main__":
    unittest.main()
