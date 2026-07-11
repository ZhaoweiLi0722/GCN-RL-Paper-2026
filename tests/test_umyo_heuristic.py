"""Tests for the patient-aware uMYO heuristic (Phase 5, group 2).

Note: uMYO empirically ties condition-blind MYO on eligibility (urgency
correlates with the waiting counts MYO already balances on). That near-tie is a
recorded finding; the assertion here is uMYO >= MYO (never worse), not a strict
improvement. See specs/2026-07-11-benchmark-algorithms/requirements.md.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.heuristics import available_heuristics, get_heuristic_class
from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import PatientConditionCapacityEnv, PatientEnvConfig


def _constrained_base() -> CapacityPlanningConfig:
    # Asymmetric demand + limited shared capacity so triage could matter.
    return CapacityPlanningConfig(
        num_facilities=2, production_lead_time=2, episode_horizon=26,
        demand_rates=(8.0, 1.0), initial_reagents=(20.0, 20.0),
        initial_idle_bioreactors=(6.0, 6.0), max_reagents=(200.0, 200.0),
        max_idle_bioreactors=(20.0, 20.0), max_reagent_replenishment=(6.0, 6.0),
        action_mode="facility_net", include_supplier_state=True, supplier_disruption_rate=0.1,
    )


def _eligibility(name: str, seed: int) -> float:
    env = PatientConditionCapacityEnv(PatientEnvConfig(base=_constrained_base()), seed=seed)
    state = env.reset(seed=seed)
    agent = get_heuristic_class(name)(env.observation_size, env.action_size, {})
    info = {}
    for _ in range(env.config.episode_horizon):
        state, _r, done, info = env.step(agent.select_action(state, env=env))
        if done:
            break
    return float(info["eligibility_rate"])


class UMyoTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIn("umyo", available_heuristics())

    def test_emits_valid_action(self) -> None:
        env = PatientConditionCapacityEnv(PatientEnvConfig(base=_constrained_base()), seed=0)
        state = env.reset(seed=0)
        action = get_heuristic_class("umyo")(env.observation_size, env.action_size, {}).select_action(
            state, env=env
        )
        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_deterministic_under_seed(self) -> None:
        self.assertEqual(_eligibility("umyo", 5), _eligibility("umyo", 5))

    def test_falls_back_to_myopic_on_base_env(self) -> None:
        from src.env.capacity_planning import CapacityPlanningEnv

        env = CapacityPlanningEnv(_constrained_base(), seed=0)  # base env: no patient signal
        state = env.reset(seed=0)
        umyo = get_heuristic_class("umyo")(env.observation_size, env.action_size, {}).select_action(
            state, env=env
        )
        myo = get_heuristic_class("myo")(env.observation_size, env.action_size, {}).select_action(
            state, env=env
        )
        np.testing.assert_array_equal(umyo, myo)

    def test_at_least_as_good_as_myo(self) -> None:
        # uMYO ties MYO (recorded finding); assert it is never worse across seeds.
        for seed in range(5):
            self.assertGreaterEqual(
                _eligibility("umyo", seed) + 1e-9, _eligibility("myo", seed),
                f"uMYO worse than MYO at seed {seed}",
            )


if __name__ == "__main__":
    unittest.main()
