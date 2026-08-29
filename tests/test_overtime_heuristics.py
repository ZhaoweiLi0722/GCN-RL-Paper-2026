"""Overtime heuristic comparators (spec 2026-08-29): MDL-2-OT, uMYO-OT, static-OT.

Includes the E2-readiness sanity gate: in a stressed capacity-bound fixture, at
least one overtime heuristic must strictly beat plain MDL-2 on paired episodes
— if no heuristic can use the channel, no screen may claim an agent could.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from src.baselines.heuristics import (
    HEURISTIC_POLICIES,
    MeanDemandLookahead2Policy,
    OvertimeMeanDemandLookahead2Policy,
    OvertimeUrgencyAwareMyopicPolicy,
    StaticOvertimePolicy,
)
from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv


def _stressed_config(**overrides) -> CapacityPlanningConfig:
    """Capacity-bound fixture: many waiting specimens, ample reagents, few reactors."""

    defaults = dict(
        num_facilities=2,
        action_mode="facility_net",
        enable_overtime_control=True,
        initial_specimens=(60.0, 60.0),
        max_specimens=(200.0, 200.0),
        initial_reagents=(200.0, 200.0),
        max_reagents=(400.0, 400.0),
        initial_idle_bioreactors=(4.0, 4.0),
        max_idle_bioreactors=(20.0, 20.0),
        max_overtime_fraction=0.5,
        demand_rates=(2.0, 2.0),
        episode_horizon=30,
    )
    defaults.update(overrides)
    return CapacityPlanningConfig(**defaults)


def _run_episode(policy, config: CapacityPlanningConfig, seed: int) -> float:
    env = CapacityPlanningEnv(config, seed=seed)
    policy.reset()
    total_cost = 0.0
    done = False
    while not done:
        action = policy.select_action(env.observation(), env=env)
        _, reward, done, _ = env.step(action)
        total_cost += -reward
    return total_cost


class OvertimeHeuristicsTest(unittest.TestCase):
    def test_registry(self) -> None:
        self.assertIs(HEURISTIC_POLICIES["mdl2_ot"], OvertimeMeanDemandLookahead2Policy)
        self.assertIs(HEURISTIC_POLICIES["umyo_ot"], OvertimeUrgencyAwareMyopicPolicy)
        self.assertIs(HEURISTIC_POLICIES["static_ot"], StaticOvertimePolicy)

    def test_mdl2_ot_reduces_to_mdl2_at_zero_fraction(self) -> None:
        config = _stressed_config(max_overtime_fraction=0.0)
        env = CapacityPlanningEnv(config, seed=0)
        base = MeanDemandLookahead2Policy().select_action(env.observation(), env=env)
        overtime = OvertimeMeanDemandLookahead2Policy().select_action(
            env.observation(), env=env
        )
        np.testing.assert_array_equal(base, overtime)

    def test_static_ot_zero_equals_mdl2(self) -> None:
        config = _stressed_config()
        env = CapacityPlanningEnv(config, seed=0)
        base = MeanDemandLookahead2Policy().select_action(env.observation(), env=env)
        static = StaticOvertimePolicy(
            config={"static_overtime_fraction": 0.0}
        ).select_action(env.observation(), env=env)
        np.testing.assert_array_equal(base, static)
        with self.assertRaises(ValueError):
            StaticOvertimePolicy(config={"static_overtime_fraction": 1.5})

    def test_mdl2_ot_surges_only_with_usable_shortfall(self) -> None:
        n = 2
        # Capacity binds and reagents suffice -> surge.
        env = CapacityPlanningEnv(_stressed_config(), seed=0)
        block = OvertimeMeanDemandLookahead2Policy()._overtime_action_block(env)
        self.assertTrue(np.all(block > -1.0))
        # No reagents -> the reagent-sufficiency condition suppresses the surge.
        env = CapacityPlanningEnv(
            _stressed_config(initial_reagents=(0.0, 0.0)), seed=0
        )
        block = OvertimeMeanDemandLookahead2Policy()._overtime_action_block(env)
        np.testing.assert_array_equal(block, np.full(n, -1.0, dtype=np.float32))
        # Idle capacity exceeds waiting -> no shortfall, no surge.
        env = CapacityPlanningEnv(
            _stressed_config(
                initial_specimens=(2.0, 2.0), initial_idle_bioreactors=(10.0, 10.0)
            ),
            seed=0,
        )
        block = OvertimeMeanDemandLookahead2Policy()._overtime_action_block(env)
        np.testing.assert_array_equal(block, np.full(n, -1.0, dtype=np.float32))

    def test_umyo_ot_needs_patient_signal(self) -> None:
        # On the base environment there is no urgency signal: no surge.
        env = CapacityPlanningEnv(_stressed_config(), seed=0)
        block = OvertimeUrgencyAwareMyopicPolicy()._overtime_action_block(env)
        np.testing.assert_array_equal(block, np.full(2, -1.0, dtype=np.float32))

    def test_sanity_gate_overtime_beats_plain_mdl2_when_capacity_bound(self) -> None:
        config = _stressed_config()
        for seed in (0, 1, 2):
            cost_plain = _run_episode(MeanDemandLookahead2Policy(), config, seed)
            cost_ot = _run_episode(OvertimeMeanDemandLookahead2Policy(), config, seed)
            self.assertLess(
                cost_ot,
                cost_plain,
                f"MDL-2-OT failed to beat MDL-2 in the capacity-bound fixture (seed {seed})",
            )


if __name__ == "__main__":
    unittest.main()
