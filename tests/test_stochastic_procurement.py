"""Stochastic procurement lead times with order crossing (spec 2026-08-29)."""

from __future__ import annotations

import copy
import dataclasses
import unittest

import numpy as np

from src.baselines.heuristics import get_heuristic_class
from src.env.capacity_planning import (
    CapacityPlanningConfig,
    CapacityPlanningEnv,
    make_20_clinic_config,
)

BASE = dataclasses.replace(make_20_clinic_config(), action_mode="facility_net")


def _run(config, seed=7, steps=40):
    env = CapacityPlanningEnv(config, seed=seed)
    env.reset(seed=seed)
    policy = get_heuristic_class("mdl2")()
    policy.reset()
    total, infos = 0.0, []
    for _ in range(steps):
        _, reward, done, info = env.step(policy.select_action(env.observation(), env=env))
        total += -reward
        infos.append(info)
        if done:
            break
    return env, total, infos


class FlagOffEquivalenceTest(unittest.TestCase):
    def test_disabled_is_bit_identical(self) -> None:
        env_a, cost_a, _ = _run(BASE)
        env_b, cost_b, _ = _run(dataclasses.replace(BASE, reagent_purchase_lead_time=0))
        self.assertEqual(cost_a, cost_b)
        self.assertEqual(env_a.rng.bit_generator.state, env_b.rng.bit_generator.state)
        np.testing.assert_array_equal(env_a.reagents, env_b.reagents)

    def test_disabled_exposes_no_procurement_info(self) -> None:
        _, _, infos = _run(BASE, steps=5)
        for info in infos:
            self.assertNotIn("procurement_arrivals", info)
            self.assertNotIn("reagents_on_order", info)


class DeterministicLeadTest(unittest.TestCase):
    def test_order_arrives_exactly_l_epochs_later(self) -> None:
        config = dataclasses.replace(BASE, reagent_purchase_lead_time=2)
        env = CapacityPlanningEnv(config, seed=5)
        env.reset(seed=5)
        n = env.config.num_facilities
        action = env.noop_action()
        action[3 * n : 4 * n] = 1.0  # order the maximum
        _, _, _, first = env.step(action)
        ordered = float(first["replenishment"].sum())
        self.assertGreater(ordered, 0.0)
        self.assertEqual(float(first["procurement_arrivals"].sum()), 0.0)
        quiet = env.noop_action()
        _, _, _, second = env.step(quiet)
        self.assertEqual(float(second["procurement_arrivals"].sum()), 0.0)
        _, _, _, third = env.step(quiet)
        self.assertAlmostEqual(float(third["procurement_arrivals"].sum()), ordered, places=6)

    def test_nothing_is_lost_in_the_pipeline(self) -> None:
        config = dataclasses.replace(BASE, reagent_purchase_lead_time=2)
        env, _, infos = _run(config, steps=30)
        ordered = sum(float(i["replenishment"].sum()) for i in infos)
        arrived = sum(float(i["procurement_arrivals"].sum()) for i in infos)
        outstanding = float(env.reagent_purchase_pipeline.sum())
        self.assertAlmostEqual(ordered, arrived + outstanding, places=6)


class StochasticLeadTest(unittest.TestCase):
    CONFIG = dataclasses.replace(
        BASE,
        enable_stochastic_procurement=True,
        reagent_lead_time_probabilities=(0.2, 0.3, 0.3, 0.2),
        include_on_order_state=True,
    )

    def test_order_crossing_is_possible(self) -> None:
        env = CapacityPlanningEnv(self.CONFIG, seed=3)
        env.reset(seed=3)
        draws = {int(d) for _ in range(200) for d in env._draw_procurement_leads()}
        # More than one attainable delay means a later order can overtake an earlier one.
        self.assertGreater(len(draws), 1)
        self.assertTrue(draws <= {0, 1, 2, 3})

    def test_pipeline_depth_and_feature_width(self) -> None:
        env = CapacityPlanningEnv(self.CONFIG, seed=3)
        off = CapacityPlanningEnv(BASE, seed=3)
        self.assertEqual(env.reagent_purchase_pipeline.shape[0], 4)
        self.assertEqual(env.features_per_facility, off.features_per_facility + 4)
        self.assertEqual(env.observation().shape[0], env.observation_size)

    def test_crn_pairing_survives_a_change_of_order_quantity(self) -> None:
        """The load-bearing property: lead draws must not depend on the action.

        Leads are drawn once per facility per epoch, never per order or per
        unit, so the random stream cannot depend on how much is ordered. If
        that ever regresses, every paired screen in this project silently
        loses its pairing.
        """

        env = CapacityPlanningEnv(self.CONFIG, seed=11)
        env.reset(seed=11)
        policy = get_heuristic_class("mdl2")()
        policy.reset()
        for _ in range(20):
            env.step(policy.select_action(env.observation(), env=env))
        snapshot = {
            "rng": copy.deepcopy(env.rng.bit_generator.state),
            "reagents": env.reagents.copy(),
            "specimens": env.specimens.copy(),
            "bioreactors": env.bioreactors.copy(),
            "pipeline": env.reagent_purchase_pipeline.copy(),
            "demand": env.demand.copy(),
        }

        def replay(delta):
            env.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
            env.reagents = snapshot["reagents"].copy()
            env.specimens = snapshot["specimens"].copy()
            env.bioreactors = snapshot["bioreactors"].copy()
            env.reagent_purchase_pipeline = snapshot["pipeline"].copy()
            env.demand = snapshot["demand"].copy()
            env.rng = np.random.default_rng(4242)
            n = env.config.num_facilities
            action = np.asarray(
                policy.select_action(env.observation(), env=env), dtype=np.float32
            ).copy()
            action[3 * n : 4 * n] = np.clip(action[3 * n : 4 * n] + delta, -1.0, 1.0)
            total = 0.0
            for _ in range(15):
                _, reward, _, _ = env.step(action)
                total += -reward
                action = np.asarray(
                    policy.select_action(env.observation(), env=env), dtype=np.float32
                )
            return total, env.rng.bit_generator.state

        cost_a, state_a = replay(0.0)
        cost_b, state_b = replay(0.5)
        self.assertEqual(state_a, state_b)
        self.assertNotEqual(cost_a, cost_b)


class ValidationTest(unittest.TestCase):
    def test_degenerate_distribution_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two lead-time"):
            CapacityPlanningEnv(
                dataclasses.replace(
                    BASE, enable_stochastic_procurement=True,
                    reagent_lead_time_probabilities=(1.0,),
                ),
                seed=0,
            )

    def test_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            CapacityPlanningEnv(
                dataclasses.replace(
                    BASE, enable_stochastic_procurement=True,
                    reagent_lead_time_probabilities=(0.5, 0.2),
                ),
                seed=0,
            )

    def test_probabilities_require_the_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires enable_stochastic_procurement"):
            CapacityPlanningEnv(
                dataclasses.replace(BASE, reagent_lead_time_probabilities=(0.5, 0.5)),
                seed=0,
            )

    def test_negative_lead_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            CapacityPlanningEnv(
                dataclasses.replace(BASE, reagent_purchase_lead_time=-1), seed=0
            )

    def test_on_order_state_requires_a_lead(self) -> None:
        with self.assertRaisesRegex(ValueError, "include_on_order_state requires"):
            CapacityPlanningEnv(
                dataclasses.replace(BASE, include_on_order_state=True), seed=0
            )


if __name__ == "__main__":
    unittest.main()
