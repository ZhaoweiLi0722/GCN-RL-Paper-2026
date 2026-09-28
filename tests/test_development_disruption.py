"""Mechanics and information checks, not research-performance fixtures."""

import copy
import io
import json
import unittest
from dataclasses import replace

import numpy as np

from evaluation.disruption_feasibility_pilot import run_episode, validate_spec
from evaluation.information_matched_rollout_planner import make_matched_clone
from src.baselines.heuristics import get_heuristic_class
from src.env.capacity_planning import CapacityPlanningConfig
from src.env.development_disruption import DevelopmentDisruptionEnv, DisruptionProfile
from src.env.patient_capacity_planning import PatientConditionCapacityEnv, PatientEnvConfig
from src.env.patient_condition import PatientConditionConfig


def config(**overrides):
    params = dict(num_facilities=2, production_lead_time=3, episode_horizon=10,
                  demand_rates=(1., 1.), initial_specimens=(0., 0.),
                  initial_idle_bioreactors=(6., 6.), initial_reagents=(30., 30.),
                  max_idle_bioreactors=(30., 30.), max_reagents=(100., 100.),
                  max_specimens=(100., 100.), max_reagent_replenishment=(10., 10.),
                  action_mode="facility_net", enable_stochastic_procurement=True,
                  reagent_lead_time_probabilities=(.8, .2, 0., 0.), include_on_order_state=True)
    params.update(overrides)
    return PatientEnvConfig(base=CapacityPlanningConfig(**params))


def profile(fractions=(0., 0.), probabilities=(.8, .2, 0., 0.)):
    return DisruptionProfile(2, fractions, probabilities)


class DisruptionTests(unittest.TestCase):
    def test_legacy_idle_limit_can_drop_arriving_capacity(self):
        for limit, expected_stock in ((6., 9.), (12., 12.)):
            env = PatientConditionCapacityEnv(config(transfer_lead_time=1,
                max_idle_bioreactors=(limit, limit)), seed=1)
            env.bioreactors[1, 0] -= 3.
            env.capacity_transfer_pipeline[0, 0] = 3.
            self.assertEqual(env.bioreactors.sum() + env.capacity_transfer_pipeline.sum(), 12.)
            env._receive_transfer_arrivals()
            self.assertEqual(env.bioreactors.sum() + env.capacity_transfer_pipeline.sum(), expected_stock)

    def test_no_change_is_bit_identical_to_parent(self):
        a = PatientConditionCapacityEnv(config(), seed=1)
        b = DevelopmentDisruptionEnv(config(), profile(), seed=1)
        policy = get_heuristic_class("mdl2")()
        for _ in range(10):
            np.testing.assert_array_equal(a.observation(), b.observation())
            action = policy.select_action(a.observation(), env=a)
            oa, ra, da, ia = a.step(action)
            ob, rb, db, ib = b.step(action)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual((ra, da), (rb, db))
            self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)
            self.assertEqual(ia["cost"], ib["cost"])

    def test_outage_preserves_busy_stock_and_holding_cost(self):
        cfg = config(demand_rates=(0., 0.), initial_specimens=(2., 0.))
        env = DevelopmentDisruptionEnv(cfg, profile((.5, 0)), seed=1)
        env.step(env.noop_action())
        before_busy = env.bioreactors[:, 1:].sum()
        self.assertGreater(before_busy, 0)
        env.step(env.noop_action())
        self.assertEqual(env.t, 2)
        self.assertEqual(env.blocked_idle.tolist(), [3., 0.])
        self.assertEqual(env.bioreactors.sum() + env.blocked_idle.sum(), 12.)
        _, reward, _, info = env.step(env.noop_action())
        self.assertEqual(info["blocked_capacity_holding_cost"], 3 * cfg.base.costs.bioreactor_holding)
        self.assertEqual(-reward, info["cost"])
        self.assertEqual(env.bioreactors.sum() + env.blocked_idle.sum(), 12.)
        env.reset(seed=1)
        self.assertEqual(float(env.blocked_idle.sum()), 0.)

    def test_restore_after_outage_and_with_pipeline(self):
        p = profile((.5, 0), (0., 0., 0., 1.))
        a = DevelopmentDisruptionEnv(config(), p, seed=5)
        for _ in range(4):
            a.step(np.zeros(a.action_size))
        state = a.state_dict()
        b = DevelopmentDisruptionEnv(config(), p, seed=999)
        b.load_state_dict(state)
        for _ in range(3):
            oa, ra, _, ia = a.step(np.zeros(a.action_size))
            ob, rb, _, ib = b.step(np.zeros(b.action_size))
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual(ra, rb)
            np.testing.assert_array_equal(ia["procurement_leads"], ib["procurement_leads"])
            self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)
        wrong = DevelopmentDisruptionEnv(config(), profile(), seed=5)
        with self.assertRaises(ValueError):
            wrong.load_state_dict(state)

    def test_changed_delay_applies_only_to_new_orders(self):
        cfg = config(demand_rates=(0., 0.), reagent_lead_time_probabilities=(0., 0., 0., 1.))
        env = DevelopmentDisruptionEnv(cfg, profile(probabilities=(1., 0., 0., 0.)), seed=1)
        order = env.noop_action()
        order[6:] = 1.
        env.step(order)
        _, _, _, info = env.step(env.noop_action())
        self.assertEqual(info["procurement_arrivals"].sum(), 0.)
        _, _, _, info = env.step(env.noop_action())
        self.assertEqual(info["procurement_leads"].tolist(), [0, 0])
        self.assertEqual(info["procurement_arrivals"].sum(), 0.)
        _, _, _, info = env.step(env.noop_action())
        self.assertEqual(info["procurement_arrivals"].sum(), 20.)

    def test_crn_pairs_despite_profile_and_action_differences(self):
        a = DevelopmentDisruptionEnv(config(), profile(), seed=11)
        b = DevelopmentDisruptionEnv(config(), profile((.5, 0), (.1, .2, .4, .3)), seed=11)
        for _ in range(10):
            np.testing.assert_array_equal(a.demand, b.demand)
            a.step(a.noop_action())
            b.step(np.zeros(b.action_size))
            self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)

    def test_future_profile_does_not_change_current_observation_or_rule(self):
        a = DevelopmentDisruptionEnv(config(), profile(), seed=11)
        b = DevelopmentDisruptionEnv(config(), profile((.5, 0), (.1, .2, .4, .3)), seed=11)
        p = get_heuristic_class("mdl2_lt")()
        np.testing.assert_array_equal(a.observation(), b.observation())
        np.testing.assert_array_equal(p.select_action(a.observation(), env=a), p.select_action(b.observation(), env=b))

    def test_invalid_profiles_and_seed_overlap_rejected(self):
        for p in (profile((float("nan"), 0)), profile(probabilities=(1, 0)), profile(probabilities=(0, 0, 0, 0))):
            with self.assertRaises(ValueError):
                DevelopmentDisruptionEnv(config(), p)
        with self.assertRaises(ValueError):
            validate_spec({"discovery_seeds": [1], "validation_seeds": [1]})

    def test_episode_log_reconciles_and_is_finite(self):
        log = io.StringIO()
        key = dict(cell="test", split="test", world_seed=123, policy="mdl2")
        row = run_episode(DevelopmentDisruptionEnv(config(), profile((.5, 0)), seed=123),
                          {"algorithm": "mdl2", "settings": {}}, key, log)
        steps = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(len(steps), 10)
        self.assertAlmostEqual(sum(s["cost"] for s in steps), row["total_cost"])
        self.assertEqual(row["maximum_capacity_error"], 0.)


class PlannerInformationContractTest(unittest.TestCase):
    def test_equal_aggregate_observation_does_not_fix_sampler_inputs(self):
        cfg = replace(config(initial_specimens=(2., 0.)),
                      patient=PatientConditionConfig(healthy_decay_rate=0., frail_decay_rate=0.))
        a = PatientConditionCapacityEnv(cfg, seed=3)
        b = copy.deepcopy(a)
        for patient in b.patient_registry.values():
            patient.age = 1  # Below the near-expiry threshold; same survival.
        np.testing.assert_array_equal(a.observation(), b.observation())
        np.testing.assert_array_equal(a.graph_observation()["node_features"], b.graph_observation()["node_features"])

        class Recorder:
            def __init__(self):
                self.inputs = []

            def draw(self, age, survival, risk, rng):
                self.inputs.append((age, survival, risk))
                return 0., 10., risk, 1.

        x, y = Recorder(), Recorder()
        make_matched_clone(a, 123, x, privileged=False)
        make_matched_clone(b, 123, y, privileged=False)
        self.assertNotEqual(x.inputs, y.inputs)


if __name__ == "__main__":
    unittest.main()
