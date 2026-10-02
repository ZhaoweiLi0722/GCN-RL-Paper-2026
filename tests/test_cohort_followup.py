"""Invented clock/cost engine only; no patient environment or neural execution."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.env.cohort_followup import ClosedCohortClockMixin, CohortTailSpec
from src.rl.cohort_objective import cohort_reward_receipt
from src.rl.cohort_collection import CohortCollection
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays


class FakeEngine:
    def __init__(self):
        self.config = SimpleNamespace(episode_horizon=2, enable_overtime_control=False, num_facilities=1)
        self.action_size, self.t = 4, 0
        self.demand, self.demand_forecast = np.array([1.]), np.array([2.])
        self.demand_forecast_error = 0.1
        self.supplier_available = np.ones(1)
        self.patient_registry = {"initial": "waiting"}
        self.cumulative_enrolled = 1
        self.rng = np.random.default_rng(123)
        self.history, self.supply_ticks, self.demand_draws = [], 0, 0
        self.resolve_at = 3

    def observation(self):
        return np.array([self.t, self.demand[0], self.demand_forecast[0]])

    def _enroll_arrivals(self, demand):
        for _ in range(int(demand[0])):
            self.patient_registry[str(self.cumulative_enrolled)] = "waiting"
            self.cumulative_enrolled += 1

    def _advance_regional_supplier_disruptions(self):
        self.supply_ticks += 1

    def _sample_supplier_available(self):
        return np.array([float(self.rng.random() > .1)])

    def _record_demand_observation(self):
        self.history.append(float(self.demand[0]))

    def _advance_clock(self):
        self.t += 1
        self.demand_draws += 1
        self.demand = np.array([1. + 0. * self.rng.random()])
        self._advance_regional_supplier_disruptions()
        self.supplier_available = self._sample_supplier_available()
        self._record_demand_observation()
        return self.t >= self.config.episode_horizon

    def step(self, action):
        if self.t >= self.resolve_at:
            self.patient_registry = {key: "delivered" for key in self.patient_registry}
        self._enroll_arrivals(self.demand)
        done = self._advance_clock()
        return self.observation(), -10., done, {"cost": 10., "request": np.asarray(action).copy()}

    def assert_identity_conservation(self):
        assert len(self.patient_registry) == self.cumulative_enrolled
        return {"active_count": sum(x == "waiting" for x in self.patient_registry.values())}

    def state_dict(self):
        return {k: copy.deepcopy(v) for k, v in self.__dict__.items()
                if not k.startswith("_cohort") and k not in ("cohort_spec", "rng", "config")} | {
                    "config": vars(self.config).copy(),
                    "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        for key, value in state.items():
            if key == "config":
                self.config = SimpleNamespace(**value)
            elif key != "rng_state":
                setattr(self, key, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class FakeCohort(ClosedCohortClockMixin, FakeEngine):
    pass


def make():
    return FakeCohort(cohort_spec=CohortTailSpec(2, 2, 3), enabled=True)


IDLE = np.array([0., 0., 0., -1.])


class CohortFollowupTests(unittest.TestCase):
    def prefix(self):
        env = make()
        env.step(IDLE)
        env.step(IDLE)
        return env

    def test_invalid_spec_and_opt_in(self):
        for args in ((0, 2, 3), (2, True, 3), (2, 3, 2)):
            with self.assertRaises(ValueError):
                CohortTailSpec(*args)
        with self.assertRaises(ValueError):
            FakeCohort(cohort_spec=CohortTailSpec(2, 2, 3))

    def test_prefix_exact_actions_rng_cost_clock_and_last_arrival(self):
        plain, cohort = FakeEngine(), make()
        for _ in range(2):
            a, b = plain.step(IDLE), cohort.step(IDLE)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:3], b[1:3])
            self.assertEqual(plain.rng.bit_generator.state, cohort.rng.bit_generator.state)
        self.assertEqual(cohort.cumulative_enrolled, 3)
        self.assertEqual(cohort.config.episode_horizon, 2)
        with self.assertRaises(ValueError):
            cohort.step(IDLE)

    def test_close_once_preserves_patient_resources_rng_and_raw_prefix(self):
        env = self.prefix()
        state = env.state_dict()
        prefix = env.close_enrollment()
        self.assertEqual(prefix["patient_registry"], state["patient_registry"])
        self.assertEqual(prefix["rng_state"], state["rng_state"])
        self.assertEqual(env.rng.bit_generator.state, state["rng_state"])
        np.testing.assert_array_equal(prefix["demand"], [1.])
        np.testing.assert_array_equal(env.demand, [0.])
        np.testing.assert_array_equal(env.demand_forecast, [0.])
        with self.assertRaises(ValueError):
            env.close_enrollment()

    def test_no_new_demand_but_supply_clock_continues_to_fixed_endpoint(self):
        env = self.prefix()
        env.close_enrollment()
        debits, events = [], []
        for i in range(3):
            event = env.step_followup(IDLE, before_step=lambda: debits.append(1))
            events.append(event)
            self.assertEqual(event[2], i == 2)
            self.assertEqual(env.cumulative_enrolled, 3)
        self.assertEqual(env.demand_draws, 2)
        self.assertEqual(env.supply_ticks, 5)
        self.assertEqual(env.history, [1., 1., 0., 0., 0.])
        self.assertEqual(env._cohort_resolution_step, 2)
        self.assertEqual(env._cohort_costs, [10.] * 3)
        self.assertEqual(len(debits), 3)
        with self.assertRaises(ValueError):
            env.step_followup(IDLE, before_step=lambda: None)

    def test_resolved_cohort_keeps_common_cost_window_and_forbids_new_orders(self):
        env = self.prefix()
        env.patient_registry = {key: "delivered" for key in env.patient_registry}
        env.close_enrollment()
        self.assertEqual(env._cohort_resolution_step, 0)
        with self.assertRaises(ValueError):
            env.step_followup(np.zeros(4), before_step=lambda: None)
        for _ in range(3):
            env.step_followup(IDLE, before_step=lambda: None)
        self.assertEqual(sum(env._cohort_costs), 30.)

    def test_float64_request_not_rounded(self):
        env = self.prefix()
        env.close_enrollment()
        request = np.array([1e-12, -1e-12, 0., -1.])
        event = env.step_followup(request, before_step=lambda: None)[3]
        np.testing.assert_array_equal(event["action"], request)
        np.testing.assert_array_equal(event["info"]["request"], request)

    def test_unresolved_bound_latches_and_does_not_refund(self):
        env = self.prefix()
        env.resolve_at = 999
        env.close_enrollment()
        debits = []
        env.step_followup(IDLE, before_step=lambda: debits.append(1))
        prior = env.followup_state_dict()
        with self.assertRaisesRegex(ValueError, "bound exceeded"):
            env.step_followup(IDLE, before_step=lambda: debits.append(1))
        self.assertEqual(len(debits), 2)
        self.assertEqual(env.t, 3)
        self.assertEqual(env._cohort_steps, 1)
        self.assertFalse(env._cohort_failure["refunded"])
        with self.assertRaises(ValueError):
            env.load_followup_state_dict(prior)
        with self.assertRaises(ValueError):
            env.step_followup(IDLE, before_step=lambda: None)

    def test_callback_failure_terminal_before_environment_call(self):
        env = self.prefix()
        env.close_enrollment()
        def fail():
            raise RuntimeError("budget exhausted")
        with patch.object(FakeEngine, "step", side_effect=AssertionError("must not step")) as step:
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                env.step_followup(IDLE, before_step=fail)
            step.assert_not_called()
        self.assertIsNotNone(env._cohort_failure)
        self.assertEqual(env.t, 2)

    def test_no_enrollment_and_nonfinite_reward_failure(self):
        env = self.prefix()
        env.close_enrollment()
        with self.assertRaises(ValueError):
            env._enroll_arrivals(np.ones(1))
        original = FakeEngine.step
        def invalid(engine, action):
            raw, reward, done, info = original(engine, action)
            return raw, reward, done, info | {"cost": float("nan")}
        with patch.object(FakeEngine, "step", invalid), self.assertRaises(ValueError):
            env.step_followup(IDLE, before_step=lambda: None)
        self.assertEqual(env.t, 2)
        self.assertIsNotNone(env._cohort_failure)

    def test_checkpoint_exact_readback_cannot_refund_or_change_prefix(self):
        env = self.prefix()
        env.close_enrollment()
        start = env.followup_state_dict()
        env.load_followup_state_dict(start)
        env.step_followup(IDLE, before_step=lambda: None)
        current = env.followup_state_dict()
        env.load_followup_state_dict(current)
        with self.assertRaises(ValueError):
            env.load_followup_state_dict(start)
        changed = copy.deepcopy(current)
        changed["prefix_state"]["t"] = 900
        with self.assertRaises(ValueError):
            env.load_followup_state_dict(changed)
        changed = copy.deepcopy(current)
        changed["environment"]["rng_state"]["state"]["state"] += 1
        with self.assertRaises(ValueError):
            env.load_followup_state_dict(changed)

    def test_common_rule_zeroes_rate_priors_without_mutating_prefix_config(self):
        env = self.prefix()
        env.close_enrollment()
        config = {"num_facilities": 1, "demand_rates": [30.], "demand_rate_estimates": [20.]}
        with patch("src.baselines.heuristics.facility_net_action_from_state", return_value=IDLE) as fn:
            np.testing.assert_array_equal(env.common_followup_request(config), IDLE)
            self.assertEqual(fn.call_args.args[1]["demand_rates"], [0.])
            self.assertEqual(fn.call_args.args[1]["demand_rate_estimates"], [0.])
        self.assertEqual(config["demand_rates"], [30.])

    def test_remaining_pipeline_at_fixed_endpoint_is_not_silently_dropped(self):
        env = self.prefix()
        env.close_enrollment()
        env.step_followup(IDLE, before_step=lambda: None)
        env.step_followup(IDLE, before_step=lambda: None)
        env.reagent_transfer_pipeline = np.array([[1.]])
        with self.assertRaisesRegex(ValueError, "unsettled"):
            env.step_followup(IDLE, before_step=lambda: None)


class CohortObjectiveTests(unittest.TestCase):
    def receipt(self, method="cohort", **kw):
        fields = dict(objective=method, prefix_steps=2, accounting_steps=3,
                      active_at_end=0, trajectory_id="invented/train/0", split="training")
        fields.update(kw)
        return cohort_reward_receipt([2., 3.], [10., 11., 12.], **fields)

    def test_tail_once_and_no_scaling_or_raw_reward_mutation(self):
        old, new = self.receipt("window"), self.receipt()
        self.assertEqual(old["training_raw_rewards"], (-2., -3.))
        self.assertEqual(new["training_raw_rewards"], (-2., -36.))
        self.assertEqual(new["raw_prefix_rewards"], old["raw_prefix_rewards"])
        self.assertEqual(new["cohort_cost"], 38.)
        self.assertFalse(new["reward_scale_applied"])

    def test_test_data_and_unresolved_or_incomplete_episodes_rejected(self):
        for extra in (dict(split="test"), dict(active_at_end=1), dict(prefix_steps=3),
                      dict(accounting_steps=2), dict(trajectory_id="")):
            with self.assertRaises(ValueError):
                self.receipt(**extra)

    def test_invalid_costs_and_overflow(self):
        for value in (True, -1., float("nan"), float("inf"), "2"):
            with self.assertRaises(ValueError):
                cohort_reward_receipt([value], [1.], objective="cohort", prefix_steps=1,
                    accounting_steps=1, active_at_end=0, trajectory_id="invented", split="training")
        with self.assertRaises((ValueError, OverflowError)):
            cohort_reward_receipt([1e308], [1e308], objective="cohort", prefix_steps=1,
                accounting_steps=1, active_at_end=0, trajectory_id="invented", split="training")


class FakePrefix:
    def __init__(self):
        self.env, self.index = make(), 0
        self.events = []

    @property
    def closed(self):
        return self.index == 2

    def step(self, *, before_step):
        before_step()
        _, reward, done, info = self.env.step(IDLE)
        self.index += 1
        event = dict(raw_reward=reward, done=done, info=info)
        self.events.append(copy.deepcopy(event))
        return event

    def state_dict(self):
        return dict(index=self.index, environment=self.env.state_dict(), events=copy.deepcopy(self.events))

    def load_state_dict(self, saved):
        from src.rl.prospective_patient_session import decode_arrays
        state = decode_arrays(saved)
        self.env.load_state_dict(state["environment"])
        self.index, self.events = state["index"], copy.deepcopy(state["events"])


class CohortCollectionTests(unittest.TestCase):
    def collection(self, **kwargs):
        return CohortCollection(FakePrefix(), objective="cohort", split="training",
            trajectory_id="invented/0", followup_action=lambda env: IDLE,
            finish_prefix=lambda _: {"finished": True}, enabled=True, **kwargs)

    def test_raw_prefix_last_row_precedes_finish_and_tail_recording(self):
        calls = []
        run = self.collection(record_prefix=lambda p, e: calls.append(("prefix", p.index)),
                              record_tail=lambda c, e: calls.append(("tail", e["index"])))
        run.finish_prefix = lambda p: calls.append(("finish", p.index)) or {"finished": True}
        for _ in range(5):
            run.step(before_step=lambda: None)
        self.assertEqual(calls, [("prefix", 1), ("prefix", 2), ("finish", 2),
                                 ("tail", 1), ("tail", 2), ("tail", 3)])

    def test_prefix_tail_and_closed_restore_without_replay_or_recording(self):
        run = self.collection()
        for index in range(1, 6):
            run.step(before_step=lambda: None)
            saved = run.state_dict()
            restored = self.collection(record_prefix=lambda *a: self.fail("unexpected write"),
                                       record_tail=lambda *a: self.fail("unexpected write"))
            with patch.object(FakeEngine, "step", side_effect=AssertionError("replay forbidden")):
                restored.load_state_dict(saved)
                restored.load_state_dict(saved)
            self.assertEqual(restored.index, index)
            self.assertEqual(state_digest(saved), state_digest(restored.state_dict()))
            self.assertEqual(run.env.rng.bit_generator.state, restored.env.rng.bit_generator.state)
        self.assertEqual(restored.target_receipt(), run.target_receipt())

    def test_restore_rejects_tamper_atomically_and_never_rewinds_or_retries(self):
        run = self.collection()
        run.step(before_step=lambda: None)
        early = run.state_dict()
        for _ in range(3):
            run.step(before_step=lambda: None)
        before = state_digest(run.state_dict())
        with self.assertRaises(ValueError):
            run.load_state_dict(early)
        for change in ("cost", "clock", "demand", "patient", "index"):
            saved = decode_arrays(run.state_dict())
            if change == "cost":
                saved["tail_events"][0]["cost"] += 1
            elif change == "clock":
                saved["followup"]["environment"]["t"] += 1
            elif change == "demand":
                saved["followup"]["environment"]["demand"][0] = 1.
            elif change == "patient":
                saved["followup"]["environment"]["patient_registry"]["extra"] = "waiting"
            else:
                saved["tail_events"][0]["index"] = 999
            with self.subTest(change=change), self.assertRaises((ValueError, AssertionError)):
                run.load_state_dict(encode_arrays(saved))
            self.assertEqual(before, state_digest(run.state_dict()))
        def fail(*args):
            raise RuntimeError("invented disk failure")
        saved = run.state_dict()
        run.record_tail = fail
        with self.assertRaises(RuntimeError):
            run.step(before_step=lambda: None)
        with self.assertRaises(ValueError):
            run.load_state_dict(saved)

    def test_complete_mock_dispatch_preserves_prefix_and_defers_target(self):
        recorded, debits = [], []
        def finish(prefix):
            self.assertEqual(prefix.env.t, 2)
            self.assertFalse(prefix.env._cohort_closed)
            recorded.append(prefix.state_dict())
            return {"raw_prefix_closed": True}
        run = CohortCollection(FakePrefix(), objective="cohort", split="training",
            trajectory_id="invented/0", followup_action=lambda env: IDLE,
            finish_prefix=finish, enabled=True)
        stages = []
        for _ in range(5):
            with self.assertRaises(ValueError):
                run.target_receipt()
            stages.append(run.step(before_step=lambda: debits.append(1))["stage"])
        self.assertEqual(stages, ["prefix"] * 2 + ["tail"] * 3)
        self.assertEqual(len(debits), 5)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(run.target_receipt()["training_raw_rewards"], (-10., -40.))
        self.assertEqual(run.target_receipt(), run.target_receipt())
        self.assertEqual(run.state_dict()["prefix_snapshot"]["environment"]["t"], 2)
        self.assertEqual(run.state_dict()["followup"]["environment"]["t"], 5)

    def test_failed_prefix_recording_stops_before_closure_and_no_retry(self):
        def fail(prefix):
            raise RuntimeError("disk full")
        run = CohortCollection(FakePrefix(), objective="none", split="test",
            trajectory_id="invented/test", followup_action=lambda env: IDLE,
            finish_prefix=fail, enabled=True)
        run.step(before_step=lambda: None)
        with self.assertRaises(RuntimeError):
            run.step(before_step=lambda: None)
        self.assertFalse(run.env._cohort_closed)
        self.assertIsNotNone(run.failure)
        with self.assertRaises(ValueError):
            run.step(before_step=lambda: None)

    def test_evaluation_cannot_admit_training_objective(self):
        with self.assertRaises(ValueError):
            CohortCollection(FakePrefix(), objective="cohort", split="test",
                trajectory_id="invented/test", followup_action=lambda env: IDLE,
                finish_prefix=lambda _: {}, enabled=True)


if __name__ == "__main__":
    unittest.main()
