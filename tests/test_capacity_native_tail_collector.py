"""Artificial records only; native construction and real optimizer are forbidden."""

import copy
from dataclasses import dataclass
import random

import numpy as np

from src.rl.capacity_native_tail_collector import collect_native_tail, snapshot_sha256
from src.rl.capacity_pilot_runner import COST_KEYS
from src.rl.candidate_pilot_campaign import global_rng_state
from tests.test_capacity_value_mpc import ZeroUpdateCase


@dataclass
class Common:
    epoch: int


@dataclass
class Operations:
    patients: tuple = ()


@dataclass
class Public:
    common: Common
    operations: Operations


def capture(env):
    return Public(Common(env.t), Operations())


def features(public, controller):
    return np.full((4, 31), public.common.epoch, dtype=np.float32), 0.


class Environment:
    def __init__(self, root, before):
        self.t, self._before_native, self._failed = root, before, False
        self.rng = np.random.default_rng(701)
        self.requests = []
        self.fail_epoch = None

    def state_dict(self):
        return dict(t=self.t, world_tape_sha256="c" * 64,
                    rng=copy.deepcopy(self.rng.bit_generator.state), requests=copy.deepcopy(self.requests))

    def step(self, action):
        self._before_native("step")
        self.requests.append(list(action.requested_hours))
        self.rng.random()
        random.random()
        np.random.random()
        if self.t == self.fail_epoch:
            self._failed = True
            raise RuntimeError("artificial charged failure")
        self.t += 1
        cost = float(self.t) + 2. ** -35
        components = dict.fromkeys(COST_KEYS, 0.)
        components["patient_loss_cost"] = cost
        return None, -cost, self.t == 64, dict(components, cost=cost,
            support_public_receipt=dict(committed_hours=list(action.requested_hours)))

    def settlement(self):
        return dict(settled=self.t == 64, enrolled=0, delivered=0, lost=0)


class Controller:
    def __init__(self, proposal, base_control, **kwargs):
        self.proposal, self.base_control = proposal, base_control
        self.epoch, self.recorded, self.last_plan = 0, -1, {}

    def state_dict(self):
        return dict(filter=dict(epoch=self.epoch), lifecycle=dict(epoch=self.epoch, recorded_epoch=self.recorded),
                    capture_epochs=(), last_training_tails=[], last_plan=copy.deepcopy(self.last_plan))

    def load_state_dict(self, state):
        self.epoch = state["filter"]["epoch"]
        self.recorded = state["lifecycle"]["recorded_epoch"]
        self.last_plan = copy.deepcopy(state["last_plan"])

    def candidates(self, public):
        return [(np.full(4, 2.), switch) for switch in (False, True) for _ in range(8)]

    def adaptive(self, public):
        return np.full(4, .5)

    def record_operation(self, public, base):
        self.recorded = public.common.epoch

    def observe(self, public, before_filter):
        if self.epoch != public.common.epoch:
            before_filter(dict(hypothesis_transitions=100))
        self.epoch = public.common.epoch

    def act(self, public, *, before_query, **kwargs):
        before_query(dict(model_epochs=384))
        self.last_plan = dict(epoch=public.common.epoch)
        return np.full(4, 1.)


class NativeBranchTests(ZeroUpdateCase):
    def setup_root(self, root=9):
        # Deep-copying this bound callback would copy its owner. The collector
        # must substitute the branch callback through deepcopy's memo instead.
        env = Environment(root, self.forbidden_reference_step)
        controller = Controller({}, lambda *a, **k: np.zeros(16))
        controller.epoch, controller.recorded = root, root - 1
        return env, controller

    def forbidden_reference_step(self, kind):
        self.fail("reference primitive invoked")

    def run_branch(self, env, controller, **overrides):
        rows, checkpoints, charges = [], [], []
        def charge(kind):
            return lambda payload: charges.append((kind, copy.deepcopy(payload)))
        args = dict(value=object(), continuation_sha256="a" * 64, tape_sha256="c" * 64,
            before_clone=charge("clone"), before_native=charge("native"),
            before_plan=charge("plan"), before_query=charge("query"), after_plan=charge("plan_end"),
            before_filter=charge("filter"), after_filter=charge("filter_end"),
            record_raw=rows.append, checkpoint=checkpoints.append,
            controller_factory=Controller, capture=capture, feature_adapter=features, scientific=False)
        args.update(overrides)
        result = collect_native_tail(capture(env), env, controller, 9, **args)
        return result, rows, checkpoints, charges

    def test_native_prefix_switch_and_frozen_suffix_isolate_root_and_rng(self):
        env, controller = self.setup_root()
        prior = snapshot_sha256(dict(env=env.state_dict(), controller=controller.state_dict(), rng=global_rng_state()))
        # A copied act closure must never be used in the branch.
        controller.act = lambda *a, **k: self.fail("copied reference closure")
        result, rows, checkpoints, charges = self.run_branch(env, controller)
        self.assertEqual(result["epochs"], list(range(17, 65)))
        self.assertEqual(len(rows), 55)
        self.assertEqual(len(result["costs"]), 47)
        np.testing.assert_array_equal(rows[0]["requested_hours"], np.full(4, 2.))
        np.testing.assert_array_equal(rows[1]["requested_hours"], np.full(4, 2.))
        np.testing.assert_array_equal(rows[2]["requested_hours"], np.full(4, .5))
        np.testing.assert_array_equal(rows[8]["requested_hours"], np.full(4, 1.))
        np.testing.assert_array_equal(rows[48 - 9]["requested_hours"], np.zeros(4))
        self.assertEqual(result["costs"].dtype, np.dtype("float64"))
        self.assertEqual(result["costs"][0], 18. + 2. ** -35)
        self.assertEqual(sum(k == "clone" for k, _ in charges), 1)
        self.assertEqual(sum(k == "native" for k, _ in charges), 55)
        self.assertEqual(sum(k == "plan" for k, _ in charges), 31)
        self.assertEqual(sum(k == "filter" for k, _ in charges), 55)
        self.assertEqual(checkpoints[-1]["status"], "completed")
        self.assertEqual(prior, snapshot_sha256(dict(env=env.state_dict(), controller=controller.state_dict(), rng=global_rng_state())))

    def test_latest_root_settlement_and_no_terminal_bootstrap(self):
        env, controller = self.setup_root(47)
        result, rows, _, charges = self.run_branch(env, controller)
        self.assertEqual(len(result["costs"]), 9)
        self.assertEqual(result["heuristics"][-1], 0.)
        self.assertEqual(len(rows), 17)
        self.assertFalse(any(k == "plan" for k, _ in charges))
        self.assertTrue(all(np.count_nonzero(row["requested_hours"]) == 0 for row in rows[1:]))

    def test_charged_failure_preserves_partial_without_retry_or_reference_mutation(self):
        env, controller = self.setup_root()
        env.fail_epoch = 12
        before = snapshot_sha256(dict(env=env.state_dict(), rng=global_rng_state()))
        checkpoints, charges = [], []
        with self.assertRaisesRegex(RuntimeError, "artificial charged failure"):
            self.run_branch(env, controller, checkpoint=checkpoints.append, before_native=charges.append)
        self.assertEqual(charges, ["step"] * 4)
        self.assertEqual(checkpoints[-1]["status"], "failed")
        self.assertTrue(checkpoints[-1]["primitive_may_be_partially_applied"])
        self.assertEqual(len(checkpoints[-1]["environment"]["requests"]), 4)
        self.assertEqual(before, snapshot_sha256(dict(env=env.state_dict(), rng=global_rng_state())))

    def test_bad_tape_or_operated_root_rejects_before_clone(self):
        env, controller = self.setup_root()
        for bad in (dict(tape_sha256="b" * 64), dict(before_clone=None)):
            with self.assertRaises(ValueError):
                self.run_branch(env, controller, **bad)
        controller.recorded = env.t
        with self.assertRaises(ValueError):
            self.run_branch(env, controller)
