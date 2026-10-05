"""Artificial serialized parameters and tensors only; zero optimizer updates."""

import copy
import hashlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from src.rl.capacity_planner_tail_targets import mc_rows, td_rows
from src.rl.capacity_policy_tail_learner import CapacityPolicyTailLearner, TAIL_SCHEMA


ANCESTOR = "1" * 64


class ArtificialValue(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fixture = torch.nn.Parameter(torch.tensor([.25, -.5], dtype=torch.float32))

    def forward(self, *args):
        raise AssertionError("model forward forbidden")


class ArtificialOptimizer:
    def __init__(self, parameters, **kwargs):
        self.state = {}

    def state_dict(self):
        return dict(state=copy.deepcopy(self.state))

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state["state"])

    def zero_grad(self, *args, **kwargs):
        raise AssertionError("optimizer use forbidden")

    def step(self, *args, **kwargs):
        raise AssertionError("optimizer step forbidden")


class PolicyTailLearnerTests(unittest.TestCase):
    def setUp(self):
        self.model_factory = Mock(side_effect=ArtificialValue)
        for target, replacement in (
                ("src.rl.capacity_value_comparison_learner.CapacityTerminalValue", self.model_factory),
                ("torch.optim.Adam", ArtificialOptimizer),
                ("src.models.capacity_terminal_value.CapacityTerminalValue.forward",
                 Mock(side_effect=AssertionError("scientific forward forbidden"))),
                ("src.models.capacity_terminal_value.CapacityTerminalValue.load_state_dict",
                 Mock(side_effect=AssertionError("scientific weight loading forbidden"))),
                ("src.rl.capacity_planner_tail_learner.CapacityPlannerTailLearner.update",
                 Mock(side_effect=AssertionError("updates forbidden")))):
            active = patch(target, replacement)
            active.start()
            self.addCleanup(active.stop)

    def config(self, method="planner_tail_td", policy="frozen_mpc", **overrides):
        config = dict(method=method, architecture="graph", width=32, lr=.0003,
                      gradient_norm_cap=5., cost_scale=1000000., td_horizon=8, gamma=1.,
                      batch_size=64, updates_per_world=32, max_new_updates=768,
                      tail_policy=policy, tails_per_world=6, tail_schema=TAIL_SCHEMA,
                      continuation_sha256=ANCESTOR if policy == "frozen_mpc" else None)
        config.update(overrides)
        return config

    def make(self, method="planner_tail_td", policy="frozen_mpc", **overrides):
        return CapacityPolicyTailLearner(self.config(method, policy, **overrides), seed=7,
            feature_dim=31, ancestor_sha256=ANCESTOR,
            before_forward=Mock(side_effect=AssertionError("real forward forbidden")),
            before_optimizer=Mock(side_effect=AssertionError("optimizer callback forbidden")))

    def records(self, policy="frozen_mpc", decisions=(0, 24)):
        records = []
        for root, decision in enumerate(decisions):
            start = decision + 8
            for quantile in (.1, .5, .9):
                length = 64 - start
                records.append(dict(decision_epoch=decision, candidate=root + 2,
                    quantile=quantile, start_epoch=start, epochs=list(range(start, 65)),
                    label_policy=policy,
                    continuation_sha256=ANCESTOR if policy == "frozen_mpc" else None,
                    features=np.full((length + 1, 4, 31), root + quantile, dtype=np.float32),
                    heuristics=np.arange(length + 1, dtype=np.float64) * 1.125,
                    costs=np.arange(1, length + 1, dtype=np.float64) * 3.25))
        return records

    def artificial_forward(self, learner, calls, callback=None):
        def forward(states):
            self.assertIsNone(learner.pending)
            self.assertEqual(learner.updates, 0)
            self.assertEqual(learner.counts["optimizer_steps"], 0)
            self.assertEqual(states.dtype, np.float32)
            calls.append(len(states))
            if callback:
                callback()
            return torch.full((len(states),), .125, dtype=torch.float32)
        learner._forward = forward

    def assert_no_training(self, learner):
        self.assertEqual(learner.updates, 0)
        self.assertEqual(learner.counts["optimizer_steps"], 0)
        learner.before_optimizer.assert_not_called()
        self.assertEqual(learner.optimizer.state, {})

    def test_configuration_rejects_before_model_construction(self):
        for override in (
                dict(tail_policy="ADAPTIVE"), dict(tail_policy=None),
                dict(tail_policy="adaptive", method="planner_tail_mc", continuation_sha256=None),
                dict(method="observed_td"), dict(continuation_sha256=None),
                dict(continuation_sha256="2" * 64), dict(continuation_sha256="z" * 64),
                dict(tails_per_world=96), dict(tails_per_world=6.),
                dict(tail_schema="old"), dict(batch_size=32),
                dict(updates_per_world=31), dict(max_new_updates=769)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.make(**override)
        with self.assertRaises(ValueError):
            self.make(policy="adaptive", continuation_sha256=ANCESTOR)
        self.model_factory.assert_not_called()
        adaptive = self.make(policy="adaptive")
        self.assertIsNone(adaptive.config["continuation_sha256"])
        self.assertEqual(adaptive.config["tails_per_world"], 6)
        self.assertEqual(adaptive.config["tail_schema"], TAIL_SCHEMA)
        self.assertEqual(adaptive.config, self.config(policy="adaptive"))

    def test_h8_root_bounds_on_admission_and_restore(self):
        learner = self.make()
        learner._forward = Mock(side_effect=AssertionError("forward before validation"))
        for decision in (-1, 48, 55):
            with self.subTest(decision=decision), self.assertRaisesRegex(
                    ValueError, "invalid predicted root"):
                learner.admit_tails(self.records(decisions=(0, decision)))
        for candidate in (-1, 16, True, 15.):
            records = self.records()
            for record in records[3:]:
                record["candidate"] = candidate
            with self.subTest(candidate=candidate), self.assertRaisesRegex(
                    ValueError, "invalid predicted root"):
                learner.admit_tails(records)
        learner._forward.assert_not_called()
        self.assertIsNone(learner.pending)
        self.assert_no_training(learner)

        boundary = self.make("planner_tail_mc")
        records = self.records(decisions=(0, 47))
        for record in records:
            record["candidate"] = 0 if record["decision_epoch"] == 0 else 15
        boundary.admit_tails(records)
        self.assertEqual(len(boundary.pending["states"]), 195)
        before = boundary.state_dict()
        for override in (dict(decision_epoch=48, start_epoch=56), dict(candidate=16)):
            state = copy.deepcopy(before)
            for record in state["tail_admission"]["records"][3:]:
                record.update(override)
            with self.subTest(restore=override), self.assertRaisesRegex(
                    ValueError, "invalid predicted root"):
                boundary.load_state_dict(state)
        self.assertEqual(boundary.state_dict()["tail_admission"], before["tail_admission"])
        self.assert_no_training(boundary)

    def test_prevalidate_entire_cohort_before_any_model_call(self):
        learner = self.make()
        learner._forward = Mock(side_effect=AssertionError("forward before validation"))
        cases = []
        for key, value in (
                ("label_policy", "adaptive"), ("continuation_sha256", "2" * 64),
                ("continuation_sha256", "z" * 64), ("continuation_sha256", None),
                ("start_epoch", 31), ("candidate", 99), ("quantile", .2),
                ("decision_epoch", True), ("features", np.zeros((1, 4, 31))),
                ("costs", np.full(32, np.nan)), ("heuristics", np.full(33, np.inf)),
                ("epochs", list(range(31, 64)))):
            records = self.records()
            records[-1][key] = value
            cases.append(records)
        for key in ("continuation_sha256", "label_policy", "epochs"):
            records = self.records()
            del records[-1][key]
            cases.append(records)
        duplicate = self.records()
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        cases.extend([duplicate, self.records()[:-1], self.records() * 16,
                      self.records(decisions=(0, 0))])
        third_root = self.records()
        third_root[-1].update(decision_epoch=23, start_epoch=31)
        cases.append(third_root)
        wrong_start = self.records()
        wrong_start[-1].update(decision_epoch=25, start_epoch=33)
        for record in wrong_start[3:]:
            record.update(decision_epoch=25, start_epoch=33)
        cases.append(wrong_start)
        for index, records in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                learner.admit_tails(records)
            self.assertIsNone(learner.pending)
        learner._forward.assert_not_called()
        self.assert_no_training(learner)

    def test_six_tail_targets_shared_data_rng_and_dtype(self):
        td, mc, calls = self.make(), self.make("planner_tail_mc"), []
        records, config = self.records(), copy.deepcopy(td.config)
        original = copy.deepcopy(records)
        self.artificial_forward(td, calls)
        td.admit_tails(records)
        mc.admit_tails(records)
        self.assertEqual(calls, [64, 64, 64, 64, 8])
        self.assertEqual(td.pending["states"].shape, (264, 4, 31))
        self.assertEqual(td.pending["source_hashes"], mc.pending["source_hashes"])
        np.testing.assert_array_equal(td.pending["states"], mc.pending["states"])
        for learner, builder in ((td, td_rows), (mc, mc_rows)):
            expected = []
            for record in original:
                data = {key: record[key] for key in ("features", "heuristics", "costs", "epochs")}
                if builder is td_rows:
                    data["frozen_residuals"] = np.r_[np.full(len(record["costs"]), 125000.), 0.]
                expected.extend(builder(**data)["targets"])
            np.testing.assert_array_equal(learner.pending["targets"], expected)
            self.assertEqual(learner.pending["targets"].dtype, np.float64)
            self.assertEqual(learner.pending["states"].dtype, np.float32)
            self.assertTrue(all(p.dtype == torch.float32 for p in learner.model.parameters()))
            self.assert_no_training(learner)
        for _ in range(32):
            np.testing.assert_array_equal(td.rng.integers(0, 264, size=64),
                                          mc.rng.integers(0, 264, size=64))
        for record, before in zip(records, original):
            for key in record:
                np.testing.assert_array_equal(record[key], before[key])
        self.assertEqual(td.config, config)
        records[0]["features"][:] = 999.
        records[0]["costs"][:] = 999.
        self.assertFalse(np.any(td.pending["states"] == 999.))
        self.assertFalse(np.any(mc.pending["states"] == 999.))

    def test_td_owns_snapshot_before_callbacks_and_terminal_rows(self):
        learner, calls = self.make(policy="adaptive"), []
        records = self.records(policy="adaptive", decisions=(46, 47))
        baseline = copy.deepcopy(records)
        def change_sources():
            for record in records:
                record["costs"][:] = 999.
                record["features"][:] = 999.
                record["heuristics"][:] = 999.
        self.artificial_forward(learner, calls, change_sources)
        learner.admit_tails(records)
        self.assertEqual(calls, [57])
        expected = []
        for record in baseline:
            data = {k: record[k] for k in ("features", "heuristics", "costs", "epochs")}
            targets = td_rows(**data, frozen_residuals=np.r_[
                np.full(len(record["costs"]), 125000.), 0.])["targets"]
            np.testing.assert_array_equal(targets[-8:], mc_rows(**data)["targets"][-8:])
            expected.extend(targets)
        np.testing.assert_array_equal(learner.pending["targets"], expected)
        self.assertFalse(np.any(learner.pending["states"] == 999.))
        self.assert_no_training(learner)

    def test_adaptive_and_frozen_trajectories_cannot_be_mixed(self):
        adaptive = self.make(policy="adaptive")
        for records in (self.records(), self.records(policy="adaptive")):
            records[-1]["continuation_sha256"] = ANCESTOR
            with self.assertRaises(ValueError):
                adaptive.admit_tails(records)
        frozen = self.make()
        with self.assertRaises(ValueError):
            frozen.admit_tails(self.records(policy="adaptive"))
        self.assert_no_training(adaptive)
        self.assert_no_training(frozen)

    def test_pending_and_update_cap_reject_before_forward(self):
        learner = self.make("planner_tail_mc")
        learner.admit_tails(self.records())
        before = learner.state_dict()
        with self.assertRaises(ValueError):
            learner.admit_tails(self.records())
        self.assertEqual(before["tail_admission"], learner.state_dict()["tail_admission"])
        learner.pending = None
        learner.updates = 768
        with self.assertRaises(ValueError):
            learner.admit_tails(self.records())
        learner.before_forward.assert_not_called()
        learner.before_optimizer.assert_not_called()

    def test_inherited_snapshot_roundtrip_and_pending_hash_binding(self):
        learner = self.make("planner_tail_mc")
        learner.admit_tails(self.records())
        raw, sha = learner.snapshot()
        restored = CapacityPolicyTailLearner.from_bytes(raw,
            before_forward=learner.before_forward, before_optimizer=learner.before_optimizer)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
        self.assertIs(type(restored), CapacityPolicyTailLearner)
        self.assertEqual(restored.state_dict()["tail_admission"], learner.state_dict()["tail_admission"])
        np.testing.assert_array_equal(restored.pending["targets"], learner.pending["targets"])
        np.testing.assert_array_equal(restored.rng.integers(264, size=64), learner.rng.integers(264, size=64))
        mutations = [
            lambda s: s.update(format=CapacityPlannerTailLearner.format),
            lambda s: s.update(ancestor_sha256="2" * 64),
            lambda s: s["config"].update(tails_per_world=96),
            lambda s: s["config"].update(tail_schema="old"),
            lambda s: s["config"].update(tail_policy="adaptive"),
            lambda s: s["config"].update(continuation_sha256="2" * 64),
            lambda s: s.update(tail_admission=None),
            lambda s: s["tail_admission"]["records"].pop(),
            lambda s: s["tail_admission"]["records"][0].update(label_policy="adaptive"),
            lambda s: s["tail_admission"].update(pending_sha256="0" * 64),
            lambda s: s["pending"]["targets"].__setitem__(0, 123.),
            lambda s: s["pending"]["states"].__setitem__(0, 123.),
            lambda s: s["pending"]["source_hashes"].__setitem__(0, "2" * 64),
            lambda s: s["pending"].update(targets=s["pending"]["targets"].astype(np.float32)),
            lambda s: s["pending"].update(completed=2),
        ]
        before = restored.state_dict()
        for index, mutate in enumerate(mutations):
            state = learner.state_dict()
            mutate(state)
            with self.subTest(case=index), self.assertRaises(ValueError):
                restored.load_state_dict(state)
            self.assertEqual(restored.state_dict()["tail_admission"], before["tail_admission"])
            self.assertEqual(restored.rng.bit_generator.state, before["rng"])
            np.testing.assert_array_equal(restored.pending["targets"], before["pending"]["targets"])
        self.assert_no_training(restored)

    def test_weight_only_fork_uses_artificial_serialized_parameters(self):
        original = self.make()
        fixture = original.state_dict()
        fixture.update(format="capacity-value-comparison-td-v1", updates=1536, pending=None,
                       counts=dict(forwards=10000, optimizer_steps=1536))
        fixture["optimizer"] = dict(state={0: dict(exp_avg=torch.ones(2))})
        stream = io.BytesIO()
        torch.save(fixture, stream)
        raw = stream.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        config = self.config("planner_tail_mc", continuation_sha256=sha)
        fork = CapacityPolicyTailLearner.fork_weights(raw, config, seed=11,
            expected_sha256=sha, before_forward=original.before_forward,
            before_optimizer=original.before_optimizer)
        self.assertIs(type(fork), CapacityPolicyTailLearner)
        self.assertEqual(fork.ancestor_sha256, sha)
        self.assertEqual(fork.config["continuation_sha256"], sha)
        self.assertIsNone(fork.pending)
        self.assertIsNone(fork.state_dict()["tail_admission"])
        self.assertEqual(fork.rng.bit_generator.state, np.random.default_rng(12).bit_generator.state)
        for key, value in fixture["model"].items():
            self.assertTrue(torch.equal(value, fork.model.state_dict()[key]))
        self.assert_no_training(fork)
        for method in ("fork_weights", "update", "from_bytes"):
            self.assertNotIn(method, CapacityPolicyTailLearner.__dict__)
        with self.assertRaises(ValueError):
            CapacityPolicyTailLearner.fork_weights(raw, config, seed=11,
                expected_sha256="0" * 64, before_forward=original.before_forward,
                before_optimizer=original.before_optimizer)


if __name__ == "__main__":
    unittest.main()
