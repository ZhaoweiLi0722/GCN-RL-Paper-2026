"""Handcrafted records and in-memory artificial weights; zero optimizer steps."""

import copy
import hashlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.rl.capacity_native_tail_learner import CapacityNativeTailLearner, TAIL_SCHEMA
from src.rl.capacity_native_tail_targets import mc_rows, td_rows
from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from tests.test_capacity_native_tail_targets import ANCESTOR, TAPE, with_diagnostics, world


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


class NativeTailLearnerTests(unittest.TestCase):
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

    def config(self, method="planner_tail_td", **overrides):
        config = dict(method=method, architecture="graph", width=32, lr=.0003,
                      gradient_norm_cap=5., cost_scale=1000000., td_horizon=8, gamma=1.,
                      batch_size=64, updates_per_world=32, max_new_updates=768,
                      tail_policy="frozen_mpc", tails_per_world=2, tail_schema=TAIL_SCHEMA,
                      continuation_sha256=ANCESTOR)
        config.update(overrides)
        return config

    def make(self, method="planner_tail_td", **overrides):
        return CapacityNativeTailLearner(self.config(method, **overrides), seed=7,
            feature_dim=31, ancestor_sha256=ANCESTOR,
            before_forward=Mock(side_effect=AssertionError("real forward forbidden")),
            before_optimizer=Mock(side_effect=AssertionError("optimizer callback forbidden")))

    def artificial_forward(self, learner, calls, callback=None):
        def forward(states):
            self.assertIsNone(learner.pending)
            self.assertEqual(learner.updates, 0)
            self.assertEqual(states.dtype, np.float32)
            calls.append(len(states))
            if callback:
                callback(states)
            return torch.full((len(states),), .125, dtype=torch.float32)
        learner._forward = forward

    def assert_no_training(self, learner):
        self.assertEqual(learner.updates, 0)
        self.assertEqual(learner.counts["optimizer_steps"], 0)
        self.assertEqual(learner.optimizer.state, {})
        learner.before_optimizer.assert_not_called()

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_state_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def test_explicit_native_configuration_before_model_construction(self):
        for override in (
                dict(method="observed_td"), dict(method="native_tail_td"),
                dict(tail_policy="adaptive"), dict(tail_policy=None),
                dict(tail_schema="capacity-policy-tail-record-v1"), dict(tail_schema=None),
                dict(continuation_sha256=None), dict(continuation_sha256="c" * 64),
                dict(tails_per_world=6), dict(tails_per_world=True), dict(tails_per_world=2.),
                dict(architecture="flat"), dict(width=64), dict(lr=.001),
                dict(batch_size=32), dict(updates_per_world=31), dict(max_new_updates=769)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.make(**override)
        for field in ("tail_policy", "tail_schema", "tails_per_world", "continuation_sha256"):
            config = self.config()
            del config[field]
            with self.assertRaises(ValueError):
                CapacityNativeTailLearner(config, ancestor_sha256=ANCESTOR)
        self.model_factory.assert_not_called()
        original = self.config()
        learner = CapacityNativeTailLearner(original, seed=0, feature_dim=31,
            ancestor_sha256=ANCESTOR, before_forward=Mock(), before_optimizer=Mock())
        original["tail_schema"] = "mutated"
        self.assertEqual(learner.config, self.config())
        self.assert_no_training(learner)

    def test_prevalidate_entire_cohort_before_callbacks(self):
        learner = self.make()
        learner._forward = Mock(side_effect=AssertionError("premature forward"))
        invalid = [[], world()[:1], world() * 3, [world()[0], world()[0]]]
        for field, value in (("provenance", "public_forecast_not_native"),
                             ("label_policy", "adaptive"), ("continuation_sha256", "c" * 64),
                             ("tape_sha256", "c" * 64), ("source_state_sha256", "bad"),
                             ("candidate", 0), ("quantile", .5), ("root_epoch", 23),
                             ("epochs", list(range(31, 64))),
                             ("features", np.zeros((33, 4, 31), dtype=np.float64)),
                             ("heuristics", np.full(33, np.nan)),
                             ("costs", np.full(32, np.finfo(np.float64).max))):
            records = world()
            records[-1][field] = value
            invalid.append(records)
        before = learner.state_dict()
        for records in invalid:
            with self.assertRaises(ValueError):
                learner.admit_tails(records)
            self.assert_state_equal(learner.state_dict(), before)
        for kwargs in (dict(world_index=1), dict(world_index=True), dict(tape_sha256="c" * 64)):
            with self.assertRaises(ValueError):
                learner.admit_tails(world(), **kwargs)
        learner._forward.assert_not_called()
        self.assert_no_training(learner)

    def test_td_mc_share_native_states_sources_rng_and_float64_targets(self):
        records = world()
        td, mc, calls = self.make(), self.make("planner_tail_mc"), []
        self.artificial_forward(td, calls)
        td.admit_tails(records, world_index=0, tape_sha256=TAPE)
        mc.admit_tails(records)
        self.assertEqual(calls, [64, 24])
        self.assertEqual(td.pending["states"].shape, (88, 4, 31))
        self.assertEqual(td.pending["source_hashes"], mc.pending["source_hashes"])
        np.testing.assert_array_equal(td.pending["states"], mc.pending["states"])
        for learner, method in ((td, "td"), (mc, "mc")):
            expected = [mc_rows(r) if method == "mc" else td_rows(r,
                frozen_residuals=np.r_[np.full(len(r["costs"]), 125000.), 0.]) for r in records]
            np.testing.assert_array_equal(learner.pending["targets"],
                                          np.concatenate([r["targets"] for r in expected]))
            self.assertEqual(learner.pending["targets"].dtype, np.float64)
            self.assertEqual(learner.pending["states"].dtype, np.float32)
            self.assertTrue(all(p.dtype == torch.float32 for p in learner.model.parameters()))
            self.assertEqual(learner.state_dict()["tail_admission"]["world_index"], 0)
            self.assert_no_training(learner)
        np.testing.assert_array_equal(td.rng.integers(0, 88, size=64), mc.rng.integers(0, 88, size=64))
        mc.before_forward.assert_not_called()

    def test_float64_target_precision_survives_learner_admission(self):
        records = world()
        for r in records:
            r["costs"][:] = .25
            r["costs"][0] = 2. ** 40 + .125
            r["heuristics"][:] = 0.
            r["heuristics"][0] = 2. ** 40
        learner = self.make("planner_tail_mc")
        learner.admit_tails(records)
        self.assertEqual(learner.pending["targets"][0], (.125 + 55 * .25) / 1e6)
        self.assertEqual(learner.pending["targets"][56], (.125 + 31 * .25) / 1e6)
        self.assert_no_training(learner)

    def test_collector_diagnostics_admit_and_roundtrip_without_affecting_targets(self):
        learner = self.make("planner_tail_mc")
        records = [with_diagnostics(r) for r in world()]
        learner.admit_tails(records)
        expected = np.concatenate([mc_rows(r)["targets"] for r in world()])
        np.testing.assert_array_equal(learner.pending["targets"], expected)
        before = learner.state_dict()
        records[0]["settlement"]["support"]["outstanding"] = 123.
        records[0]["prefix_costs"][:] = 123.
        learner.load_state_dict(before)
        self.assert_state_equal(learner.state_dict(), before)
        self.assert_no_training(learner)

    def test_owned_snapshot_before_callback_and_after_admission(self):
        records = world(23)[::-1]
        saved = copy.deepcopy(records)
        def mutate_sources(states):
            for record in records:
                record.update(provenance="forecast", tape_sha256="c" * 64, candidate=99)
                for key in ("features", "heuristics", "costs"):
                    record[key][:] = 999.
            states[:] = -999.
        learner, calls = self.make(), []
        self.artificial_forward(learner, calls, mutate_sources)
        learner.admit_tails(records)
        self.assertEqual(calls, [42])
        expected = [td_rows(r, frozen_residuals=np.r_[
            np.full(len(r["costs"]), 125000.), 0.]) for r in saved]
        np.testing.assert_array_equal(learner.pending["targets"],
                                      np.concatenate([r["targets"] for r in expected]))
        np.testing.assert_array_equal(learner.pending["states"],
                                      np.concatenate([r["states"] for r in expected]))
        for record in learner.state_dict()["tail_admission"]["records"]:
            self.assertEqual(record["provenance"], "native_patient_branch")
            self.assertEqual(record["tape_sha256"], TAPE)
        self.assert_no_training(learner)

    def test_failed_bootstrap_leaves_no_partial_pending_admission(self):
        for output in (torch.full((88,), float("nan")), torch.zeros((1,)),
                       torch.zeros((64, 1))):
            learner = self.make()
            learner._forward = Mock(return_value=output)
            before = learner.state_dict()
            with self.assertRaises(ValueError):
                learner.admit_tails(world())
            self.assert_state_equal(learner.state_dict(), before)
            self.assert_no_training(learner)

    def test_pending_update_cap_and_other_admission_paths_reject(self):
        learner = self.make("planner_tail_mc")
        learner.admit_tails(world())
        before = learner.state_dict()
        with self.assertRaises(ValueError):
            learner.admit_tails(world())
        self.assert_state_equal(learner.state_dict(), before)
        with self.assertRaises(ValueError):
            learner.admit_episode({})
        with self.assertRaises(ValueError):
            learner.admit_observed({})
        learner.pending = None
        learner.updates = 768
        with self.assertRaises(ValueError):
            learner.admit_tails(world())
        self.assertIsNone(learner.state_dict()["tail_admission"])
        learner.before_forward.assert_not_called()
        learner.before_optimizer.assert_not_called()

    def test_snapshot_roundtrip_mc_and_td_with_owned_restore(self):
        for method in ("planner_tail_mc", "planner_tail_td"):
            with self.subTest(method=method):
                learner = self.make(method)
                if method == "planner_tail_td":
                    self.artificial_forward(learner, [])
                learner.admit_tails(world(13))
                raw, sha = learner.snapshot()
                restored = CapacityNativeTailLearner.from_bytes(raw,
                    before_forward=learner.before_forward, before_optimizer=learner.before_optimizer)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
                self.assertIs(type(restored), CapacityNativeTailLearner)
                self.assert_state_equal(restored.state_dict(), learner.state_dict())
                source = learner.state_dict()
                restored.load_state_dict(source)
                expected = restored.state_dict()
                source["pending"]["states"][:] = -999
                source["pending"]["targets"][:] = -999
                source["tail_admission"]["records"][0]["tape_sha256"] = "c" * 64
                source["model"]["fixture"][:] = 999
                self.assert_state_equal(restored.state_dict(), expected)
                self.assert_no_training(restored)

    def test_restore_rejects_metadata_policy_tape_source_and_array_tamper(self):
        learner = self.make("planner_tail_mc")
        learner.admit_tails(world())
        mutations = [
            lambda s: s.update(format=CapacityPlannerTailLearner.format),
            lambda s: s.update(ancestor_sha256="c" * 64),
            lambda s: s["config"].update(method="planner_tail_td"),
            lambda s: s["config"].update(tail_policy="adaptive"),
            lambda s: s["config"].update(tail_schema="capacity-policy-tail-record-v1"),
            lambda s: s["config"].update(continuation_sha256="c" * 64),
            lambda s: s["config"].update(tails_per_world=6),
            lambda s: s.update(tail_admission=None),
            lambda s: s.pop("pending"),
            lambda s: s["tail_admission"]["records"].pop(),
            lambda s: s["tail_admission"].update(world_index=1),
            lambda s: s["tail_admission"].update(pending_sha256="0" * 64),
            lambda s: s["tail_admission"].update(pending_sha256=True),
            lambda s: s["pending"]["targets"].__setitem__(0, 123.),
            lambda s: s["pending"]["states"].__setitem__(0, 123.),
            lambda s: s["pending"]["targets"].__setitem__(0, np.nan),
            lambda s: s["pending"]["source_hashes"].__setitem__(0, "c" * 64),
            lambda s: s["pending"]["source_hashes"].__setitem__(0, "z" * 64),
            lambda s: s["pending"].update(targets=s["pending"]["targets"].astype(np.float32)),
            lambda s: s["pending"].update(states=s["pending"]["states"].astype(np.float64)),
            lambda s: s["pending"].update(targets=s["pending"]["targets"].tolist()),
            lambda s: s["pending"].update(targets=s["pending"]["targets"][:-1]),
            lambda s: s["pending"].update(completed=2),
        ]
        for key, value in (("label_policy", "adaptive"), ("format", "forecast"),
                           ("provenance", "public_forecast_not_native"), ("quantile", .5),
                           ("continuation_sha256", "c" * 64), ("source_state_sha256", "c" * 64),
                           ("tape_sha256", "c" * 64), ("candidate", 15), ("start_epoch", 9),
                           ("root_epoch", True)):
            mutations.append(lambda s, k=key, v=value: s["tail_admission"]["records"][0].update({k: v}))
        # Changing both tape declarations passes pair consistency, but not binding.
        mutations.append(lambda s: [r.update(tape_sha256="c" * 64)
                                    for r in s["tail_admission"]["records"]])
        before = learner.state_dict()
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(before)
            mutate(bad)
            with self.subTest(case=index), self.assertRaises(ValueError):
                learner.load_state_dict(bad)
            self.assert_state_equal(learner.state_dict(), before)
        self.assert_no_training(learner)

    def test_restore_parent_failures_are_transactional(self):
        learner = self.make("planner_tail_mc")
        learner.admit_tails(world())
        before = learner.state_dict()
        for mutate, error in (
                (lambda s: s.update(feature_dim=30), ValueError),
                (lambda s: s.update(updates=1), ValueError),
                (lambda s: s["counts"].update(optimizer_steps=-1), ValueError),
                (lambda s: s["model"].update(fixture=torch.tensor([float("nan"), 0.])), ValueError),
                (lambda s: s["model"].update(fixture=torch.zeros(3)), RuntimeError),
                (lambda s: s["optimizer"].update(state={0: {"moment": torch.tensor(float("inf"))}}), ValueError),
                (lambda s: s.update(rng={"bit_generator": "not-an-rng"}), ValueError)):
            bad = copy.deepcopy(before)
            mutate(bad)
            with self.assertRaises(error):
                learner.load_state_dict(bad)
            self.assert_state_equal(learner.state_dict(), before)
        self.assert_no_training(learner)

    def test_pending_hash_binds_config_but_excludes_synthetic_progress_counter(self):
        learner = self.make("planner_tail_mc")
        learner.admit_tails(world())
        before = learner.state_dict()
        # Artificial checkpoint progress only, not an optimizer invocation.
        progressed = copy.deepcopy(before)
        progressed.update(updates=7, counts=dict(forwards=7, optimizer_steps=7))
        progressed["pending"]["completed"] = 7
        learner.load_state_dict(progressed)
        self.assertEqual(learner.pending["completed"], 7)
        self.assertEqual(learner.state_dict()["tail_admission"], before["tail_admission"])
        learner.before_optimizer.assert_not_called()
        td = self.make()
        wrong = copy.deepcopy(before)
        wrong["config"] = td.config
        with self.assertRaisesRegex(ValueError, "pending hash"):
            td.load_state_dict(wrong)
        self.assert_no_training(td)

    def test_empty_roundtrip_and_missing_or_unpaired_binding(self):
        learner = self.make("planner_tail_mc")
        before = learner.state_dict()
        learner.load_state_dict(before)
        self.assert_state_equal(learner.state_dict(), before)
        for binding in ({}, dict(records=[])):
            bad = copy.deepcopy(before)
            bad["tail_admission"] = binding
            with self.assertRaises(ValueError):
                learner.load_state_dict(bad)
        self.assert_state_equal(learner.state_dict(), before)
        self.assert_no_training(learner)

    def test_inherited_weight_only_fork_from_artificial_serialized_parameters(self):
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
        fork = CapacityNativeTailLearner.fork_weights(raw, config, seed=11,
            expected_sha256=sha, before_forward=original.before_forward,
            before_optimizer=original.before_optimizer)
        self.assertIs(type(fork), CapacityNativeTailLearner)
        self.assertEqual(fork.ancestor_sha256, sha)
        self.assertEqual(fork.config["continuation_sha256"], sha)
        self.assertIsNone(fork.pending)
        self.assertIsNone(fork.state_dict()["tail_admission"])
        self.assertEqual(fork.rng.bit_generator.state, np.random.default_rng(12).bit_generator.state)
        self.assert_state_equal(fork.model.state_dict(), fixture["model"])
        self.assert_no_training(fork)
        for method in ("fork_weights", "update", "_admit", "from_bytes"):
            self.assertNotIn(method, CapacityNativeTailLearner.__dict__)
        with self.assertRaises(ValueError):
            CapacityNativeTailLearner.fork_weights(raw, config, seed=11,
                expected_sha256="0" * 64, before_forward=original.before_forward,
                before_optimizer=original.before_optimizer)


if __name__ == "__main__":
    unittest.main()
