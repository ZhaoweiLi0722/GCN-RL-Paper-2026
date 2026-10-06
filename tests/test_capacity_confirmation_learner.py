"""Artificial records/tensors only; no environments, saved scientific weights or fits."""

import copy
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.models.capacity_confirmation_value import CapacityConfirmationValue
from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.capacity_confirmation_design import counts, tail_config, warm_config
from src.rl.capacity_confirmation_learner import (
    CapacityConfirmationTailLearner as TailLearner,
    CapacityConfirmationWarmLearner as WarmLearner,
)
from src.rl.capacity_native import CapacityPilotEnv
from src.rl.capacity_native_tail_targets import mc_rows, td_rows
from src.rl.capacity_value_features import PublicPatientForecast
from src.rl.capacity_value_learner import td_rows as observed_td_rows
from tests.test_capacity_native_tail_targets import ANCESTOR, TAPE, world


VALUE = dict(width=32, lr=.0003, gradient_norm_cap=5., cost_scale=1000000.,
             td_horizon=8, gamma=1., batch_size=64, updates_per_world=32, evaluation_updates=0)
STUDY = dict(value=VALUE)


def episode():
    return dict(features=np.arange(65 * 4 * 31, dtype=np.float32).reshape(65, 4, 31) / 100.,
                heuristics=np.arange(65, dtype=np.float64) * 100.,
                costs=np.arange(1, 65, dtype=np.float64) * 10.)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.forbidden_step = stack.enter_context(patch.object(
            torch.optim.Adam, "step", side_effect=AssertionError("real optimizer forbidden")))
        stack.enter_context(patch.object(CapacityTerminalValue, "forward",
                                        side_effect=AssertionError("real forward forbidden")))
        for cls in (CapacityPilotEnv, PublicPatientForecast):
            for method in ("__init__", "step"):
                stack.enter_context(patch.object(cls, method,
                                    side_effect=AssertionError("environment/forecast forbidden")))

    def warm(self, architecture="graph", seed=7, **overrides):
        config = dict(warm_config(STUDY, architecture), **overrides)
        return WarmLearner(config, seed=seed, feature_dim=31,
                           before_forward=Mock(), before_optimizer=Mock())

    def tail(self, role="graph_td", seed=11, **overrides):
        config = dict(tail_config(STUDY, role, ANCESTOR), **overrides)
        return TailLearner(config, seed=seed, ancestor_sha256="c" * 64,
                           before_forward=Mock(), before_optimizer=Mock())

    def fake_forward(self, learner, *, value=.125, mutate=None):
        def forward(x):
            if mutate:
                mutate(x)
            return torch.full((len(x),), value, dtype=torch.float32, requires_grad=True)
        learner.model.forward = Mock(side_effect=forward)
        return learner.model.forward

    def equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def artificial_adam(self, learner, updates):
        # Set fixture moments/counters directly; never invoke a real optimizer.
        learner.updates = updates
        learner.counts["optimizer_steps"] = updates
        learner.optimizer.state.clear()
        for parameter in learner.model.parameters():
            learner.optimizer.state[parameter] = dict(step=torch.tensor(float(updates)),
                exp_avg=torch.full_like(parameter, .01), exp_avg_sq=torch.full_like(parameter, .02))

    def completed_warm(self, architecture="graph"):
        learner = self.warm(architecture)
        self.fake_forward(learner)
        learner.admit_episode(episode())
        source = learner.admissions[0]
        learner.admissions = []
        for index in range(48):
            binding = copy.deepcopy(source)
            binding["world_index"] = index
            binding["pending_sha256"] = learner._pending_hash(learner.pending, binding)
            learner.admissions.append(binding)
        learner.pending = None
        self.artificial_adam(learner, 1536)
        learner.rng.integers(0, 64, size=13)
        # In-memory artificial checkpoint, not a scientific trained model.
        learner.load_state_dict(learner.state_dict())
        return learner

    def test_exact_paired_initialization_and_adjacency_only_ablation(self):
        before = torch.random.get_rng_state().clone()
        graph, self_only = self.warm(), self.warm("self_only")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.equal(dict(graph.model.named_parameters()), dict(self_only.model.named_parameters()))
        self.assertEqual(sum(p.numel() for p in graph.model.parameters()), 3169)
        self.assertEqual(sum(p.numel() for p in self_only.model.parameters()), 3169)
        self.assertTrue(torch.equal(self_only.model.encoder.adjacency, torch.eye(4)))
        self.assertFalse(torch.equal(graph.model.encoder.adjacency, torch.eye(4)))
        self.assertIs(CapacityConfirmationValue.forward, CapacityTerminalValue.forward)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            historical_architecture = CapacityTerminalValue(31, width=32)
        self.equal(graph.model.state_dict(), historical_architecture.state_dict())
        for learner in (graph, self_only):
            self.assertEqual(len(learner.model.encoder.layers), 2)
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu"
                                for p in learner.model.parameters()))
            self.assertFalse(learner.optimizer.state)
            self.assertEqual(learner.counts, dict(forwards=0, optimizer_steps=0))

    def test_design_config_interfaces_and_no_hyperparameter_changes(self):
        path = Path(__file__).resolve().parents[1] / "experiments/configs/capacity_confirmation_20261006.json"
        study = json.loads(path.read_text())
        for architecture in ("graph", "self_only"):
            learner = WarmLearner(warm_config(study, architecture), seed=1,
                                  before_forward=Mock(), before_optimizer=Mock())
            self.assertEqual(learner.config["method"], "observed_td")
        for role in ("graph_td", "self_only_td", "graph_mc"):
            TailLearner(tail_config(study, role, ANCESTOR), seed=1, ancestor_sha256=TAPE,
                        before_forward=Mock(), before_optimizer=Mock())
        invalid = [dict(width=64), dict(lr=.001), dict(batch_size=64.),
                   dict(gradient_norm_cap=10), dict(cost_scale=1), dict(td_horizon=4),
                   dict(gamma=True), dict(updates_per_world=31), dict(max_new_updates=768),
                   dict(method="planner_tail_mc"), dict(architecture="flat")]
        with patch.object(torch.random, "fork_rng", side_effect=AssertionError("construction forbidden")):
            for override in invalid:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    self.warm(**override)
            for override in (dict(tail_schema="forecast"), dict(tails_per_world=True),
                             dict(continuation_sha256="bad"), dict(tail_policy="adaptive")):
                with self.assertRaises(ValueError):
                    self.tail(**override)
            for kwargs in (dict(seed="7"), dict(seed=True), dict(seed=-1), dict(feature_dim=30)):
                args = dict(seed=7, feature_dim=31, before_forward=Mock(), before_optimizer=Mock())
                args.update(kwargs)
                with self.assertRaises(ValueError):
                    WarmLearner(warm_config(STUDY, "graph"), **args)

    def test_model_restore_rejects_wrong_buffer_shape_dtype_or_nonfinite_before_mutation(self):
        graph, target = self.warm(), self.warm("self_only")
        with self.assertRaisesRegex(ValueError, "adjacency"):
            target.model.load_state_dict(graph.model.state_dict())
        before = graph.model.state_dict()
        for key, value in (("encoder.adjacency", torch.eye(4)),
                           ("encoder.adjacency", torch.full((4, 4), float("nan"))),
                           ("head.2.bias", torch.zeros(2)),
                           ("head.2.bias", torch.zeros(1, dtype=torch.float64)),
                           ("head.2.bias", torch.tensor([float("inf")]))):
            state = copy.deepcopy(before)
            state[key] = value
            with self.assertRaises(ValueError):
                graph.model.load_state_dict(state)
            self.equal(graph.model.state_dict(), before)

    def test_observed_formula_shared_data_precision_and_one_bootstrap_batch(self):
        data = episode()
        data["costs"][0] = 2. ** 40 + .125
        data["heuristics"][0] = 2. ** 40
        graph, self_only = self.warm(), self.warm("self_only")
        expected = observed_td_rows(**data)
        target = expected["observed_cost"] + np.where(expected["done"], 0.,
            expected["bootstrap_base"] + .125) - expected["base"]
        for learner in (graph, self_only):
            self.fake_forward(learner)
            learner.admit_episode(data)
            self.equal(learner.pending["targets"], target)
            self.assertEqual(learner.pending["targets"].dtype, np.float64)
            learner.before_forward.assert_called_once_with("value", 64)
            learner.before_optimizer.assert_not_called()
        self.assertEqual(graph.pending["source_hashes"], self_only.pending["source_hashes"])
        self.equal(graph.rng.bit_generator.state, self_only.rng.bit_generator.state)
        self.assertNotEqual(float(target[0]), float(np.float32(target[0])))

    def test_warm_data_shape_finiteness_and_nonnegative_costs_before_callbacks(self):
        learner = self.warm()
        for key, value in (("features", np.zeros((65, 4, 30), dtype=np.float32)),
                           ("features", np.zeros((65, 4, 31), dtype=np.float64)),
                           ("heuristics", np.full(65, np.nan)),
                           ("heuristics", np.zeros((65, 1))),
                           ("costs", -np.ones(64)), ("costs", np.ones(63)),
                           ("costs", np.full(64, np.finfo(np.float64).max))):
            data = episode()
            data[key] = value
            with self.subTest(key=key), np.errstate(over="ignore"), self.assertRaises(ValueError):
                learner.admit_episode(data)
        learner.before_forward.assert_not_called()
        self.assertIsNone(learner.pending)

    def test_historical_runner_list_container_matches_stacked_episode(self):
        data = episode()
        listed = dict(features=list(data["features"]),
                      heuristics=data["heuristics"].tolist(), costs=data["costs"].tolist())
        stacked, historical = self.warm(), self.warm()
        for learner, source in ((stacked, data), (historical, listed)):
            self.fake_forward(learner)
            learner.admit_episode(source)
            learner.before_forward.assert_called_once_with("value", 64)
        self.equal(stacked.state_dict(), historical.state_dict())

    def test_warm_admission_owns_data_before_callbacks(self):
        data = episode()
        saved = copy.deepcopy(data)
        def mutate(x):
            for value in data.values():
                value[:] = 999
            x.fill_(-999)
        learner = self.warm()
        self.fake_forward(learner, mutate=mutate)
        learner.admit_episode(data)
        rows = observed_td_rows(**saved)
        self.equal(learner.pending["states"], rows["states"])
        self.equal(learner.pending["targets"], rows["observed_cost"] + np.where(
            rows["done"], 0., rows["bootstrap_base"] + .125) - rows["base"])

    def test_td_mc_shared_native_rows_sources_sampling_and_formulas(self):
        learners = [self.tail(role) for role in ("graph_td", "self_only_td", "graph_mc")]
        for learner in learners:
            self.fake_forward(learner)
            learner.admit_tails(world()[::-1], world_index=0, tape_sha256=TAPE)
            expected = [mc_rows(r) if learner.config["method"] == "planner_tail_mc" else
                        td_rows(r, frozen_residuals=np.r_[np.full(len(r["costs"]), 125000.), 0.])
                        for r in world()]
            self.equal(learner.pending["targets"], np.concatenate([r["targets"] for r in expected]))
            self.equal(learner.pending["states"], np.concatenate([r["states"] for r in expected]))
            self.assertEqual(learner.pending["targets"].dtype, np.float64)
            learner.before_optimizer.assert_not_called()
        learners[-1].before_forward.assert_not_called()
        for learner in learners[:2]:
            self.assertEqual([c.args for c in learner.before_forward.call_args_list],
                             [("value", 64), ("value", 24)])
        for learner in learners[1:]:
            self.equal(learners[0].pending["source_hashes"], learner.pending["source_hashes"])
            self.equal(learners[0].rng.bit_generator.state, learner.rng.bit_generator.state)
        batches = [l.rng.integers(0, 88, size=64) for l in learners]
        for batch in batches[1:]:
            self.equal(batch, batches[0])

    def test_native_batch_arithmetic_matches_840_training_bootstraps(self):
        per_arm = 0
        for index in range(24):
            learner = self.tail()
            self.fake_forward(learner)
            learner.admit_tails(world(index), world_index=index, tape_sha256=TAPE)
            n = 88 - 2 * index
            self.assertEqual(len(learner.pending["states"]), n)
            expected = [64, n - 64] if n > 64 else [n]
            self.assertEqual([c.args[1] for c in learner.before_forward.call_args_list], expected)
            per_arm += len(expected)
        self.assertEqual(per_arm, 36)
        self.assertEqual(48 * 2 * 5 + per_arm * 2 * 5, 840)
        self.assertEqual(counts()["training_bootstrap_forwards"], 840)

    def test_native_admission_rejects_bad_context_before_any_forward(self):
        learner = self.tail()
        before = learner.state_dict()
        for key, value in (("continuation_sha256", learner.ancestor_sha256),
                           ("tape_sha256", "d" * 64), ("candidate", 15),
                           ("costs", np.full(32, np.nan)), ("quantile", .5),
                           ("features", np.zeros((33, 4, 31), dtype=np.float64))):
            records = world()
            records[-1][key] = value
            with self.assertRaises(ValueError):
                learner.admit_tails(records, world_index=0, tape_sha256=TAPE)
            self.equal(learner.state_dict(), before)
        for kwargs in (dict(world_index=True, tape_sha256=TAPE),
                       dict(world_index=1, tape_sha256=TAPE), dict(world_index=0, tape_sha256=None)):
            with self.assertRaises(ValueError):
                learner.admit_tails(world(), **kwargs)
        with self.assertRaises(ValueError):
            learner.admit_episode(episode())
        learner.before_forward.assert_not_called()

    def test_native_records_owned_before_bootstrap(self):
        records = world()
        saved = copy.deepcopy(records)
        def mutate(x):
            for record in records:
                record["continuation_sha256"] = "d" * 64
                record["costs"][:] = 999
                record["features"][:] = 999
            x.fill_(-999)
        learner = self.tail()
        self.fake_forward(learner, mutate=mutate)
        learner.admit_tails(records, world_index=0, tape_sha256=TAPE)
        self.equal(learner.pending["states"], np.concatenate([r["features"][:-1] for r in saved]))
        learner.load_state_dict(learner.state_dict())

    def test_nonfinite_or_wrong_shape_bootstraps_never_partially_admit(self):
        for factory in (self.warm, self.tail):
            for bad in (torch.zeros(1), torch.zeros((64, 1)), torch.full((64,), float("nan"))):
                learner = factory()
                learner.model.forward = Mock(return_value=bad)
                with self.assertRaises(ValueError):
                    if isinstance(learner, WarmLearner):
                        learner.admit_episode(episode())
                    else:
                        learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
                self.assertIsNone(learner.pending)
                self.assertEqual(learner.admissions, [])
                learner.before_optimizer.assert_not_called()

    def test_pending_snapshot_roundtrip_all_arms_and_owned_restore(self):
        learners = [self.warm(), self.warm("self_only"), self.tail(),
                    self.tail("self_only_td"), self.tail("graph_mc")]
        for learner in learners:
            self.fake_forward(learner)
            if isinstance(learner, WarmLearner):
                learner.admit_episode(episode())
            else:
                learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
            raw, sha = learner.snapshot()
            restored = type(learner).from_bytes(raw, expected_sha256=sha,
                before_forward=Mock(), before_optimizer=Mock())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
            self.equal(restored.state_dict(), learner.state_dict())
            state = learner.state_dict()
            restored.load_state_dict(state)
            before = restored.state_dict()
            state["pending"]["states"][:] = 999
            state["admissions"][0]["source_hashes"][0] = "d" * 64
            state["model"]["head.2.bias"].fill_(999)
            self.equal(restored.state_dict(), before)

    def test_restore_tamper_rejected_transactionally(self):
        learner = self.tail("graph_mc")
        learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
        before = learner.state_dict()
        mutations = [
            lambda s: s.update(format=WarmLearner.format),
            lambda s: s.update(ancestor_sha256=ANCESTOR),
            lambda s: s.update(continuation_sha256="c" * 64),
            lambda s: s.update(feature_dim=30), lambda s: s.update(seed=9),
            lambda s: s.update(updates=1), lambda s: s.update(updates=True),
            lambda s: s["counts"].update(optimizer_steps=-1),
            lambda s: s["config"].update(batch_size=64.),
            lambda s: s["pending"].update(completed=1),
            lambda s: s["pending"].update(targets=s["pending"]["targets"].astype(np.float32)),
            lambda s: s["pending"]["states"].__setitem__(0, 999),
            lambda s: s["pending"]["targets"].__setitem__(0, 999),
            lambda s: s["pending"]["targets"].__setitem__(0, np.nan),
            lambda s: s["admissions"][0].update(pending_sha256="0" * 64),
            lambda s: s["admissions"][0]["records"][0].update(continuation_sha256="c" * 64),
            lambda s: s["admissions"].clear(),
            lambda s: s["model"].update({"encoder.adjacency": torch.eye(4)}),
            lambda s: s["optimizer"]["param_groups"][0].update(lr=.001),
            lambda s: s.update(rng={"bit_generator": "wrong"}),
        ]
        for i, mutate in enumerate(mutations):
            bad = copy.deepcopy(before)
            mutate(bad)
            with self.subTest(case=i), self.assertRaises(ValueError):
                learner.load_state_dict(bad)
            self.equal(learner.state_dict(), before)

    def test_adam_state_shape_dtype_finiteness_and_steps_are_hard_validated(self):
        learner = self.completed_warm()
        before = learner.state_dict()
        for name, value in (("exp_avg", torch.zeros(1)),
                            ("exp_avg_sq", torch.full((32, 31), float("nan"))),
                            ("exp_avg_sq", -torch.ones((32, 31))),
                            ("exp_avg", torch.zeros((32, 31), dtype=torch.float64)),
                            ("step", torch.tensor(1535.)), ("step", torch.tensor([1536.]))):
            bad = copy.deepcopy(before)
            bad["optimizer"]["state"][0][name] = value
            with self.assertRaises(ValueError):
                learner.load_state_dict(bad)
            self.equal(learner.state_dict(), before)
        learner.optimizer.state.clear()
        with self.assertRaisesRegex(ValueError, "moments"):
            learner.load_state_dict(learner.state_dict())

    def test_matching_warm_forks_reset_adam_rng_counts_but_not_own_weights(self):
        for architecture, roles in (("graph", ("graph_td", "graph_mc")),
                                    ("self_only", ("self_only_td",))):
            warm = self.completed_warm(architecture)
            raw, sha = warm.snapshot()
            for role in roles:
                fork = TailLearner.fork_weights(raw, tail_config(STUDY, role, ANCESTOR),
                    seed=11, expected_sha256=sha, before_forward=Mock(), before_optimizer=Mock())
                self.equal(fork.model.state_dict(), warm.model.state_dict())
                self.assertEqual(fork.ancestor_sha256, sha)
                self.assertEqual(fork.continuation_sha256, ANCESTOR)
                self.assertNotEqual(sha, ANCESTOR)
                self.assertEqual(fork.updates, 0)
                self.assertFalse(fork.optimizer.state)
                self.assertEqual(fork.counts, dict(forwards=0, optimizer_steps=0))
                self.assertEqual(fork.admissions, [])
                self.assertIsNone(fork.pending)
                self.equal(fork.rng.bit_generator.state, np.random.default_rng(12).bit_generator.state)
                fork.before_forward.assert_not_called()
                fork.before_optimizer.assert_not_called()

    def test_no_trained_graph_to_self_only_conversion_or_historical_forks(self):
        warm = self.completed_warm()
        raw, sha = warm.snapshot()
        with self.assertRaisesRegex(ValueError, "corresponding"):
            TailLearner.fork_weights(raw, tail_config(STUDY, "self_only_td", ANCESTOR),
                seed=11, expected_sha256=sha, before_forward=Mock(), before_optimizer=Mock())
        for changed in (dict(format="capacity-value-comparison-td-v1"),
                        dict(config=dict(warm.config, architecture="self_only"))):
            state = warm.state_dict()
            state.update(changed)
            buffer = io.BytesIO()
            torch.save(state, buffer)
            with self.assertRaises(ValueError):
                WarmLearner.from_bytes(buffer.getvalue(), before_forward=Mock(), before_optimizer=Mock())
        with self.assertRaisesRegex(ValueError, "hash"):
            TailLearner.fork_weights(raw, tail_config(STUDY, "graph_td", ANCESTOR),
                seed=11, expected_sha256="0" * 64, before_forward=Mock(), before_optimizer=Mock())
        fresh, sha = self.warm().snapshot()
        with self.assertRaisesRegex(ValueError, "completed"):
            TailLearner.fork_weights(fresh, tail_config(STUDY, "graph_td", ANCESTOR),
                seed=11, expected_sha256=sha, before_forward=Mock(), before_optimizer=Mock())

    def test_float64_loss_and_failed_step_restore_all_except_charged_counts(self):
        for learner in (self.warm(), self.warm("self_only"), self.tail(),
                        self.tail("self_only_td"), self.tail("graph_mc")):
            self.fake_forward(learner)
            if isinstance(learner, WarmLearner):
                learner.admit_episode(episode())
            else:
                learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
            before = learner.state_dict()
            with patch.object(torch.nn.functional, "mse_loss", wraps=torch.nn.functional.mse_loss) as loss:
                with self.assertRaisesRegex(AssertionError, "real optimizer forbidden"):
                    learner.update()
                self.assertEqual(loss.call_args.args[0].dtype, torch.float64)
                self.assertEqual(loss.call_args.args[1].dtype, torch.float64)
            before["counts"]["forwards"] += 1
            before["counts"]["optimizer_steps"] += 1
            self.equal(learner.state_dict(), before)

    def test_hook_denial_does_not_invoke_optimizer_or_consume_sampling_state(self):
        learner = self.tail("graph_mc")
        learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
        self.fake_forward(learner)
        before = learner.state_dict()
        learner.before_optimizer = Mock(side_effect=PermissionError("denied"))
        with self.assertRaises(PermissionError):
            learner.update()
        before["counts"]["forwards"] += 1
        self.equal(learner.state_dict(), before)
        self.forbidden_step.assert_not_called()

    def test_mocked_world_boundary_and_exact_caps_without_real_optimizer(self):
        learner = self.tail("graph_mc")
        # Synthesize prior progress and metadata, not 767 optimizer steps.
        for index in range(24):
            learner.admit_tails(world(index), world_index=index, tape_sha256=TAPE)
            if index < 23:
                learner.pending = None
                learner.updates += 32
        learner.pending["completed"] = 31
        self.artificial_adam(learner, 767)
        self.fake_forward(learner)
        learner.load_state_dict(learner.state_dict())
        def moments_only():
            for entry in learner.optimizer.state.values():
                entry["step"].fill_(768.)
        with patch.object(learner.optimizer, "step", side_effect=moments_only):
            receipt = learner.update()
        self.assertEqual(receipt["update"], 768)
        self.assertEqual(receipt["cohort_update"], 32)
        self.assertIsNone(learner.pending)
        learner.load_state_dict(learner.state_dict())
        with self.assertRaises(ValueError):
            learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
        with self.assertRaises(RuntimeError):
            learner.update()
        warm = self.completed_warm()
        with self.assertRaises(ValueError):
            warm.admit_episode(episode())
        with self.assertRaises(RuntimeError):
            warm.update()
        self.forbidden_step.assert_not_called()

    def test_pending_and_repeated_world_rejected(self):
        learner = self.tail("graph_mc")
        learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)
        with self.assertRaises(ValueError):
            learner.admit_tails(world(1), world_index=1, tape_sha256=TAPE)
        learner.pending = None
        self.artificial_adam(learner, 32)
        with self.assertRaisesRegex(ValueError, "already admitted"):
            learner.admit_tails(world(), world_index=0, tape_sha256=TAPE)

    def test_raw_cost_residuals_and_endpoint_terminal_masking(self):
        learner = self.warm()
        self.fake_forward(learner)
        x = np.zeros((3, 4, 31), dtype=np.float32)
        self.equal(learner.residuals(x), np.full(3, 125000., dtype=np.float64))
        h = np.array([1., 2., 3.])
        self.equal(learner.score_endpoints(x, h, terminal=np.array([False, True, False])),
                   np.array([125001., 0., 125003.]))
        learner.before_forward.reset_mock()
        self.equal(learner.score_endpoints(x, h, terminal=np.ones(3, dtype=bool)), np.zeros(3))
        learner.before_forward.assert_not_called()
        for bad in (np.array([True]), np.array([0., 1., 0.])):
            with self.assertRaises(ValueError):
                learner.score_endpoints(x, h, terminal=bad)
        learner.before_optimizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
