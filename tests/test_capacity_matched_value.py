"""Artificial tensors/snapshots only: no fitted models, real updates or worlds."""

import copy
from contextlib import ExitStack
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.baselines.capacity_matched_value import (
    CapacityMatchedValue, MATCHED_FLAT_HIDDEN_SIZES,
)
from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.capacity_native import CapacityPilotEnv
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_features import FEATURE_NAMES, PublicPatientForecast
from src.rl.capacity_value_learner import CapacityValueLearner, td_rows


FEATURE_DIM = len(FEATURE_NAMES)
# Derive the counts without constructing a model or using any scientific config.
GRAPH_PARAMETERS = (31+1)*32 + 2*(32+1)*32 + (32+1)
FLAT_PARAMETERS = (4*31+1)*21 + (21+1)*23 + (23+1)


def artificial_rows():
    features = np.arange(65*4*FEATURE_DIM, dtype=np.float32).reshape(65, 4, FEATURE_DIM)/1000
    return td_rows(features, np.arange(65, dtype=np.float64)*10,
                   np.arange(1, 65, dtype=np.float64), scale=100.)


class ArtificialOnlyCase(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for cls in {value for value in vars(torch.optim).values() if isinstance(value, type)}:
            if issubclass(cls, torch.optim.Optimizer):
                stack.enter_context(patch.object(
                    cls, "step", side_effect=AssertionError("real optimizer forbidden")))
        for cls in (CapacityPilotEnv, PublicPatientForecast):
            for method in ("__init__", "step"):
                stack.enter_context(patch.object(
                    cls, method, side_effect=AssertionError("environment/forecast forbidden")))

    def learner(self, architecture="flat", *, seed="7", **overrides):
        config = dict(architecture=architecture, width=32,
                      flat_hidden_sizes=list(MATCHED_FLAT_HIDDEN_SIZES), lr=.001,
                      cost_scale=100., updates_per_world=4, batch_size=8,
                      gradient_norm_cap=1.)
        config.update(overrides)
        return CapacityValueComparisonLearner(
            config, seed=seed, feature_dim=FEATURE_DIM,
            before_forward=lambda *_: None, before_optimizer=lambda *_: None)

    def populated_learner(self, architecture):
        learner = self.learner(architecture)
        with torch.no_grad():
            for i, parameter in enumerate(learner.model.parameters()):
                parameter.fill_((i+1)/100.)
        learner.admit_episode(artificial_rows())
        learner.pending["completed"] = 1
        learner.updates = 3
        learner.counts = dict(forwards=5, optimizer_steps=4)
        # Populate every Adam entry directly: never call the real step method.
        for i, parameter in enumerate(learner.model.parameters()):
            learner.optimizer.state[parameter] = dict(
                step=torch.tensor(3., dtype=torch.float32),
                exp_avg=torch.full_like(parameter, (i+1)/1000.),
                exp_avg_sq=torch.full_like(parameter, (i+1)/100.))
        learner.optimizer.param_groups[0]["lr"] = .0007
        learner.rng.integers(0, 64, size=9)
        return learner

    def assertTreeEqual(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assertTreeEqual(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(actual, expected)


class FlatModelTests(ArtificialOnlyCase):
    def test_parameter_arithmetic_matches_instantiated_trainable_counts(self):
        self.assertEqual(FEATURE_DIM, 31)
        self.assertEqual(MATCHED_FLAT_HIDDEN_SIZES, (21, 23))
        self.assertEqual((GRAPH_PARAMETERS, FLAT_PARAMETERS), (3169, 3155))
        self.assertLessEqual(abs(FLAT_PARAMETERS-GRAPH_PARAMETERS)/GRAPH_PARAMETERS, .01)
        for model, expected in ((CapacityTerminalValue(31, width=32), GRAPH_PARAMETERS),
                                (CapacityMatchedValue(31), FLAT_PARAMETERS)):
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)
        flat = CapacityMatchedValue(31)
        self.assertEqual(list(flat.buffers()), [])
        self.assertEqual([(m.in_features, m.out_features) for m in flat.modules()
                          if isinstance(m, torch.nn.Linear)],
                         [(124, 21), (21, 23), (23, 1)])

    def test_cpu_float32_zero_head_batch_shape_and_initial_gradient(self):
        model = CapacityMatchedValue(FEATURE_DIM)
        self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu"
                            for p in model.parameters()))
        self.assertEqual(torch.count_nonzero(model.head.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.head.bias).item(), 0)
        for batch in (0, 1, 3):
            values = model(torch.arange(batch*124, dtype=torch.float32).reshape(batch, 4, 31))
            self.assertEqual(tuple(values.shape), (batch,))
            self.assertEqual(values.dtype, torch.float32)
            self.assertEqual(values.device.type, "cpu")
            self.assertTrue(torch.equal(values, torch.zeros(batch)))
        loss = (model(torch.ones(2, 4, 31))-1).square().mean()
        loss.backward()
        self.assertNotEqual(model.head.bias.grad.item(), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(all(torch.count_nonzero(p.grad).item() == 0
                            for p in model.encoder.parameters()))

    def test_flatten_preserves_every_public_site_and_feature_in_order(self):
        model = CapacityMatchedValue(FEATURE_DIM)
        received = []
        hook = model.encoder[0].register_forward_pre_hook(
            lambda _module, args: received.append(args[0].detach().clone()))
        self.addCleanup(hook.remove)
        # Noncontiguous input also preserves site-major order, without pooling.
        x = torch.arange(2*31*4, dtype=torch.float32).reshape(2, 31, 4).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        model(x)
        self.assertEqual(len(received), 1)
        torch.testing.assert_close(received[0], x.flatten(start_dim=1), rtol=0, atol=0)

    def test_artificial_nonzero_head_connects_all_inputs_and_all_parameters(self):
        model = CapacityMatchedValue(FEATURE_DIM)
        # Positive artificial weights expose connectivity obscured by the zero head.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(.01)
        x = torch.ones(2, 4, 31, requires_grad=True)
        model(x).sum().backward()
        self.assertTrue(torch.all(x.grad > 0))
        self.assertTrue(all(p.grad is not None and torch.all(p.grad > 0)
                            for p in model.parameters()))
        baseline = model(x.detach())
        for site in range(4):
            changed = x.detach().clone()
            changed[:, site, :] += 1
            self.assertTrue(torch.all(model(changed) > baseline))

    def test_input_shape_type_device_and_finiteness_are_enforced(self):
        model = CapacityMatchedValue(FEATURE_DIM)
        invalid = [None, np.zeros((1, 4, 31), dtype=np.float32),
                   torch.zeros(4, 31), torch.zeros(1, 124), torch.zeros(1, 3, 31),
                   torch.zeros(1, 4, 30), torch.zeros(1, 4, 31, 1),
                   torch.zeros(1, 4, 31, dtype=torch.float64),
                   torch.zeros(1, 4, 31, dtype=torch.int64),
                   torch.zeros(1, 4, 31, device="meta"),
                   torch.full((1, 4, 31), float("nan")),
                   torch.full((1, 4, 31), float("inf"))]
        for i, x in enumerate(invalid):
            with self.subTest(case=i), self.assertRaisesRegex(ValueError, "public beliefs"):
                model(x)

    def test_invalid_feature_schema_or_hidden_widths_rejected(self):
        for feature_dim in (30, 32, True, 31.):
            with self.subTest(feature_dim=feature_dim), self.assertRaises(ValueError):
                CapacityMatchedValue(feature_dim)
        for widths in (None, [], (), [16, 0], [-1], [True], [16., 26], "16,26,26"):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                CapacityMatchedValue(31, hidden_sizes=widths)


class ComparisonLearnerTests(ArtificialOnlyCase):
    def test_td_snapshot_and_recovery_methods_are_inherited_unchanged(self):
        for name in ("_forward", "residuals", "admit_episode", "update", "state_dict",
                     "load_state_dict", "snapshot"):
            self.assertIs(getattr(CapacityValueComparisonLearner, name),
                          getattr(CapacityValueLearner, name))
        self.assertIs(CapacityValueComparisonLearner.from_bytes.__func__,
                      CapacityValueLearner.from_bytes.__func__)
        self.assertNotEqual(CapacityValueComparisonLearner.format, CapacityValueLearner.format)

    def test_architecture_selection_seed_and_global_rng_preservation(self):
        for architecture, cls in (("flat", CapacityMatchedValue), ("graph", CapacityTerminalValue)):
            with self.subTest(architecture=architecture):
                global_rng = torch.random.get_rng_state().clone()
                learner = self.learner(architecture, seed="7")
                self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
                self.assertIsInstance(learner.model, cls)
                self.assertTreeEqual(learner.state_dict(), self.learner(architecture, seed=7).state_dict())
                self.assertEqual(learner.optimizer.param_groups[0]["foreach"], False)
                self.assertEqual(learner.updates, 0)
                self.assertIsNone(learner.pending)
                self.assertEqual(learner.counts, dict(forwards=0, optimizer_steps=0))
                self.assertFalse(learner.optimizer.state)
                np.testing.assert_array_equal(learner.residuals(np.ones((2, 4, 31), dtype=np.float32)),
                                              np.zeros(2))

    def test_explicit_architecture_widths_and_schema_required_before_construction(self):
        invalid = [dict(architecture=value) for value in (None, "mlp", "", 1)]
        invalid += [dict(flat_hidden_sizes=value) for value in (None, [], [0], [True], [16.])]
        invalid += [dict(architecture="graph", width=value) for value in (None, 0, True, 32.)]
        with patch.object(torch.random, "fork_rng", side_effect=AssertionError("construction forbidden")):
            for kwargs in invalid:
                with self.subTest(config=kwargs), self.assertRaises(ValueError):
                    self.learner(**kwargs)
            with self.assertRaises(ValueError):
                CapacityValueComparisonLearner({}, seed=7, feature_dim=31,
                                               before_forward=None, before_optimizer=None)
            with self.assertRaises(ValueError):
                CapacityValueComparisonLearner(dict(architecture="graph", width=32),
                                               seed=7, feature_dim=30,
                                               before_forward=None, before_optimizer=None)

    def test_config_is_owned_and_flat_widths_are_explicit(self):
        widths = list(MATCHED_FLAT_HIDDEN_SIZES)
        learner = self.learner(flat_hidden_sizes=widths)
        widths[0] = 999
        self.assertEqual(learner.config["flat_hidden_sizes"], [21, 23])
        custom = self.learner(flat_hidden_sizes=[12, 10, 8])
        self.assertEqual(custom.model.hidden_sizes, (12, 10, 8))

    def test_inherited_public_td_target_and_residual_scaling(self):
        for architecture in ("flat", "graph"):
            with self.subTest(architecture=architecture):
                learner = self.learner(architecture)
                with torch.no_grad():
                    list(learner.model.parameters())[-1].fill_(1.25)
                np.testing.assert_array_equal(learner.residuals(np.ones((2, 4, 31), dtype=np.float32)),
                                              np.full(2, 125.))
                rows = artificial_rows()
                learner.admit_episode(rows)
                expected = (rows["observed_cost"]
                            + np.where(rows["done"], 0., rows["bootstrap_base"]+1.25) - rows["base"])
                np.testing.assert_allclose(learner.pending["targets"], expected, rtol=1e-6)
                self.assertEqual(learner.updates, 0)
                self.assertFalse(learner.optimizer.state)

    def test_complete_in_memory_roundtrip_and_independent_copies(self):
        for architecture in ("flat", "graph"):
            with self.subTest(architecture=architecture):
                source = self.populated_learner(architecture)
                before = source.state_dict()
                raw, digest = source.snapshot()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                global_rng = torch.random.get_rng_state().clone()
                callbacks = []
                restored = CapacityValueComparisonLearner.from_bytes(
                    raw, before_forward=lambda *args: callbacks.append(args),
                    before_optimizer=lambda *_: self.fail("optimizer callback forbidden"))
                self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
                self.assertTreeEqual(restored.state_dict(), before)
                self.assertEqual(callbacks, [])
                model_parameters = list(restored.model.parameters())
                self.assertEqual(len(restored.optimizer.state), len(model_parameters))
                self.assertEqual([id(p) for p in restored.optimizer.param_groups[0]["params"]],
                                 [id(p) for p in model_parameters])
                np.testing.assert_array_equal(restored.rng.integers(0, 64, 32),
                                              source.rng.integers(0, 64, 32))
                exported = restored.state_dict()
                exported["pending"]["targets"][:] = -999
                source.pending["states"][:] = -999
                with torch.no_grad():
                    for parameter in source.model.parameters():
                        parameter.fill_(-999)
                    next(iter(source.optimizer.state.values()))["exp_avg"].fill_(-999)
                self.assertTreeEqual(restored.pending, before["pending"])
                self.assertTreeEqual(restored.model.state_dict(), before["model"])
                self.assertTreeEqual(restored.optimizer.state_dict(), before["optimizer"])
                clean = self.learner(architecture)
                clean.load_state_dict(before)
                self.assertTreeEqual(clean.state_dict(), before)

    def test_wrong_architecture_or_metadata_rejects_before_live_mutation(self):
        for architecture, other in (("flat", "graph"), ("graph", "flat")):
            target = self.populated_learner(architecture)
            prior = target.state_dict()
            wrong_arch = self.populated_learner(other).state_dict()
            wrong_format = copy.deepcopy(prior)
            wrong_format["format"] = CapacityValueLearner.format
            missing_arch = copy.deepcopy(prior)
            del missing_arch["config"]["architecture"]
            wrong_width = copy.deepcopy(prior)
            key = "width" if architecture == "graph" else "flat_hidden_sizes"
            wrong_width["config"][key] = 16 if architecture == "graph" else [21, 24]
            wrong_schema = copy.deepcopy(prior)
            wrong_schema["feature_dim"] = 30
            live = (target.model, target.optimizer, target.rng)
            for state in (wrong_arch, wrong_format, missing_arch, wrong_width, wrong_schema):
                with self.subTest(architecture=architecture, config=state["config"]):
                    with patch.object(type(target.model), "load_state_dict",
                                      side_effect=AssertionError("model load forbidden")), \
                            patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer forbidden")):
                        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                            target.load_state_dict(state)
                    self.assertEqual((target.model, target.optimizer, target.rng), live)
                    self.assertTreeEqual(target.state_dict(), prior)

    def test_malformed_recovery_payload_is_atomic(self):
        for architecture in ("flat", "graph"):
            learner = self.populated_learner(architecture)
            prior = learner.state_dict()
            parameter_key = next(iter(dict(learner.model.named_parameters())))
            for defect in ("model", "optimizer", "rng", "pending", "counts", "updates", "missing_pending"):
                with self.subTest(architecture=architecture, defect=defect):
                    state = copy.deepcopy(prior)
                    state["model"][parameter_key].fill_(.75)
                    if defect == "model":
                        state["model"][parameter_key].fill_(float("nan"))
                    elif defect == "optimizer":
                        next(iter(state["optimizer"]["state"].values()))["exp_avg"].fill_(float("inf"))
                    elif defect == "rng":
                        state["rng"] = dict(bit_generator="invalid")
                    elif defect == "pending":
                        state["pending"]["targets"][0] = np.nan
                    elif defect == "counts":
                        state["counts"]["optimizer_steps"] = 0
                    elif defect == "updates":
                        state["updates"] = -1
                    else:
                        del state["pending"]
                    with self.assertRaises((ValueError, RuntimeError, KeyError)):
                        learner.load_state_dict(state)
                    self.assertTreeEqual(learner.state_dict(), prior)

    def test_failed_fake_step_restores_everything_except_durable_charges(self):
        for architecture in ("flat", "graph"):
            with self.subTest(architecture=architecture):
                learner = self.populated_learner(architecture)
                prior = learner.state_dict()
                callbacks = []
                learner.before_forward = lambda *args: callbacks.append(("forward", args))
                learner.before_optimizer = lambda *args: callbacks.append(("optimizer", args))

                def fake_failure():
                    with torch.no_grad():
                        next(learner.model.parameters()).fill_(23.)
                        next(iter(learner.optimizer.state.values()))["exp_avg"].fill_(42.)
                    learner.pending["completed"] += 1
                    learner.updates += 1
                    learner.rng.integers(0, 64, size=5)
                    raise RuntimeError("artificial failed step")

                with patch.object(learner.optimizer, "step", side_effect=fake_failure) as fake:
                    with self.assertRaisesRegex(RuntimeError, "artificial failed step"):
                        learner.update()
                self.assertEqual(fake.call_count, 1)
                self.assertEqual(callbacks, [("forward", ("value", 8)), ("optimizer", ("value", 8))])
                prior["counts"]["forwards"] += 1
                prior["counts"]["optimizer_steps"] += 1
                self.assertTreeEqual(learner.state_dict(), prior)

    def test_forbidden_step_and_failed_admission_preserve_recovery_contract(self):
        for architecture in ("flat", "graph"):
            for deny_admission in (False, True):
                with self.subTest(architecture=architecture, deny_admission=deny_admission):
                    learner = self.populated_learner(architecture)
                    prior = learner.state_dict()
                    if deny_admission:
                        def deny(*_):
                            raise PermissionError("artificial denied admission")
                        learner.before_optimizer = deny
                    expected = PermissionError if deny_admission else AssertionError
                    message = "denied admission" if deny_admission else "real optimizer forbidden"
                    with self.assertRaisesRegex(expected, message):
                        learner.update()
                    prior["counts"]["forwards"] += 1
                    prior["counts"]["optimizer_steps"] += int(not deny_admission)
                    self.assertTreeEqual(learner.state_dict(), prior)

    def test_fake_noop_step_closes_pending_boundary_without_parameter_updates(self):
        for architecture in ("flat", "graph"):
            with self.subTest(architecture=architecture):
                learner = self.populated_learner(architecture)
                learner.pending["completed"] = learner.config["updates_per_world"]-1
                before = learner.state_dict()
                with patch.object(learner.optimizer, "step", return_value=None) as fake:
                    receipt = learner.update()
                self.assertEqual(fake.call_count, 1)
                self.assertEqual(receipt["update"], before["updates"]+1)
                self.assertEqual(receipt["cohort_update"], learner.config["updates_per_world"])
                self.assertEqual(len(receipt["indices"]), 8)
                self.assertIsNone(learner.pending)
                self.assertTreeEqual(learner.model.state_dict(), before["model"])
                self.assertTreeEqual(learner.optimizer.state_dict(), before["optimizer"])


if __name__ == "__main__":
    unittest.main()
