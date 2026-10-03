"""Bounded artificial tensors and fake backends; no scientific fits or hosts."""

import copy
from contextlib import ExitStack
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.baselines import capacity_completion_control_recovery2 as plain_module
from src.baselines import capacity_value_mpc as mpc_module
from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.capacity_native import CapacityPilotEnv, CapacityWorldTape
from src.rl.capacity_value_analysis import run_analysis
from src.rl.capacity_value_execution import build_runner, stream_manifest
from src.rl.capacity_value_features import FEATURE_NAMES
from src.rl.capacity_value_learner import CapacityValueLearner, td_rows
from src.rl.capacity_value_resources import (
    CapacityValueBudget, PHASES, merged_proposal, numeric_contract, worlds,
)
from src.rl.capacity_value_runner import CapacityValueRunner
from tests.test_capacity_completion_control import PROPOSAL, view
from tests.test_capacity_pilot_runner import FakeBudget, FakeController, FakeEnv


ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "experiments/configs/capacity_value_mpc_20261003.json").read_text())
ADMISSION = dict(verified=True, scope="capacity-value-mpc-v1")


def artificial_cohort():
    return dict(features=np.broadcast_to(np.arange(65, dtype=np.float32)[:, None, None],
                                        (65, 4, 2)).copy(),
                heuristics=np.arange(65, dtype=np.float64) * 100 + 1000,
                costs=np.arange(1, 65, dtype=np.float64))


class ZeroUpdateCase(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.enterContext = self.contexts.enter_context
        # All true optimizers fail closed, including any added by future code.
        for cls in {value for value in vars(torch.optim).values() if isinstance(value, type)}:
            if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
                self.enterContext(patch.object(cls, "step", side_effect=AssertionError("real optimizer forbidden")))
        for method in ("__init__", "step"):
            self.enterContext(patch.object(CapacityPilotEnv, method,
                                          side_effect=AssertionError("real patient environment forbidden")))
        for method in ("__init__", "step"):
            self.enterContext(patch.object(plain_module.PublicPatientForecast, method,
                                          side_effect=AssertionError("real predictor forbidden")))

    def learner(self, **overrides):
        config = {**STUDY["value"], "width": 4, **overrides}
        return CapacityValueLearner(config, seed=7, feature_dim=2,
                                    before_forward=lambda *_: None, before_optimizer=lambda *_: None)

    def assertTreeEqual(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assertTreeEqual(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(actual, expected)


class TDTests(ZeroUpdateCase):
    def test_eight_step_costs_include_settlement_and_mask_exactly_last_eight(self):
        data = artificial_cohort()
        rows = td_rows(**data, horizon=8, scale=100.)
        ends = np.minimum(np.arange(64) + 8, 64)
        np.testing.assert_array_equal(rows["next_states"][:, 0, 0], ends)
        np.testing.assert_array_equal(rows["states"], data["features"][:-1])
        np.testing.assert_array_equal(rows["done"], np.arange(64) >= 56)
        expected = [sum(range(t + 1, min(t + 8, 64) + 1)) / 100 for t in range(64)]
        np.testing.assert_allclose(rows["observed_cost"], expected)
        self.assertAlmostEqual(rows["observed_cost"][47], sum(range(48, 56)) / 100)
        self.assertAlmostEqual(rows["observed_cost"][63], .64)

    def test_residual_target_subtracts_current_heuristic_once_and_freezes_bootstrap(self):
        learner = self.learner(cost_scale=100.)
        with torch.no_grad():
            learner.model.head[-1].bias.fill_(2.)
        data = artificial_cohort()
        rows = td_rows(**data, horizon=8, scale=100.)
        calls = []
        learner.before_forward = lambda name, n: calls.append((name, n))
        learner.admit_episode(rows)
        expected = rows["observed_cost"] + np.where(rows["done"], 0., rows["bootstrap_base"] + 2.) - rows["base"]
        np.testing.assert_allclose(learner.pending["targets"], expected, rtol=1e-6)
        np.testing.assert_allclose(rows["base"] + learner.pending["targets"],
                                   rows["observed_cost"] + np.where(rows["done"], 0., rows["bootstrap_base"] + 2.),
                                   atol=4e-6)
        self.assertEqual(calls, [("value", 64)])
        targets = learner.pending["targets"].copy()
        with torch.no_grad():
            learner.model.head[-1].bias.fill_(99.)
        rows["states"][:] = -5
        np.testing.assert_array_equal(learner.pending["targets"], targets)
        self.assertEqual(learner.pending["states"][1, 0, 0], 1.)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            learner.admit_episode(rows)
        self.assertEqual(calls, [("value", 64)])
        self.assertEqual(learner.updates, 0)

    def test_incomplete_nonfinite_or_negative_cohorts_rejected(self):
        for field in ("features", "heuristics", "costs"):
            with self.subTest(field=field, defect="short"):
                data = artificial_cohort()
                data[field] = data[field][:-1]
                with self.assertRaisesRegex(ValueError, "64-step"):
                    td_rows(**data)
            with self.subTest(field=field, defect="nan"):
                data = artificial_cohort()
                data[field].flat[0] = np.nan
                with self.assertRaises(ValueError):
                    td_rows(**data)
        data = artificial_cohort()
        data["costs"][0] = -1
        with self.assertRaises(ValueError):
            td_rows(**data)

    def test_zero_head_shape_dtype_and_raw_cost_scaling(self):
        learner = self.learner(cost_scale=100.)
        x = np.arange(48 * 4 * 2, dtype=np.float32).reshape(48, 4, 2)
        np.testing.assert_array_equal(learner.residuals(x), np.zeros(48))
        self.assertIsInstance(learner.model, CapacityTerminalValue)
        with torch.no_grad():
            learner.model.head[-1].bias.fill_(1.25)
        np.testing.assert_array_equal(learner.residuals(x), np.full(48, 125.))
        for bad in (torch.zeros(4, 2), torch.zeros(1, 3, 2),
                    torch.zeros(1, 4, 2, dtype=torch.float64), torch.full((1, 4, 2), float("nan"))):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(ValueError):
                learner.model(bad)


class SnapshotTests(ZeroUpdateCase):
    def test_snapshot_restores_metadata_rng_optimizer_pending_and_owns_copies(self):
        learner = self.learner()
        learner.admit_episode(td_rows(**artificial_cohort()))
        learner.pending["completed"] = 3
        learner.updates = 3
        learner.counts["optimizer_steps"] = 3
        parameter = next(learner.model.parameters())
        # Artificial Adam metadata, not a fitted optimizer state.
        learner.optimizer.state[parameter] = dict(step=torch.tensor(3.),
            exp_avg=torch.full_like(parameter, .1), exp_avg_sq=torch.full_like(parameter, .2))
        learner.rng.integers(0, 64, size=9)
        before = learner.state_dict()
        raw, digest = learner.snapshot()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        calls = []
        restored = CapacityValueLearner.from_bytes(raw, before_forward=lambda *v: calls.append(v),
                                                   before_optimizer=lambda *_: self.fail("optimizer callback"))
        self.assertTreeEqual(restored.state_dict(), before)
        np.testing.assert_array_equal(restored.rng.integers(0, 64, 64), learner.rng.integers(0, 64, 64))
        learner.pending["targets"][:] = -999
        self.assertTreeEqual(restored.pending, before["pending"])
        exported = restored.state_dict()
        exported["pending"]["states"][:] = -999
        self.assertTreeEqual(restored.pending, before["pending"])
        self.assertEqual(calls, [])

    def test_invalid_metadata_model_or_rng_restore_is_atomic(self):
        learner = self.learner()
        learner.admit_episode(td_rows(**artificial_cohort()))
        prior = learner.state_dict()
        for defect in ("format", "config", "feature_dim", "model", "rng"):
            with self.subTest(defect=defect):
                state = copy.deepcopy(prior)
                state["model"]["head.2.bias"].fill_(17.)
                if defect == "format":
                    state["format"] = "incorrect"
                elif defect == "config":
                    state["config"]["cost_scale"] *= 2
                elif defect == "feature_dim":
                    state["feature_dim"] += 1
                elif defect == "model":
                    state["model"]["head.2.bias"].fill_(float("nan"))
                else:
                    state["rng"] = {"bit_generator": "not-a-generator"}
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    learner.load_state_dict(state)
                self.assertTreeEqual(learner.state_dict(), prior)

    def test_missing_pending_rejects_before_swapping_any_live_state(self):
        learner = self.learner()
        learner.admit_episode(td_rows(**artificial_cohort()))
        prior = learner.state_dict()
        malformed = copy.deepcopy(prior)
        malformed["model"]["head.2.bias"].fill_(17.)
        del malformed["pending"]
        with self.assertRaises((KeyError, ValueError)):
            learner.load_state_dict(malformed)
        self.assertTreeEqual(learner.state_dict(), prior)

    def test_failed_fake_step_rolls_back_state_but_keeps_durable_charges(self):
        proposal = merged_proposal(PROPOSAL, STUDY)
        with tempfile.TemporaryDirectory() as tmp:
            budget = CapacityValueBudget(Path(tmp) / "ledger", proposal, clock=lambda: 0.)
            self.addCleanup(budget.close)
            budget.enter(PHASES[0], PHASES[0])
            learner = self.learner()
            learner.before_forward = lambda *_: budget.debit({"neural_forward_module_calls": 1})
            learner.before_optimizer = lambda _, n: budget.debit(dict(
                value_optimizer_steps=1, total_optimizer_steps=1, optimizer_example_presentations=n))
            learner.admit_episode(td_rows(**artificial_cohort()))
            prior = learner.state_dict()

            def fake_failed_step():
                self.assertEqual(budget.counts["value_optimizer_steps"], 1)
                with torch.no_grad():
                    next(learner.model.parameters()).fill_(23.)
                raise RuntimeError("fake step failed")

            with patch.object(learner.optimizer, "step", side_effect=fake_failed_step) as fake:
                with self.assertRaisesRegex(RuntimeError, "fake step failed"):
                    learner.update()
            self.assertEqual(fake.call_count, 1)
            after = learner.state_dict()
            charged = after.pop("counts")
            prior.pop("counts")
            self.assertTreeEqual(after, prior)
            self.assertEqual(charged, dict(forwards=2, optimizer_steps=1))
            self.assertEqual(budget.counts["neural_forward_module_calls"], 2)
            self.assertEqual(budget.counts["total_optimizer_steps"], 1)
            self.assertEqual(budget.counts["optimizer_example_presentations"], 64)
            ledger = [json.loads(line) for line in (Path(tmp) / "ledger/budget.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row.get("charges", {}).get("value_optimizer_steps", 0) for row in ledger), 1)

    def test_guarded_real_step_cannot_run_and_pre_admission_failure_does_not_charge_step(self):
        for fail_before_step in (False, True):
            with self.subTest(fail_before_step=fail_before_step):
                learner = self.learner()
                learner.admit_episode(td_rows(**artificial_cohort()))
                prior = learner.state_dict()
                if fail_before_step:
                    def deny(*_):
                        raise PermissionError("fake admission denied")
                    learner.before_optimizer = deny
                with self.assertRaises((PermissionError, AssertionError)):
                    learner.update()
                after = learner.state_dict()
                self.assertEqual(after.pop("counts"), dict(forwards=2, optimizer_steps=0 if fail_before_step else 1))
                prior.pop("counts")
                self.assertTreeEqual(after, prior)


class MockForecast:
    instances = []
    terminal_epoch = None

    def __init__(self, public, proposal, filter_, lifecycle, quantile):
        self.epoch, self.quantile = public.common.epoch, quantile
        self.initial, self.calls = None, []
        self.total_components = {}
        self.instances.append(self)

    def adaptive(self):
        return np.full(4, .5)

    def step(self, hours, base):
        if self.initial is None:
            self.initial = float(hours[0])
        self.calls.append(np.asarray(hours).copy())
        self.epoch = self.epoch + 1 if self.terminal_epoch is None else self.terminal_epoch
        return float(sum(hours)) + 10 * self.quantile

    def terminal_value(self):
        return 100 + self.initial * 3 + self.quantile


def mock_forecast_features(model):
    return np.tile(np.asarray([model.initial, model.quantile], dtype=np.float32), (4, 1))


class PlannerTests(ZeroUpdateCase):
    def control(self, cls, **kwargs):
        controller = cls(copy.deepcopy(PROPOSAL), scientific=True,
                         base_control=lambda *_: np.zeros(16), **kwargs)
        self.enterContext(patch.object(controller, "observe"))
        candidates = [(np.full(4, i / 4), False) for i in range(8)]
        candidates += [(h.copy(), True) for h, _ in candidates]
        self.enterContext(patch.object(controller, "candidates", return_value=candidates))
        return controller

    def setUp(self):
        super().setUp()
        MockForecast.instances, MockForecast.terminal_epoch = [], None
        self.enterContext(patch.object(plain_module, "PublicPatientForecast", MockForecast))
        self.enterContext(patch.object(mpc_module, "PublicPatientForecast", MockForecast))
        self.enterContext(patch.object(mpc_module, "forecast_features", mock_forecast_features))

    def test_zero_head_matches_corrected_plain_mpc_on_same_mock_predictor(self):
        learner = self.learner()
        calls = []
        learner.before_forward = lambda *args: calls.append(args)
        plain = self.control(plain_module.CapacityCompletionControl)
        value = self.control(mpc_module.CapacityValueMPC, value=learner)
        admissions = [[], []]
        expected = plain.act(view(), role="id_mpc", before_query=admissions[0].append)
        actual = value.act(view(), before_query=admissions[1].append)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(value.last_plan["scores"], plain.last_plan["scores"])
        self.assertEqual(value.last_plan["chosen"], plain.last_plan["chosen"])
        self.assertEqual(admissions[0], admissions[1])
        self.assertEqual(len(admissions[1]), 384)
        self.assertEqual(calls, [("value", 48)])
        self.assertEqual(learner.updates, 0)

    def test_candidate_major_quantile_order_one_batched_residual_and_one_base(self):
        calls = []

        class Value:
            def residuals(self, x):
                calls.append(x.copy())
                return -1000 * x[:, 0, 0].astype(float) + x[:, 0, 1]

        controller = self.control(mpc_module.CapacityValueMPC, value=Value())
        admissions = []
        hours = controller.act(view(), before_query=admissions.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape, (48, 4, 2))
        q = PROPOSAL["id_mpc"]["response_quantiles"]
        np.testing.assert_allclose(calls[0][:, 0, 1], q * 16)
        np.testing.assert_allclose(calls[0][:, 0, 0], np.repeat(list(np.arange(8) / 4) * 2, 3))
        weights = np.asarray(PROPOSAL["id_mpc"]["summary_weights"])
        expected = []
        for candidate in range(16):
            initial = (candidate % 8) / 4
            prefix = 8 * 4 * initial if candidate < 8 else 2 * 4 * initial + 6 * 4 * .5
            expected.append(sum(weight * (prefix + 80 * quantile + 100 + 3 * initial + quantile
                                          - 1000 * initial + np.float32(quantile))
                                for quantile, weight in zip(q, weights)))
        np.testing.assert_allclose(controller.last_plan["scores"], expected)
        best = int(np.argmin(expected))
        self.assertEqual(controller.last_plan["chosen"], best)
        np.testing.assert_array_equal(hours, np.full(4, (best % 8) / 4))
        self.assertEqual(len(admissions), 384)
        self.assertEqual(len(MockForecast.instances), 48)
        self.assertTrue(all(len(model.calls) == 8 for model in MockForecast.instances))

    def test_terminal_mask_and_fixed_or_tail_roles_never_score_value(self):
        class Value:
            def residuals(self, x):
                return np.full(len(x), 123456.)

        controller = self.control(mpc_module.CapacityValueMPC, value=Value())
        MockForecast.terminal_epoch = 64
        controller.act(view(), before_query=lambda _: None)
        self.assertEqual(controller.last_plan["heuristic_terminal_costs"], [0.] * 48)
        self.assertEqual(controller.last_plan["learned_terminal_residuals"], [0.] * 48)
        with patch.object(controller.value, "residuals", side_effect=AssertionError("unexpected scoring")):
            np.testing.assert_array_equal(controller.act(view(), role="fixed_allocation_reference"), [2.] * 4)
            np.testing.assert_array_equal(controller.act(view(48)), [0.] * 4)

    def test_failed_first_query_prevents_predictor_and_value_dispatch(self):
        controller = self.control(mpc_module.CapacityValueMPC)

        def deny(_):
            raise RuntimeError("query denied")

        with self.assertRaisesRegex(RuntimeError, "query denied"):
            controller.act(view(), before_query=deny)
        self.assertEqual(MockForecast.instances, [])


class ValueBudget(FakeBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {phase: dict.fromkeys(self.limits, 0) for phase in PHASES}
        self.phase, self.chunks, self.jobs = None, {}, []

    def debit(self, charges):
        super().debit(charges)
        for key, n in charges.items():
            self.phase_counts[self.phase][key] += n
            assert self.phase_counts[self.phase][key] <= self.contract["phase_limits"][self.phase][key]

    def job(self, identifier, kind):
        self.jobs.append((self.phase, identifier, kind))


class FakeValue:
    restores = []

    def __init__(self, config, *, seed, feature_dim, before_forward, before_optimizer):
        assert type(seed) is int
        self.config, self.seed, self.feature_dim = copy.deepcopy(config), seed, feature_dim
        self.forward, self.optimizer = before_forward, before_optimizer
        self.updates, self.pending = 0, None

    def admit_episode(self, rows):
        assert self.pending is None
        assert rows["done"].tolist() == [False] * 56 + [True] * 8
        self.forward("value", 64)
        self.pending = dict(targets=np.zeros(64, dtype=np.float32), completed=0)

    def update(self):
        self.forward("value", 64)
        self.optimizer("value", 64)
        self.updates += 1
        self.pending["completed"] += 1
        completed = self.pending["completed"]
        if self.pending["completed"] == self.config["updates_per_world"]:
            self.pending = None
        return dict(fake_accounting_only=True, update=self.updates, cohort_update=completed,
                    loss=0., gradient_norm=0.)

    def residuals(self, x):
        assert x.shape == (48, 4, len(FEATURE_NAMES))
        self.forward("value", 48)
        return np.zeros(len(x))

    def state_dict(self):
        return dict(config=self.config, seed=self.seed, feature_dim=self.feature_dim, updates=self.updates)

    def snapshot(self):
        assert self.pending is None
        raw = json.dumps(self.state_dict(), sort_keys=True).encode()
        return raw, hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_bytes(cls, raw, *, before_forward, before_optimizer):
        state = json.loads(raw)
        out = cls(state["config"], seed=state["seed"], feature_dim=state["feature_dim"],
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.updates = state["updates"]
        cls.restores.append(out.updates)
        return out


class ValueController(FakeController):
    def __init__(self, *args, value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value, self.last_plan = value, dict(fake_predictor=True)

    def act(self, public, role, **kwargs):
        hours = super().act(public, role, **kwargs)
        if self.value is not None and role == "id_mpc":
            self.value.residuals(np.zeros((48, 4, len(FEATURE_NAMES)), dtype=np.float32))
        return hours


class ReceiptEnv(FakeEnv):
    def __init__(self, *args):
        super().__init__(*args)
        self.requests = [[0.] * 4, [0.] * 4]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["support_public_receipt"]["applied_hours"] = self.requests.pop(0)
        self.requests.append(list(action.requested_hours))
        return observation, reward, done, info

    def settlement(self):
        return dict(super().settlement(), live_ids=[], pending_obligations=0,
                    resource_conservation=True)


@dataclass(frozen=True)
class FakeCommon:
    epoch: int


@dataclass(frozen=True)
class FakeOperations:
    patients: tuple = ()


@dataclass(frozen=True)
class FakePublic:
    common: FakeCommon
    operations: FakeOperations = FakeOperations()


def fake_features(public, controller):
    return np.full((4, len(FEATURE_NAMES)), public.common.epoch / 64, dtype=np.float32), 0.


class ResourceRunnerTests(ZeroUpdateCase):
    def test_real_json_metadata_streams_disjoint_including_model_replay_rng(self):
        proposal = json.loads(json.dumps(merged_proposal(PROPOSAL, STUDY)))
        manifest = stream_manifest(proposal)
        self.assertEqual(len(manifest["rows"]), 180)
        self.assertEqual(len(manifest["allocations"]), 1087)
        self.assertEqual(len(set(manifest["allocations"])), 1087)
        self.assertTrue(all(type(seed) is int for seed in manifest["allocations"]))
        bad = copy.deepcopy(proposal)
        seeds = bad["value_mpc_study"]["design"]["model_seeds"]
        seeds[1] = seeds[0] + 1
        with self.assertRaisesRegex(ValueError, "collision"):
            stream_manifest(bad)

    def test_exact_numeric_contract_and_disjoint_integer_seed_matrix(self):
        contract = numeric_contract(STUDY)
        expected = dict(trajectories=288, native_steps=18432, native_constructions=288,
            construction_triggered_resets=288, total_native_operations=19008, control_steps=13824,
            tail_steps=4608, value_optimizer_steps=4608, total_optimizer_steps=4608,
            optimizer_example_presentations=294912, neural_forward_module_calls=11664,
            planner_total_decisions=12096, planner_candidate_rollouts=580608,
            planner_total_model_epochs=4644864, estimator_receipt_updates=18432,
            estimator_hypothesis_transitions=1843200, initial_seals=3, final_seals=3)
        self.assertEqual(contract["limits"], expected)
        for key, total in expected.items():
            self.assertEqual(sum(row[key] for row in contract["phase_limits"].values()), total)
        self.assertEqual([contract["phase_limits"][p]["trajectories"] for p in PHASES], [72, 72, 144])
        self.assertEqual(contract["phase_limits"][PHASES[2]]["value_optimizer_steps"], 0)
        seeds = [w for phase in ("initial", "continuation", "evaluation")
                 for block in range(3) for w in worlds(STUDY, phase, block)]
        self.assertEqual(len(seeds), 180)
        self.assertEqual(len({w["seed"] for w in seeds}), 180)
        self.assertTrue(all(type(w["seed"]) is int for w in seeds))
        for field in ("native_steps", "native_operations", "value_optimizer_steps", "forwards",
                      "planner_epochs", "filter_transitions"):
            changed = copy.deepcopy(STUDY)
            changed["budget"][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                numeric_contract(changed)

    def test_real_entry_fake_backends_full_schedule_all_seals_before_144_evaluations(self):
        budget = ValueBudget()
        FakeValue.restores = []
        proposal = json.loads(json.dumps(merged_proposal(PROPOSAL, STUDY)))
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync", return_value=None), \
                patch.object(CapacityValueLearner, "__init__", side_effect=AssertionError("real runner learner forbidden")):
            root = Path(tmp)
            runner = build_runner(root, proposal, budget, admission=ADMISSION,
                value_factory=FakeValue, env_factory=ReceiptEnv, controller_factory=ValueController,
                tape_factory=lambda p, w: CapacityWorldTape(w, (), (), (), None),
                capture=lambda host: FakePublic(FakeCommon(host.t)), feature_adapter=fake_features,
                base_control=lambda *args, **kwargs: np.zeros(16))
            events, evaluation_starts = [], []
            real_status = runner.status

            def status(event, **values):
                events.append(event)
                if event == "trajectory_started" and runner.active["phase"] == "evaluation":
                    self.assertEqual(set(runner.final), {0, 1, 2})
                    self.assertEqual(set(runner.initial), {0, 1, 2})
                    self.assertIn("all_models_sealed", events)
                    self.assertEqual(budget.counts["total_optimizer_steps"], 4608)
                    self.assertEqual(budget.phase, PHASES[2])
                    evaluation_starts.append(copy.deepcopy(runner.active))
                return real_status(event, **values)

            with patch.object(runner, "status", side_effect=status):
                result = runner.run()
            self.assertEqual(result["trajectories"], 288)
            self.assertEqual(budget.counts, budget.limits)
            self.assertEqual(budget.phase_counts, budget.contract["phase_limits"])
            self.assertEqual(len(evaluation_starts), 144)
            self.assertEqual(FakeValue.restores.count(768), 39)
            self.assertEqual(FakeValue.restores.count(1536), 36)
            self.assertEqual(len(list((root / "tapes").glob("*.json"))), 180)
            summaries = [json.loads(p.read_text()) for p in (root / "summaries").glob("*.json")]
            self.assertEqual(len(summaries), 288)
            self.assertTrue(all(s["cost"] == 64. and s["settled"] for s in summaries))
            self.assertTrue(all(s["optimizer_updates_during_trajectory"] == 0 for s in summaries))
            analysis = run_analysis(root, proposal["value_mpc_study"])
            self.assertEqual(analysis["trajectory_count"], 288)
            self.assertEqual(analysis["raw_rows"], 18432)
            self.assertEqual(analysis["evaluation_trajectories"], 144)
            self.assertEqual(analysis["updates"], dict(initial=2304, continuation=2304))
            self.assertTrue(analysis["complete"])
            self.assertFalse(analysis["training_signal"])
            self.assertFalse(analysis["clinical_safety_established"])
            self.assertFalse(analysis["deployment_online_adaptation"])
            self.assertEqual(len(analysis["contrasts"]), 4)
            for conditions in analysis["contrasts"].values():
                for contrast in conditions.values():
                    self.assertEqual(contrast["n_worlds"], 12)
                    self.assertEqual(contrast["means"]["savings"], 0.)
                    self.assertEqual(contrast["means"]["extra_lost"], 0.)
            for block in range(3):
                for kind, updates in (("initial", 768), ("final", 1536)):
                    raw = (root / f"models/block{block}-{kind}.pt").read_bytes()
                    metadata = json.loads((root / f"models/block{block}-{kind}.json").read_text())
                    self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())
                    self.assertEqual(metadata["updates"], updates)
            first_eval = next(s for s in summaries if s["world"]["phase"] == "evaluation")
            with gzip.open(root / first_eval["raw_path"], "rt") as handle:
                raw_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(raw_rows), 64)
            self.assertTrue(all(row["requested_hours"] == [0.] * 4 for row in raw_rows[48:]))
            state_path = next((root / "states").glob("*-after-fit.pkl.gz"))
            with gzip.open(state_path, "rb") as handle:
                state = pickle.load(handle)
            self.assertEqual(state["format"], "capacity-value-runner-v1")
            self.assertFalse(state["resume_authorized"])
            before = copy.deepcopy(budget.counts)
            with self.assertRaises(PermissionError):
                runner._optimizer("value", 64)
            self.assertEqual(budget.counts, before)
            with self.assertRaisesRegex(RuntimeError, "single attempt"):
                runner.run()
            self.assertEqual(budget.counts, before)

    def test_unverified_admission_has_no_directory_or_backend_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absent"
            with self.assertRaises(PermissionError):
                build_runner(path, {}, None, admission={})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
