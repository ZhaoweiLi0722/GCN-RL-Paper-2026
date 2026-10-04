"""Saved metadata and complete fake scheduling; no native step/forward/optimizer."""

import ast
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

from src.baselines.capacity_completion_control import CompletionIntervalFilter
from src.baselines.capacity_value_mpc_recovery1 import CapacityValueMPCRecovery1
from src.env.capacity_planning import CapacityPlanningEnv
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl import capacity_value_comparison_recovery1_runner as recovery
from src.rl.capacity_native import CapacityPilotEnv, CapacityWorldTape
from src.rl.capacity_pilot_runner import trajectory_id
from src.rl.capacity_value_comparison_analysis import run_analysis
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_comparison_resources import ARCHITECTURES, PHASES, SCOPE, merged_proposal, numeric_contract, worlds
from src.rl.capacity_value_comparison_runner import CapacityValueComparisonRunner
from src.rl.capacity_value_features import FEATURE_NAMES
from tests.test_capacity_value_comparison import ComparisonBudget, STUDY
from tests.test_capacity_value_mpc import (
    ZeroUpdateCase, FakeValue, ReceiptEnv, ValueController, FakePublic, FakeCommon, fake_features, PROPOSAL,
)


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/"results/capacity_value_comparison_20261004/payload"
ADMISSION = dict(verified=True, scope=SCOPE, remaining_scope=recovery.RECOVERY_SCOPE)


class RemainingBudget(ComparisonBudget):
    def __init__(self, old_counts):
        super().__init__()
        self.limits = {k:v-old_counts[k] for k,v in self.limits.items()}
        for key, amount in (("planner_total_decisions", 1), ("planner_candidate_rollouts", 48),
                            ("planner_total_model_epochs", 384)):
            self.limits[key] += amount
        self.counts = dict.fromkeys(self.limits, 0)


class FakeRecoveredValue(FakeValue):
    loaded = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(self.seed+1)
        self.optimizer_state = dict(moment=0.)

    def state_dict(self):
        return dict(super().state_dict(), pending=copy.deepcopy(self.pending),
                    rng=copy.deepcopy(self.rng.bit_generator.state), optimizer=copy.deepcopy(self.optimizer_state))

    def load_state_dict(self, state):
        self.config = copy.deepcopy(state["config"])
        self.seed, self.feature_dim = state["seed"], state["feature_dim"]
        self.updates, self.pending = state["updates"], copy.deepcopy(state["pending"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.optimizer_state = copy.deepcopy(state["optimizer"])
        self.loaded.append(copy.deepcopy(state))

    @classmethod
    def from_bytes(cls, raw, *, before_forward, before_optimizer):
        state = json.loads(raw)
        out = cls(state["config"], seed=0, feature_dim=state["feature_dim"],
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.load_state_dict(state)
        return out

    def update(self):
        receipt = super().update()
        self.optimizer_state["moment"] += float(self.rng.random())
        return receipt


class FakeRecoveredEnv(ReceiptEnv):
    world = None
    restored = []

    def __init__(self, proposal, tape, before_native):
        type(self).world = copy.deepcopy(tape.world)
        super().__init__(proposal, tape, before_native)

    @classmethod
    def from_state_dict(cls, proposal, tape, state, before_native):
        out = cls.__new__(cls)
        out.before, out.t = before_native, state["fake_epoch"]
        out.requests = copy.deepcopy(state["requests"])
        cls.world = copy.deepcopy(tape.world)
        cls.restored.append(out.t)
        return out

    def state_dict(self):
        return dict(super().state_dict(), requests=copy.deepcopy(self.requests))


class FakeRecoveryController(ValueController):
    observed = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded_epoch = -1

    def state_dict(self):
        return dict(filter=dict(last_view=dict(epoch=self.epoch)), epoch=self.epoch,
                    recorded_epoch=self.recorded_epoch, last_plan=copy.deepcopy(self.last_plan))

    def load_state_dict(self, state):
        self.epoch, self.recorded_epoch = state["epoch"], state["recorded_epoch"]
        self.last_plan = copy.deepcopy(state["last_plan"])
        # The runner must explicitly reconnect the learner even after restoration.
        self.value = None

    def observe(self, public, **kwargs):
        assert public.common.epoch == self.epoch+1, "duplicate receipt observation"
        self.observed.append(public.common.epoch)
        return super().observe(public, **kwargs)

    def record_operation(self, public, base):
        assert self.epoch == public.common.epoch
        assert self.recorded_epoch == self.epoch-1
        self.recorded_epoch = self.epoch


class FakeFailure(RuntimeError):
    pass


class MetadataValue:
    """Opaque saved learner metadata, without allocating a scientific model."""

    def __init__(self, config, *, seed, feature_dim, **callbacks):
        assert type(seed) is int
        self.config, self.feature_dim = copy.deepcopy(config), feature_dim

    def load_state_dict(self, state):
        self.saved = copy.deepcopy(state)
        self.updates, self.pending = state["updates"], copy.deepcopy(state["pending"])

    def state_dict(self):
        return copy.deepcopy(self.saved)


class MetadataEnv:
    @classmethod
    def from_state_dict(cls, proposal, tape, state, before_native):
        out = cls()
        out.t = state["host"]["scalars"]["t"]
        out.saved = copy.deepcopy(state)
        return out

    def state_dict(self):
        return copy.deepcopy(self.saved)


class FailingController(FakeRecoveryController):
    def act(self, public, role, **kwargs):
        w = FakeRecoveredEnv.world
        if (w["phase"] == "continuation" and w["block"] == 0 and w["condition"] == 2
                and w["replicate"] == 7 and public.common.epoch == 37
                and self.value.config["architecture"] == "flat"):
            kwargs["before_query"](dict(model_epochs=13))
            raise FakeFailure("saved artificial epoch37 failure")
        return super().act(public, role, **kwargs)


def backends(controller=FakeRecoveryController):
    return dict(value_factory=FakeRecoveredValue, env_factory=FakeRecoveredEnv, controller_factory=controller,
                tape_factory=lambda p,w:CapacityWorldTape(w, (), (), (), None),
                capture=lambda env:FakePublic(FakeCommon(env.t)), feature_adapter=fake_features,
                parameter_counter=lambda learner:3169 if learner.config["architecture"] == "graph" else 3155,
                base_control=lambda *args, **kwargs:np.zeros(16))


def copy_payload(prior, target, partial):
    target.mkdir()
    for name in ("raw", "summaries", "updates", "states", "models", "tapes"):
        shutil.copytree(prior/name, target/name,
                        ignore=lambda directory,names: [partial.name] if Path(directory) == prior/"raw" else [])
    shutil.copyfile(prior/"progress.jsonl", target/"progress.jsonl")


class RecoveryTests(ZeroUpdateCase):
    def setUp(self):
        super().setUp()
        # These tests also forbid any real neural forward, not merely updates.
        self.enterContext(patch("torch.nn.Module._call_impl", side_effect=AssertionError("real neural forward forbidden")))
        for cls in (CapacityPlanningEnv, PatientConditionCapacityEnv):
            for method in ("__init__", "reset", "step"):
                self.enterContext(patch.object(cls, method, side_effect=AssertionError("native execution forbidden")))

    def test_actual_persisted_native_and_learner_restore_without_execution(self):
        if not (PRIOR/"failure-state.pkl.gz").exists():
            self.skipTest("immutable persisted metadata is not present in this checkout")
        state = pickle.loads(gzip.decompress((PRIOR/"failure-state.pkl.gz").read_bytes()))
        proposal = state["controller"]["filter"]["proposal"]
        self.assertEqual(len(state["completed"]), 71)
        self.assertEqual(state["epoch"], 37)
        self.assertEqual(state["budget"]["pending_chunks"]["planner_total_model_epochs"], dict(reserved=384, dispatched=13))
        budget = RemainingBudget(state["budget"]["counts"])
        for key,value in dict(native_steps=33819, total_native_operations=34875,
                total_optimizer_steps=12320, neural_forward_module_calls=27692, trajectories=528).items():
            self.assertEqual(budget.limits[key], value)
        w = {k:v for k,v in state["active"].items() if k != "role"}
        tape = CapacityWorldTape(**json.loads((PRIOR/"tapes"/(trajectory_id(w, "exogenous")+".json")).read_text()))
        before = copy.deepcopy(state)
        no_call = lambda *_: self.fail("restore performed a scientific operation")
        env = recovery.CapacityValueRecoveryEnv.from_state_dict(proposal, tape, state["environment"], no_call)
        self.assertEqual(env.t, 37)
        self.assertTreeEqual(env.state_dict(), state["environment"])
        # Check static constructor/reset attributes too; they need not be serialized.
        for relative, name in (("src/env/capacity_planning.py", "CapacityPlanningEnv"),
                ("src/env/patient_capacity_planning.py", "PatientConditionCapacityEnv"),
                ("src/env/patient_support_capacity.py", "PatientSupportCapacityEnv"),
                ("src/env/patient_support_public.py", "PublicPatientSupportCapacityEnv"),
                ("src/rl/capacity_native.py", "CapacityPilotEnv")):
            tree = ast.parse((ROOT/relative).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            attributes = {n.attr for method in cls.body if isinstance(method, ast.FunctionDef)
                          and method.name in ("__init__", "reset") for n in ast.walk(method)
                          if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                          and isinstance(n.value, ast.Name) and n.value.id == "self"}
            self.assertEqual(attributes-set(vars(env)), set(), name)
        public = recovery.PublicSupportControlInput.capture(env)
        self.assertTreeEqual(asdict(public), asdict(recovery.decode_public(state["controller"]["filter"]["last_view"])))
        with patch.object(CompletionIntervalFilter, "update", side_effect=AssertionError("no receipt replay")):
            controller = CapacityValueMPCRecovery1(proposal, base_control=no_call, value=None)
            controller.load_state_dict(state["controller"])
        self.assertTreeEqual(controller.state_dict(), state["controller"])
        # act() may invoke observe on its current view, but this must be idempotent.
        controller.observe(public, before_filter=no_call)
        self.assertTreeEqual(controller.state_dict(), state["controller"])
        for a in ARCHITECTURES:
            saved = state["training_learners"][a]
            learner = CapacityValueComparisonLearner(saved["config"], seed=0, feature_dim=saved["feature_dim"],
                before_forward=no_call, before_optimizer=no_call)
            learner.load_state_dict(saved)
            self.assertTreeEqual(learner.state_dict(), saved)
            expected = np.random.default_rng()
            expected.bit_generator.state = copy.deepcopy(saved["rng"])
            np.testing.assert_array_equal(learner.rng.integers(0, 64, 64), expected.integers(0, 64, 64))
        self.assertTreeEqual(state, before)

    def test_actual_metadata_through_fake_backend_restore_entry(self):
        if not (PRIOR/"failure-state.pkl.gz").exists():
            self.skipTest("immutable persisted metadata is not present in this checkout")
        state = pickle.loads(gzip.decompress((PRIOR/"failure-state.pkl.gz").read_bytes()))
        proposal = state["controller"]["filter"]["proposal"]
        public = recovery.decode_public(state["controller"]["filter"]["last_view"])
        ident = trajectory_id(state["active"], "flat_value_mpc")
        tape_name = trajectory_id(state["active"], "exogenous")+".json"
        budget = RemainingBudget(state["budget"]["counts"])
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"):
            root = Path(tmp)
            for name in ("models", "tapes"):
                (root/name).mkdir()
            for a in ARCHITECTURES:
                name = f"block0-{a}-initial.pt"
                shutil.copyfile(PRIOR/"models"/name, root/"models"/name)
            shutil.copyfile(PRIOR/"tapes"/tape_name, root/"tapes"/tape_name)
            shutil.copyfile(PRIOR/"progress.jsonl", root/"progress.jsonl")
            runner = recovery.CapacityValueComparisonRecovery1Runner(root, proposal, budget,
                admission=ADMISSION, failure_state=state, partial_raw_path=PRIOR/"raw"/(ident+".jsonl.gz"),
                env_factory=MetadataEnv, value_factory=MetadataValue, capture=lambda _:public)
            runner.budget.enter(PHASES[1], PHASES[1])
            runner._restore_learners()
            def restored(event, **values):
                self.assertEqual(event, "partial_trajectory_restored")
                self.assertEqual(values["prefix_rows"], 37)
                self.assertEqual(values["remaining_steps"], 27)
                raise FakeFailure("stop before any simulated step")
            with patch.object(runner, "status", side_effect=restored), \
                    patch.object(CompletionIntervalFilter, "update", side_effect=AssertionError("no filter replay")), \
                    patch.object(CapacityValueMPCRecovery1, "act", side_effect=AssertionError("no planner")):
                with self.assertRaisesRegex(FakeFailure, "stop before"):
                    runner._resume_partial()
            self.assertIs(runner.controller.value, runner.training_learners["flat"])
            self.assertTreeEqual(runner.controller.state_dict(), state["controller"])
            for a in ARCHITECTURES:
                self.assertTreeEqual(runner.training_learners[a].state_dict(), state["training_learners"][a])
            self.assertEqual(budget.counts, dict.fromkeys(budget.counts, 0))
            self.assertEqual(budget.jobs, [(PHASES[1], ident+"-remaining", "world")])
            self.assertEqual(list((root/"raw").iterdir()), [])
            # An already copied partial must never be silently overwritten.
            target = root/"raw"/(ident+".jsonl.gz")
            target.write_bytes(b"untouchable")
            with self.assertRaises(FileExistsError):
                runner._resume_partial()
            self.assertEqual(target.read_bytes(), b"untouchable")

    def test_full_remaining_schedule_and_original_analyzer(self):
        proposal = json.loads(json.dumps(merged_proposal(PROPOSAL, STUDY)))
        original_proposal = copy.deepcopy(proposal)
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"), \
                patch.object(CapacityValueComparisonLearner, "__init__", side_effect=AssertionError("no scientific learner")):
            prior, root = Path(tmp)/"prior", Path(tmp)/"recovery"
            prior.mkdir()
            old_budget = ComparisonBudget()
            old = CapacityValueComparisonRunner(prior, proposal, old_budget,
                admission=dict(verified=True, scope=SCOPE), **backends(FailingController))
            with self.assertRaises(FakeFailure):
                old.run()
            state = old.state_dict()
            state["budget"] = dict(counts=copy.deepcopy(old_budget.counts))
            self.assertEqual(len(state["completed"]), 71)
            self.assertEqual(state["epoch"], 37)
            self.assertEqual({a:l["updates"] for a,l in state["training_learners"].items()}, dict(graph=1536, flat=1504))
            partial = prior/"raw/continuation-b0-c2-j7-flat_value_mpc.jsonl.gz"
            # Deliberately noncanonical whitespace tests BYTE preservation, not reserialization.
            prefix = gzip.decompress(partial.read_bytes()).replace(b'"cost": 1.0', b'"cost" : 1.0')
            partial.write_bytes(gzip.compress(prefix))
            hashes = {p.relative_to(prior):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in prior.rglob("*") if p.is_file()}
            copy_payload(prior, root, partial)
            historical_progress = (root/"progress.jsonl").read_bytes()
            budget = RemainingBudget(old_budget.counts)
            runner = recovery.CapacityValueComparisonRecovery1Runner(root, proposal, budget,
                admission=ADMISSION, failure_state=state, partial_raw_path=partial, **backends())
            saved = copy.deepcopy(state)
            opened, resumed_steps, resumed_decisions = [], [], []
            FakeRecoveryController.observed = []
            FakeRecoveredEnv.restored = []
            native, query, status = runner._native, runner._query, runner.status

            def before_native(kind):
                if runner.active == state["active"]:
                    self.assertEqual(kind, "step")
                    self.assertIs(runner.controller.value, runner.training_learners["flat"])
                    resumed_steps.append(runner.epoch)
                return native(kind)

            def before_query(payload):
                if runner.active == state["active"]:
                    resumed_decisions.append(runner.epoch)
                return query(payload)

            def progress(event, **values):
                if event == "partial_trajectory_restored":
                    self.assertEqual(budget.counts, dict.fromkeys(budget.counts, 0))
                    for a in ARCHITECTURES:
                        self.assertTreeEqual(runner.training_learners[a].state_dict(), state["training_learners"][a])
                if event == "trajectory_started":
                    self.assertFalse(runner.active["block"] == 0 and runner.active["phase"] != "evaluation")
                    if runner.active["phase"] == "evaluation":
                        self.assertEqual(len(runner.initial), 10)
                        self.assertEqual(len(runner.final), 10)
                        self.assertEqual(budget.counts["total_optimizer_steps"], 12320)
                        opened.append(copy.deepcopy(runner.active))
                return status(event, **values)

            with patch.object(recovery, "decode_public", side_effect=lambda v:FakePublic(FakeCommon(v["epoch"]))), \
                    patch.object(runner, "_native", side_effect=before_native), \
                    patch.object(runner, "_query", side_effect=before_query), \
                    patch.object(runner, "status", side_effect=progress):
                result = runner.run()
            self.assertEqual(result["trajectories"], 600)
            self.assertEqual(result["new_completed_trajectories"], 529)
            self.assertEqual(budget.counts, budget.limits)
            self.assertFalse(budget.chunks)
            self.assertEqual(FakeRecoveredEnv.restored, [37])
            self.assertEqual(resumed_steps, list(range(37,64)))
            self.assertEqual(resumed_decisions, list(range(37,48)))
            self.assertEqual(FakeRecoveryController.observed[:27], list(range(38,65)))
            self.assertEqual(len(opened), 240)
            self.assertEqual(budget.counts["native_constructions"], 528)
            self.assertEqual(budget.counts["native_steps"], 33819)
            self.assertEqual(budget.counts["total_native_operations"], 34875)
            self.assertEqual(budget.counts["neural_forward_module_calls"], 27692)
            self.assertEqual(budget.counts["planner_total_model_epochs"], 8630400)
            complete_raw = gzip.decompress((root/"raw"/partial.name).read_bytes())
            self.assertTrue(complete_raw.startswith(prefix))
            self.assertEqual(len(complete_raw.splitlines()), 64)
            self.assertTrue((root/"progress.jsonl").read_bytes().startswith(historical_progress))
            events = [json.loads(line) for line in (root/"progress.jsonl").read_text().splitlines()]
            eval_events = [e for e in events if e["event"] == "trajectory_started" and e["active"]["phase"] == "evaluation"]
            self.assertTrue(all(e["counts"]["total_optimizer_steps"] == 15360 for e in eval_events))
            self.assertTrue(all(e["new_counts"]["total_optimizer_steps"] == 12320 for e in eval_events))
            self.assertEqual(events[-1]["counts"]["planner_total_model_epochs"],
                             numeric_contract(STUDY)["limits"]["planner_total_model_epochs"]+384)
            analysis = run_analysis(root, STUDY)
            self.assertEqual(analysis["trajectory_count"], 600)
            self.assertEqual(analysis["raw_rows"], 38400)
            self.assertEqual(analysis["evaluation_trajectories"], 240)
            self.assertTreeEqual(state, saved)
            self.assertEqual(proposal, original_proposal)
            self.assertEqual(hashes, {p.relative_to(prior):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in prior.rglob("*") if p.is_file()})
            with self.assertRaisesRegex(RuntimeError, "single attempt"):
                runner.run()
            with self.assertRaises(PermissionError):
                runner._optimizer("value", 64)

    def test_rejects_wrong_metadata_before_any_backends_or_new_work(self):
        if not (PRIOR/"failure-state.pkl.gz").exists():
            self.skipTest("immutable persisted metadata is not present in this checkout")
        saved = pickle.loads(gzip.decompress((PRIOR/"failure-state.pkl.gz").read_bytes()))
        proposal = saved["controller"]["filter"]["proposal"]
        for defect in ("epoch", "updates", "pending", "completed", "architecture"):
            state = copy.deepcopy(saved)
            if defect == "epoch":
                state["epoch"] = 36
            elif defect == "updates":
                state["training_learners"]["flat"]["updates"] = 1503
            elif defect == "pending":
                state["training_learners"]["flat"]["pending"] = dict(completed=1)
            elif defect == "completed":
                state["completed"].pop()
            else:
                state["training_learners"]["flat"]["config"]["architecture"] = "graph"
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)/"absent"
                budget = RemainingBudget(saved["budget"]["counts"])
                with self.assertRaises(ValueError):
                    recovery.CapacityValueComparisonRecovery1Runner(target, proposal, budget,
                        admission=ADMISSION, failure_state=state, partial_raw_path=PRIOR/"unused", **backends())
                self.assertFalse(target.exists())
                self.assertFalse(any(budget.counts.values()))
