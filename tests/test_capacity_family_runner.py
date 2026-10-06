"""Necessary integration checks on fake science, plus fixed selection arithmetic."""

import copy
import itertools
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.rl.capacity_family_analysis import pair_readout, reconcile_raw, select_finalists
from src.rl.capacity_family_design import METHODS, PHASES, SCOPE, TRAIN_COUNTS, expected_learning_counts, merged_proposal, numeric_contract, worlds
from src.rl.capacity_family_execution import freeze, stream_manifest
from src.rl.capacity_family_resources import CapacityFamilyBudget
from src.rl.capacity_family_runner import CapacityFamilyRunner, actor_rows
from src.rl.capacity_family_value import CapacityFamilyValue
from src.rl.capacity_native import CapacityWorldTape
from tests.test_capacity_pilot_runner import FakeEnv, FakeController
from tests.test_capacity_value_mpc import ValueBudget, ZeroUpdateCase, FakePublic, FakeCommon, fake_features, PROPOSAL

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT/"experiments/configs/capacity_family_selection_20261006.json").read_text())


class Budget(ValueBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in PHASES}
        self.phase, self.chunks, self.jobs = None, {}, []
    def enter(self, owner, phase=None):
        self.phase = phase


class Environment(FakeEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["support_public_receipt"]["applied_hours"] = list(action.requested_hours)
        return obs, reward, done, info


class Controller(FakeController):
    def __init__(self, *a, value=None, planning_horizon=8, **k):
        super().__init__()
        self.value, self.horizon, self.last_plan = value, planning_horizon, None
    def act(self, public, *, before_query, before_filter):
        before_query(dict(model_epochs=48*self.horizon))
        if self.value is not None:
            self.value.residuals(np.zeros((48, 4, 31), dtype=np.float32))
        return np.full(4, 2.)


class Learner:
    def __init__(self, runner, method):
        self.r, self.method, self.counts = runner, method, {}
        self.replay, self.pending = [], None
    def act(self, x, explore=False):
        self.r._forward("actor", 1)
        if self.method == "ppo" and explore:
            self.r._forward("value", 1)
        return np.full(4, 2.), {}
    def residuals(self, x):
        self.r._forward("value", len(x))
        return np.zeros(len(x))
    def observe_episode(self, rows):
        assert len(rows) == 48 and rows[-1]["full_cost"] == 17
        self.replay.extend(rows)
    def admit_episode(self, rows):
        self.r._forward("value", 64)
        self.pending = dict(targets=np.zeros(64))
    def update(self):
        self.r._forward("value", 64)
        self.r._optimizer("value", 64)
        return dict(loss=0.)
    def fit_episode(self, updates, batch_size):
        fwd, actor, critic, value = TRAIN_COUNTS[self.method]
        acts = 96 if self.method == "ppo" else 48
        for _ in range(fwd-acts):
            self.r._forward("actor", 64)
        for key, n in (("actor", actor), ("critic", critic), ("value", value)):
            for _ in range(n):
                self.r._optimizer(key, 64)
        return [dict(fixture_only=True) for _ in range(32)]
    def state_dict(self):
        return dict(fixture_only=True, replay=self.replay)


def development_rows():
    roles = ["plain_h8", "plain_h16"]+[f"{m}-lr{li}-graph" for m in METHODS for li in range(2)]
    return [dict(world=w, role=role, cost=100. if role.startswith("plain") else 90.+i,
                 lost=0, tape_sha256=str(w["seed"]))
            for b in range(3) for w in worlds(STUDY, PHASES[1], b) for i, role in enumerate(roles)]


class FamilyRunnerTests(ZeroUpdateCase):
    def test_counts_streams_and_all_possible_finalists_fit_envelope(self):
        contract = numeric_contract(STUDY)
        self.assertEqual(contract["limits"]["trajectories"], 6744)
        self.assertEqual(contract["limits"]["native_steps"], 431616)
        self.assertEqual(contract["limits"]["total_native_operations"], 445104)
        self.assertEqual(contract["limits"]["planner_total_model_epochs"], 50429952)
        self.assertEqual(contract["limits"]["neural_forward_module_calls"], 2013312)
        self.assertEqual(sum(v for k, v in STUDY["budget"]["seconds"].items() if k != "global"), 172800)
        for a, b in itertools.combinations(METHODS, 2):
            for key, value in expected_learning_counts([dict(method=a), dict(method=b)]).items():
                self.assertLessEqual(value, contract["limits"][key])
        streams = stream_manifest(merged_proposal(PROPOSAL, STUDY))
        self.assertEqual(len(streams["rows"]), 960)
        self.assertEqual(len(streams["allocations"]), 5777)

    def test_actual_episode_and_fit_interfaces_all_five_no_science(self):
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"), \
                patch("src.rl.capacity_family_runner.observed_features", side_effect=fake_features):
            budget = Budget()
            budget.enter(PHASES[0], PHASES[0])
            runner = CapacityFamilyRunner(tmp, merged_proposal(PROPOSAL, STUDY), budget,
                admission=dict(verified=True, scope=SCOPE), workspace=tmp, env_factory=Environment,
                controller_factory=Controller, capture=lambda e: FakePublic(FakeCommon(e.t)),
                base_control=lambda *a, **k: np.zeros(16),
                tape_factory=lambda p, w: CapacityWorldTape(w, (), (), (), None))
            world = worlds(STUDY, PHASES[0], 0)[0]
            for method in METHODS:
                before = dict(budget.counts)
                runner.learner = Learner(runner, method)
                data = runner._episode(world, method, runner._tape(world), method=method)
                self.assertEqual(sum(r["full_cost"] for r in actor_rows(data)), 64)
                runner._fit(data)
                self.assertEqual(budget.counts["neural_forward_module_calls"]-before["neural_forward_module_calls"], TRAIN_COUNTS[method][0])
            rows = [json.loads(p.read_text()) for p in (Path(tmp)/"summaries").glob("*.json")]
            self.assertEqual(len(reconcile_raw(tmp, rows)), 5)
            budget.enter(PHASES[1], PHASES[1])
            world = worlds(STUDY, PHASES[1], 0)[0]
            for role in ("plain_h8", "plain_h16"):
                runner.learner = None
                runner._episode(world, role, runner._tape(world))
            with self.assertRaises(PermissionError):
                runner._optimizer("actor", 64)
            self.assertEqual(budget.chunks, {})

    def test_full_schedule_barriers_with_zero_episode_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget()
            runner = CapacityFamilyRunner(tmp, merged_proposal(PROPOSAL, STUDY), budget,
                admission=dict(verified=True, scope=SCOPE), workspace=tmp)
            training, evaluation = [], []
            def train(phase, block, method, arch, li):
                training.append((phase, block, method, arch, li))
                for version in ("initial", "final"):
                    runner.records[f"{phase}-b{block}-{method}-lr{li}-{arch}-{version}"] = dict(fake=True)
                runner.completed.extend([None]*96)
            def evaluate(phase, block, entries):
                self.assertEqual(len(training), 30 if phase == PHASES[1] else 50)
                self.assertEqual(len(entries), 12 if phase == PHASES[1] else 9)
                evaluation.append((phase, block))
                runner.completed.extend([None]*(24*len(entries)))
            with patch.object(runner, "_train_job", side_effect=train), \
                    patch.object(runner, "_evaluation", side_effect=evaluate), \
                    patch.object(runner, "_json"), patch.object(runner, "status"), \
                    patch("src.rl.capacity_family_runner.summaries", return_value=development_rows()):
                selected = select_finalists(development_rows(), STUDY)
                budget.counts.update(expected_learning_counts(selected["finalists"]), native_steps=431616)
                result = runner.run()
            self.assertEqual(result["trajectories"], 6744)
            self.assertEqual(len(training), 50)
            self.assertEqual(len(evaluation), 8)
            with self.assertRaises(RuntimeError):
                runner.run()

    def test_patient_first_selection_two_distinct_families_not_test(self):
        rows = development_rows()
        for row in rows:
            if row["role"].startswith("ddpg"):
                row.update(cost=1., lost=1)
        selection = select_finalists(rows, STUDY)
        self.assertNotIn("ddpg", [r["method"] for r in selection["finalists"]])
        self.assertEqual(len({r["method"] for r in selection["finalists"]}), 2)
        with self.assertRaises(ValueError):
            select_finalists(rows[:-1], STUDY)

    def test_five_block_paired_inference_and_mismatched_tape_rejected(self):
        rows = [dict(world=w, role=role, cost=90. if role == "candidate" else 100.,
                     lost=0, tape_sha256=str(w["seed"])) for b in range(5)
                for w in worlds(STUDY, PHASES[3], b) for role in ("candidate", "reference")]
        result = pair_readout(rows, "candidate", "reference", 5, np.random.default_rng(5))
        self.assertEqual(result["cost_ci95"], [10., 10.])
        self.assertEqual(result["saving_fraction"], .1)
        rows[0]["tape_sha256"] = "different"
        with self.assertRaises(ValueError):
            pair_readout(rows, "candidate", "reference", 5, np.random.default_rng(5))

    def test_value_architecture_snapshot_binding_without_optimizer(self):
        config = dict(STUDY["value"], lr=.0001, architecture="graph", sampler_seed=8)
        args = dict(seed=7, before_forward=lambda *a: None, before_optimizer=lambda *a: None)
        graph = CapacityFamilyValue(config, **args)
        self_only = CapacityFamilyValue(dict(config, architecture="self_only"), **args)
        for a, b in zip(graph.model.parameters(), self_only.model.parameters()):
            self.assertTreeEqual(a, b)
        saved = graph.state_dict()
        with self.assertRaises(ValueError):
            self_only.load_state_dict(saved)
        restored = CapacityFamilyValue(config, **args)
        restored.load_state_dict(saved)
        self.assertTreeEqual(saved, restored.state_dict())

    def test_per_owner_timeout_and_no_implicit_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [0.]
            budget = CapacityFamilyBudget(Path(tmp)/"launcher", merged_proposal(PROPOSAL, STUDY), clock=lambda: now[0])
            self.addCleanup(budget.close)
            budget.enter(PHASES[0], PHASES[0])
            budget.job("world", "actor_world")
            now[0] = 121.
            with self.assertRaises(TimeoutError):
                budget.check()
        with patch("src.rl.capacity_family_execution.base._clean"), \
                patch("src.rl.capacity_family_execution.base.committed_json", side_effect=[STUDY, {}]), \
                patch("src.rl.capacity_family_execution.base.file_record", return_value=dict(sha256="a"*64)):
            with self.assertRaises(PermissionError):
                freeze(ROOT)
