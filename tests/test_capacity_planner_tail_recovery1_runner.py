"""Final-world recovery on artificial saved metadata and fake backends only."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from unittest.mock import patch

import numpy as np

from src.rl.capacity_native import CapacityWorldTape
from src.rl.capacity_pilot_runner import jsonable, trajectory_id
from src.rl.capacity_planner_tail_design import EVAL_ROLES, TRAIN_ROLES, worlds
from src.rl.capacity_planner_tail_resources import SCOPE, merged_proposal, numeric_contract
from src.rl.capacity_planner_tail_recovery1_runner import CapacityPlannerTailRecovery1Runner, PARTIAL, RECOVERY_SCOPE
from tests.test_capacity_planner_tail_design import STUDY
from tests.test_capacity_planner_tail_runner import Controller, RawReceiptEnv, Value
from tests.test_capacity_value_mpc import (
    ZeroUpdateCase, ValueBudget, FakePublic, FakeCommon, fake_features, PROPOSAL)


class Budget(ValueBudget):
    def __init__(self):
        self.limits = dict.fromkeys(numeric_contract(STUDY)["limits"], 0)
        self.limits.update(native_steps=54, total_native_operations=54, control_steps=38,
            tail_steps=16, neural_forward_module_calls=960, planner_total_decisions=38,
            planner_candidate_rollouts=1824, planner_total_model_epochs=29184,
            estimator_receipt_updates=54, estimator_hypothesis_transitions=5400)
        evaluation = dict(self.limits, neural_forward_module_calls=0)
        analysis = dict.fromkeys(self.limits, 0)
        analysis["neural_forward_module_calls"] = 960
        self.contract = dict(phase_limits=dict(frozen_evaluation=evaluation, analysis_archive=analysis))
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p:dict.fromkeys(self.limits, 0) for p in self.contract["phase_limits"]}
        self.phase, self.chunks, self.jobs = None, {}, []


class RestoredEnv(RawReceiptEnv):
    @classmethod
    def from_state_dict(cls, proposal, tape, state, before_native):
        out = cls.__new__(cls)
        out.t, out.before = state["fake_epoch"], before_native
        out.requests = copy.deepcopy(state["requests"])
        return out

    def state_dict(self):
        return dict(fake_epoch=self.t, requests=self.requests)


class RestoredController(Controller):
    def load_state_dict(self, state):
        self.epoch = state["filter"]["last_view"]["epoch"]

    def state_dict(self):
        return dict(filter=dict(last_view=dict(epoch=self.epoch)))


def fixture(root):
    root.mkdir()
    for name in ("raw", "summaries", "models", "states", "tapes", "updates", "training-tails"):
        (root/name).mkdir()
    (root/"progress.jsonl").write_text("")
    references = [trajectory_id(w, "plain_h8") for b in range(5) for w in worlds(STUDY, "reference", b)]
    evaluations = [trajectory_id(w, role) for b in range(5) for w in worlds(STUDY, "evaluation", b) for role in EVAL_ROLES]
    world = list(worlds(STUDY, "evaluation", 4))[-1]
    tape = CapacityWorldTape(world, (), (), (), None)
    (root/"tapes"/(trajectory_id(world, "exogenous")+".json")).write_text(json.dumps(jsonable(tape)))
    features = [np.full((4, 31), t/64, dtype=np.float32) for t in range(11)]
    data = dict(features=features, heuristics=[0.]*11, costs=[1.]*10)
    rows = [dict(epoch=t, world=world, role="plain_h16", value_features=features[t],
        next_value_features=features[t+1], value_base=0., next_value_base=0., cost=1.,
        patient_records=[]) for t in range(10)]
    prefix = "".join(json.dumps(jsonable(row), indent=None)+"  \n" for row in rows).encode()
    prefix_path = root.parent/"saved-prefix.gz"
    prefix_path.write_bytes(gzip.compress(prefix))
    ancestors, final = {}, {}
    for b in range(5):
        for role in ("ancestor", *TRAIN_ROLES):
            raw = json.dumps(dict(config=dict(STUDY["value"], method="observed_td" if role == "ancestor" else role),
                seed=b, feature_dim=31, updates=1536 if role == "ancestor" else 768,
                ancestor_sha256="a"*64)).encode()
            name = f"block{b}-ancestor" if role == "ancestor" else f"block{b}-{role}-final"
            (root/"models"/(name+".pt")).write_bytes(raw)
            (ancestors if role == "ancestor" else final)[b if role == "ancestor" else f"block{b}-{role}"] = hashlib.sha256(raw).hexdigest()
    (root/"models/all-sealed.json").write_text(json.dumps(dict(final_models=final,
        ancestors={f"block{k}":v for k,v in ancestors.items()})))
    tail = dict(features=np.zeros((2, 4, 31), dtype=np.float32), heuristics=[0., 0.],
                costs=[1.], prefix_cost=8., decision_epoch=0)
    paths = ["training-tails/"+n+".pkl.gz" for n in references]
    for name in paths:
        with gzip.open(root/name, "wb") as f:
            pickle.dump([tail for _ in range(96)], f)
    counts = dict(numeric_contract(STUDY)["limits"])
    for k,v in Budget().limits.items():
        counts[k] -= v
    counts["planner_total_model_epochs"] += 768
    counts["planner_total_decisions"] += 1
    counts["planner_candidate_rollouts"] += 48
    state = dict(format="capacity-planner-tail-runner-v1", active=dict(world, role="plain_h16"),
        role="plain_h16", epoch=10, learner=None, training_learners={}, reference_tails=[],
        pending_tail_length=None, completed=references+evaluations[:-1], reference_files=paths,
        ancestors=ancestors, final=final, episode_data=data, budget=dict(counts=counts),
        environment=dict(fake_epoch=10, requests=[[2.]*4, [2.]*4]),
        controller=dict(filter=dict(last_view=dict(epoch=10))))
    return state, prefix_path, prefix


class RecoveryTests(ZeroUpdateCase):
    def make_runner(self, root, state, prefix, **kw):
        from src.rl.capacity_planner_tail_recovery1_execution import build_runner
        return build_runner(root, merged_proposal(PROPOSAL, STUDY), Budget(),
            admission=dict(verified=True, scope=SCOPE, remaining_scope=RECOVERY_SCOPE),
            failure_state=state, partial_raw_path=prefix, env_factory=RestoredEnv,
            controller_factory=RestoredController, value_factory=Value, frozen_factory=Value,
            capture=lambda env:FakePublic(FakeCommon(env.t)), feature_adapter=fake_features,
            base_control=lambda *a, **k:np.zeros(16),
            public_decoder=lambda s:FakePublic(FakeCommon(s["epoch"])), **kw)

    def test_final54steps_and960diagnostics_no_training_or_replay(self):
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"):
            root = Path(tmp)/"payload"
            state, prefix_path, prefix = fixture(root)
            old = prefix_path.read_bytes()
            runner = self.make_runner(root, state, prefix_path)
            with patch.object(RestoredEnv, "__init__", side_effect=AssertionError("no fresh environment")):
                result = runner.run()
            self.assertEqual(result["trajectories"], 480)
            self.assertEqual(runner.budget.counts, runner.budget.limits)
            self.assertEqual(runner.budget.phase_counts, runner.budget.contract["phase_limits"])
            self.assertEqual(prefix_path.read_bytes(), old)
            raw = gzip.decompress((root/"raw"/(PARTIAL+".jsonl.gz")).read_bytes())
            self.assertTrue(raw.startswith(prefix))
            rows = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual([r["epoch"] for r in rows], list(range(64)))
            self.assertEqual(rows[10]["info"]["support_public_receipt"]["applied_hours"], [2.]*4)
            self.assertEqual(len(list((root/"training-tails").glob("*-diagnostics.json"))), 120)
            summary = json.loads((root/"summaries"/(PARTIAL+".json")).read_text())
            self.assertEqual(summary["cost"], 64.)
            events = [json.loads(line) for line in (root/"progress.jsonl").read_text().splitlines()]
            self.assertNotIn("trajectory_started", [r["event"] for r in events])
            self.assertEqual(events[-1]["counts"]["value_optimizer_steps"], 11520)
            self.assertEqual(events[-1]["counts"]["planner_total_model_epochs"], 9953280+768)
            with self.assertRaises(PermissionError):
                runner._optimizer("value", 1)
            with self.assertRaises(RuntimeError):
                runner.run()

    def test_wrong_boundary_or_saved_model_fails_before_native_call(self):
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"):
            root = Path(tmp)/"payload"
            state, prefix, _ = fixture(root)
            for key, bad in (("epoch", 9), ("role", "planner_tail_td"), ("learner", {})):
                mutated = copy.deepcopy(state)
                mutated[key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.make_runner(root, mutated, prefix)
            (root/"models/block4-planner_tail_td-final.pt").write_bytes(b"wrong")
            runner = self.make_runner(root, state, prefix)
            with self.assertRaisesRegex(ValueError, "bytes"):
                runner.run()
            self.assertEqual(runner.budget.counts["native_steps"], 0)

    def test_saved_prefix_mismatch_and_existing_output_are_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"):
            root = Path(tmp)/"payload"
            state, prefix, _ = fixture(root)
            state["episode_data"]["costs"][0] = 2.
            runner = self.make_runner(root, state, prefix)
            with self.assertRaisesRegex(ValueError, "episode data"):
                runner.run()
            self.assertEqual(runner.budget.counts["native_steps"], 0)
            state["episode_data"]["costs"][0] = 1.
            (root/"raw"/(PARTIAL+".jsonl.gz")).write_bytes(b"do not overwrite")
            runner = self.make_runner(root, state, prefix)
            with self.assertRaises(FileExistsError):
                runner.run()

    def test_unapproved_admission_rejected(self):
        with self.assertRaises(PermissionError):
            CapacityPlannerTailRecovery1Runner("unused", {}, Budget(), admission={},
                failure_state={}, partial_raw_path="unused")
