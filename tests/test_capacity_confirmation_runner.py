"""Full new entry on fake science, including actual native clone orchestration."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.rl.capacity_confirmation_analysis import run_analysis, interaction
from src.rl.capacity_confirmation_design import SCOPE, counts, merged_proposal, numeric_contract
from src.rl.capacity_confirmation_execution import build_runner, stream_manifest, freeze
from src.rl.capacity_confirmation_learner import CapacityConfirmationWarmLearner, CapacityConfirmationTailLearner
from src.rl.capacity_confirmation_resources import CapacityConfirmationBudget
from src.rl.capacity_native import CapacityWorldTape
from tests.test_capacity_native_tail_runner import Environment, Controller
from tests.test_capacity_policy_tail_runner import Value
from tests.test_capacity_value_mpc import ValueBudget, ZeroUpdateCase, FakePublic, FakeCommon, fake_features, PROPOSAL

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "experiments/configs/capacity_confirmation_20261006.json").read_text())


class Budget(ValueBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in self.contract["phase_limits"]}
        self.phase, self.chunks, self.jobs = None, {}, []


class FakeValue(Value):
    def admit_episode(self, data):
        assert data["features"].shape == (65, 4, 31)
        assert data["features"].dtype == np.float32
        assert data["costs"].dtype == np.float64
        self.admit_observed(data)

    def admit_tails(self, records, **kwargs):
        from src.rl.capacity_native_tail_targets import validate_records
        validate_records(records, continuation_sha256=self.config["continuation_sha256"], **kwargs)
        super().admit_tails(records)

    def update(self):
        return dict(super().update(), indices=[0] * 64)


class ConfirmationRunnerTests(ZeroUpdateCase):
    def test_full780main240branch_entry_and_saved_reader_no_real_science(self):
        study, budget = copy.deepcopy(STUDY), Budget()
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"), \
                patch.object(CapacityConfirmationWarmLearner, "__init__", side_effect=AssertionError("real model forbidden")), \
                patch.object(CapacityConfirmationTailLearner, "__init__", side_effect=AssertionError("real model forbidden")):
            work = Path(tmp)
            (work / "legacy").mkdir()
            study["initial_models"]["root"] = "legacy"
            for b, name in enumerate(study["initial_models"]["names"]):
                raw = json.dumps(dict(config=study["value"], seed=b, feature_dim=31,
                    updates=1536, ancestor_sha256="a" * 64)).encode()
                (work / "legacy" / name).write_bytes(raw)
                study["initial_models"]["sha256"][b] = hashlib.sha256(raw).hexdigest()
            p = merged_proposal(PROPOSAL, study)
            manifest = stream_manifest(p)
            self.assertEqual(len(manifest["rows"]), 420)
            self.assertEqual(len(manifest["allocations"]), 2541)
            payload = work / "payload"
            payload.mkdir()
            runner = build_runner(payload, p, budget, admission=dict(verified=True, scope=SCOPE), workspace=work,
                warm_factory=FakeValue, value_factory=FakeValue, frozen_factory=FakeValue,
                env_factory=Environment, controller_factory=Controller,
                tape_factory=lambda p, w: CapacityWorldTape(w, (), (), (), None),
                capture=lambda e: FakePublic(FakeCommon(e.t)), feature_adapter=fake_features,
                base_control=lambda *a, **k: np.zeros(16))
            result = runner.run()
            self.assertEqual(result["trajectories"], 780)
            self.assertEqual(budget.counts, budget.limits)
            self.assertEqual(budget.phase_counts, budget.contract["phase_limits"])
            self.assertEqual(budget.chunks, {})
            self.assertEqual(len(runner.warm), 10)
            self.assertEqual(len(runner.final), 15)
            report = run_analysis(payload, study)
            self.assertEqual(len(report["contrasts"]), 21)
            self.assertEqual(report["fit_evidence"]["total_updates"], 26880)
            self.assertEqual(report["fit_evidence"]["after_fit_states"], 840)
            self.assertEqual(report["saved_native_branches"]["native_steps"], 9720)
            self.assertFalse(report["joint_primary_development_signal"])
            self.assertTrue(report["barrier_verified"])
            with self.assertRaises(RuntimeError):
                runner.run()
            barrier = payload / "models/all-sealed.json"
            old = barrier.read_text()
            barrier.write_text("{}")
            with self.assertRaises(ValueError):
                run_analysis(payload, study)
            barrier.write_text(old)
            update = next((payload / "updates").glob("warmup*-self_only.jsonl"))
            receipts = [json.loads(l) for l in update.read_text().splitlines()]
            receipts[0]["indices"][0] = 1
            update.write_text("\n".join(json.dumps(r) for r in receipts) + "\n")
            with self.assertRaisesRegex(ValueError, "sampling"):
                run_analysis(payload, study)

    def test_budget_arithmetic_and_nonrefundable_owner_timeout(self):
        d = counts()
        self.assertEqual(d["total_native_steps"], (240 + 120 + 7 * 60) * 64 + 9720)
        self.assertEqual(d["value_optimizer_steps"], 5 * (2 * 1536 + 3 * 768))
        self.assertEqual(d["forward_calls"], 26880 + 480 + 360 + 4100 + 14400)
        self.assertEqual(d["total_predictor_epochs"], (780 + 60) * 48 * 384 + 4100 * 384)
        with tempfile.TemporaryDirectory() as tmp:
            now = [0.]
            budget = CapacityConfirmationBudget(Path(tmp) / "run/launcher", merged_proposal(PROPOSAL, STUDY), clock=lambda: now[0])
            self.addCleanup(budget.close)
            budget.enter("reference_and_tails", "reference_and_tails")
            budget.job("r0", "reference_world_with_tails")
            budget.chunk_call("native_branch_model_epochs", 384, 13)
            with self.assertRaises(Exception):
                budget.finish_chunk("native_branch_model_epochs")
            self.assertEqual(budget.counts["native_branch_model_epochs"], 384)
            now[0] = 901.
            with self.assertRaises(Exception):
                budget.check()

    def test_no_admission_from_general_direction(self):
        with patch("src.rl.capacity_confirmation_execution.base._clean"), \
                patch("src.rl.capacity_confirmation_execution.base.committed_json",
                    side_effect=[STUDY, dict(scope=SCOPE, scope_approved=False)]), \
                patch("src.rl.capacity_confirmation_execution.base.file_record", return_value=dict(sha256="a" * 64)):
            with self.assertRaises(PermissionError):
                freeze(ROOT)

    def test_explicit_condition_interaction_not_subgroup_significance(self):
        conditions = {str(c): dict(pairs=[dict(block=b, replicate=j, savings=10.*c + b)
            for b in range(5) for j in range(4)]) for c in range(3)}
        result = interaction(conditions, np.random.default_rng(17), 2000)
        self.assertEqual(result["fast_minus_0"]["savings_difference"], 20.)
        np.testing.assert_allclose(result["fast_minus_1"]["descriptive95"], [10., 10.])
        self.assertFalse(result["fast_minus_0"]["condition_worlds_paired"])
