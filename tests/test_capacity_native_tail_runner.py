"""Complete new entry with real branch orchestration and fake science backends."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.rl.capacity_native import CapacityWorldTape
from src.rl.capacity_native_tail_analysis import run_analysis, saved_native_branches
from src.rl.capacity_native_tail_execution import build_runner, stream_manifest, freeze
from src.rl.capacity_native_tail_resources import numeric_contract, merged_proposal, SCOPE, CapacityNativeTailBudget
from src.rl.capacity_native_tail_learner import CapacityNativeTailLearner
from src.rl.capacity_policy_tail_learner import CapacityPolicyTailLearner
from tests.test_capacity_policy_tail_runner import Value, Controller as OldController, fake_tail_pair, RawReceiptEnv
from tests.test_capacity_value_mpc import ValueBudget, ZeroUpdateCase, FakePublic, FakeCommon, fake_features, PROPOSAL


ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "experiments/configs/capacity_native_tail_20261005.json").read_text())


class Budget(ValueBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in self.contract["phase_limits"]}
        self.phase, self.chunks, self.jobs = None, {}, []


class NativeValue(Value):
    def admit_tails(self, records, **kwargs):
        if self.config["tails_per_world"] == 2:
            from src.rl.capacity_native_tail_targets import validate_records
            validate_records(records, continuation_sha256=self.ancestor_sha256, **kwargs)
        return super().admit_tails(records)


class Controller(OldController):
    def __init__(self, proposal, *, base_control, **kwargs):
        super().__init__(proposal, base_control=base_control, **kwargs)
        self.proposal, self.base_control, self.recorded = proposal, base_control, -1

    def state_dict(self):
        return dict(filter=dict(epoch=self.epoch), lifecycle=dict(epoch=self.epoch, recorded_epoch=self.recorded),
                    capture_epochs=(), last_training_tails=[], last_plan=copy.deepcopy(self.last_plan))

    def load_state_dict(self, state):
        self.epoch, self.recorded = state["filter"]["epoch"], state["lifecycle"]["recorded_epoch"]
        self.last_plan = copy.deepcopy(state["last_plan"])

    def record_operation(self, public, base):
        self.recorded = public.common.epoch

    def candidates(self, public):
        return [(np.full(4, 2.), switch) for switch in (False, True) for _ in range(8)]

    def adaptive(self, public):
        return np.full(4, .5)


class Environment(RawReceiptEnv):
    def __init__(self, proposal, tape, before_native):
        super().__init__(proposal, tape, before_native)
        self._before_native, self._failed = before_native, False
        self.tape_sha = tape.digest()

    def state_dict(self):
        return dict(world_tape_sha256=self.tape_sha, epoch=self.t, requests=copy.deepcopy(self.requests))


class NativeRunnerTests(ZeroUpdateCase):
    def test_full_entry_native240branches_480worlds_no_real_optimizer_or_model_load(self):
        study = copy.deepcopy(STUDY)
        budget = Budget()
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"), \
                patch.object(CapacityNativeTailLearner, "__init__", side_effect=AssertionError("real learner forbidden")), \
                patch.object(CapacityPolicyTailLearner, "__init__", side_effect=AssertionError("real learner forbidden")):
            work = Path(tmp)
            initial = work / "initial"
            initial.mkdir()
            study["initial_models"]["root"] = "initial"
            for block, name in enumerate(study["initial_models"]["names"]):
                raw = json.dumps(dict(config=dict(study["value"], method="observed_td"), seed=block,
                    feature_dim=31, updates=1536, ancestor_sha256="a" * 64), sort_keys=True).encode()
                (initial / name).write_bytes(raw)
                study["initial_models"]["sha256"][block] = hashlib.sha256(raw).hexdigest()
            proposal = json.loads(json.dumps(merged_proposal(PROPOSAL, study)))
            self.assertEqual(len(stream_manifest(proposal)["rows"]), 180)
            payload = work / "payload"
            payload.mkdir()
            runner = build_runner(payload, proposal, budget, admission=dict(verified=True, scope=SCOPE),
                workspace=work, value_factory=NativeValue, forecast_value_factory=NativeValue,
                frozen_factory=NativeValue, controller_factory=Controller, env_factory=Environment,
                tape_factory=lambda p, w: CapacityWorldTape(w, (), (), (), None),
                capture=lambda env: FakePublic(FakeCommon(env.t)), feature_adapter=fake_features,
                tail_collector=fake_tail_pair, base_control=lambda *a, **k: np.zeros(16))
            status = runner.status
            tests = []
            def record(event, **values):
                if event == "trajectory_started" and runner.active["phase"] == "evaluation":
                    self.assertEqual(len(runner.final), 15)
                    self.assertEqual(len(runner.ancestors), 5)
                    self.assertEqual(budget.counts["total_optimizer_steps"], 11520)
                    self.assertTrue((payload / "models/all-sealed.json").is_file())
                    tests.append(runner.active.copy())
                return status(event, **values)
            runner.status = record
            result = runner.run()
            self.assertEqual(result["trajectories"], 480)
            self.assertEqual(len(tests), 360)
            self.assertEqual(budget.counts, budget.limits)
            self.assertEqual(budget.phase_counts, budget.contract["phase_limits"])
            self.assertEqual(budget.chunks, {})
            self.assertEqual(len(list((payload / "native-branches").glob("*.jsonl.gz"))), 240)
            report = run_analysis(payload, study)
            self.assertEqual(report["trajectory_count"], 480)
            self.assertEqual(len(report["contrasts"]), 15)
            self.assertEqual(report["saved_tail_diagnostics"]["branch_count"], 240)
            self.assertEqual(report["saved_tail_diagnostics"]["native_steps"], 9720)
            self.assertFalse(report["strong_baseline_development_signal"])
            self.assertTrue(report["progress_evidence"]["barrier_verified"])
            self.assertEqual(len(report["model_evidence"]["byte_verified"]), 20)
            raw_file = next((payload / "native-branches").glob("*.jsonl.gz"))
            original = raw_file.read_bytes()
            rows = [json.loads(line) for line in gzip.decompress(original).decode().splitlines()]
            for field in ("cost", "new_lost_patients"):
                bad = copy.deepcopy(rows)
                bad[0][field] += 1
                raw_file.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in bad).encode()))
                with self.assertRaises(ValueError):
                    saved_native_branches(payload, study)
                raw_file.write_bytes(original)
            with self.assertRaises(RuntimeError):
                runner.run()

    def test_native_budget_nonrefund_and_phase_owner_caps(self):
        proposal = merged_proposal(PROPOSAL, STUDY)
        now = [0.]
        with tempfile.TemporaryDirectory() as tmp:
            budget = CapacityNativeTailBudget(Path(tmp) / "run" / "launcher", proposal, clock=lambda: now[0])
            self.addCleanup(budget.close)
            budget.enter("reference_and_tails", "reference_and_tails")
            budget.job("reference0", "reference_world_with_tails")
            budget.chunk_call("native_branch_model_epochs", 384, 13)
            self.assertEqual(budget.counts["native_branch_model_epochs"], 384)
            with self.assertRaises(Exception):
                budget.finish_chunk("native_branch_model_epochs")
            self.assertEqual(budget.counts["native_branch_model_epochs"], 384)
            now[0] = 901.
            with self.assertRaises(Exception):
                budget.check()

    def test_no_scientific_admission_from_continuation_direction(self):
        # This call stops before any model/native construction or lock creation.
        with patch("src.rl.capacity_native_tail_execution.base._clean"), \
                patch("src.rl.capacity_native_tail_execution.base.committed_json",
                      side_effect=[STUDY, dict(scope=SCOPE, scope_approved=False)]), \
                patch("src.rl.capacity_native_tail_execution.base.file_record", return_value=dict(sha256="a" * 64)):
            with self.assertRaises(PermissionError):
                freeze(ROOT)
