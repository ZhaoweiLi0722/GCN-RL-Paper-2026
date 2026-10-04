"""Complete new entry on fake backends; real hosts and optimizers are forbidden."""

import copy
import gzip
import json
from pathlib import Path
import pickle
import tempfile
from unittest.mock import patch

import numpy as np

from src.rl import capacity_value_comparison_execution as entry
from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_native import CapacityWorldTape
from src.rl.capacity_value_comparison_analysis import run_analysis, summarize
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_comparison_resources import (
    PHASES, SCOPE, CapacityValueComparisonBudget, merged_proposal, numeric_contract, worlds,
)
from tests.test_capacity_value_mpc import (
    ZeroUpdateCase, ValueBudget, FakeValue, ReceiptEnv, ValueController,
    FakePublic, FakeCommon, fake_features, PROPOSAL,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT/entry.PROPOSAL).read_text())
ADMISSION = dict(verified=True, scope=SCOPE)


class ComparisonBudget(ValueBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p:dict.fromkeys(self.limits,0) for p in PHASES}
        self.phase, self.chunks, self.jobs = None, {}, []


class ComparisonTests(ZeroUpdateCase):
    def test_exact_contract_shared_initial_calls_and_no_evaluation_updates(self):
        contract = numeric_contract(STUDY)
        self.assertEqual(contract["limits"]["trajectories"],600)
        self.assertEqual(contract["limits"]["native_steps"],38400)
        self.assertEqual(contract["limits"]["total_native_operations"],39600)
        self.assertEqual(contract["limits"]["total_optimizer_steps"],15360)
        self.assertEqual(contract["limits"]["neural_forward_module_calls"],33120)
        self.assertEqual(contract["limits"]["planner_total_model_epochs"],9953280)
        self.assertEqual(contract["phase_limits"][PHASES[0]]["trajectories"],120)
        self.assertEqual(contract["phase_limits"][PHASES[0]]["total_optimizer_steps"],7680)
        self.assertEqual(contract["phase_limits"][PHASES[2]]["total_optimizer_steps"],0)
        self.assertEqual(sum(STUDY["budget"]["seconds"][k] for k in STUDY["budget"]["seconds"] if k != "global"),32400)
        for section,key in (("design","blocks"),("design","initial_worlds_per_block"),
                            ("value","updates_per_world"),("budget","native_steps"),("budget","planner_epochs")):
            bad = copy.deepcopy(STUDY)
            bad[section][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                numeric_contract(bad)

    def test_integer_stream_matrix_and_intentional_role_pairing(self):
        p = json.loads(json.dumps(merged_proposal(PROPOSAL,STUDY)))
        manifest = entry.stream_manifest(p)
        self.assertEqual(len(manifest["rows"]),300)
        self.assertEqual(len(manifest["allocations"]),1821)
        self.assertEqual(len(set(manifest["allocations"])),1821)
        self.assertTrue(all(type(s) is int for s in manifest["allocations"]))
        bad = copy.deepcopy(p)
        bad["value_comparison_study"]["design"]["model_seeds"]["flat"][0] = 530270511
        with self.assertRaisesRegex(ValueError,"collision"):
            entry.stream_manifest(bad)

    def test_full_fake_entry_600_worlds_both_models_bound_no_real_updates(self):
        budget = ComparisonBudget()
        p = json.loads(json.dumps(merged_proposal(PROPOSAL,STUDY)))
        with tempfile.TemporaryDirectory() as tmp, patch("os.fsync"), \
                patch.object(CapacityValueComparisonLearner,"__init__",side_effect=AssertionError("no scientific learner")):
            root = Path(tmp)
            runner = entry.build_runner(root,p,budget,admission=ADMISSION,
                value_factory=FakeValue, env_factory=ReceiptEnv, controller_factory=ValueController,
                tape_factory=lambda p,w:CapacityWorldTape(w,(),(),(),None),
                capture=lambda host:FakePublic(FakeCommon(host.t)), feature_adapter=fake_features,
                parameter_counter=lambda l:3169 if l.config["architecture"]=="graph" else 3155,
                base_control=lambda *args,**kwargs:np.zeros(16))
            original_status, opened = runner.status, []
            def status(event,**values):
                if event == "trajectory_started" and runner.active["phase"] == "evaluation":
                    self.assertEqual(len(runner.initial),10)
                    self.assertEqual(len(runner.final),10)
                    self.assertEqual(budget.counts["total_optimizer_steps"],15360)
                    role = runner.active["role"]
                    if role in ("graph_value_mpc","flat_value_mpc"):
                        self.assertEqual(runner.controller.value.config["architecture"],role.split("_")[0])
                        self.assertEqual(runner.learner.updates,1536)
                    else:
                        self.assertIsNone(runner.controller.value)
                    opened.append(role)
                return original_status(event,**values)
            with patch.object(runner,"status",side_effect=status):
                result = runner.run()
            self.assertEqual(result["trajectories"],600)
            self.assertEqual(budget.counts,budget.limits)
            self.assertEqual(budget.phase_counts,budget.contract["phase_limits"])
            self.assertEqual(len(opened),240)
            self.assertEqual(len(list((root/"tapes").glob("*.json"))),300)
            self.assertEqual(len(list((root/"models").glob("*.pt"))),20)
            for a in ("graph","flat"):
                meta = json.loads((root/f"updates/initial-b0-c0-j0-{a}_value_fit-input.json").read_text())
                self.assertEqual(meta["source_trajectory"],"initial-b0-c0-j0-plain_mpc")
            graph = json.loads((root/"updates/initial-b0-c0-j0-graph_value_fit-input.json").read_text())
            flat = json.loads((root/"updates/initial-b0-c0-j0-flat_value_fit-input.json").read_text())
            self.assertEqual(graph["data_sha256"],flat["data_sha256"])
            with gzip.open(root/"states/initial-b0-c0-j0-graph_value_fit-after-fit.pkl.gz","rb") as f:
                state = pickle.load(f)
            self.assertEqual(state["format"],"capacity-value-comparison-runner-v1")
            self.assertEqual(set(state["training_learners"]),{"graph","flat"})
            self.assertEqual(state["training_learners"]["graph"]["updates"],32)
            self.assertEqual(state["training_learners"]["flat"]["updates"],0)
            analysis = run_analysis(root,STUDY)
            self.assertEqual(analysis["trajectory_count"],600)
            self.assertEqual(analysis["raw_rows"],38400)
            self.assertEqual(analysis["evaluation_trajectories"],240)
            self.assertFalse(analysis["strong_baseline_development_signal"])
            self.assertFalse(analysis["graph_inductive_bias_development_signal"])
            self.assertEqual(len(analysis["contrasts"]),6)
            for contrast in analysis["contrasts"].values():
                for c in contrast.values():
                    self.assertEqual(c["independent_blocks"],5)
                    self.assertEqual(c["n_worlds"],20)
                    self.assertEqual(c["means"]["savings"],0.)
            prior = copy.deepcopy(budget.counts)
            with self.assertRaises(PermissionError):
                runner._optimizer("value",64)
            with self.assertRaisesRegex(RuntimeError,"single attempt"):
                runner.run()
            self.assertEqual(budget.counts,prior)
            path = root/"updates/initial-b0-c0-j0-flat_value_fit-input.json"
            flat["data_sha256"] = "tampered"
            path.write_text(json.dumps(flat))
            with self.assertRaisesRegex(ValueError,"fit inputs"):
                run_analysis(root,STUDY)

    def test_fifth_block_not_dropped_by_old_three_block_bootstrap(self):
        pairs = [dict(block=b,replicate=j,savings=100. if b==4 else 0.,extra_lost=0.,
                      extra_delivered=0.,percent_savings=10. if b==4 else 0.) for b in range(5) for j in range(4)]
        out = summarize(pairs,np.random.default_rng(3))
        self.assertEqual(out["means"]["savings"],20.)
        self.assertEqual(out["independent_blocks"],5)
        self.assertEqual(len(out["blocks"]),5)
        self.assertGreater(out["descriptive95"]["savings"][1],0.)
        with self.assertRaises(ValueError):
            summarize(pairs[:-1],np.random.default_rng(3))

    def test_unapproved_freeze_and_runner_have_no_scientific_side_effects(self):
        with patch.object(entry.base,"_clean"), \
                patch.object(entry.base,"committed_json",side_effect=[STUDY,dict(scope_approved=False)]), \
                patch.object(entry.base,"file_record",return_value=dict(sha256="artificial")), \
                patch.object(entry,"source_locks",side_effect=AssertionError("reject authority first")), \
                patch.object(entry,"build_runner",side_effect=AssertionError("scientific entry forbidden")):
            with self.assertRaises(PermissionError):
                entry.freeze(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"absent"
            with self.assertRaises(PermissionError):
                entry.build_runner(path,{},None,admission={})
            self.assertFalse(path.exists())

    def test_derived_authority_binds_new_scope_and_exact_contract(self):
        packet = dict(format="capacity-value-comparison-frozen-v1",intent=dict(scope_approved=True,
            scope=SCOPE,user_literal="artificial fixture only, not actual approval"),
            seed_audit=dict(passed=True),implementation_commit="artificial",
            config=dict(value_comparison_study=STUDY),contract=numeric_contract(STUDY))
        packet["packet_sha256"] = digest(packet)
        auth = entry.authorization(packet)
        self.assertTrue(auth["scientific_execution_authorized"])
        self.assertFalse(auth["automatic_follow_on"])
        bad = copy.deepcopy(packet)
        bad["contract"]["limits"]["trajectories"] -= 1
        with self.assertRaises((ValueError,PermissionError)):
            entry.authorization(bad)

    def test_real_budget_rejects_excess_without_refunding_previous_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = CapacityValueComparisonBudget(Path(tmp)/"launcher",merged_proposal(PROPOSAL,STUDY))
            try:
                budget.enter(PHASES[0],PHASES[0])
                budget.debit(dict(value_optimizer_steps=7680))
                with self.assertRaises(Exception):
                    budget.debit(dict(value_optimizer_steps=1))
                self.assertGreaterEqual(budget.counts["value_optimizer_steps"],7680)
            finally:
                budget.close()
