"""Real persisted scheduling metadata, fake science, and explicit update guards."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.paired_cohort_recovery_campaign import PairedCohortRecoveryCampaign
from src.rl.paired_cohort_recovery_data import recovery_manifest, branch_schedule
from src.rl.paired_cohort_recovery_training import restore_template
from src.rl.prospective_ddpg_kernel import state_digest
from src.utils.research_archive import inventory
from tests import test_paired_cohort_campaign as fixtures
from tests.test_paired_cohort_resources import ROOT


class RecoveryCampaignTests(unittest.TestCase):
    setUp = fixtures.PairedCampaignTests.setUp
    fake_episode = fixtures.PairedCampaignTests.fake_episode
    fake_fit = fixtures.PairedCampaignTests.fake_fit
    verify = fixtures.PairedCampaignTests.verify

    @classmethod
    def setUpClass(cls):
        cls.manifest = recovery_manifest(ROOT)

    def campaign(self):
        from src.rl.paired_cohort_recovery_resources import budget_plan
        self.budget.close()
        self.p["recovery_proposal"] = json.loads((ROOT / "specs/2026-10-02-paired-cohort-improvement/recovery-proposal.json").read_text())
        self.p["recovery_manifest"] = copy.deepcopy(self.manifest)
        self.budget = DynamicCandidateBudget(self.root / "recovery-ledger.jsonl",
            budget_plan(self.p["config"], self.p["recovery_proposal"]), enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        def importer(workspace, root, manifest, *, admit, budget):
            contexts = {}
            for row in manifest["contexts"]:
                admit("context_load")
                b, c, t = row["key"]
                contexts[b, c, t] = dict(block=b, cohort=c, after_prefix_steps=t, invented=True)
            for row in manifest["completed"]:
                admit("branch_import")
            write_json_once(root / "payload/reused-data.json", dict(invented=True))
            return contexts, [dict(invented=True) for _ in manifest["completed"]]
        def template(backend, b, saved, *, admit):
            admit("template_build")
            return fixtures.Owner(dict(invented_template=b))
        def collect(root, cfg, streams, block, contexts, template, producer, reference, budget, **kw):
            self.assertEqual(len(contexts), 12)
            self.assertEqual(len(kw["remaining"]), self.p["recovery_proposal"]["remaining_by_block"][str(block)]["branches"])
            for row in kw["remaining"]:
                kw["admit"]("conditional_branch_clone")
                for _ in range(row["environment_calls"]):
                    kw["admit"]("branch_step")
                    budget.debit_environment("clone")
            return [dict(invented=True) for _ in kw["remaining"]]
        def assemble(contexts, branches, **kw):
            self.assertEqual(len(branches), 402)
            result = dict(format="paired-cohort-label-dataset-v1", labels=contexts)
            result["dataset_sha256"] = digest(result)
            return result
        return PairedCohortRecoveryCampaign(ROOT, self.root, self.p, self.budget, self.admit,
            engineering=True, binder=self.bind, importer=importer, template=template,
            collect=collect, episode=self.fake_episode, fit=self.fake_fit, assemble=assemble,
            branch_reader=lambda root, values: values, wrapper=lambda owner, seed: fixtures.Owner(owner.state_dict()),
            verifier=self.verify, archive=lambda source, dest: dict(files=inventory(source), engineering_fixture=True))

    def test_complete_remaining_matrix_then_all_actor_updates_and_test_barrier(self):
        run = self.campaign()
        final = run.run()
        self.assertIsNone(final["active"])
        self.assertEqual(len(run.branch_indexes), 402)
        self.assertEqual(len(run.fitted), 6)
        self.assertEqual(len(run.evaluation_indexes), 216)
        self.assertEqual(self.admit.counts["conditional_branch_clone"], 285)
        self.assertEqual(self.admit.counts["context_load"], 36)
        self.assertEqual(self.admit.counts["branch_import"], 117)
        self.assertEqual(self.admit.counts["template_build"], 3)
        self.assertEqual(self.admit.counts["episode_build"], 216)
        self.assertEqual(self.admit.counts.get("preflight_clone", 0), 0)
        self.assertEqual(self.budget.counts, dict(environment=25655, optimizer=768))
        self.assertEqual(read_dynamic_ledger(self.budget.path)["owner_counts"].get("global:critic", 0), 0)
        before = run.checkpoint()
        run.restore_boundary(before)
        self.assertEqual(state_digest(before), state_digest(run.checkpoint()))

    def test_branch_failure_preserves_debit_and_cannot_enter_training(self):
        run = self.campaign()
        def failed(*args, **kwargs):
            args[8].debit_environment("clone")
            raise RuntimeError("invented terminal failure")
        run.collect = failed
        with self.assertRaisesRegex(RuntimeError, "terminal failure"):
            run.run()
        self.assertEqual(self.budget.counts["environment"], 1)
        self.assertEqual(self.budget.counts["optimizer"], 0)
        self.assertFalse((self.root / "payload/model-seals.json").exists())
        with self.assertRaises(ValueError):
            run.restore_boundary(run.checkpoint())

    def test_saved_decimal_seed_and_nested_environment_schema_restore_without_step(self):
        path = ROOT / self.manifest["contexts"][0]["file"]["path"]
        saved = load_envelope(path)
        from src.rl.paired_cohort_collection import validate_context
        context, _ = validate_context(saved)
        class Env:
            def __init__(self, typed, *, seed, cohort_spec, enabled):
                self.seed = seed
                self.cohort_spec = cohort_spec
                self._cohort_closed, self._cohort_steps = False, 0
            def load_state_dict(self, state):
                self.value = copy.deepcopy(state)
                self.t = state["scalars"]["t"]
            def state_dict(self):
                return copy.deepcopy(self.value)
        backend = SimpleNamespace(contexts={60: ({"env": {}}, None,
            SimpleNamespace(checkpoint_sha256=context["reference_sha256"]))},
            config={"objective": {"scenario": "fixture"}},
            branch_producer=lambda *args: SimpleNamespace(check_environment=lambda env: None))
        calls = []
        with patch("src.rl.paired_cohort_recovery_training.cohort_environment_class", return_value=Env), \
                patch("src.env.patient_capacity_planning.patient_env_config_from_dict", return_value=SimpleNamespace(base=None)), \
                patch("src.rl.paired_cohort_recovery_training.replace", side_effect=lambda cfg, **kw: cfg), \
                patch("src.rl.experiment.apply_graph_ablation", return_value=None), \
                patch("src.rl.frozen_value_probe.assert_scenario"):
            env = restore_template(backend, 60, saved, admit=calls.append)
        self.assertEqual(calls, ["template_build"])
        self.assertEqual(env.seed, context["environment"]["scalars"]["_episode_seed"])
        self.assertEqual(env.t, 4)


if __name__ == "__main__":
    unittest.main()
