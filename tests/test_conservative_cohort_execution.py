"""Metadata and fake native admission only; never approve/freeze/run real science."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl import conservative_cohort_execution as ex
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import read_dynamic_ledger
from tests.test_conservative_cohort_campaign import CampaignTests
from tests.test_conservative_cohort_plan import ROOT
from tests.test_paired_cohort_execution import reseal


class MetadataTests(unittest.TestCase):
    def test_real_metadata_prepare_cannot_load_checkpoints_or_run_runtime(self):
        with patch("src.rl.conservative_cohort_binding.load_envelope", side_effect=AssertionError("no load")), \
                patch.object(ex, "runtime_record", side_effect=AssertionError("no runtime")), \
                patch.object(ex, "seed_inventory", side_effect=AssertionError("no historical audit")):
            packet = ex.prepare(ROOT)
        self.assertEqual(len(packet["inherited"]["model_inputs"]), 3)
        self.assertTrue(all(k.endswith("/initializer") for k in packet["inherited"]["model_inputs"]))
        self.assertFalse(packet["ready_to_launch"])
        self.assertFalse(packet["scientific_execution_authorized"])
        self.assertEqual(packet["budget_plan"]["limits"]["actor"], 768)
        self.assertEqual(packet["inherited"]["backend_config"]["totals"]["fresh_episode_builds"], 195)

    def test_missing_approval_denies_before_scientific_execution(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ex, "_clean"), \
                patch.object(ex, "prepare", side_effect=AssertionError("must deny before prepare")):
            with self.assertRaises(PermissionError):
                ex.approved(Path(directory))


class NativeAdmissionTests(CampaignTests):
    # Reuse the same fake-only campaign, but its numerical operations now pass
    # the exact production admission path and prospective owner caps.
    def test_complete_native_admission_fake_end_to_end(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        packet = copy.deepcopy(self.packet)
        packet.update(workspace=str(workspace), format="conservative-cohort-frozen-v1", source_frozen=True,
            implementation_commit="f" * 40, proposal={"sha256": "a" * 64}, protocol={"sha256": "b" * 64},
            intent=dict(approved=True, user_literal="SYNTHETIC FIXTURE NOT REAL AUTHORIZATION",
                proposal_sha256="a" * 64, protocol_sha256="b" * 64),
            runtime={"invented": True}, local_seed_collision_audit=dict(passed=True, collisions=[]))
        self.packet = reseal(packet)
        self.workspace = workspace
        original_binder = self.binder
        self.binder = lambda workspace, *args, **kwargs: original_binder(ROOT, *args, **kwargs)
        self.root = workspace / ex.RUN
        self.budget = ex.base.PairedBudget(self.root / "launcher/budget.jsonl", self.packet["budget_plan"], enabled=True)
        self.addCleanup(self.budget.close)
        auth = ex.authorization(self.packet)
        claim = dict(pid=os.getppid(), packet_sha256=self.packet["packet_sha256"],
            authorization_sha256=digest(auth), started=self.budget.started, clock_id=ex.CLOCK_ID)
        ex.write_json_once(self.root / "launcher/claim.json", claim)
        self.admit = ex.ConservativeAdmission(workspace, self.packet, auth, claim, self.budget)
        self.addCleanup(self.admit.close)
        run = self.campaign(engineering=False)
        run.run()
        self.assertEqual(self.admit.admitted_jobs, list(ex.operation_limits(self.packet["config"])))
        caps = ex.operation_limits(self.packet["config"])
        for job, value in caps.items():
            if job.startswith("branches/"):
                value.update(conditional_branch_clone=72, branch_step=3096)
        self.assertEqual(self.admit.by_section, {k: v for k, v in caps.items() if v})
        result = ex.verify_completion(self.root, self.packet)
        self.assertEqual(result["counts"], dict(environment=31050, optimizer=768))


if __name__ == "__main__":
    unittest.main()
