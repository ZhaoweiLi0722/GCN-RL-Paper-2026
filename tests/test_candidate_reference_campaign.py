"""P2 whole application on invented bookkeeping; no patient dynamics."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

from src.rl.candidate_pilot_resources import PilotBudget, stream_manifest, budget_sections
from src.rl.candidate_reference_campaign import ReferencePriorCampaign
from src.rl.candidate_reference_verification import verify_reference_bundle
from src.rl.prospective_ddpg_kernel import state_digest
from tests import test_candidate_pilot_campaign as base


class ReferenceCampaignTests(base.CandidatePilotCampaignTests):
    def setUp(self):
        super().setUp()
        self.cfg["pilot_profile"] = "p2_reference_prior"
        self.cfg["initialization"] = {"qualification_episodes_per_block": 1, "nonreference_mass": .1}
        self.cfg["caps"]["environment_steps"].pop("demonstrations")
        self.cfg["caps"]["optimizer_steps"].pop("initialization")
        self.cfg["rng"]["ordinal_ranges"].pop("demonstration")
        self.cfg["rng"]["neural_role_paths"] = [p for p in self.cfg["rng"]["neural_role_paths"] if "bc_init" not in p]

    def campaign(self):
        streams = stream_manifest(self.cfg)
        budget = PilotBudget(self.root / "launcher/budget.jsonl", self.cfg)
        self.addCleanup(budget.close)
        return ReferencePriorCampaign(self.root, self.cfg, streams, budget, base.InventedBackend(self.cfg),
                                      verifier=verify_reference_bundle)

    def test_complete_fake_application_real_kernels_recorder_verifier_and_archive(self):
        run = self.campaign()
        with patch("src.rl.candidate_pilot_campaign.initialization_kernel", side_effect=AssertionError("no BC init")), patch("builtins.print"):
            result = run.run()
        if result:
            self.fail((self.root / "launcher/failure.json").read_text())
        self.assertEqual(len(run.sequence.completed), 56)
        self.assertEqual(run.budget.counts, {"environment": 342, "optimizer": 36})
        self.assertEqual(run.initializers, {})
        self.assertEqual(run.demonstrations, {})
        self.assertEqual(len(run.sequence.seals), 27)
        self.assertEqual(len(run.test_index), 33)
        report = json.loads((self.root / "payload/independent-verification.json").read_text())
        self.assertEqual(len(report["duplicate_frozen_reference_lineage"]), 9)
        self.assertFalse(report["duplicate_baselines_increase_sample_size"])
        self.assertEqual(len(report["training"]), 18)
        self.assertTrue(all(r["row_count"] == 4 and r["passed"] for r in run.qualified.values()))
        self.assertTrue(all(run.models[f"block{b}/graph/{role}"].optimizer.state for b in (60, 61, 62)
                            for role in ("ppo", "bc_continue")))
        self.assertFalse(any(row[1] == "demonstration" for row in run.backend.calls))
        receipt = json.loads((self.root / "launcher/archive-receipt.json").read_text())
        self.assertTrue(all(r["local_copy_verified"] for r in receipt["local_copies"]))
        self.assertFalse(receipt["cloud_sync_verified"])
        # A saved raw index cannot silently substitute a frozen comparator.
        corrupted = copy.deepcopy(run.test_index)
        corrupted[0] = corrupted[-1]
        with self.assertRaises(ValueError):
            verify_reference_bundle(self.root, corrupted, self.cfg, run.streams)

    def test_qualification_failure_preserves_all_nine_scores_and_never_trains_or_tests(self):
        run = self.campaign()
        def rejected(kernel, examples, **kwargs):
            return {"passed": False, "kernel_sha256": state_digest(kernel.state_dict())}
        with patch("src.rl.candidate_reference_campaign.qualify_prior", side_effect=rejected), patch("builtins.print"):
            self.assertEqual(run.run(), 1)
        self.assertEqual(len(run.qualified), 9)
        self.assertEqual(run.budget.counts, {"environment": 66, "optimizer": 0})
        self.assertFalse(run.models)
        self.assertFalse(any(row[1] in ("training", "test") for row in run.backend.calls))
        self.assertTrue((self.root / "launcher/failure-state.pt").is_file())

    def test_first_update_failure_closes_phase_and_retains_spent_ledger(self):
        import torch
        run = self.campaign()
        original = torch.optim.Adam.step
        def fail(optimizer, *args, **kwargs):
            original(optimizer, *args, **kwargs)
            raise RuntimeError("invented P2 post-step failure")
        with patch.object(torch.optim.Adam, "step", fail), patch("builtins.print"):
            self.assertEqual(run.run(), 1)
        self.assertEqual(run.sequence.failure["job"], "block60/graph/ppo")
        self.assertEqual(run.budget.counts, {"environment": 70, "optimizer": 1})
        self.assertEqual(run.continuation.kernel.optimizer.state_dict()["state"], {})
        self.assertFalse(any(row[1] == "test" for row in run.backend.calls))
        self.assertTrue((self.root / "launcher/failure-state.pt").is_file())

    def test_profile_removes_all_demonstration_and_initialization_budget_scopes(self):
        scopes = budget_sections(self.cfg)
        self.assertFalse(any("demonstration" in s or "initialization" in s for s in scopes))
        run = self.campaign()
        self.assertFalse(any("demonstration" in s or "initialization" in s for s in run.sequence.jobs))
        with self.assertRaises(ValueError):
            run.collect_shared("demonstration")
        with self.assertRaises(ValueError):
            run.initialize("block60/graph")
