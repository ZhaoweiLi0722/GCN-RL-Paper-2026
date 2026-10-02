"""Full fake campaign through exact native admission; all science guarded."""

import os
import unittest
from unittest.mock import patch

from src.rl import paired_cohort_recovery_execution as execution
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import read_dynamic_ledger
from src.rl.paired_cohort_recovery_campaign import PairedCohortRecoveryCampaign
from tests import test_paired_cohort_recovery_campaign as fixtures
from tests.test_paired_cohort_recovery_execution import synthetic_packet, labels_fixture, save


class RecoveryNativeTests(unittest.TestCase):
    setUp = fixtures.RecoveryCampaignTests.setUp
    setUpClass = classmethod(fixtures.RecoveryCampaignTests.setUpClass.__func__)
    fake_episode = fixtures.RecoveryCampaignTests.fake_episode
    fake_fit = fixtures.RecoveryCampaignTests.fake_fit
    verify = fixtures.RecoveryCampaignTests.verify
    campaign = fixtures.RecoveryCampaignTests.campaign

    def test_whole_fake_campaign_exact_native_admission_and_completion(self):
        template = self.campaign()
        workspace = self.root
        self.p = synthetic_packet(self, workspace, base=self.p, manifest=self.manifest)
        self.budget.close()
        self.root = workspace / execution.RUN
        for target in ("src.rl.dynamic_candidate_resources.shared_monotonic", "src.rl.paired_cohort_execution.shared_monotonic"):
            clock = patch(target, return_value=100.)
            clock.start()
            self.addCleanup(clock.stop)
        auth = execution.authorization(self.p)
        claim = dict(pid=os.getppid(), packet_sha256=self.p["packet_sha256"], authorization_sha256=digest(auth),
                     started=100., clock_id=execution.CLOCK_ID, head="invented")
        save(self.root, "launcher/claim.json", claim)
        self.budget = execution.PairedBudget(self.root / "launcher/budget.jsonl", self.p["budget_plan"], enabled=True, started=100.)
        self.addCleanup(self.budget.close)
        self.admit = execution.RecoveryAdmission(workspace, self.p, auth, claim, self.budget)
        self.addCleanup(self.admit.close)
        def assemble(contexts, branches, **kwargs):
            self.assertEqual((len(contexts), len(branches)), (36, 402))
            return labels_fixture(self.p)
        def binder(workspace, root, packet, budget, admit):
            return template.binder(fixtures.ROOT, root, packet, budget, admit)
        def final_check():
            execution.verify_completion(self.root, self.p, require_closed=False, admission=self.admit, campaign=run)
        run = PairedCohortRecoveryCampaign(workspace, self.root, self.p, self.budget, self.admit,
            binder=binder, importer=template.importer, template=template.template, collect=template.collect,
            episode=template.episode, fit=template.fit, assemble=assemble, branch_reader=template.branch_reader,
            wrapper=template.wrapper, verifier=template.verifier, archive=template.archive, final_lock_check=final_check)
        self.assertIs(type(run.admit), execution.RecoveryAdmission)
        run.run()
        report = execution.verify_completion(self.root, self.p, admission=self.admit, campaign=run)
        self.assertEqual(report["counts"], dict(environment=25655, optimizer=768))
        self.assertEqual(report["cumulative_environment_calls"], 31901)
        self.assertEqual(report["old_partial_debit_preserved"], 1)
        saved = execution.read_admission(self.admit.path, self.p)
        for job, caps in self.admit.limits.items():
            self.assertEqual(saved["by_section"].get(job, {}), caps, job)
        self.assertEqual(self.admit.counts["conditional_branch_clone"], 285)
        self.assertEqual(self.admit.counts["branch_step"], 12047)
        self.assertEqual(self.admit.counts["episode_step"], 13608)
        self.assertEqual(len(run.fitted), 6)
        ledger = read_dynamic_ledger(self.budget.path)
        self.assertEqual(ledger["owner_counts"].get("global:critic", 0), 0)
        self.assertFalse(any("preflight" in key for key in saved["jobs"]))


if __name__ == "__main__":
    unittest.main()
