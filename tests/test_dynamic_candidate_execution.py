"""Authorization/claim fixtures only; no patient or numerical optimizer call."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_backend import DynamicPatientBackend
from src.rl.dynamic_candidate_execution import (
    AUTHORIZATION, PROTOCOL, REQUIRED_AMENDMENTS, ScientificAdmission,
    approved_packet, committed_json, confined, launch, validate_authorization, verify_completed,
)
from src.rl.dynamic_candidate_preparation import RUN_DIRECTORY
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.networks import torch
from src.utils.research_archive import create_archive
from src.utils.research_clock import CLOCK_ID


def tiny_plan():
    phases = {name: {"trajectory": 0, "clone": 0, "actor": 0, "critic": 0, "seconds": seconds}
              for name, seconds in (("runtime_input_binding", 2), ("invented", 2),
                                    ("supervisor_dispatch_terminal_closure", 2))}
    return {"format": "dynamic-candidate-budget-plan-v1", "draft_sha256": "invented",
        "limits": {"trajectory": 0, "clone": 0, "actor": 0, "critic": 0, "seconds": 8},
        "phases": phases, "sections": {name: row | {"phase": name} for name, row in phases.items()}}


def fake_packet():
    return {"packet_content_sha256": "a" * 64, "implementation_commit": "b" * 40,
        "scientific_config": {"invented": True}, "streams": {"invented": True},
        "budget_plan": tiny_plan(), "protocol": {"sha256": "c" * 64}}


def fake_authorization(packet):
    return {"format": "dynamic-pilot-explicit-authorization-v1", "approved": True,
        "proposal_content_sha256": packet["packet_content_sha256"], "implementation_commit": packet["implementation_commit"],
        "scientific_config_sha256": digest(packet["scientific_config"]), "amendments": copy.deepcopy(REQUIRED_AMENDMENTS),
        "limits": {"environment_steps": 0, "optimizer_steps": 0, "seconds": 8, "attempts": 1},
        "automatic_retry": False, "remote_or_dropbox_actions": False, "protocol": PROTOCOL,
        "protocol_sha256": packet["protocol"]["sha256"],
        "user_approval": {"user": "Zhaowei", "verbatim": "INVENTED UNIT FIXTURE, NOT USER APPROVAL", "recorded_at_utc": "fixture"}}


class DynamicExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for target in ("src.rl.experiment.build_env", "src.rl.strict_frozen_policy.StrictFrozenPolicy", "torch.load",
                       "torch.optim.Adam.step", "torch.optim.SGD.step"):
            guard = patch(target, side_effect=AssertionError("real numerical work forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def test_missing_approval_stops_before_claim_source_read_or_backend(self):
        with patch("src.rl.dynamic_candidate_execution.verify_frozen_proposal") as verify:
            with self.assertRaises(PermissionError):
                launch(self.root)
            verify.assert_not_called()
        self.assertFalse((self.root / RUN_DIRECTORY).exists())

    def test_approval_must_bind_every_prospective_choice_and_limit(self):
        packet = fake_packet()
        good = fake_authorization(packet)
        validate_authorization(good, packet)
        for key in ("approved", "proposal_content_sha256", "implementation_commit", "scientific_config_sha256",
                    "amendments", "limits", "automatic_retry", "remote_or_dropbox_actions", "user_approval", "protocol_sha256"):
            bad = copy.deepcopy(good)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(PermissionError):
                validate_authorization(bad, packet)

    def test_uncommitted_approval_and_symlink_path_rejected(self):
        path = self.root / AUTHORIZATION
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        with patch("src.rl.dynamic_candidate_execution.subprocess.check_output", return_value=b'{"different":1}'):
            with self.assertRaisesRegex(ValueError, "committed"):
                committed_json(self.root, AUTHORIZATION)
        (self.root / "link").symlink_to(path.parent, target_is_directory=True)
        for name in ("../outside", "/outside", "link/record.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                confined(self.root, name)

    def test_exclusive_claim_consumed_even_when_supervisor_fails(self):
        packet = fake_packet()
        with patch("src.rl.dynamic_candidate_execution.approved_packet", return_value=(packet, fake_authorization(packet))), \
                patch("src.rl.dynamic_candidate_execution.git", return_value="fixture-head"), \
                patch("src.rl.dynamic_candidate_execution.supervise_dynamic", side_effect=RuntimeError("invented failure")) as supervisor:
            self.assertEqual(launch(self.root), 1)
            output = self.root / RUN_DIRECTORY
            self.assertTrue((output / "launcher/claim.json").is_file())
            self.assertTrue((output / "launcher/launch-failure.json").is_file())
            with self.assertRaises(FileExistsError):
                launch(self.root)
            self.assertEqual(supervisor.call_count, 1)

    def test_admission_checks_owned_parent_exact_backend_and_active_phase(self):
        packet = fake_packet()
        auth = fake_authorization(packet)
        output = self.root / RUN_DIRECTORY
        budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True)
        self.addCleanup(budget.close)
        claim = {"format": "dynamic-exclusive-claim-v1", "pid": os.getppid(),
            "proposal_content_sha256": packet["packet_content_sha256"], "authorization_sha256": digest(auth),
            "clock_id": CLOCK_ID, "started": budget.started}
        write_json_once(output / "launcher/claim.json", claim)
        admission = ScientificAdmission(self.root, packet, auth, claim, budget)
        backend = DynamicPatientBackend(self.root, packet["scientific_config"], packet["streams"], admit_real_calls=admission)
        admission.bind_campaign(output, packet["scientific_config"], packet["streams"], budget, backend)
        with self.assertRaises(PermissionError):
            admission("outside_phase")
        budget.begin("runtime_input_binding")
        admission("fixture_boundary_only")
        with self.assertRaises(PermissionError):
            admission.bind_campaign(output, {}, packet["streams"], budget, backend)
        write_json_once(output / "launcher/terminal.json", {"status": "failed"})
        with self.assertRaises(PermissionError):
            admission("closed_attempt")

    def test_terminal_verifier_requires_complete_ledger_and_archive_bytes(self):
        packet = fake_packet()
        output = self.root / "invented-output"
        budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True)
        self.addCleanup(budget.close)
        for name in packet["budget_plan"]["sections"]:
            budget.begin(name)
            budget.finish()
        # A zero-operation fixture still needs explicit zero counter normalization.
        payload = output / "payload"
        payload.mkdir()
        (payload / "invented.txt").write_text("not scientific evidence")
        archive = create_archive(payload, output / "archives/completed-payload.tar.gz")
        write_json_once(output / "launcher/archive-receipt.json", {"archive": archive})
        write_json_once(output / "launcher/completed.json", {"status": "completed", "scientific_execution": True,
            "engineering_fixture": False, "exit_code": 0})
        self.assertTrue(verify_completed(output, packet, {"passed": True}))
        with (output / "archives/completed-payload.tar.gz").open("ab") as handle:
            handle.write(b"corruption")
        self.assertFalse(verify_completed(output, packet, {"passed": True}))


if __name__ == "__main__":
    unittest.main()
