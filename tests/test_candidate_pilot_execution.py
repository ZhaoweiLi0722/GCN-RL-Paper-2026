"""Execution-lock checks on temporary invented files; no scientific launch."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl import candidate_pilot_execution as execution
from src.rl.candidate_pilot_resources import stream_manifest
from tests.test_candidate_pilot_resources import config


class CandidatePilotExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cfg = config()
        contents = {execution.PROPOSAL: json.dumps(self.cfg), self.cfg["protocol"]: "invented protocol",
                    execution.AUTHORIZATION: "invented authorization", "src/fixture.py": "# invented source\n",
                    "results/prior.json": '{"seed":123}', "inputs/model.pt": "invented non-tensor input"}
        for name, text in contents.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in contents.items()}
        self.runtime = {"invented": True}
        self.effective = {"kind": "p1_single_attempt_effective_execution", "scientific_execution_authorized": True,
            "workspace": str(self.root), "branch": execution.BRANCH, "implementation_commit": "a" * 40,
            "proposal": execution.PROPOSAL, "proposal_sha256": self.hashes[execution.PROPOSAL],
            "protocol": self.cfg["protocol"], "protocol_sha256": self.hashes[self.cfg["protocol"]],
            "authorization": execution.AUTHORIZATION, "authorization_sha256": self.hashes[execution.AUTHORIZATION],
            "scientific_config": self.cfg, "source_files": {"src/fixture.py": self.hashes["src/fixture.py"]},
            "runtime": self.runtime, "readiness_audit": {"streams": stream_manifest(self.cfg),
                "static_compatibility": {"passed": True, "invented": True},
                "collision_audit": {"passed": True, "files": [{"path": "results/prior.json", "sha256": self.hashes["results/prior.json"]}]},
                "explicit_non_seed_parse_exclusions": [], "verified_r4_inputs": {"inputs/model.pt": self.hashes["inputs/model.pt"]}}}
        def git(root, *args):
            if args == ("branch", "--show-current"):
                return execution.BRANCH
            if args == ("status", "--porcelain"):
                return ""
            raise AssertionError(args)
        guards = [patch.object(execution, "PROPOSAL_SHA", self.hashes[execution.PROPOSAL]),
            patch.object(execution, "PROTOCOL_SHA", self.hashes[self.cfg["protocol"]]),
            patch.object(execution, "git", side_effect=git),
            patch.object(execution, "runtime_record", return_value=self.runtime),
            patch.object(execution, "audit_reference_layouts", return_value={"passed": True, "invented": True}),
            patch.object(execution, "source_files", side_effect=lambda root: {"src/fixture.py": execution.sha(root / "src/fixture.py")}),
            patch.object(execution.subprocess, "run"),
            patch.object(execution.subprocess, "check_output", return_value=contents["src/fixture.py"].encode())]
        for guard in guards:
            guard.start()
            self.addCleanup(guard.stop)

    def test_exact_packet_passes_without_factory_or_updates(self):
        with patch.object(execution.PatientBackend, "__init__", side_effect=AssertionError("no environment factory")):
            self.assertEqual(execution.verify_packet(self.root, self.effective), self.cfg)

    def test_modified_scope_authorization_runtime_and_source_rejected(self):
        mutations = [lambda x: x.update(scientific_execution_authorized=False),
            lambda x: x.update(workspace=str(self.root / "other")),
            lambda x: x["scientific_config"]["caps"].update(maximum_environment_steps=999999),
            lambda x: x.update(runtime={"changed": True}),
            lambda x: x.update(authorization_sha256="0" * 64),
            lambda x: x["source_files"].update({"src/fixture.py": "0" * 64})]
        for mutate in mutations:
            packet = copy.deepcopy(self.effective)
            mutate(packet)
            with self.assertRaises(ValueError):
                execution.verify_packet(self.root, packet)

    def test_changed_inherited_input_and_historical_bytes_rejected(self):
        for name in ("inputs/model.pt", "results/prior.json", "src/fixture.py"):
            path = self.root / name
            before = path.read_bytes()
            path.write_bytes(b"invented change")
            with self.assertRaises(ValueError):
                execution.verify_packet(self.root, self.effective)
            path.write_bytes(before)

    def test_source_manifest_must_match_actual_implementation_commit(self):
        with patch.object(execution.subprocess, "check_output", return_value=b"uncommitted code"):
            with self.assertRaisesRegex(ValueError, "actual implementation"):
                execution.verify_packet(self.root, self.effective)

    def test_static_incompatibility_prevents_packet_or_claim_creation(self):
        with patch.object(execution, "audit", return_value={"static_compatibility": {"passed": False}}):
            with self.assertRaisesRegex(ValueError, "compatibility"):
                execution.freeze_packet(self.root)
        with patch.object(execution, "audit_reference_layouts", return_value={"passed": False}):
            with self.assertRaisesRegex(ValueError, "compatibility"):
                execution.verify_packet(self.root, self.effective)

    def test_static_compatibility_receipt_is_bound_to_frozen_packet(self):
        packet = copy.deepcopy(self.effective)
        del packet["readiness_audit"]["static_compatibility"]
        with self.assertRaisesRegex(ValueError, "compatibility receipt"):
            execution.verify_packet(self.root, packet)

    def recovery_packet(self):
        amendment = (Path(__file__).resolve().parents[1] / execution.recovery_spec.AMENDMENT).read_text()
        amendment_path = self.root / execution.recovery_spec.AMENDMENT
        amendment_path.write_text(amendment)
        authorization = self.root / execution.recovery_spec.AUTHORIZATION
        authorization.parent.mkdir(parents=True, exist_ok=True)
        authorization.write_text("invented recovery approval")
        packet = copy.deepcopy(self.effective)
        packet.update(kind="p1_recovery1_single_attempt_effective_execution",
            authorization=execution.recovery_spec.AUTHORIZATION, authorization_sha256=execution.sha(authorization),
            original_authorization_sha256=self.hashes[execution.AUTHORIZATION],
            amendment=execution.recovery_spec.AMENDMENT, amendment_sha256=execution.sha(amendment_path),
            scientific_config=execution.recovery_spec.recovery_configuration(self.root, self.cfg),
            new_scope_authorized=True)
        packet["readiness_audit"]["recovery_prior_attempt"] = {"invented_prior_receipt": True}
        return packet

    def test_recovery_packet_requires_explicit_profile_and_exact_scientific_delta(self):
        packet = self.recovery_packet()
        with patch.object(execution.recovery_spec, "prior_attempt_receipt", return_value={"invented_prior_receipt": True}):
            self.assertEqual(execution.verify_packet(self.root, packet, recovery=True), packet["scientific_config"])
            with self.assertRaises(ValueError):
                execution.verify_packet(self.root, packet)
            for field in ("objective", "ppo", "caps", "evaluation"):
                altered = copy.deepcopy(packet)
                altered["scientific_config"][field]["unapproved_setting"] = 1
                with self.assertRaisesRegex(ValueError, "original draft"):
                    execution.verify_packet(self.root, altered, recovery=True)
            altered = copy.deepcopy(packet)
            altered["scientific_config"]["candidate_message_graph"] = "union"
            with self.assertRaises(ValueError):
                execution.verify_packet(self.root, altered, recovery=True)

    def test_recovery_approval_and_prior_evidence_cannot_drift(self):
        packet = self.recovery_packet()
        with patch.object(execution.recovery_spec, "prior_attempt_receipt", return_value={"different": True}):
            with self.assertRaisesRegex(ValueError, "prior failure changed"):
                execution.verify_packet(self.root, packet, recovery=True)
        with patch.object(execution.recovery_spec, "prior_attempt_receipt", return_value={"invented_prior_receipt": True}):
            (self.root / execution.recovery_spec.AUTHORIZATION).write_text("changed approval")
            with self.assertRaisesRegex(ValueError, "authorization changed"):
                execution.verify_packet(self.root, packet, recovery=True)

    def test_recovery_cannot_launch_the_old_packet_or_output(self):
        with self.assertRaisesRegex(ValueError, "committed effective"):
            execution.launch(self.root, self.root / execution.EFFECTIVE, recovery=True)

    def test_unclaimed_child_and_arbitrary_launch_path_cannot_start_science(self):
        with patch.object(execution, "verify_packet", return_value=self.cfg), patch.object(execution, "EFFECTIVE", execution.PROPOSAL):
            with self.assertRaises(FileNotFoundError):
                execution.child(self.root)
        with self.assertRaisesRegex(ValueError, "committed effective"):
            execution.launch(self.root, self.root / execution.PROPOSAL)


if __name__ == "__main__":
    unittest.main()
