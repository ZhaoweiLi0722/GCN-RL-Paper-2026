"""P2 packet binding with invented files and no environment factory."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl import candidate_reference_execution as execution
from src.rl import candidate_reference_spec as spec
from src.rl import candidate_pilot_execution as common
from src.rl.candidate_pilot_resources import stream_manifest


class ReferenceExecutionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.cfg = spec.configuration(Path(__file__).resolve().parents[1])
        names = [spec.DESIGN, spec.PROTOCOL, spec.AUTHORIZATION, spec.PROPOSAL,
                 "specs/2026-09-30-candidate-return-pilot/protocol.md", "src/invented.py", "results/old.json", "inputs/old.pt"]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("invented bytes " + name)
        docs = {name: spec.sha(self.root / name) for name in names[:5]}
        audit = {"static_compatibility": {"passed": True}, "streams": stream_manifest(self.cfg),
            "prior_failures": ["invented"], "collision_audit": {"passed": True, "collisions": [],
            "files": [{"path": "results/old.json", "sha256": spec.sha(self.root / "results/old.json")}]},
            "explicit_non_seed_parse_exclusions": [], "verified_r4_inputs": {"inputs/old.pt": spec.sha(self.root / "inputs/old.pt")}}
        self.packet = {"kind": "p2_reference_prior_single_attempt_effective_execution", "scientific_execution_authorized": True,
            "workspace": str(self.root), "branch": execution.BRANCH, "implementation_commit": "a" * 40,
            "document_locks": docs, "scientific_config": self.cfg,
            "source_files": {"src/invented.py": spec.sha(self.root / "src/invented.py")},
            "runtime": {"invented": True}, "readiness_audit": audit, "automatic_retry": False}
        def git(root, *args):
            return execution.BRANCH if args == ("branch", "--show-current") else ""
        for guard in (
            patch.object(execution, "git", side_effect=git),
            patch.object(spec, "configuration", side_effect=lambda root: copy.deepcopy(self.cfg)),
            patch.object(spec, "prior_evidence", return_value=["invented"]),
            patch.object(execution, "source_files", side_effect=lambda root: {"src/invented.py": spec.sha(root / "src/invented.py")}),
            patch.object(execution, "runtime_record", return_value={"invented": True}),
            patch.object(execution, "audit_reference_layouts", return_value={"passed": True}),
            patch.object(execution.subprocess, "run"),
            patch.object(execution.subprocess, "check_output", side_effect=lambda args, **kw: (self.root / args[-1].split(":", 1)[1]).read_bytes())):
            guard.start()
            self.addCleanup(guard.stop)

    def test_exact_binding_and_all_scientific_delta_vetoes(self):
        self.assertEqual(execution.verify_packet(self.root, self.packet), self.cfg)
        for mutate in (lambda p: p.update(scientific_execution_authorized=False),
                       lambda p: p.update(automatic_retry=True),
                       lambda p: p.update(kind="p1_single_attempt_effective_execution"),
                       lambda p: p["document_locks"].pop(spec.AUTHORIZATION),
                       lambda p: p["scientific_config"]["initialization"].update(nonreference_mass=.2),
                       lambda p: p["scientific_config"]["caps"].update(maximum_optimizer_steps=2400),
                       lambda p: p["readiness_audit"].update(prior_failures=[]),
                       lambda p: p["readiness_audit"]["collision_audit"].update(passed=False),
                       lambda p: p.update(runtime={"different": True})):
            packet = copy.deepcopy(self.packet)
            mutate(packet)
            with self.assertRaises(ValueError):
                execution.verify_packet(self.root, packet)

    def test_historical_documents_inputs_and_actual_commit_are_bound(self):
        for name in (spec.AUTHORIZATION, spec.PROTOCOL, "src/invented.py", "results/old.json", "inputs/old.pt"):
            path = self.root / name
            before = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                execution.verify_packet(self.root, self.packet)
            path.write_bytes(before)
        with patch.object(execution.subprocess, "check_output", return_value=b"not committed"):
            with self.assertRaisesRegex(ValueError, "implementation commit"):
                execution.verify_packet(self.root, self.packet)

    def test_legacy_profile_cannot_launch_p2_and_dual_profiles_forbidden(self):
        with self.assertRaisesRegex(ValueError, "committed effective"):
            common.launch(self.root, self.root / spec.EFFECTIVE)
        with self.assertRaisesRegex(ValueError, "committed effective"):
            common.launch(self.root, self.root / common.EFFECTIVE, reference_prior=True)
        with self.assertRaisesRegex(ValueError, "one explicit"):
            common.launch(self.root, self.root / spec.EFFECTIVE, reference_prior=True, recovery=True)
        with self.assertRaisesRegex(ValueError, "one explicit"):
            common.child(self.root, reference_prior=True, recovery=True)
