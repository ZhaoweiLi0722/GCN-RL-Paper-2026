"""Tests for frozen Mac validation evidence used by the PC preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from evaluation import verify_patient_indexed_specimen_routing_validation as validation


VALIDATED_COMMIT = "4cc04d298417ae65d6a0c8f4c9ca6414a6da47ee"
EXPECTED_COMMIT = "f" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    mechanics = root / "evidence" / "mechanics.json"
    mechanics.parent.mkdir(parents=True)
    mechanics.write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": {
                    "routing_identity_preserved": True,
                    "parameter_gap_below_one_percent": True,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "status": "PASS",
        "validated_branch": validation.LOCKED_BRANCH,
        "validated_commit": VALIDATED_COMMIT,
        "focused_tests": {
            "status": "PASS",
            "total": validation.EXPECTED_FOCUSED_TESTS,
            "patterns": {"routing": validation.EXPECTED_FOCUSED_TESTS},
        },
        "full_test_suite": {
            "status": "PASS",
            "total": validation.EXPECTED_FULL_TESTS,
        },
        "compileall": {"status": "PASS"},
        "mechanics_gate": {
            "status": "PASS",
            "path": "evidence/mechanics.json",
            "sha256": _sha256(mechanics),
        },
    }
    evidence_path = root / "evidence" / "validation.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence_path


def _git_output(changed_paths: str = ""):
    def output(_repo_root: Path, *arguments: str) -> str:
        if arguments == ("branch", "--show-current"):
            return validation.LOCKED_BRANCH
        if arguments == ("rev-parse", "HEAD"):
            return EXPECTED_COMMIT
        if arguments[:2] == ("diff", "--name-only"):
            return changed_paths
        raise AssertionError(f"Unexpected git arguments: {arguments}")

    return output


class FrozenMacValidationTests(unittest.TestCase):
    def test_locked_evidence_and_mechanics_hash_are_consistent(self) -> None:
        evidence_path = Path(
            "experiments/evidence/"
            "patient_indexed_specimen_routing_mac_validation.json"
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        mechanics = Path(evidence["mechanics_gate"]["path"])

        self.assertEqual(evidence["validated_commit"], VALIDATED_COMMIT)
        self.assertEqual(evidence["focused_tests"]["total"], 40)
        self.assertEqual(evidence["full_test_suite"]["total"], 497)
        self.assertEqual(
            _sha256(mechanics),
            evidence["mechanics_gate"]["sha256"],
        )

    def test_verifier_accepts_only_allowlisted_descendant_changes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = _write_fixture(root)
            changed = "\n".join(sorted(validation.ALLOWED_DESCENDANT_PATHS))
            with mock.patch.object(
                validation,
                "git_output",
                side_effect=_git_output(changed),
            ), mock.patch.object(validation.subprocess, "run") as merge_base:
                evidence, mechanics = validation.verify_validation_evidence(
                    evidence_path,
                    repo_root=root,
                    expected_commit=EXPECTED_COMMIT,
                )

            self.assertEqual(evidence["status"], "PASS")
            self.assertEqual(mechanics, root / "evidence" / "mechanics.json")
            merge_base.assert_called_once()

    def test_verifier_rejects_scientific_descendant_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = _write_fixture(root)
            with mock.patch.object(
                validation,
                "git_output",
                side_effect=_git_output("src/env/patient_condition.py"),
            ), mock.patch.object(validation.subprocess, "run"):
                with self.assertRaisesRegex(ValueError, "Scientific paths changed"):
                    validation.verify_validation_evidence(
                        evidence_path,
                        repo_root=root,
                        expected_commit=EXPECTED_COMMIT,
                    )

    def test_materialization_is_atomic_and_never_overwrites(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "nested" / "report.json"
            source.write_bytes(b'{"status":"PASS"}\n')

            validation.materialize_mechanics_report(source, output)
            self.assertEqual(output.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                validation.materialize_mechanics_report(source, output)


if __name__ == "__main__":
    unittest.main()
