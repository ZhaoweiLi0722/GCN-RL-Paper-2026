"""Verify frozen Mac validation evidence before CUDA-only execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


LOCKED_BRANCH = "codex/patient-indexed-specimen-routing"
EXPECTED_FOCUSED_TESTS = 40
EXPECTED_FULL_TESTS = 497
ALLOWED_DESCENDANT_PATHS = frozenset(
    {
        "evaluation/verify_patient_indexed_specimen_routing_validation.py",
        "experiments/evidence/patient_indexed_specimen_routing_mac_validation.json",
        "experiments/evidence/patient_indexed_specimen_routing_mac_validation_mechanics.json",
        "scripts/invoke_patient_indexed_specimen_routing_phase_detached.ps1",
        "scripts/run_patient_indexed_specimen_routing.ps1",
        "scripts/start_patient_indexed_specimen_routing_phase.ps1",
        "tests/test_patient_indexed_specimen_routing_contract.py",
        "tests/test_patient_indexed_specimen_routing_validation.py",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_validation_evidence(
    evidence_path: Path,
    *,
    repo_root: Path,
    expected_commit: str,
) -> tuple[dict[str, Any], Path]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 1 or evidence.get("status") != "PASS":
        raise ValueError("Mac validation evidence is absent, failed, or unsupported")
    if evidence.get("validated_branch") != LOCKED_BRANCH:
        raise ValueError("Mac validation evidence has the wrong branch")
    if git_output(repo_root, "branch", "--show-current") != LOCKED_BRANCH:
        raise ValueError("Local branch does not match Mac validation evidence")
    if git_output(repo_root, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("Local commit does not match the locked CUDA preflight")

    validated_commit = str(evidence.get("validated_commit", ""))
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", validated_commit, expected_commit),
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    changed_paths = set(
        filter(
            None,
            git_output(
                repo_root,
                "diff",
                "--name-only",
                validated_commit,
                expected_commit,
            ).splitlines(),
        )
    )
    unexpected_paths = sorted(changed_paths - ALLOWED_DESCENDANT_PATHS)
    if unexpected_paths:
        raise ValueError(
            "Scientific paths changed after Mac validation: "
            + ", ".join(unexpected_paths)
        )

    focused = evidence.get("focused_tests", {})
    if focused.get("status") != "PASS" or focused.get("total") != EXPECTED_FOCUSED_TESTS:
        raise ValueError("Mac focused-test evidence is incomplete")
    if sum(int(value) for value in focused.get("patterns", {}).values()) != EXPECTED_FOCUSED_TESTS:
        raise ValueError("Mac focused-test counts do not reconcile")
    full_suite = evidence.get("full_test_suite", {})
    if full_suite.get("status") != "PASS" or full_suite.get("total") != EXPECTED_FULL_TESTS:
        raise ValueError("Mac full-test evidence is incomplete")
    if evidence.get("compileall", {}).get("status") != "PASS":
        raise ValueError("Mac compileall evidence did not pass")

    mechanics = evidence.get("mechanics_gate", {})
    mechanics_path = repo_root / str(mechanics.get("path", ""))
    if not mechanics_path.is_file():
        raise ValueError("Frozen Mac mechanics report is missing")
    if sha256_file(mechanics_path) != mechanics.get("sha256"):
        raise ValueError("Frozen Mac mechanics report hash mismatch")
    report = json.loads(mechanics_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("Frozen Mac mechanics gate did not pass")
    checks = report.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("Frozen Mac mechanics checks are incomplete")
    return evidence, mechanics_path


def materialize_mechanics_report(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite mechanics report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(source.read_bytes())
    try:
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    evidence, mechanics_path = verify_validation_evidence(
        (repo_root / args.evidence).resolve(),
        repo_root=repo_root,
        expected_commit=args.expected_commit,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    materialize_mechanics_report(mechanics_path, output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "validated_commit": evidence["validated_commit"],
                "focused_tests": evidence["focused_tests"]["total"],
                "full_tests": evidence["full_test_suite"]["total"],
                "mechanics_report_sha256": sha256_file(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
