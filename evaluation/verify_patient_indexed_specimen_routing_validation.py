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
EXPECTED_EVIDENCE_SHA256 = (
    "4805af6790999a4403ebb35495179444f667da079a4cd5a08015371316d14953"
)
RECOVERY5_RESULT_ROOT = "results/patient_indexed_specimen_routing_recovery5"
RECOVERY6_RESULT_ROOT = "results/patient_indexed_specimen_routing_recovery6"
RECOVERY6_PATH_ONLY_CONFIGS = frozenset(
    {
        "experiments/configs/patient_indexed_specimen_routing_attribution.json",
        "experiments/configs/patient_indexed_specimen_routing_benchmark.json",
        "experiments/configs/patient_indexed_specimen_routing_lead0_sensitivity_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_mechanics_gate.json",
        "experiments/configs/patient_indexed_specimen_routing_pilot_no_routing.json",
        "experiments/configs/patient_indexed_specimen_routing_pilot_no_routing_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_pilot_no_routing_pretrain_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_pilot_routing.json",
        "experiments/configs/patient_indexed_specimen_routing_pilot_routing_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_pilot_routing_pretrain_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_return1_sensitivity_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_smoke_no_routing.json",
        "experiments/configs/patient_indexed_specimen_routing_smoke_routing.json",
        "experiments/configs/patient_indexed_specimen_routing_teacher_no_routing.json",
    }
)
ALLOWED_DESCENDANT_PATHS = frozenset(
    {
        "evaluation/verify_patient_indexed_specimen_routing_validation.py",
        "experiments/evidence/patient_indexed_specimen_routing_mac_validation.json",
        "experiments/evidence/patient_indexed_specimen_routing_mac_validation_mechanics.json",
        "scripts/invoke_patient_indexed_specimen_routing_phase_detached.ps1",
        "scripts/run_patient_indexed_specimen_routing.ps1",
        "scripts/start_patient_indexed_specimen_routing_phase.ps1",
        "tests/test_gcn_ddpg_graph.py",
        "tests/test_patient_indexed_specimen_routing_contract.py",
        "tests/test_patient_indexed_specimen_routing_validation.py",
    }
) | RECOVERY6_PATH_ONLY_CONFIGS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_blob_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ("git", "show", f"{commit}:{relative_path}"),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def replace_recovery6_result_root(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(RECOVERY6_RESULT_ROOT, RECOVERY5_RESULT_ROOT)
    if isinstance(value, list):
        return [replace_recovery6_result_root(item) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_recovery6_result_root(item)
            for key, item in value.items()
        }
    return value


def assert_recovery6_configs_are_path_only(
    repo_root: Path,
    *,
    validated_commit: str,
    expected_commit: str,
) -> None:
    for relative_path in sorted(RECOVERY6_PATH_ONLY_CONFIGS):
        validated = json.loads(
            git_blob_bytes(repo_root, validated_commit, relative_path).decode(
                "utf-8"
            )
        )
        recovery6 = json.loads(
            git_blob_bytes(repo_root, expected_commit, relative_path).decode(
                "utf-8"
            )
        )
        serialized = json.dumps(recovery6, sort_keys=True)
        if RECOVERY5_RESULT_ROOT in serialized:
            raise ValueError(
                f"Recovery 6 config retains Recovery 5 output: {relative_path}"
            )
        if replace_recovery6_result_root(recovery6) != validated:
            raise ValueError(
                "Recovery 6 config changed beyond its result namespace: "
                f"{relative_path}"
            )


def verify_validation_evidence(
    evidence_path: Path,
    *,
    repo_root: Path,
    expected_commit: str,
) -> tuple[dict[str, Any], Path, bytes]:
    evidence_relative = evidence_path.resolve().relative_to(
        repo_root.resolve()
    ).as_posix()
    evidence_blob = git_blob_bytes(repo_root, expected_commit, evidence_relative)
    if sha256_bytes(evidence_blob) != EXPECTED_EVIDENCE_SHA256:
        raise ValueError("Frozen Mac validation evidence Git-blob hash mismatch")
    evidence = json.loads(evidence_blob.decode("utf-8"))
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
    assert_recovery6_configs_are_path_only(
        repo_root,
        validated_commit=validated_commit,
        expected_commit=expected_commit,
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
    mechanics_relative = str(mechanics.get("path", ""))
    mechanics_path = repo_root / mechanics_relative
    mechanics_blob = git_blob_bytes(
        repo_root,
        expected_commit,
        Path(mechanics_relative).as_posix(),
    )
    if sha256_bytes(mechanics_blob) != mechanics.get("sha256"):
        raise ValueError("Frozen Mac mechanics report Git-blob hash mismatch")
    report = json.loads(mechanics_blob.decode("utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("Frozen Mac mechanics gate did not pass")
    checks = report.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("Frozen Mac mechanics checks are incomplete")
    return evidence, mechanics_path, mechanics_blob


def materialize_bytes(payload: bytes, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite mechanics report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_mechanics_report(source: Path, output: Path) -> None:
    materialize_bytes(source.read_bytes(), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    evidence, mechanics_path, mechanics_blob = verify_validation_evidence(
        (repo_root / args.evidence).resolve(),
        repo_root=repo_root,
        expected_commit=args.expected_commit,
    )
    output = None
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = repo_root / output
        materialize_bytes(mechanics_blob, output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "validated_commit": evidence["validated_commit"],
                "focused_tests": evidence["focused_tests"]["total"],
                "full_tests": evidence["full_test_suite"]["total"],
                "mechanics_report_sha256": sha256_file(
                    output
                ) if output is not None else sha256_bytes(mechanics_blob),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
