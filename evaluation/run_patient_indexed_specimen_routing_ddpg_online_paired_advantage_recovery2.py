"""Evaluation-only Recovery 2 executor for the Stage F1 DDPG campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.run_patient_indexed_specimen_routing_ddpg_online_paired_advantage import (
    ALGORITHMS,
    SEEDS,
    audit_evaluation_curve,
    audit_f0_identifiability,
    audit_preonline,
    audit_training,
    run_focused_tests,
    validate_scientific_contract,
)
from evaluation.run_patient_indexed_specimen_routing_ddpg_persistent_shift import (
    artifact_inventory,
    atomic_write_json,
    claim_execution,
    environment_fingerprint,
    git_output,
    read_json,
    run_phase,
    sha256_file,
    verify_locked_assets,
)


DEFAULT_SPEC = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_online_paired_advantage_"
    "recovery2_execution.json"
)
RECOVERY1_COMMIT = "48292683152e5a29107cd2d6ad9451612d8b22a3"
BASE_EXECUTION_SPEC_SHA256 = (
    "119b681afccd40c348b7ab5e3622e8f8efa7526fb925c8bad2c3d1b6471b0868"
)
BASE_LOCK_EXCEPTIONS = frozenset(
    {
        "docs/patient_indexed_specimen_routing_locked_execution_plan.md",
        "docs/patient_indexed_specimen_routing_stage_f1_protocol.md",
        "evaluation/run_patient_indexed_specimen_routing_ddpg_online_paired_advantage.py",
        "tests/test_ddpg_online_paired_advantage_stage_f1.py",
    }
)
ALLOWED_PHASE_MODULES = frozenset(
    {
        "evaluation.evaluate_fixed_checkpoint_curve",
        "evaluation.compare_ddpg_online_paired_advantage",
    }
)


def immutable_tree_snapshot(root: Path) -> dict[str, Any]:
    """Hash every file under a reused artifact root by relative path."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    artifacts = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda value: str(value))
        if path.is_file()
    }
    payload = json.dumps(
        artifacts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "root": str(root),
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def recovery1_digest_snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    recovery = dict(spec["recovery1"])
    files = {
        str(entry["name"]): {
            "path": str(entry["path"]),
            "sha256": sha256_file(Path(str(entry["path"]))),
        }
        for entry in recovery["files"]
    }
    trees = {
        str(entry["name"]): immutable_tree_snapshot(
            Path(str(entry["root"]))
        )
        for entry in recovery["trees"]
    }
    return {"files": files, "trees": trees}


def verify_recovery_lock_manifest(
    spec_path: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    base_path = Path(str(spec["base_execution_spec"]))
    if sha256_file(base_path) != BASE_EXECUTION_SPEC_SHA256:
        raise ValueError("Historical Stage F1 execution spec changed")
    exceptions = frozenset(str(value) for value in spec["base_lock_exceptions"])
    if exceptions != BASE_LOCK_EXCEPTIONS:
        raise ValueError("Recovery 2 base-lock exception set changed")

    base_spec = read_json(base_path)
    base_verified = {}
    for entry in base_spec["locked_files"]:
        path = Path(str(entry["path"]))
        if str(path) in exceptions:
            continue
        actual = sha256_file(path)
        if actual != str(entry["sha256"]).lower():
            raise ValueError(f"Recovery 2 base SHA256 mismatch for {path}")
        base_verified[str(path)] = actual

    required_current = {
        *exceptions,
        "evaluation/launch_patient_indexed_specimen_routing_ddpg_online_paired_advantage_recovery2.py",
        "evaluation/run_patient_indexed_specimen_routing_ddpg_online_paired_advantage_recovery2.py",
        "experiments/configs/patient_indexed_specimen_routing_ddpg_online_paired_advantage_recovery2_control_curve_eval.json",
        "experiments/configs/patient_indexed_specimen_routing_ddpg_online_paired_advantage_recovery2_candidate_curve_eval.json",
    }
    current_paths = {str(entry["path"]) for entry in spec["locked_files"]}
    if current_paths != required_current:
        raise ValueError("Recovery 2 current lock manifest changed")
    if str(spec_path) in current_paths:
        raise ValueError("Recovery 2 spec must not hash-reference itself")
    current_verified = verify_locked_assets(spec)
    return {
        "historical_execution_spec": str(base_path),
        "historical_execution_spec_sha256": BASE_EXECUTION_SPEC_SHA256,
        "historical_locked_files_verified": len(base_verified),
        "historical_lock_exceptions": sorted(exceptions),
        "current_locked_files": current_verified,
    }


def build_phase_commands(spec: dict[str, Any]) -> list[tuple[str, list[str]]]:
    phases = []
    for role in ("control", "candidate"):
        phases.append(
            (
                f"evaluation_{role}_checkpoint_curve",
                [
                    sys.executable,
                    "-m",
                    "evaluation.evaluate_fixed_checkpoint_curve",
                    "--config",
                    str(spec[f"{role}_evaluation_config"]),
                ],
            )
        )
    phases.append(
        (
            "comparison",
            [
                sys.executable,
                "-m",
                "evaluation.compare_ddpg_online_paired_advantage",
                "--control-config",
                str(spec["control_evaluation_config"]),
                "--candidate-config",
                str(spec["candidate_evaluation_config"]),
                "--output",
                str(spec["comparison_output"]),
                "--resamples",
                str(spec["bootstrap_resamples"]),
                "--bootstrap-seed",
                str(spec["bootstrap_seed"]),
            ],
        )
    )
    validate_evaluation_only_commands(phases)
    return phases


def validate_evaluation_only_commands(
    phases: list[tuple[str, list[str]]],
) -> None:
    observed_modules = set()
    for name, command in phases:
        if "-m" not in command:
            raise ValueError(f"Recovery 2 phase lacks a Python module: {name}")
        module = command[command.index("-m") + 1]
        observed_modules.add(module)
        if module not in ALLOWED_PHASE_MODULES:
            raise ValueError(f"Recovery 2 forbids phase module {module}")
        if any(
            token in " ".join(command)
            for token in (
                "train_multiscenario_network_residual",
                "prepare_ddpg_online_paired_advantage_preonline",
                "--resume-training-state",
            )
        ):
            raise ValueError(f"Recovery 2 contains a training command: {name}")
    if observed_modules != ALLOWED_PHASE_MODULES:
        raise ValueError("Recovery 2 evaluation phase set is incomplete")


def verify_recovery1(spec: dict[str, Any]) -> dict[str, Any]:
    recovery = dict(spec["recovery1"])
    if str(recovery["commit"]) != RECOVERY1_COMMIT:
        raise ValueError("Recovery 1 commit lock changed")
    snapshot = recovery1_digest_snapshot(spec)
    for entry in recovery["files"]:
        name = str(entry["name"])
        if snapshot["files"][name]["sha256"] != str(entry["sha256"]):
            raise ValueError(f"Recovery 1 immutable file changed: {name}")
    for entry in recovery["trees"]:
        name = str(entry["name"])
        observed = snapshot["trees"][name]
        if observed["artifact_tree_sha256"] != str(entry["sha256"]):
            raise ValueError(f"Recovery 1 immutable tree changed: {name}")
        if int(observed["artifact_count"]) != int(entry["artifact_count"]):
            raise ValueError(f"Recovery 1 artifact count changed: {name}")

    file_paths = {
        str(entry["name"]): Path(str(entry["path"]))
        for entry in recovery["files"]
    }
    status = read_json(file_paths["status"])
    claim = read_json(file_paths["claim"])
    expected_phases = [
        "prepare_preonline",
        *(
            f"training_{role}_{algorithm}_seed{seed}"
            for role in ("control", "candidate")
            for algorithm in ALGORITHMS
            for seed in SEEDS
        ),
    ]
    phases = list(status.get("phases", ()))
    if [str(phase.get("name")) for phase in phases] != expected_phases:
        raise ValueError("Recovery 1 phase history is not the complete training run")
    if any(int(phase.get("exit_code", -1)) != 0 for phase in phases):
        raise ValueError("Recovery 1 contains a failed training phase")
    if any(
        str(phase.get("name", "")).startswith("evaluation_")
        or str(phase.get("name")) == "comparison"
        for phase in phases
    ):
        raise ValueError("Recovery 1 unexpectedly launched evaluation")
    if (
        status.get("status") != "failed"
        or int(status.get("exit_code", -1)) != 1
        or status.get("error_type") != "ValueError"
        or status.get("error") != "Candidate paired distinct count mismatch"
        or status.get("commit") != RECOVERY1_COMMIT
    ):
        raise ValueError("Recovery 1 failure classification changed")
    if claim.get("commit") != RECOVERY1_COMMIT:
        raise ValueError("Recovery 1 claim commit changed")
    if claim.get("spec_sha256") != BASE_EXECUTION_SPEC_SHA256:
        raise ValueError("Recovery 1 claim spec hash changed")

    scenario_by_seed = {
        int(seed): str(scenario)
        for seed, scenario in spec["scenario_by_seed"].items()
    }
    preonline = audit_preonline(Path(spec["preonline_manifest"]))
    training = {
        role: audit_training(
            Path(spec[f"{role}_training_manifest"]),
            role=role,
            scenario_by_seed=scenario_by_seed,
            maximum_parameter_gap=float(spec["maximum_parameter_relative_gap"]),
            minimum_behavior_delta=float(
                spec["minimum_structured_behavior_linf_delta"]
            ),
        )
        for role in ("control", "candidate")
    }
    return {
        "failure_classification": "post-training audit statistic mismatch",
        "scientific_training_failure": False,
        "training_jobs_reused_read_only": 12,
        "new_training_jobs": 0,
        "evaluation_runs_preexisting": 0,
        "immutable_snapshot": snapshot,
        "preonline_audit": preonline,
        "training_audits": training,
    }


def preflight(spec_path: Path, expected_commit: str) -> dict[str, Any]:
    spec = read_json(spec_path)
    actual_commit = git_output("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if git_output("status", "--short"):
        raise ValueError("Stage F1 Recovery 2 requires a clean worktree")
    validate_scientific_contract(spec)
    locks = verify_recovery_lock_manifest(spec_path, spec)
    for raw in spec["fresh_output_roots"]:
        path = Path(str(raw))
        if path.exists():
            raise FileExistsError(f"Recovery 2 output already exists: {path}")
    commands = build_phase_commands(spec)
    environment = environment_fingerprint()
    tests = run_focused_tests(spec)
    recovery1 = verify_recovery1(spec)
    f0 = audit_f0_identifiability(Path(spec["f0_summary"]))
    return {
        "commit": actual_commit,
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "lock_audit": locks,
        "environment": environment,
        "preflight_tests": tests,
        "recovery1": recovery1,
        "f0_identifiability_audit": f0,
        "phase_commands": [
            {"name": name, "command": command} for name, command in commands
        ],
        "evaluation_only": True,
        "formal_confirmation_authorized": False,
    }


def run_recovery(spec_path: Path, expected_commit: str) -> None:
    spec = read_json(spec_path)
    preflight_result = preflight(spec_path, expected_commit)
    launcher_root = Path(spec["launcher_root"])
    claim_path = claim_execution(
        launcher_root,
        preflight_result=preflight_result,
    )
    status: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": expected_commit,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "claim": str(claim_path),
        "phase": "post_training_audit_complete",
        "phases": [],
        "evaluation_only": True,
        "reused_training_jobs": 12,
        "new_training_jobs": 0,
        "recovery1_audit": preflight_result["recovery1"],
    }
    atomic_write_json(launcher_root / "status.json", status)
    try:
        phases = build_phase_commands(spec)
        for name, command in phases:
            run_phase(launcher_root, name, command, status)
            if name.startswith("evaluation_"):
                role = name.split("_")[1]
                status[f"evaluation_{role}_audit"] = audit_evaluation_curve(
                    Path(spec[f"{role}_evaluation_config"])
                )
                atomic_write_json(launcher_root / "status.json", status)

        comparison = read_json(Path(spec["comparison_output"]))
        status["decision"] = comparison["decision"]
        inventory = artifact_inventory(Path(spec["campaign_root"]))
        inventory_path = launcher_root / "artifact_sha256.json"
        atomic_write_json(inventory_path, inventory)
        status.update(
            {
                "status": "completed",
                "phase": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "artifact_inventory": str(inventory_path),
                "artifact_count": len(inventory),
                "formal_confirmation_launched": False,
            }
        )
        atomic_write_json(launcher_root / "status.json", status)
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(launcher_root / "status.json", status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-recovery1-digests", action="store_true")
    args = parser.parse_args()
    spec_path = Path(args.spec)
    if args.print_recovery1_digests:
        print(
            json.dumps(
                recovery1_digest_snapshot(read_json(spec_path)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.expected_commit:
        parser.error("--expected-commit is required")
    if args.preflight_only:
        print(
            json.dumps(
                preflight(spec_path, str(args.expected_commit)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_recovery(spec_path, str(args.expected_commit))


if __name__ == "__main__":
    main()
