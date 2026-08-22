"""Run provenance-locked CPU headroom shards without a chat output pipe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.headroom_teacher_bundle import (
    create_teacher_bundle,
    local_provenance,
    sha256_file,
    validate_teacher_root,
)
from src.rl.config import load_config


PHASES = ("state-probe", "teacher", "bundle")
LOCKED_BRANCH = "codex/patient-indexed-specimen-routing"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def assert_locked_checkout(repo_root: Path, expected_commit: str) -> None:
    branch = git_output(repo_root, "branch", "--show-current")
    if branch != LOCKED_BRANCH:
        raise ValueError(f"Expected branch {LOCKED_BRANCH}, found {branch}")
    commit = git_output(repo_root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"Expected commit {expected_commit}, found {commit}")
    dirty = git_output(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("Tracked worktree must be clean before a formal CPU phase")


def claim_phase(
    output_root: Path,
    phase: str,
    payload: dict[str, Any],
) -> tuple[Path, Path, Path]:
    orchestration = output_root / "orchestration"
    orchestration.mkdir(parents=True, exist_ok=True)
    claim_path = orchestration / f"{phase}.claim.json"
    status_path = orchestration / f"{phase}.status.json"
    logs_root = orchestration / f"{phase}-logs"
    if status_path.exists() or logs_root.exists():
        raise FileExistsError(f"Formal phase already has outputs: {phase}")
    with claim_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logs_root.mkdir()
    return claim_path, status_path, logs_root


def build_shard_command(
    python_executable: str,
    config_argument: str,
    phase: str,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    if phase == "state-probe":
        shard_arguments = (
            "--skip-teacher",
            "--state-probe-shard-index",
            str(shard_index),
            "--state-probe-shard-count",
            str(shard_count),
        )
    elif phase == "teacher":
        shard_arguments = (
            "--skip-state-probe",
            "--teacher-shard-index",
            str(shard_index),
            "--teacher-shard-count",
            str(shard_count),
        )
    else:
        raise ValueError(f"Unsupported shard phase: {phase}")
    return [
        python_executable,
        "-m",
        "evaluation.network_residual_headroom",
        "--config",
        config_argument,
        *shard_arguments,
    ]


def run_logged_command(
    command: list[str],
    *,
    repo_root: Path,
    logs_root: Path,
    job_name: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    stdout_path = logs_root / f"{job_name}.stdout.log"
    stderr_path = logs_root / f"{job_name}.stderr.log"
    status_path = logs_root / f"{job_name}.status.json"
    started_at = utc_now()
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        atomic_write_json(
            status_path,
            {
                "state": "running",
                "started_at": started_at,
                "pid": process.pid,
                "command": command,
            },
        )
        exit_code = process.wait()
    result = {
        "state": "completed" if exit_code == 0 else "failed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "pid": process.pid,
        "exit_code": exit_code,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    atomic_write_json(status_path, result)
    return result


def phase_environment(
    repo_root: Path,
    logs_root: Path,
    job_name: str,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(repo_root),
            "PYTHONPYCACHEPREFIX": str(
                logs_root / "python-cache" / job_name
            ),
            "MPLCONFIGDIR": str(logs_root / "matplotlib-cache" / job_name),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def run_shards(
    commands: list[list[str]],
    *,
    repo_root: Path,
    logs_root: Path,
    max_workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(commands)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_logged_command,
                command,
                repo_root=repo_root,
                logs_root=logs_root,
                job_name=f"shard_{index:02d}",
                environment=phase_environment(
                    repo_root,
                    logs_root,
                    f"shard_{index:02d}",
                ),
            ): index
            for index, command in enumerate(commands)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    completed = [result for result in results if result is not None]
    failures = [result for result in completed if result["exit_code"] != 0]
    if failures:
        raise RuntimeError(
            "Formal CPU shards failed without retry: "
            + ", ".join(str(result["pid"]) for result in failures)
        )
    return completed


def run_merge(
    phase: str,
    *,
    python_executable: str,
    config_argument: str,
    repo_root: Path,
    logs_root: Path,
) -> dict[str, Any]:
    module = (
        "evaluation.merge_headroom_state_probe_shards"
        if phase == "state-probe"
        else "evaluation.merge_headroom_teacher_shards"
    )
    result = run_logged_command(
        [
            python_executable,
            "-m",
            module,
            "--config",
            config_argument,
        ],
        repo_root=repo_root,
        logs_root=logs_root,
        job_name="merge",
        environment=phase_environment(repo_root, logs_root, "merge"),
    )
    if result["exit_code"] != 0:
        raise RuntimeError(f"Formal {phase} merge failed without retry")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--bundle-output", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    config = load_config(config_path)
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    config_argument = str(config_path.relative_to(repo_root))
    assert_locked_checkout(repo_root, args.expected_commit)
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.phase == "state-probe" and output_root.exists():
        raise FileExistsError(f"Fresh teacher root already exists: {output_root}")
    if args.phase != "state-probe" and not output_root.is_dir():
        raise FileNotFoundError(f"Teacher root does not exist: {output_root}")
    if args.phase == "teacher":
        required_state_outputs = (
            output_root / "state_probe.csv",
            output_root / "summary.json",
        )
        for path in required_state_outputs:
            if not path.is_file():
                raise FileNotFoundError(f"Missing merged state probe: {path}")
        summary = json.loads((output_root / "summary.json").read_text())
        if not isinstance(summary.get("state_probe_shards"), dict):
            raise ValueError("Merged state-probe shard provenance is missing")
        for path in (
            output_root / "shards",
            output_root / "teacher_cache.npz",
            output_root / "anchor.csv",
            output_root / "teacher.csv",
        ):
            if path.exists():
                raise FileExistsError(f"Teacher output is not fresh: {path}")

    claim_payload = {
        "state": "claimed",
        "phase": args.phase,
        "claimed_at": utc_now(),
        "git_branch": git_output(repo_root, "branch", "--show-current"),
        "git_commit": args.expected_commit,
        "config_path": config_argument,
        "config_sha256": sha256_file(config_path),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "shard_count": args.shard_count,
        "max_workers": args.max_workers,
        "bundle_output": args.bundle_output,
    }
    _claim, status_path, logs_root = claim_phase(
        output_root,
        args.phase,
        claim_payload,
    )
    try:
        details: dict[str, Any]
        if args.phase in ("state-probe", "teacher"):
            locked_count = int(
                config[
                    "state_probe_rollouts"
                    if args.phase == "state-probe"
                    else "teacher_replications"
                ]
            )
            shard_count = args.shard_count or locked_count
            if not 1 <= shard_count <= locked_count:
                raise ValueError("Shard count exceeds the locked work units")
            commands = [
                build_shard_command(
                    sys.executable,
                    config_argument,
                    args.phase,
                    index,
                    shard_count,
                )
                for index in range(shard_count)
            ]
            shard_results = run_shards(
                commands,
                repo_root=repo_root,
                logs_root=logs_root,
                max_workers=min(args.max_workers, shard_count),
            )
            merge_result = run_merge(
                args.phase,
                python_executable=sys.executable,
                config_argument=config_argument,
                repo_root=repo_root,
                logs_root=logs_root,
            )
            details = {
                "shards": shard_results,
                "merge": merge_result,
            }
            if args.phase == "teacher":
                validate_teacher_root(output_root, config)
        else:
            if args.shard_count is not None:
                raise ValueError("Bundle phase does not accept --shard-count")
            if not args.bundle_output:
                raise ValueError("Bundle phase requires --bundle-output")
            bundle_output = Path(args.bundle_output)
            if not bundle_output.is_absolute():
                bundle_output = repo_root / bundle_output
            validate_teacher_root(output_root, config)
            details = create_teacher_bundle(
                output_root,
                bundle_output,
                expected_config=config,
                provenance=local_provenance(
                    repo_root,
                    config_path,
                    source_root=output_root,
                ),
            )
    except BaseException as error:
        atomic_write_json(
            status_path,
            {
                **claim_payload,
                "state": "failed",
                "completed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            **claim_payload,
            "state": "completed",
            "completed_at": utc_now(),
            "details": details,
        },
    )
    print(json.dumps(details, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
