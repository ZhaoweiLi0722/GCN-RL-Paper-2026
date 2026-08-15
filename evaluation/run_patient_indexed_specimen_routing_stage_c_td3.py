"""One-claim executor for the routing-primary Stage C TD3 screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SPEC = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_stage_c_td3_execution.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_locked_assets(spec: dict[str, Any]) -> dict[str, str]:
    verified = {}
    for raw in spec["locked_files"]:
        entry = dict(raw)
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = str(entry["sha256"]).lower()
        if actual != expected:
            raise ValueError(
                f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
            )
        verified[str(path)] = actual
    return verified


def validate_scientific_contract(spec: dict[str, Any]) -> None:
    training = read_json(Path(spec["training_config"]))
    expected_algorithms = [
        "gcn_residual_mdl2_network_td3_bc",
        "flat_residual_mdl2_network_td3_bc",
    ]
    observed_algorithms = [
        str(entry["name"])
        for entry in training["algorithms"]
    ]
    if observed_algorithms != expected_algorithms:
        raise ValueError("Stage C algorithm order does not match the lock")
    expected_seeds = [int(value) for value in spec["training_seeds"]]
    if expected_seeds != [20, 21, 22]:
        raise ValueError("Stage C training seeds must be 20, 21, and 22")
    for entry in training["algorithms"]:
        if [int(value) for value in entry["seeds"]] != expected_seeds:
            raise ValueError("GCN and flat must use the same locked seeds")
    if int(training["online_episodes"]) != 100:
        raise ValueError("Stage C requires 100 online episodes per run")
    overrides = dict(training["config_overrides"])
    if str(overrides.get("device")) != "mps":
        raise ValueError("Stage C requires Apple MPS without CPU fallback")
    required = {
        "checkpoint_interval": 5,
        "training_state_checkpoint_interval": 5,
        "policy_delay": 2,
        "actor_update_frequency": 2,
    }
    for key, expected in required.items():
        if int(overrides.get(key, -1)) != expected:
            raise ValueError(f"Stage C lock mismatch for {key}")
    if float(overrides.get("policy_noise", -1.0)) != 0.01:
        raise ValueError("Stage C policy_noise must be 0.01")
    if float(overrides.get("noise_clip", -1.0)) != 0.02:
        raise ValueError("Stage C noise_clip must be 0.02")
    if float(
        overrides["residual_action"]["group_scales"]["specimen_transfer"]
    ) != 0.1:
        raise ValueError("Stage C specimen residual scale must be 0.1")
    if any(
        int(seed) in {91100000, 93100000}
        for seed in spec["development_crn_seeds"]
    ):
        raise ValueError("Stage C reused a forbidden CRN stream")


def environment_fingerprint() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for Stage C") from error
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; Stage C forbids CPU fallback")
    fallback_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if fallback_value.strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; Stage C forbids "
            "CPU fallback"
        )
    probe = torch.ones(4, device="mps")
    probe = probe.square().sum()
    torch.mps.synchronize()
    if probe.device.type != "mps" or float(probe.cpu()) != 4.0:
        raise RuntimeError("MPS execution probe failed")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_probe_device": str(probe.device),
        "pytorch_enable_mps_fallback": fallback_value,
        "machine": platform.machine(),
    }


def claim_execution(
    root: Path,
    *,
    expected_commit: str,
    spec_path: Path,
    fingerprint: dict[str, Any],
    verified_hashes: dict[str, str],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    claim_path = root / "claim.json"
    payload = {
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "expected_commit": expected_commit,
        "stage_spec": str(spec_path),
        "stage_spec_sha256": sha256_file(spec_path),
        "environment": fingerprint,
        "verified_hashes": verified_hashes,
    }
    descriptor = os.open(
        claim_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def run_phase(
    root: Path,
    name: str,
    command: list[str],
    status: dict[str, Any],
) -> None:
    stdout_path = root / f"{name}.stdout.log"
    stderr_path = root / f"{name}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"Phase log already exists: {name}")
    phase = {
        "name": name,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    status["phase"] = name
    status.setdefault("phases", []).append(phase)
    atomic_write_json(root / "status.json", status)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    with stdout_path.open("xb") as stdout_handle, stderr_path.open(
        "xb"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
            check=False,
        )
    phase["completed_at"] = datetime.now(timezone.utc).isoformat()
    phase["exit_code"] = int(completed.returncode)
    phase["stdout_sha256"] = sha256_file(stdout_path)
    phase["stderr_sha256"] = sha256_file(stderr_path)
    atomic_write_json(root / "status.json", status)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage C phase {name} exited {completed.returncode}"
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_finite(row: dict[str, str], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = row.get(key, "")
        if value == "" or not math.isfinite(float(value)):
            raise ValueError(f"Non-finite or missing {key}")


def audit_training(spec: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(spec["training_manifest"])
    manifest = read_json(manifest_path)
    expected = {
        (algorithm, int(seed))
        for algorithm in spec["algorithms"]
        for seed in spec["training_seeds"]
    }
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    if set(runs) != expected:
        raise ValueError("Stage C training manifest is incomplete or duplicated")
    parameter_counts: dict[str, set[int]] = {
        algorithm: set()
        for algorithm in spec["algorithms"]
    }
    audits = []
    for algorithm, seed in sorted(expected):
        run = runs[(algorithm, seed)]
        parameter_counts[algorithm].add(int(run["parameter_count"]))
        run_root = Path(run["config"]).parent
        rows = read_csv_rows(run_root / "training.csv")
        if len(rows) != 100:
            raise ValueError(f"{algorithm} seed {seed} has {len(rows)} rows")
        if [int(row["episode"]) for row in rows] != list(range(100)):
            raise ValueError(f"{algorithm} seed {seed} episode sequence failed")
        routing = sum(float(row["specimen_route_count"]) for row in rows)
        if routing <= 0.0:
            raise ValueError(f"{algorithm} seed {seed} has zero routing")
        updates = sum(float(row["online_rl_updates"]) for row in rows)
        if updates <= 0.0:
            raise ValueError(f"{algorithm} seed {seed} has zero online updates")
        for row in rows:
            require_finite(
                row,
                (
                    "total_cost",
                    "completion_service_level",
                    "patients_lost",
                    "online_rl_updates",
                ),
            )
            for key, value in row.items():
                if key.startswith("online_rl_") and value != "":
                    if not math.isfinite(float(value)):
                        raise ValueError(
                            f"{algorithm} seed {seed} has non-finite {key}"
                        )
        checkpoint_root = run_root / "checkpoints"
        checkpoints = []
        for episode in range(5, 101, 5):
            path = checkpoint_root / (
                f"{algorithm}_seed{seed}_episode{episode}.pt"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints.append(path)
        training_state = Path(run["training_state_checkpoint"])
        if not training_state.is_file():
            raise FileNotFoundError(training_state)
        if not (run_root / "summary.json").is_file():
            raise FileNotFoundError(run_root / "summary.json")
        audits.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "episodes": len(rows),
                "specimen_route_count": routing,
                "online_rl_updates": updates,
                "checkpoint_count": len(checkpoints),
                "training_state_sha256": sha256_file(training_state),
            }
        )
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("Parameter counts vary across matched training seeds")
    resolved_counts = {
        algorithm: next(iter(values))
        for algorithm, values in parameter_counts.items()
    }
    largest = max(resolved_counts.values())
    smallest = min(resolved_counts.values())
    relative_gap = (largest - smallest) / largest
    if relative_gap > float(spec["maximum_parameter_relative_gap"]):
        raise ValueError(
            f"Stage C parameter gap {relative_gap:.6f} exceeds the lock"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "parameter_counts": resolved_counts,
        "parameter_relative_gap": relative_gap,
        "runs": audits,
    }


def nested_metric_total(value: Any, key: str) -> float:
    if isinstance(value, dict):
        total = float(value.get(key, 0.0)) if key in value else 0.0
        return total + sum(
            nested_metric_total(child, key)
            for child_key, child in value.items()
            if child_key != key
        )
    if isinstance(value, list):
        return sum(nested_metric_total(child, key) for child in value)
    return 0.0


def audit_evaluation(
    spec: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    config = read_json(Path(spec[f"{variant}_evaluation_config"]))
    root = Path(config["output_root"])
    phase_summary = read_json(root / "summary.json")
    runs = list(phase_summary["runs"])
    if len(runs) != 6:
        raise ValueError(f"{variant} evaluation requires six runs")
    expected_scenarios = set(spec["scenarios"])
    run_audits = []
    for run in runs:
        algorithm = str(run["algorithm"])
        seed = int(run["training_seed"])
        run_root = root / algorithm / f"seed{seed}"
        learned = read_csv_rows(run_root / "holdout_rows.csv")
        anchor = read_csv_rows(run_root / "holdout_anchor_rows.csv")
        if len(learned) != 200 or len(anchor) != 200:
            raise ValueError(
                f"{variant} {algorithm} seed {seed} row count mismatch"
            )
        for rows in (learned, anchor):
            pairs = {
                (str(row["scenario"]), int(row["replication"]))
                for row in rows
            }
            if len(pairs) != 200:
                raise ValueError("Duplicate evaluation scenario/replication")
            if {scenario for scenario, _ in pairs} != expected_scenarios:
                raise ValueError("Evaluation scenario mismatch")
            for row in rows:
                require_finite(
                    row,
                    (
                        "total_cost",
                        "completion_service_level",
                        "patients_lost",
                        "specimen_route_count",
                    ),
                )
            if sum(float(row["specimen_route_count"]) for row in rows) <= 0:
                raise ValueError("Evaluation has zero specimen routing")
        corrected = nested_metric_total(run, "corrected_decisions")
        if corrected <= 0.0:
            raise ValueError("Learned evaluation has zero residual correction")
        run_audits.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "learned_rows": len(learned),
                "anchor_rows": len(anchor),
                "corrected_decisions": corrected,
            }
        )
    return {
        "variant": variant,
        "summary": str(root / "summary.json"),
        "summary_sha256": sha256_file(root / "summary.json"),
        "runs": run_audits,
    }


def artifact_inventory(root: Path) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_stage(
    spec_path: Path,
    expected_commit: str,
    *,
    recover: bool = False,
) -> None:
    spec = read_json(spec_path)
    actual_commit = git_output("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    tracked_status = git_output("status", "--short", "--untracked-files=no")
    if tracked_status:
        raise ValueError("Tracked worktree must be clean before Stage C")
    verify_locked_assets(spec)
    validate_scientific_contract(spec)
    training_root = Path(spec["campaign_root"]) / "training"
    for raw_path in spec["fresh_output_roots"]:
        path = Path(raw_path)
        if recover and path == training_root:
            if not path.exists():
                raise FileNotFoundError(
                    f"Recovery requires existing training output: {path}"
                )
            continue
        if path.exists():
            raise FileExistsError(f"Stage C output already exists: {path}")
    fingerprint = environment_fingerprint()
    verified_hashes = verify_locked_assets(spec)
    launcher_root = Path(spec["launcher_root"])
    if recover:
        launcher_root = launcher_root / "recovery"
    claim_path = claim_execution(
        launcher_root,
        expected_commit=expected_commit,
        spec_path=spec_path,
        fingerprint=fingerprint,
        verified_hashes=verified_hashes,
    )
    status: dict[str, Any] = {
        "status": "running",
        "mode": "recovery" if recover else "initial",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": actual_commit,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "claim": str(claim_path),
        "phases": [],
    }
    atomic_write_json(launcher_root / "status.json", status)
    try:
        run_phase(
            launcher_root,
            "focused_tests",
            [sys.executable, "-m", "unittest", *spec["focused_tests"]],
            status,
        )
        if not recover:
            run_phase(
                launcher_root,
                "training",
                [
                    sys.executable,
                    "-m",
                    "evaluation.train_multiscenario_network_residual",
                    "--config",
                    str(spec["training_config"]),
                ],
                status,
            )
        status["training_audit"] = audit_training(spec)
        atomic_write_json(launcher_root / "status.json", status)
        for variant in ("final", "pretrain"):
            run_phase(
                launcher_root,
                f"evaluation_{variant}",
                [
                    sys.executable,
                    "-m",
                    "evaluation.evaluate_multiscenario_network_residual",
                    "--config",
                    str(spec[f"{variant}_evaluation_config"]),
                ],
                status,
            )
            status[f"evaluation_{variant}_audit"] = audit_evaluation(
                spec,
                variant=variant,
            )
            atomic_write_json(launcher_root / "status.json", status)
        inventory = artifact_inventory(Path(spec["campaign_root"]))
        atomic_write_json(
            launcher_root / "artifact_sha256.json",
            inventory,
        )
        status.update(
            {
                "status": "completed",
                "phase": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "artifact_inventory": str(
                    launcher_root / "artifact_sha256.json"
                ),
                "artifact_count": len(inventory),
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
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    run_stage(
        Path(args.spec),
        str(args.expected_commit),
        recover=bool(args.recover),
    )


if __name__ == "__main__":
    main()
