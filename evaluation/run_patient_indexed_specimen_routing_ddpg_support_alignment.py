"""One-claim executor for the paired DDPG support-alignment experiment."""

from __future__ import annotations

import argparse
import copy
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
    "patient_indexed_specimen_routing_ddpg_support_alignment_execution.json"
)
ALGORITHMS = (
    "gcn_residual_mdl2_network_ddpg_afd",
    "flat_residual_mdl2_network_ddpg_afd",
)
SEEDS = (30, 31, 32)
RUNTIME_CONFIG_KEYS = (
    "control_training_config",
    "candidate_training_config",
    "control_final_evaluation_config",
    "control_pretrain_evaluation_config",
    "candidate_final_evaluation_config",
    "candidate_pretrain_evaluation_config",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rewrite_path_prefix(value: Any, source: str, target: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_path_prefix(child, source, target)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_path_prefix(child, source, target)
            for child in value
        ]
    if isinstance(value, str) and value.startswith(source):
        return target + value[len(source):]
    return value


def materialize_runtime_configs(
    spec: dict[str, Any],
    launcher_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create path-only recovery configs from the locked scientific assets."""

    rewrite = spec.get("runtime_config_path_rewrite")
    explicit = spec.get("runtime_config_overrides")
    if rewrite is not None and explicit is not None:
        raise ValueError("Choose one runtime config path mechanism")
    if rewrite is None and explicit is None:
        return copy.deepcopy(spec), {"enabled": False, "configs": []}
    if rewrite is not None and not isinstance(rewrite, dict):
        raise ValueError("Runtime config path rewrite must be an object")
    if explicit is not None and not isinstance(explicit, dict):
        raise ValueError("Runtime config overrides must be an object")
    runtime_root = launcher_root / "runtime-configs"
    runtime_root.mkdir(parents=True, exist_ok=False)
    resolved = copy.deepcopy(spec)
    provenance = {
        "enabled": True,
        "mode": "prefix_rewrite" if rewrite is not None else "explicit",
        "scientific_values_modified": False,
        "configs": [],
    }
    if rewrite is not None:
        source = str(rewrite["source_prefix"])
        target = str(rewrite["target_prefix"])
        if not source or not target or source == target:
            raise ValueError(
                "Runtime config path rewrite must use distinct prefixes"
            )
        provenance["source_prefix"] = source
        provenance["target_prefix"] = target
    for key in RUNTIME_CONFIG_KEYS:
        source_path = Path(str(spec[key]))
        source_config = read_json(source_path)
        if rewrite is not None:
            runtime_config = _rewrite_path_prefix(
                source_config,
                source,
                target,
            )
            expected = (
                ["output_root"]
                if key.endswith("training_config")
                else ["output_root", "training_manifest"]
            )
        else:
            overrides = explicit.get(key, {})
            if not isinstance(overrides, dict):
                raise ValueError(f"Runtime overrides for {key} must be an object")
            allowed = (
                {"output_root"}
                if key.endswith("training_config")
                else {"output_root", "training_manifest"}
            )
            unexpected = set(overrides) - allowed
            if unexpected:
                raise ValueError(
                    f"Scientific runtime override for {key}: "
                    + ", ".join(sorted(unexpected))
                )
            runtime_config = copy.deepcopy(source_config)
            runtime_config.update(overrides)
            expected = sorted(overrides)
        differences = sorted(
            path
            for path in set(_flatten(source_config)) | set(_flatten(runtime_config))
            if _flatten(source_config).get(path) != _flatten(runtime_config).get(path)
        )
        if differences != expected:
            raise ValueError(
                f"Unexpected runtime config rewrite for {key}: "
                + ", ".join(differences)
            )
        runtime_path = runtime_root / f"{key}__{source_path.name}"
        atomic_write_json(runtime_path, runtime_config)
        resolved[key] = str(runtime_path)
        provenance["configs"].append(
            {
                "key": key,
                "source": str(source_path),
                "source_sha256": sha256_file(source_path),
                "runtime": str(runtime_path),
                "runtime_sha256": sha256_file(runtime_path),
                "changed_paths": differences,
            }
        )
    return resolved, provenance


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_locked_assets(spec: dict[str, Any]) -> dict[str, str]:
    verified = {}
    for raw in spec["locked_files"]:
        path = Path(raw["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = str(raw["sha256"]).lower()
        if actual != expected:
            raise ValueError(
                f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
            )
        verified[str(path)] = actual
    return verified


def verify_reused_preonline_source(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    """Verify the immutable manifest and every episode-0 artifact it names."""

    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256.lower():
        raise ValueError(
            "Reused pre-online manifest SHA256 mismatch: expected "
            f"{expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    manifest = read_json(manifest_path)
    paths = {manifest_path}
    for run in manifest.get("runs", ()):
        paths.add(Path(str(run["preonline_training_state"])))
        paths.add(Path(str(run["pretrain_checkpoint"])))
    inventory = {
        str(path): sha256_file(path)
        for path in sorted(paths, key=lambda value: str(value))
    }
    tree_payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tree_sha256 = hashlib.sha256(tree_payload).hexdigest()
    if tree_sha256 != expected_tree_sha256.lower():
        raise ValueError(
            "Reused pre-online artifact tree SHA256 mismatch: expected "
            f"{expected_tree_sha256}, got {tree_sha256}"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "artifact_tree_sha256": tree_sha256,
        "artifact_count": len(inventory),
        "artifacts": inventory,
    }


def verify_immutable_artifact_tree(
    root: Path,
    *,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    inventory = {
        str(path): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda value: str(value))
        if path.is_file()
    }
    tree_payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tree_sha256 = hashlib.sha256(tree_payload).hexdigest()
    if tree_sha256 != expected_tree_sha256.lower():
        raise ValueError(
            f"Immutable artifact tree SHA256 mismatch for {root}: "
            f"expected {expected_tree_sha256}, got {tree_sha256}"
        )
    return {
        "root": str(root),
        "artifact_tree_sha256": tree_sha256,
        "artifact_count": len(inventory),
        "artifacts": inventory,
    }


def verify_reused_control_recovery(spec: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(str(spec["reuse_control_status"]))
    stderr_path = Path(str(spec["reuse_control_stderr"]))
    manifest_path = Path(str(spec["reuse_control_training_manifest"]))
    provenance_path = Path(str(spec["reuse_paired_state_provenance"]))
    locked_files = (
        (
            status_path,
            str(spec["reuse_control_status_sha256"]),
        ),
        (
            stderr_path,
            str(spec["reuse_control_stderr_sha256"]),
        ),
        (
            manifest_path,
            str(spec["reuse_control_training_manifest_sha256"]),
        ),
        (
            provenance_path,
            str(spec["reuse_paired_state_provenance_sha256"]),
        ),
    )
    verified_files = {}
    for path, expected in locked_files:
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(
                f"Reused Recovery 1 SHA256 mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
        verified_files[str(path)] = actual

    status = read_json(status_path)
    expected_phases = [
        "focused_tests",
        *(
            f"training_control_{algorithm}_seed{seed}"
            for algorithm in ALGORITHMS
            for seed in SEEDS
        ),
    ]
    phases = list(status.get("phases", ()))
    if [str(phase.get("name")) for phase in phases] != expected_phases:
        raise ValueError("Recovery 1 did not stop after exactly six controls")
    if any(int(phase.get("exit_code", -1)) != 0 for phase in phases):
        raise ValueError("A reused Recovery 1 phase did not exit cleanly")
    if (
        status.get("status") != "failed"
        or status.get("phase")
        != "training_control_flat_residual_mdl2_network_ddpg_afd_seed32"
        or status.get("error_type") != "Error"
        or "field larger than field limit" not in str(status.get("error"))
    ):
        raise ValueError("Recovery 1 failure classification does not match")
    if any("candidate" in str(phase.get("name")) for phase in phases):
        raise ValueError("Recovery 1 unexpectedly consumed a candidate run")

    control_tree = verify_immutable_artifact_tree(
        Path(str(spec["reuse_control_training_root"])),
        expected_tree_sha256=str(
            spec["reuse_control_training_artifact_tree_sha256"]
        ),
    )
    paired_tree = verify_immutable_artifact_tree(
        Path(str(spec["reuse_paired_state_root"])),
        expected_tree_sha256=str(
            spec["reuse_paired_state_artifact_tree_sha256"]
        ),
    )
    provenance = read_json(provenance_path)
    runs = list(provenance.get("runs", ()))
    if len(runs) != len(ALGORITHMS) * len(SEEDS):
        raise ValueError("Recovery 1 paired-state provenance is incomplete")
    for run in runs:
        candidate = Path(str(run["candidate_clone"]))
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if sha256_file(candidate) != str(run["candidate_clone_sha256"]):
            raise ValueError(f"Recovery 1 candidate clone changed: {candidate}")
    return {
        "failure_classification": "post-control CSV audit field-size limit",
        "scientific_training_failure": False,
        "candidate_runs_consumed": 0,
        "control_runs_reused": len(ALGORITHMS) * len(SEEDS),
        "verified_files": verified_files,
        "control_training_tree": control_tree,
        "paired_state_tree": paired_tree,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
        return result
    return {prefix: value}


def validate_scientific_contract(spec: dict[str, Any]) -> None:
    control = read_json(Path(spec["control_training_config"]))
    candidate = read_json(Path(spec["candidate_training_config"]))
    expected_algorithms = list(ALGORITHMS)
    expected_seeds = list(SEEDS)
    for name, config in (("control", control), ("candidate", candidate)):
        observed = [str(entry["name"]) for entry in config["algorithms"]]
        if observed != expected_algorithms:
            raise ValueError(f"{name} algorithm order does not match the lock")
        if [int(value) for value in config["seeds"]] != expected_seeds:
            raise ValueError(f"{name} training seeds do not match the lock")
        for entry in config["algorithms"]:
            if [int(value) for value in entry["seeds"]] != expected_seeds:
                raise ValueError(f"{name} algorithm seeds are not matched")
        if int(config["online_episodes"]) != 100:
            raise ValueError(f"{name} requires 100 online episodes")
        overrides = dict(config["config_overrides"])
        if str(overrides.get("device")) != "mps":
            raise ValueError(f"{name} requires Apple MPS")
        if int(overrides.get("checkpoint_interval", -1)) != 5:
            raise ValueError(f"{name} checkpoint interval must be five")
        if int(overrides.get("training_state_checkpoint_interval", -1)) != 5:
            raise ValueError(f"{name} state interval must be five")
    control_flat = _flatten(control)
    candidate_flat = _flatten(candidate)
    differences = {
        key
        for key in set(control_flat) | set(candidate_flat)
        if control_flat.get(key) != candidate_flat.get(key)
    }
    expected_differences = {
        "name",
        "experimental_role",
        "config_overrides.critic_teacher_advantage_calibration."
        "allowed_option_groups",
        "output_root",
    }
    if differences != expected_differences:
        raise ValueError(
            "Control/candidate scientific diff is not single-factor: "
            + ", ".join(sorted(differences))
        )
    allowed = candidate["config_overrides"][
        "critic_teacher_advantage_calibration"
    ].get("allowed_option_groups")
    if allowed != ["specimen_transfer"]:
        raise ValueError("Candidate support must be anchor + specimen_transfer")

    forbidden = {int(value) for value in spec["forbidden_crn_seeds"]}
    development = {int(value) for value in spec["development_crn_seeds"]}
    if forbidden & development:
        raise ValueError("Development evaluation reused a forbidden CRN stream")
    for role in ("control", "candidate"):
        for variant in ("final", "pretrain"):
            evaluation = read_json(
                Path(spec[f"{role}_{variant}_evaluation_config"])
            )
            if int(evaluation["holdout_seed"]) != 95_100_000:
                raise ValueError("Support evaluation holdout stream is not locked")
            if int(evaluation["validation_seed"]) != 95_000_000:
                raise ValueError("Support validation stream is not locked")
            if int(evaluation["holdout_replications"]) != 50:
                raise ValueError("Support evaluation requires 50 replications")
            if str(evaluation["fixed_checkpoint_variant"]) != variant:
                raise ValueError(f"{role} {variant} checkpoint variant mismatch")


def environment_fingerprint() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required") from error
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; CPU fallback is forbidden")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if fallback.strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    probe = torch.ones(4, device="mps").square().sum()
    torch.mps.synchronize()
    if probe.device.type != "mps" or float(probe.cpu()) != 4.0:
        raise RuntimeError("MPS execution probe failed")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_probe_device": str(probe.device),
        "pytorch_enable_mps_fallback": fallback,
    }


def preflight(spec_path: Path, expected_commit: str) -> dict[str, Any]:
    spec = read_json(spec_path)
    actual_commit = git_output("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if git_output("status", "--porcelain"):
        raise ValueError("Worktree must be clean before execution")
    hashes = verify_locked_assets(spec)
    validate_scientific_contract(spec)
    source_manifest = spec.get("reuse_preonline_manifest")
    reused_preonline = None
    if source_manifest not in (None, ""):
        reused_preonline = verify_reused_preonline_source(
            Path(str(source_manifest)),
            expected_manifest_sha256=str(
                spec["reuse_preonline_manifest_sha256"]
            ),
            expected_tree_sha256=str(
                spec["reuse_preonline_artifact_tree_sha256"]
            ),
        )
    reused_control = None
    if spec.get("reuse_control_training_manifest") not in (None, ""):
        reused_control = verify_reused_control_recovery(spec)
    for raw in spec["fresh_output_roots"]:
        if Path(raw).exists():
            raise FileExistsError(f"Fresh output already exists: {raw}")
    result = {
        "status": "passed",
        "commit": actual_commit,
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "locked_files": hashes,
        "environment": environment_fingerprint(),
    }
    if reused_preonline is not None:
        result["reused_preonline"] = reused_preonline
    if reused_control is not None:
        result["reused_control_recovery"] = reused_control
    return result


def claim_execution(
    root: Path,
    *,
    preflight_result: dict[str, Any],
    expected_commit: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "claim.json"
    payload = {
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "expected_commit": expected_commit,
        "preflight": preflight_result,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


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
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
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
        raise RuntimeError(f"Phase {name} exited {completed.returncode}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_finite(row: dict[str, str], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = row.get(key, "")
        if value == "" or not math.isfinite(float(value)):
            raise ValueError(f"Non-finite or missing {key}")


def audit_preonline_preparation(manifest_path: Path) -> dict[str, Any]:
    import torch

    manifest = read_json(manifest_path)
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    if set(runs) != expected:
        raise ValueError("Pre-online manifest is incomplete or duplicated")
    audits = []
    for key in sorted(expected):
        run = runs[key]
        if int(run["online_episodes"]) != 0:
            raise ValueError(f"Pre-online source {key} consumed a trajectory")
        state_path = Path(str(run["preonline_training_state"]))
        pretrain_path = Path(str(run["pretrain_checkpoint"]))
        if not state_path.is_file() or not pretrain_path.is_file():
            raise FileNotFoundError(f"Incomplete pre-online source for {key}")
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
        training = dict(checkpoint["training"])
        if (
            int(training.get("next_episode", -1)) != 0
            or int(training.get("global_step", -1)) != 0
            or list(training.get("rows", ()))
        ):
            raise ValueError(f"Pre-online source {key} is not episode-0 atomic")
        contract = dict(checkpoint["training_contract"])
        if int(contract.get("num_episodes", -1)) != 0:
            raise ValueError(f"Pre-online source {key} contract is not zero-episode")
        calibration = dict(contract["critic_teacher_advantage_calibration"])
        if bool(calibration.get("enabled", True)):
            raise ValueError(f"Pre-online source {key} calibrated its critic")
        if float(calibration.get("online_ranking_weight", -1.0)) != 0.0:
            raise ValueError(f"Pre-online source {key} retained online ranking")
        if "allowed_option_groups" in calibration:
            raise ValueError(f"Pre-online source {key} is already filtered")
        audits.append(
            {
                "algorithm": key[0],
                "seed": key[1],
                "state": str(state_path),
                "state_sha256": sha256_file(state_path),
                "pretrain_checkpoint": str(pretrain_path),
                "pretrain_checkpoint_sha256": sha256_file(pretrain_path),
            }
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "runs": audits,
    }


def clone_paired_preonline_states(
    preonline_manifest_path: Path,
    *,
    control_config_path: Path,
    candidate_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Clone one state into exact control and support-aligned contracts."""

    import torch

    from src.rl.training_state import training_contract_sha256

    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in read_json(preonline_manifest_path)["runs"]
    }
    target_calibrations = {
        "control": dict(
            read_json(control_config_path)["config_overrides"][
                "critic_teacher_advantage_calibration"
            ]
        ),
        "candidate": dict(
            read_json(candidate_config_path)["config_overrides"][
                "critic_teacher_advantage_calibration"
            ]
        ),
    }
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experimental_role": (
            "metadata-only contract clones for paired control and "
            "support-aligned online forks"
        ),
        "changed_scientific_paths": [
            "num_episodes",
            "history_screen.online_episodes",
            "history_screen.pretrain_only",
            "critic_teacher_advantage_calibration.enabled",
            "critic_teacher_advantage_calibration.online_ranking_weight",
            "critic_teacher_advantage_calibration.allowed_option_groups",
        ],
        "agent_replay_optimizer_rng_payload_modified": False,
        "runs": [],
    }
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            source = Path(str(runs[(algorithm, seed)]["preonline_training_state"]))
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if int(checkpoint["training"]["next_episode"]) != 0:
                raise ValueError("Only an episode-0 state may be cloned")
            entry = {
                "algorithm": algorithm,
                "seed": seed,
                "source": str(source),
                "source_sha256": sha256_file(source),
                "source_training_contract_sha256": checkpoint[
                    "training_contract_sha256"
                ],
            }
            for role in ("control", "candidate"):
                cloned = copy.deepcopy(checkpoint)
                contract = copy.deepcopy(dict(cloned["training_contract"]))
                contract["num_episodes"] = 100
                history_screen = dict(contract["history_screen"])
                history_screen["online_episodes"] = 100
                history_screen["pretrain_only"] = False
                contract["history_screen"] = history_screen
                calibration = dict(
                    contract["critic_teacher_advantage_calibration"]
                )
                calibration.update(target_calibrations[role])
                if role == "control":
                    calibration.pop("allowed_option_groups", None)
                contract["critic_teacher_advantage_calibration"] = calibration
                cloned["training_contract"] = contract
                cloned["training_contract_sha256"] = training_contract_sha256(
                    contract
                )
                target = (
                    output_root
                    / role
                    / f"{algorithm}_seed{seed}_{role}_preonline.pt"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(
                    f".{target.name}.{os.getpid()}.tmp"
                )
                torch.save(cloned, temporary)
                os.replace(temporary, target)
                entry[f"{role}_clone"] = str(target)
                entry[f"{role}_clone_sha256"] = sha256_file(target)
                entry[f"{role}_training_contract_sha256"] = cloned[
                    "training_contract_sha256"
                ]
            provenance["runs"].append(entry)
    provenance_path = output_root / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    provenance["provenance"] = str(provenance_path)
    provenance["provenance_sha256"] = sha256_file(provenance_path)
    return provenance


def audit_training(
    manifest_path: Path,
    *,
    require_preonline_state: bool,
    maximum_parameter_gap: float,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    if set(runs) != expected:
        raise ValueError("Training manifest is incomplete or duplicated")
    counts: dict[str, set[int]] = {algorithm: set() for algorithm in ALGORITHMS}
    audits = []
    for algorithm, seed in sorted(expected):
        run = runs[(algorithm, seed)]
        counts[algorithm].add(int(run["parameter_count"]))
        run_root = Path(run["config"]).parent
        rows = read_csv_rows(run_root / "training.csv")
        if len(rows) != 100 or [int(row["episode"]) for row in rows] != list(range(100)):
            raise ValueError(f"{algorithm} seed {seed} episode sequence failed")
        routing = sum(float(row["specimen_route_count"]) for row in rows)
        updates = sum(float(row["online_rl_updates"]) for row in rows)
        if routing <= 0.0 or updates <= 0.0:
            raise ValueError(f"{algorithm} seed {seed} lacks routing or updates")
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
                        raise ValueError(f"Non-finite persisted loss {key}")
        checkpoints = [
            run_root
            / "checkpoints"
            / f"{algorithm}_seed{seed}_episode{episode}.pt"
            for episode in range(5, 101, 5)
        ]
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        state = Path(run["training_state_checkpoint"])
        if not state.is_file():
            raise FileNotFoundError(state)
        preonline = run.get("preonline_training_state")
        if require_preonline_state:
            if not preonline or not Path(str(preonline)).is_file():
                raise FileNotFoundError(str(preonline))
        elif preonline not in (None, ""):
            raise ValueError("Online fork unexpectedly created pre-online state")
        audits.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "episodes": len(rows),
                "specimen_route_count": routing,
                "online_rl_updates": updates,
                "checkpoint_count": len(checkpoints),
                "preonline_training_state": preonline,
                "preonline_training_state_sha256": (
                    sha256_file(Path(str(preonline))) if preonline else None
                ),
            }
        )
    resolved = {name: next(iter(values)) for name, values in counts.items()}
    if any(len(values) != 1 for values in counts.values()):
        raise ValueError("Parameter counts vary across seeds")
    gap = (max(resolved.values()) - min(resolved.values())) / max(resolved.values())
    if gap > maximum_parameter_gap:
        raise ValueError(f"Parameter gap {gap:.6f} exceeds the lock")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "parameter_counts": resolved,
        "parameter_relative_gap": gap,
        "runs": audits,
    }


def _nested_values(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        result = [value[key]] if key in value else []
        for child in value.values():
            result.extend(_nested_values(child, key))
        return result
    if isinstance(value, list):
        return [item for child in value for item in _nested_values(child, key)]
    return []


def _payload_equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return bool(
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return bool(
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_payload_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def audit_paired_forks(
    preonline_manifest_path: Path,
    control_manifest_path: Path,
    candidate_manifest_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    import torch

    preonline = {
        (str(run["algorithm"]), int(run["seed"])): run
        for run in read_json(preonline_manifest_path)["runs"]
    }
    control = {
        (str(run["algorithm"]), int(run["seed"])): run
        for run in read_json(control_manifest_path)["runs"]
    }
    candidate = {
        (str(run["algorithm"]), int(run["seed"])): run
        for run in read_json(candidate_manifest_path)["runs"]
    }
    provenance = {
        (str(run["algorithm"]), int(run["seed"])): run
        for run in read_json(provenance_path)["runs"]
    }
    audits = []
    for key in sorted(control):
        source = Path(str(preonline[key]["preonline_training_state"]))
        control_clone = Path(str(provenance[key]["control_clone"]))
        candidate_clone = Path(str(provenance[key]["candidate_clone"]))
        control_resume = Path(str(control[key]["resumed_training_state"]))
        candidate_resume = Path(str(candidate[key]["resumed_training_state"]))
        if control_clone.resolve() != control_resume.resolve():
            raise ValueError(f"Control {key} did not consume its contract clone")
        if candidate_clone.resolve() != candidate_resume.resolve():
            raise ValueError(f"Candidate {key} did not consume its contract clone")
        control_overrides = _nested_values(
            control[key].get("pretrain", {}),
            "preonline_fork_overrides",
        )
        if control_overrides:
            raise ValueError(f"Control {key} contract clone did not match")
        candidate_overrides = _nested_values(
            candidate[key].get("pretrain", {}),
            "preonline_fork_overrides",
        )
        if candidate_overrides:
            raise ValueError(f"Candidate {key} contract clone did not match")
        source_state = torch.load(source, map_location="cpu", weights_only=False)
        immutable_source = {
            name: value
            for name, value in source_state.items()
            if name not in {"training_contract", "training_contract_sha256"}
        }
        for clone in (control_clone, candidate_clone):
            clone_state = torch.load(
                clone,
                map_location="cpu",
                weights_only=False,
            )
            immutable_clone = {
                name: value
                for name, value in clone_state.items()
                if name not in {
                    "training_contract",
                    "training_contract_sha256",
                }
            }
            if not _payload_equal(immutable_source, immutable_clone):
                raise ValueError(f"Contract clone changed state payload for {key}")
        excluded = _nested_values(
            candidate[key].get("pretrain", {}),
            "critic_teacher_advantage_excluded_samples",
        )
        if not excluded or max(float(value) for value in excluded) <= 0.0:
            raise ValueError(f"Candidate {key} did not filter unsupported rows")
        left = torch.load(control[key]["pretrain_checkpoint"], map_location="cpu")
        right = torch.load(candidate[key]["pretrain_checkpoint"], map_location="cpu")
        compared = 0
        for module_name in ("actor", "correction_gate"):
            left_module = left.get(module_name)
            right_module = right.get(module_name)
            if left_module is None or right_module is None:
                if left_module is not right_module:
                    raise ValueError(f"Fork module mismatch: {key} {module_name}")
                continue
            if left_module.keys() != right_module.keys():
                raise ValueError(f"Fork module keys differ: {key} {module_name}")
            for parameter in left_module:
                if not torch.equal(left_module[parameter], right_module[parameter]):
                    raise ValueError(f"Episode-0 parameters differ: {key} {module_name}")
                compared += int(left_module[parameter].numel())
        critic_differences = sum(
            int(not torch.equal(left["critic"][name], right["critic"][name]))
            for name in left["critic"]
        )
        if critic_differences <= 0:
            raise ValueError(f"Support treatment did not alter critic: {key}")
        audits.append(
            {
                "algorithm": key[0],
                "seed": key[1],
                "source_state": str(source),
                "source_state_sha256": sha256_file(source),
                "control_contract_clone": str(control_clone),
                "control_contract_clone_sha256": sha256_file(control_clone),
                "candidate_contract_clone": str(candidate_clone),
                "candidate_contract_clone_sha256": sha256_file(
                    candidate_clone
                ),
                "state_payload_identical": True,
                "identical_actor_and_gate_parameters": compared,
                "critic_parameter_tensors_changed_by_treatment": (
                    critic_differences
                ),
                "support_rows_excluded": max(float(value) for value in excluded),
                "control_contract_matched_without_runtime_override": True,
                "candidate_contract_matched_without_runtime_override": True,
            }
        )
    return {"passed": True, "runs": audits}


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


def audit_evaluation(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    root = Path(config["output_root"])
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    if len(summary["runs"]) != 6:
        raise ValueError("Evaluation requires six matched runs")
    expected_scenarios = set(config["scenarios"])
    audits = []
    for run in summary["runs"]:
        algorithm = str(run["algorithm"])
        seed = int(run["training_seed"])
        run_root = root / algorithm / f"seed{seed}"
        learned = read_csv_rows(run_root / "holdout_rows.csv")
        anchor = read_csv_rows(run_root / "holdout_anchor_rows.csv")
        if len(learned) != 200 or len(anchor) != 200:
            raise ValueError(f"Evaluation row count mismatch: {algorithm} {seed}")
        for rows in (learned, anchor):
            pairs = {(row["scenario"], int(row["replication"])) for row in rows}
            if len(pairs) != 200 or {item[0] for item in pairs} != expected_scenarios:
                raise ValueError("Evaluation scenarios or replications mismatch")
            if sum(float(row["specimen_route_count"]) for row in rows) <= 0.0:
                raise ValueError("Evaluation has zero specimen routing")
            for row in rows:
                require_finite(
                    row,
                    (
                        "total_cost",
                        "completion_service_level",
                        "patients_lost",
                        "patient_ineligibility_during_manufacturing_rate",
                        "specimen_route_count",
                    ),
                )
        corrected = nested_metric_total(run, "corrected_decisions")
        if corrected <= 0.0:
            raise ValueError("Learned evaluation has zero residual correction")
        audits.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "learned_rows": len(learned),
                "anchor_rows": len(anchor),
                "corrected_decisions": corrected,
            }
        )
    return {
        "config": str(config_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "runs": audits,
    }


def artifact_inventory(root: Path) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_sha256.json"
    }


def run_experiment(spec_path: Path, expected_commit: str) -> None:
    preflight_result = preflight(spec_path, expected_commit)
    spec = read_json(spec_path)
    launcher_root = Path(spec["launcher_root"])
    claim = claim_execution(
        launcher_root,
        preflight_result=preflight_result,
        expected_commit=expected_commit,
    )
    status: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": expected_commit,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "claim": str(claim),
        "phases": [],
    }
    atomic_write_json(launcher_root / "status.json", status)
    try:
        spec, runtime_config_provenance = materialize_runtime_configs(
            spec,
            launcher_root,
        )
        status["runtime_config_provenance"] = runtime_config_provenance
        atomic_write_json(launcher_root / "status.json", status)
        run_phase(
            launcher_root,
            "focused_tests",
            [sys.executable, "-m", "unittest", *spec["focused_tests"]],
            status,
        )
        reuse_manifest = spec.get("reuse_preonline_manifest")
        if reuse_manifest in (None, ""):
            run_phase(
                launcher_root,
                "prepare_preonline_states",
                [
                    sys.executable,
                    "-m",
                    "evaluation.prepare_ddpg_support_alignment_preonline",
                    "--config",
                    str(spec["control_training_config"]),
                    "--output-root",
                    str(spec["preonline_output_root"]),
                ],
                status,
            )
            preonline_manifest = Path(spec["preonline_manifest"])
        else:
            preonline_manifest = Path(str(reuse_manifest))
        status["preonline_audit"] = audit_preonline_preparation(
            preonline_manifest
        )
        status["preonline_reused_read_only"] = reuse_manifest not in (None, "")
        reused_control_manifest = spec.get("reuse_control_training_manifest")
        if reused_control_manifest in (None, ""):
            clone_provenance = clone_paired_preonline_states(
                preonline_manifest,
                control_config_path=Path(spec["control_training_config"]),
                candidate_config_path=Path(spec["candidate_training_config"]),
                output_root=Path(spec["paired_state_root"]),
            )
            control_manifest = Path(spec["control_training_manifest"])
            status["control_training_reused_read_only"] = False
        else:
            provenance_path = Path(
                str(spec["reuse_paired_state_provenance"])
            )
            clone_provenance = read_json(provenance_path)
            clone_provenance["provenance"] = str(provenance_path)
            clone_provenance["provenance_sha256"] = sha256_file(
                provenance_path
            )
            control_manifest = Path(str(reused_control_manifest))
            status["control_training_reused_read_only"] = True
            status["reused_control_recovery"] = preflight_result[
                "reused_control_recovery"
            ]
        status["contract_clone_provenance"] = clone_provenance
        atomic_write_json(launcher_root / "status.json", status)
        clone_runs = {
            (str(run["algorithm"]), int(run["seed"])): run
            for run in clone_provenance["runs"]
        }
        if reused_control_manifest in (None, ""):
            for algorithm in ALGORITHMS:
                for seed in SEEDS:
                    run_phase(
                        launcher_root,
                        f"training_control_{algorithm}_seed{seed}",
                        [
                            sys.executable,
                            "-m",
                            "evaluation.train_multiscenario_network_residual",
                            "--config",
                            str(spec["control_training_config"]),
                            "--algorithm",
                            algorithm,
                            "--seed",
                            str(seed),
                            "--resume-training-state",
                            str(
                                clone_runs[(algorithm, seed)][
                                    "control_clone"
                                ]
                            ),
                        ],
                        status,
                    )
        status["control_training_audit"] = audit_training(
            control_manifest,
            require_preonline_state=False,
            maximum_parameter_gap=float(spec["maximum_parameter_relative_gap"]),
        )
        atomic_write_json(launcher_root / "status.json", status)
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                run_phase(
                    launcher_root,
                    f"training_candidate_{algorithm}_seed{seed}",
                    [
                        sys.executable,
                        "-m",
                        "evaluation.train_multiscenario_network_residual",
                        "--config",
                        str(spec["candidate_training_config"]),
                        "--algorithm",
                        algorithm,
                        "--seed",
                        str(seed),
                        "--resume-training-state",
                        str(clone_runs[(algorithm, seed)]["candidate_clone"]),
                    ],
                    status,
                )
        candidate_manifest = Path(spec["candidate_training_manifest"])
        status["candidate_training_audit"] = audit_training(
            candidate_manifest,
            require_preonline_state=False,
            maximum_parameter_gap=float(spec["maximum_parameter_relative_gap"]),
        )
        status["paired_fork_audit"] = audit_paired_forks(
            preonline_manifest,
            control_manifest,
            candidate_manifest,
            Path(clone_provenance["provenance"]),
        )
        verify_locked_assets(spec)
        atomic_write_json(launcher_root / "status.json", status)
        for role in ("control", "candidate"):
            for variant in ("final", "pretrain"):
                config_path = Path(spec[f"{role}_{variant}_evaluation_config"])
                run_phase(
                    launcher_root,
                    f"evaluation_{role}_{variant}",
                    [
                        sys.executable,
                        "-m",
                        "evaluation.evaluate_multiscenario_network_residual",
                        "--config",
                        str(config_path),
                    ],
                    status,
                )
                status[f"evaluation_{role}_{variant}_audit"] = audit_evaluation(
                    config_path
                )
                atomic_write_json(launcher_root / "status.json", status)
        run_phase(
            launcher_root,
            "comparison",
            [
                sys.executable,
                "-m",
                "evaluation.compare_ddpg_support_alignment",
                "--control-final",
                str(spec["control_final_summary"]),
                "--control-pretrain",
                str(spec["control_pretrain_summary"]),
                "--candidate-final",
                str(spec["candidate_final_summary"]),
                "--candidate-pretrain",
                str(spec["candidate_pretrain_summary"]),
                "--output",
                str(spec["comparison_output"]),
            ],
            status,
        )
        status["comparison"] = read_json(Path(spec["comparison_output"]))
        inventory = artifact_inventory(Path(spec["campaign_root"]))
        atomic_write_json(launcher_root / "artifact_sha256.json", inventory)
        status.update(
            {
                "status": "completed",
                "phase": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "artifact_inventory": str(launcher_root / "artifact_sha256.json"),
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
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    spec_path = Path(args.spec)
    if args.preflight_only:
        print(
            json.dumps(
                preflight(spec_path, str(args.expected_commit)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_experiment(spec_path, str(args.expected_commit))


if __name__ == "__main__":
    main()
