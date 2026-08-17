"""Locked executor for the Stage C2 persistent-shift DDPG screen."""

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
    "patient_indexed_specimen_routing_ddpg_persistent_shift_execution.json"
)
ALGORITHMS = (
    "gcn_residual_mdl2_network_ddpg_afd",
    "flat_residual_mdl2_network_ddpg_afd",
)
SEEDS = (40, 41, 42)
ROLES = ("control", "realigned")
VARIANTS = (
    "pretrain",
    "episode10",
    "episode25",
    "episode50",
    "episode75",
    "final",
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


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, path))
        return flattened
    return {prefix: value}


def verify_locked_assets(spec: dict[str, Any]) -> dict[str, str]:
    verified = {}
    for raw in spec["locked_files"]:
        path = Path(str(raw["path"]))
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


def validate_scientific_contract(spec: dict[str, Any]) -> None:
    configs = {
        role: read_json(Path(spec[f"{role}_training_config"]))
        for role in ROLES
    }
    expected_assignment = {
        "40": "routing_persistent_hotspot_cluster1",
        "41": "routing_persistent_hotspot_cluster2",
        "42": "routing_persistent_hotspot_cluster3",
    }
    for role, config in configs.items():
        observed_algorithms = [
            str(entry["name"]) for entry in config["algorithms"]
        ]
        if observed_algorithms != list(ALGORITHMS):
            raise ValueError(f"{role} algorithm order does not match")
        if [int(value) for value in config["seeds"]] != list(SEEDS):
            raise ValueError(f"{role} seed lock does not match")
        if any(
            [int(value) for value in entry["seeds"]] != list(SEEDS)
            for entry in config["algorithms"]
        ):
            raise ValueError(f"{role} GCN/flat seeds are not matched")
        if config.get("scenario_by_seed") != expected_assignment:
            raise ValueError(f"{role} scenario-by-seed lock does not match")
        if int(config["online_episodes"]) != 100:
            raise ValueError(f"{role} must use 100 online episodes")
        overrides = dict(config["config_overrides"])
        if str(overrides.get("device")) != "mps":
            raise ValueError(f"{role} requires MPS")
        if int(overrides.get("checkpoint_interval", -1)) != 5:
            raise ValueError(f"{role} checkpoint interval must be five")
        if int(overrides.get("training_state_checkpoint_interval", -1)) != 5:
            raise ValueError(f"{role} state interval must be five")

    control_flat = _flatten(configs["control"])
    realigned_flat = _flatten(configs["realigned"])
    differences = {
        key
        for key in set(control_flat) | set(realigned_flat)
        if control_flat.get(key) != realigned_flat.get(key)
    }
    expected_differences = {
        "name",
        "experimental_role",
        "output_root",
        "config_overrides.online_critic_realignment.enabled",
        "config_overrides.online_critic_realignment.actor_warmup_updates",
    }
    if differences != expected_differences:
        raise ValueError(
            "Stage C2 training arms differ outside the lock: "
            + ", ".join(sorted(differences))
        )
    control_realign = configs["control"]["config_overrides"][
        "online_critic_realignment"
    ]
    candidate_realign = configs["realigned"]["config_overrides"][
        "online_critic_realignment"
    ]
    if control_realign != {
        "enabled": False,
        "mode": "zero_action_columns",
        "actor_warmup_updates": 0,
    }:
        raise ValueError("Standard control realignment contract changed")
    if candidate_realign != {
        "enabled": True,
        "mode": "zero_action_columns",
        "actor_warmup_updates": 500,
    }:
        raise ValueError("Realigned candidate contract changed")

    forbidden = {int(value) for value in spec["forbidden_crn_seeds"]}
    development = {int(value) for value in spec["development_crn_seeds"]}
    if forbidden & development:
        raise ValueError("Stage C2 reused a forbidden CRN stream")
    if float(spec["minimum_final_improvement_pct"]) != 0.02:
        raise ValueError("Stage C2 final improvement threshold changed")
    for role in ROLES:
        evaluation = read_json(Path(spec[f"{role}_evaluation_config"]))
        if list(evaluation["checkpoint_variants"]) != list(VARIANTS):
            raise ValueError(f"{role} checkpoint curve changed")
        if int(evaluation["validation_seed"]) != 95_200_000:
            raise ValueError(f"{role} validation CRN changed")
        if int(evaluation["holdout_seed"]) != 95_300_000:
            raise ValueError(f"{role} development CRN changed")
        if int(evaluation["holdout_replications"]) != 50:
            raise ValueError(f"{role} requires 50 development replications")
        if evaluation["scenario_by_training_seed"] != expected_assignment:
            raise ValueError(f"{role} evaluation scenario mapping changed")


def environment_fingerprint() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for Stage C2") from error
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; Stage C2 forbids CPU fallback")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if fallback.strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    probe = torch.ones(4, device="mps").square().sum()
    torch.mps.synchronize()
    if probe.device.type != "mps" or float(probe.cpu()) != 4.0:
        raise RuntimeError("Stage C2 MPS probe failed")
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


def run_preflight_tests(spec: dict[str, Any]) -> dict[str, Any]:
    modules = [str(value) for value in spec["focused_tests"]]
    if not modules:
        raise ValueError("Stage C2 preflight test list is empty")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    environment["PYTHONPYCACHEPREFIX"] = "/private/tmp/gcn_rl_pycache"
    command = [sys.executable, "-m", "unittest", *modules]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Stage C2 preflight tests failed:\n" + completed.stdout[-8_000:]
        )
    return {
        "command": command,
        "modules": modules,
        "exit_code": int(completed.returncode),
        "output_tail": completed.stdout[-2_000:],
    }


def preflight(
    spec_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    spec = read_json(spec_path)
    actual_commit = git_output("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if git_output("status", "--short"):
        raise ValueError("Stage C2 requires a completely clean worktree")
    validate_scientific_contract(spec)
    verified = verify_locked_assets(spec)
    for raw in spec["fresh_output_roots"]:
        path = Path(str(raw))
        if path.exists():
            raise FileExistsError(f"Stage C2 output already exists: {path}")
    environment = environment_fingerprint()
    tests = run_preflight_tests(spec)
    return {
        "commit": actual_commit,
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "verified_hashes": verified,
        "environment": environment,
        "preflight_tests": tests,
    }


def claim_execution(
    root: Path,
    *,
    preflight_result: dict[str, Any],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    claim_path = root / "claim.json"
    payload = {
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        **preflight_result,
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
        raise FileExistsError(f"Stage C2 phase log exists: {name}")
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
        raise RuntimeError(
            f"Stage C2 phase {name} exited {completed.returncode}"
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_finite(row: dict[str, str], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = row.get(key, "")
        if value == "" or not math.isfinite(float(value)):
            raise ValueError(f"Non-finite or missing {key}")


def _require_all_numeric_fields_finite(
    row: dict[str, str],
    *,
    nonnumeric_fields: frozenset[str],
) -> None:
    for key, value in row.items():
        if key in nonnumeric_fields or value == "":
            continue
        try:
            numeric = float(value)
        except ValueError as error:
            raise ValueError(
                f"Unexpected nonnumeric persisted value for {key}"
            ) from error
        if not math.isfinite(numeric):
            raise ValueError(f"Non-finite persisted value for {key}")


def audit_preonline(manifest_path: Path) -> dict[str, Any]:
    import torch

    manifest = read_json(manifest_path)
    expected = {
        (algorithm, seed)
        for algorithm in ALGORITHMS
        for seed in SEEDS
    }
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    if set(runs) != expected:
        raise ValueError("Stage C2 pre-online manifest is incomplete")
    audits = []
    for algorithm, seed in sorted(expected):
        run = runs[(algorithm, seed)]
        if int(run["online_episodes"]) != 0:
            raise ValueError("Stage C2 pre-online source consumed a trajectory")
        state_path = Path(str(run["preonline_training_state"]))
        checkpoint = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )
        training = dict(checkpoint["training"])
        if (
            int(training.get("next_episode", -1)) != 0
            or int(training.get("global_step", -1)) != 0
            or list(training.get("rows", ()))
        ):
            raise ValueError("Stage C2 pre-online state is not episode-0 atomic")
        contract = dict(checkpoint["training_contract"])
        if int(contract.get("num_episodes", -1)) != 0:
            raise ValueError("Stage C2 source contract is not zero-episode")
        calibration = dict(contract["critic_teacher_advantage_calibration"])
        if not bool(calibration.get("enabled", False)):
            raise ValueError("Stage C2 source must retain critic calibration data")
        if int(calibration.get("updates", -1)) != 0:
            raise ValueError("Stage C2 source consumed boundary calibration")
        if float(calibration.get("online_ranking_weight", -1.0)) != 3.0:
            raise ValueError("Stage C2 source changed offline critic ranking")
        realignment = dict(contract["online_critic_realignment"])
        if bool(realignment.get("enabled", True)):
            raise ValueError("Stage C2 source applied critic realignment early")
        agent_state = dict(checkpoint["agent"])
        if bool(agent_state.get("online_critic_realignment_applied", True)):
            raise ValueError("Stage C2 source agent is already realigned")
        if int(agent_state.get("online_updates_since_prepare", -1)) != 0:
            raise ValueError("Stage C2 source has online updates")
        audits.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "state": str(state_path),
                "state_sha256": sha256_file(state_path),
                "total_offline_updates": int(agent_state["total_updates"]),
            }
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "runs": audits,
    }


def _payload_equal(left: Any, right: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and torch.is_tensor(left):
        return torch.is_tensor(right) and bool(torch.equal(left, right))
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_payload_equal(a, b) for a, b in zip(left, right))
        )
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None and isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and bool(
            np.array_equal(left, right)
        )
    return left == right


def clone_paired_states(
    preonline_manifest_path: Path,
    *,
    control_config_path: Path,
    realigned_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    import torch

    from src.rl.training_state import training_contract_sha256

    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    source_runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in read_json(preonline_manifest_path)["runs"]
    }
    target_configs = {
        "control": read_json(control_config_path),
        "realigned": read_json(realigned_config_path),
    }
    provenance: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experimental_role": "metadata-only paired Stage C2 episode-0 forks",
        "agent_environment_rng_payload_modified": False,
        "changed_contract_paths": [
            "num_episodes",
            "history_screen.online_episodes",
            "history_screen.pretrain_only",
            "critic_teacher_advantage_calibration.updates",
            "online_critic_realignment.enabled",
            "online_critic_realignment.actor_warmup_updates",
        ],
        "runs": [],
    }
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            source = Path(
                str(source_runs[(algorithm, seed)]["preonline_training_state"])
            )
            checkpoint = torch.load(
                source,
                map_location="cpu",
                weights_only=False,
            )
            if int(checkpoint["training"]["next_episode"]) != 0:
                raise ValueError("Only episode-0 states may be paired")
            entry = {
                "algorithm": algorithm,
                "seed": seed,
                "source": str(source),
                "source_sha256": sha256_file(source),
            }
            for role in ROLES:
                cloned = copy.deepcopy(checkpoint)
                contract = copy.deepcopy(dict(cloned["training_contract"]))
                contract["num_episodes"] = 100
                history = dict(contract["history_screen"])
                history["online_episodes"] = 100
                history["pretrain_only"] = False
                contract["history_screen"] = history
                target_overrides = target_configs[role]["config_overrides"]
                calibration = dict(
                    contract["critic_teacher_advantage_calibration"]
                )
                calibration["updates"] = int(
                    target_overrides[
                        "critic_teacher_advantage_calibration"
                    ]["updates"]
                )
                contract["critic_teacher_advantage_calibration"] = calibration
                contract["online_critic_realignment"] = copy.deepcopy(
                    target_overrides["online_critic_realignment"]
                )
                cloned["training_contract"] = contract
                cloned["training_contract_sha256"] = training_contract_sha256(
                    contract
                )
                if not _payload_equal(cloned["agent"], checkpoint["agent"]):
                    raise RuntimeError("Paired clone changed agent payload")
                if not _payload_equal(
                    cloned.get("environment"),
                    checkpoint.get("environment"),
                ):
                    raise RuntimeError("Paired clone changed environment payload")
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
            provenance["runs"].append(entry)
    provenance_path = output_root / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    provenance["provenance"] = str(provenance_path)
    provenance["provenance_sha256"] = sha256_file(provenance_path)
    return provenance


def audit_training(
    manifest_path: Path,
    *,
    role: str,
    scenario_by_seed: dict[int, str],
    maximum_parameter_gap: float,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    expected = {
        (algorithm, seed)
        for algorithm in ALGORITHMS
        for seed in SEEDS
    }
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    if set(runs) != expected:
        raise ValueError(f"{role} training manifest is incomplete")
    parameter_counts = {algorithm: set() for algorithm in ALGORITHMS}
    audits = []
    for algorithm, seed in sorted(expected):
        run = runs[(algorithm, seed)]
        parameter_counts[algorithm].add(int(run["parameter_count"]))
        run_root = Path(run["config"]).parent
        rows = _read_csv_rows(run_root / "training.csv")
        if len(rows) != 100:
            raise ValueError(f"{role} {algorithm} seed {seed} row count")
        if [int(row["episode"]) for row in rows] != list(range(100)):
            raise ValueError(f"{role} {algorithm} seed {seed} episode order")
        if {str(row["scenario"]) for row in rows} != {
            scenario_by_seed[seed]
        }:
            raise ValueError(f"{role} {algorithm} seed {seed} changed scenario")
        routing = sum(float(row["specimen_route_count"]) for row in rows)
        updates = sum(float(row["online_rl_updates"]) for row in rows)
        if routing <= 0.0 or updates <= 0.0:
            raise ValueError(f"{role} {algorithm} seed {seed} inactive")
        for row in rows:
            _require_finite(
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
                            f"{role} {algorithm} seed {seed} non-finite {key}"
                        )
            _require_all_numeric_fields_finite(
                row,
                nonnumeric_fields=frozenset(
                    {
                        "algorithm",
                        "scenario",
                        "graph_ablation",
                        "pretrain_policy",
                        "pretrain_checkpoint_path",
                        "train_randomization_disruption_range",
                        "train_randomization_forecast_error_range",
                        "train_randomization_demand_rate_multiplier_range",
                        "transferred_patient_ids_json",
                        "specimen_route_events_json",
                        "finished_product_return_assumption",
                    }
                ),
            )
        checkpoint_root = run_root / "checkpoints"
        checkpoints = [
            checkpoint_root / f"{algorithm}_seed{seed}_episode{episode}.pt"
            for episode in range(5, 101, 5)
        ]
        if any(not path.is_file() for path in checkpoints):
            raise FileNotFoundError(
                f"{role} {algorithm} seed {seed} checkpoint missing"
            )
        state_path = Path(run["training_state_checkpoint"])
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        pretrain = dict(run["pretrain"])
        if role == "realigned":
            if pretrain.get("online_critic_realignment") != "zero_action_columns":
                raise ValueError("Candidate did not report critic realignment")
            if float(
                pretrain["online_critic_action_columns_l2_after"]
            ) != 0.0:
                raise ValueError("Candidate critic realignment was incomplete")
            if float(pretrain["online_actor_warmup_updates"]) != 500.0:
                raise ValueError("Candidate online actor warmup changed")
            if any(
                float(row.get("online_rl_actor_updated_mean", 0.0)) != 0.0
                for row in rows[:9]
            ):
                raise ValueError("Candidate actor updated during warmup")
            if sum(
                float(row.get("online_rl_actor_updated_mean", 0.0))
                for row in rows
            ) <= 0.0:
                raise ValueError("Candidate actor never updated after warmup")
        elif "online_critic_realignment" in pretrain:
            raise ValueError("Control unexpectedly applied critic realignment")
        audits.append(
            {
                "role": role,
                "algorithm": algorithm,
                "seed": seed,
                "scenario": scenario_by_seed[seed],
                "episodes": len(rows),
                "specimen_route_count": routing,
                "online_rl_updates": updates,
                "checkpoint_count": len(checkpoints),
                "training_state_sha256": sha256_file(state_path),
                "actor_drift_from_pretrain": dict(
                    run["actor_drift_from_pretrain"]
                ),
            }
        )
    resolved_counts = {
        algorithm: next(iter(values))
        for algorithm, values in parameter_counts.items()
        if len(values) == 1
    }
    if len(resolved_counts) != len(ALGORITHMS):
        raise ValueError(f"{role} parameter counts vary across seeds")
    largest = max(resolved_counts.values())
    smallest = min(resolved_counts.values())
    relative_gap = (largest - smallest) / largest
    if relative_gap > maximum_parameter_gap:
        raise ValueError(f"{role} parameter gap exceeds the lock")
    return {
        "role": role,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "parameter_counts": resolved_counts,
        "parameter_relative_gap": relative_gap,
        "runs": audits,
    }


def _nested_metric_total(value: Any, key: str) -> float:
    if isinstance(value, dict):
        total = float(value.get(key, 0.0)) if key in value else 0.0
        return total + sum(
            _nested_metric_total(child, key)
            for child_key, child in value.items()
            if child_key != key
        )
    if isinstance(value, list):
        return sum(_nested_metric_total(child, key) for child in value)
    return 0.0


def audit_evaluation_curve(
    config_path: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    root = Path(config["output_root"])
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config["scenario_by_training_seed"].items()
    }
    run_audits = []
    learned_total = 0
    anchor_total = 0
    for variant in VARIANTS:
        summary_path = root / variant / "summary.json"
        summary = read_json(summary_path)
        if len(summary["runs"]) != len(ALGORITHMS) * len(SEEDS):
            raise ValueError(f"{variant} evaluation run count mismatch")
        for run in summary["runs"]:
            algorithm = str(run["algorithm"])
            seed = int(run["training_seed"])
            run_root = root / variant / algorithm / f"seed{seed}"
            learned = _read_csv_rows(run_root / "holdout_rows.csv")
            anchor = _read_csv_rows(run_root / "holdout_anchor_rows.csv")
            if len(learned) != 50 or len(anchor) != 50:
                raise ValueError(f"{variant} {algorithm} seed {seed} rows")
            expected_pairs = {
                (assignment[seed], replication)
                for replication in range(50)
            }
            for rows in (learned, anchor):
                pairs = {
                    (str(row["scenario"]), int(row["replication"]))
                    for row in rows
                }
                if pairs != expected_pairs:
                    raise ValueError(
                        f"{variant} {algorithm} seed {seed} CRN mismatch"
                    )
                for row in rows:
                    _require_finite(
                        row,
                        (
                            "total_cost",
                            "completion_service_level",
                            "patients_lost",
                            "specimen_route_count",
                        ),
                    )
                    _require_all_numeric_fields_finite(
                        row,
                        nonnumeric_fields=frozenset(
                            {
                                "algorithm",
                                "scenario",
                                "graph_ablation",
                                "transferred_patient_ids_json",
                                "specimen_route_events_json",
                                "finished_product_return_assumption",
                            }
                        ),
                    )
                if sum(float(row["specimen_route_count"]) for row in rows) <= 0:
                    raise ValueError("Stage C2 evaluation has zero routing")
            corrected = _nested_metric_total(run, "corrected_decisions")
            if corrected <= 0.0:
                raise ValueError("Stage C2 learned policy has zero correction")
            learned_total += len(learned)
            anchor_total += len(anchor)
            run_audits.append(
                {
                    "variant": variant,
                    "algorithm": algorithm,
                    "seed": seed,
                    "scenario": assignment[seed],
                    "learned_rows": len(learned),
                    "anchor_rows": len(anchor),
                    "corrected_decisions": corrected,
                }
            )
    curve_summary = root / "checkpoint_curve_summary.json"
    if not curve_summary.is_file():
        raise FileNotFoundError(curve_summary)
    return {
        "config": str(config_path),
        "checkpoint_curve_summary": str(curve_summary),
        "checkpoint_curve_summary_sha256": sha256_file(curve_summary),
        "learned_rows": learned_total,
        "anchor_rows": anchor_total,
        "runs": run_audits,
    }


def artifact_inventory(root: Path) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda value: str(value))
        if path.is_file()
    }


def run_experiment(spec_path: Path, expected_commit: str) -> None:
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
        "phases": [],
    }
    atomic_write_json(launcher_root / "status.json", status)
    try:
        run_phase(
            launcher_root,
            "prepare_preonline",
            [
                sys.executable,
                "-m",
                "evaluation.prepare_ddpg_persistent_shift_preonline",
                "--config",
                str(spec["control_training_config"]),
                "--output-root",
                str(spec["preonline_output_root"]),
            ],
            status,
        )
        status["preonline_audit"] = audit_preonline(
            Path(spec["preonline_manifest"])
        )
        status["phase"] = "clone_paired_states"
        atomic_write_json(launcher_root / "status.json", status)
        paired = clone_paired_states(
            Path(spec["preonline_manifest"]),
            control_config_path=Path(spec["control_training_config"]),
            realigned_config_path=Path(spec["realigned_training_config"]),
            output_root=Path(spec["paired_state_root"]),
        )
        status["paired_state_provenance"] = paired
        atomic_write_json(launcher_root / "status.json", status)

        paired_runs = {
            (str(run["algorithm"]), int(run["seed"])): dict(run)
            for run in paired["runs"]
        }
        for role in ROLES:
            config_path = Path(spec[f"{role}_training_config"])
            for algorithm in ALGORITHMS:
                for seed in SEEDS:
                    state_path = paired_runs[(algorithm, seed)][
                        f"{role}_clone"
                    ]
                    run_phase(
                        launcher_root,
                        f"training_{role}_{algorithm}_seed{seed}",
                        [
                            sys.executable,
                            "-m",
                            "evaluation.train_multiscenario_network_residual",
                            "--config",
                            str(config_path),
                            "--algorithm",
                            algorithm,
                            "--seed",
                            str(seed),
                            "--resume-training-state",
                            str(state_path),
                        ],
                        status,
                    )
            status[f"training_{role}_audit"] = audit_training(
                Path(spec[f"{role}_training_manifest"]),
                role=role,
                scenario_by_seed={
                    int(seed): str(scenario)
                    for seed, scenario in spec["scenario_by_seed"].items()
                },
                maximum_parameter_gap=float(
                    spec["maximum_parameter_relative_gap"]
                ),
            )
            atomic_write_json(launcher_root / "status.json", status)

        for role in ROLES:
            run_phase(
                launcher_root,
                f"evaluation_{role}_checkpoint_curve",
                [
                    sys.executable,
                    "-m",
                    "evaluation.evaluate_fixed_checkpoint_curve",
                    "--config",
                    str(spec[f"{role}_evaluation_config"]),
                ],
                status,
            )
            status[f"evaluation_{role}_audit"] = audit_evaluation_curve(
                Path(spec[f"{role}_evaluation_config"])
            )
            atomic_write_json(launcher_root / "status.json", status)

        run_phase(
            launcher_root,
            "comparison",
            [
                sys.executable,
                "-m",
                "evaluation.compare_ddpg_persistent_shift",
                "--control-config",
                str(spec["control_evaluation_config"]),
                "--realigned-config",
                str(spec["realigned_evaluation_config"]),
                "--output",
                str(spec["comparison_output"]),
                "--resamples",
                str(spec["bootstrap_resamples"]),
                "--bootstrap-seed",
                str(spec["bootstrap_seed"]),
                "--minimum-improvement-pct",
                str(spec["minimum_final_improvement_pct"]),
            ],
            status,
        )
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
    if args.preflight_only:
        result = preflight(Path(args.spec), str(args.expected_commit))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    run_experiment(Path(args.spec), str(args.expected_commit))


if __name__ == "__main__":
    main()
