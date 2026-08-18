"""Locked executor for the Stage F1 paired online DDPG experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

from evaluation.run_patient_indexed_specimen_routing_ddpg_persistent_shift import (
    _flatten,
    _nested_metric_total,
    _payload_equal,
    _read_csv_rows,
    _require_all_numeric_fields_finite,
    _require_finite,
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
    "execution.json"
)
ALGORITHMS = (
    "gcn_residual_mdl2_network_ddpg_afd",
    "flat_residual_mdl2_network_ddpg_afd",
)
SEEDS = (60, 61, 62)
ROLES = ("control", "candidate")
VARIANTS = (
    "pretrain",
    "episode10",
    "episode25",
    "episode50",
    "episode75",
    "final",
)
EXPECTED_ASSIGNMENT = {
    "60": "routing_persistent_hotspot_cluster1",
    "61": "routing_persistent_hotspot_cluster2",
    "62": "routing_persistent_hotspot_cluster3",
}
EXPECTED_OPTION_LABELS = {
    "mdl2",
    "specimen_transfer:-0.05",
    "specimen_transfer:+0.05",
    "specimen_transfer:-0.10",
    "specimen_transfer:+0.10",
}


def validate_scientific_contract(spec: dict[str, Any]) -> None:
    configs = {
        role: read_json(Path(spec[f"{role}_training_config"]))
        for role in ROLES
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
        if config.get("scenario_by_seed") != EXPECTED_ASSIGNMENT:
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
        if dict(overrides["online_critic_realignment"]) != {
            "enabled": False,
            "mode": "zero_action_columns",
            "actor_warmup_updates": 0,
        }:
            raise ValueError(f"{role} changed the standard DDPG critic")
        structured = dict(
            overrides["residual_action"]["structured_exploration"]
        )
        validate_structured_settings(structured, expected_enabled=True)
        paired = dict(overrides["online_paired_advantage_critic"])
        validate_paired_advantage_settings(
            paired,
            expected_enabled=(role == "candidate"),
        )

    control_flat = _flatten(configs["control"])
    candidate_flat = _flatten(configs["candidate"])
    differences = {
        key
        for key in set(control_flat) | set(candidate_flat)
        if control_flat.get(key) != candidate_flat.get(key)
    }
    expected_differences = {
        "name",
        "experimental_role",
        "output_root",
        "config_overrides.online_paired_advantage_critic.enabled",
    }
    if differences != expected_differences:
        raise ValueError(
            "Stage F1 training arms differ outside the lock: "
            + ", ".join(sorted(differences))
        )

    forbidden = {int(value) for value in spec["forbidden_crn_seeds"]}
    development = {int(value) for value in spec["development_crn_seeds"]}
    if forbidden & development:
        raise ValueError("Stage F1 reused a forbidden CRN stream")
    if development != {96_600_000, 96_700_000}:
        raise ValueError("Stage F1 development CRN lock changed")
    if 91_100_000 not in forbidden:
        raise ValueError("Formal holdout must remain forbidden")
    for role in ROLES:
        evaluation = read_json(Path(spec[f"{role}_evaluation_config"]))
        if list(evaluation["checkpoint_variants"]) != list(VARIANTS):
            raise ValueError(f"{role} checkpoint curve changed")
        if int(evaluation["validation_seed"]) != 96_600_000:
            raise ValueError(f"{role} validation CRN changed")
        if int(evaluation["holdout_seed"]) != 96_700_000:
            raise ValueError(f"{role} development CRN changed")
        if int(evaluation["holdout_replications"]) != 50:
            raise ValueError(f"{role} requires 50 development replications")
        if evaluation["scenario_by_training_seed"] != EXPECTED_ASSIGNMENT:
            raise ValueError(f"{role} evaluation scenario mapping changed")
    smoke_configs = {
        role: read_json(Path(spec[f"{role}_smoke_config"]))
        for role in ROLES
    }
    smoke_differences = {
        key
        for key in set(_flatten(smoke_configs["control"]))
        | set(_flatten(smoke_configs["candidate"]))
        if _flatten(smoke_configs["control"]).get(key)
        != _flatten(smoke_configs["candidate"]).get(key)
    }
    if smoke_differences != expected_differences:
        raise ValueError(
            "Stage F1 smoke arms differ outside the lock: "
            + ", ".join(sorted(smoke_differences))
        )
    for role, config in smoke_configs.items():
        overrides = dict(config["config_overrides"])
        if str(overrides.get("device")) != "cpu":
            raise ValueError(f"{role} smoke must use CPU")
        if int(overrides.get("max_steps_per_episode", -1)) != 10:
            raise ValueError(f"{role} smoke must use ten steps")
        if int(config["online_episodes"]) != 1:
            raise ValueError(f"{role} smoke must use one episode")
        validate_paired_advantage_settings(
            dict(overrides["online_paired_advantage_critic"]),
            expected_enabled=(role == "candidate"),
        )


def validate_structured_settings(
    settings: dict[str, Any],
    *,
    expected_enabled: bool,
) -> None:
    if bool(settings.get("enabled")) != bool(expected_enabled):
        raise ValueError("Structured-exploration arm assignment changed")
    if float(settings.get("selection_probability", -1.0)) != 0.2:
        raise ValueError("Structured-exploration probability changed")
    if str(settings.get("selection_mode")) != "uniform":
        raise ValueError("Structured-exploration selection mode changed")
    if not bool(settings.get("apply_after_correction_gate", False)):
        raise ValueError("Structured options must follow the actor gate")
    if int(settings.get("seed_offset", -1)) != 684_211:
        raise ValueError("Structured-exploration RNG offset changed")
    observed = {
        (
            str(option["group"]),
            float(option["epsilon"]),
            float(option["sign"]),
        )
        for option in settings.get("options", ())
    }
    expected = {
        ("specimen_transfer", 0.05, -1.0),
        ("specimen_transfer", 0.05, 1.0),
        ("specimen_transfer", 0.10, -1.0),
        ("specimen_transfer", 0.10, 1.0),
    }
    if observed != expected or len(settings.get("options", ())) != 4:
        raise ValueError("Structured specimen option set changed")


def validate_paired_advantage_settings(
    settings: dict[str, Any],
    *,
    expected_enabled: bool,
) -> None:
    if bool(settings.get("enabled")) != bool(expected_enabled):
        raise ValueError("Paired-advantage arm assignment changed")
    expected = {
        "horizon": 4,
        "loss_weight": 3.0,
        "followup_policy": "mdl2",
        "reward_consistency_atol": 0.000001,
        "sign_tolerance": 1e-12,
    }
    observed = {
        "horizon": int(settings.get("horizon", -1)),
        "loss_weight": float(settings.get("loss_weight", -1.0)),
        "followup_policy": str(settings.get("followup_policy", "")),
        "reward_consistency_atol": float(
            settings.get("reward_consistency_atol", -1.0)
        ),
        "sign_tolerance": float(settings.get("sign_tolerance", -1.0)),
    }
    if observed != expected:
        raise ValueError("Paired-advantage critic settings changed")


def run_focused_tests(spec: dict[str, Any]) -> dict[str, Any]:
    modules = [str(value) for value in spec["focused_tests"]]
    if not modules:
        raise ValueError("Stage F1 preflight test list is empty")
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
            "Stage F1 preflight tests failed:\n" + completed.stdout[-8_000:]
        )
    return {
        "command": command,
        "modules": modules,
        "exit_code": int(completed.returncode),
        "output_tail": completed.stdout[-2_000:],
    }


def run_cpu_smoke(spec: dict[str, Any]) -> dict[str, Any]:
    """Run and audit both ten-step CPU arms through the real trainer."""

    import torch

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    environment["PYTHONPYCACHEPREFIX"] = "/private/tmp/gcn_rl_pycache"
    results = {}
    for role in ROLES:
        config_path = Path(spec[f"{role}_smoke_config"])
        config = read_json(config_path)
        command = [
            sys.executable,
            "-m",
            "evaluation.train_multiscenario_network_residual",
            "--config",
            str(config_path),
        ]
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
                f"Stage F1 {role} smoke failed:\n"
                + completed.stdout[-8_000:]
            )
        manifest_path = (
            Path(config["output_root"])
            / str(config["name"])
            / "training_manifest.json"
        )
        manifest = read_json(manifest_path)
        if len(manifest["runs"]) != 1:
            raise ValueError(f"Stage F1 {role} smoke run count changed")
        run = dict(manifest["runs"][0])
        state_path = Path(run["training_state_checkpoint"])
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        replay = dict(state["agent"]["replay_buffer"])
        replay_size = int(replay["size"])
        masks = np.asarray(
            replay["paired_advantage_mask"],
            dtype=bool,
        )[:replay_size]
        targets = np.asarray(
            replay["paired_advantages"],
            dtype=float,
        )[:replay_size, 0]
        pair_count = int(masks.sum())
        if not np.all(np.isfinite(targets[masks])):
            raise ValueError(f"Stage F1 {role} smoke target is non-finite")
        rows = _read_csv_rows(Path(run["config"]).parent / "training.csv")
        if len(rows) != 1:
            raise ValueError(f"Stage F1 {role} smoke row count changed")
        row = rows[0]
        for key, value in row.items():
            if key.startswith("online_rl_") and value != "":
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"Stage F1 {role} smoke has non-finite {key}"
                    )
        if role == "control":
            if pair_count:
                raise ValueError("Stage F1 control smoke stored paired targets")
            sampled_pairs = 0.0
            weighted_loss = 0.0
        else:
            sampled_pairs = float(
                row[
                    "online_rl_critic_online_paired_advantage_samples_mean"
                ]
            )
            weighted_loss = float(
                row[
                    "online_rl_critic_online_paired_advantage_weighted_loss_mean"
                ]
            )
            reward_error = float(
                row[
                    "online_rl_online_paired_advantage_reward_error_max_final"
                ]
            )
            if pair_count <= 0 or sampled_pairs <= 0.0 or weighted_loss <= 0.0:
                raise ValueError("Stage F1 candidate smoke missed paired learning")
            if reward_error > 0.000001:
                raise ValueError("Stage F1 candidate smoke CRN clone mismatch")
        results[role] = {
            "command": command,
            "output_tail": completed.stdout[-2_000:],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "training_state": str(state_path),
            "training_state_sha256": sha256_file(state_path),
            "paired_replay_samples": pair_count,
            "paired_samples_per_update_mean": sampled_pairs,
            "paired_weighted_loss_mean": weighted_loss,
        }
    return results


def zero_trajectory_mps_probe(spec: dict[str, Any]) -> dict[str, Any]:
    """Instantiate both candidate agents on MPS without taking an env step."""

    from evaluation.run_full_benchmark import (
        load_benchmark_plan,
        make_training_config,
        resolve_budget,
        select_scenarios,
    )
    from evaluation.train_multiscenario_network_residual import (
        deep_update,
        merge_multiscenario_env_overrides,
    )
    from evaluation.train_network_residual_history_screen import (
        make_history_screen_config,
    )
    from src.rl.agents import get_agent_class
    from src.rl.experiment import build_env

    run_config = read_json(Path(spec["candidate_training_config"]))
    plan = load_benchmark_plan(run_config["plan"])
    budget_name = str(run_config["budget"])
    budget = resolve_budget(plan, budget_name)
    scenario = select_scenarios(
        plan,
        (EXPECTED_ASSIGNMENT["60"],),
    )[0]
    reference = select_scenarios(
        plan,
        (str(run_config["reference_scenario"]),),
    )[0]
    teacher = Path(run_config["teacher_cache"])
    results = []
    for algorithm in ALGORITHMS:
        config = make_history_screen_config(
            plan,
            budget_name=f"stage_f1_preflight_{budget_name}",
            budget=budget,
            algorithm=algorithm,
            scenario=reference,
            seed=60,
            teacher_cache=teacher,
            demand_history_window=int(run_config["demand_history_window"]),
            online_episodes=0,
            pretrain_epochs=1,
        )
        config = deep_update(config, run_config["config_overrides"])
        config["algorithm"] = algorithm
        config["seed"] = 60
        scenario_config = make_training_config(
            plan,
            budget_name,
            budget,
            algorithm,
            scenario,
            60,
        )
        config["env"] = merge_multiscenario_env_overrides(
            scenario_config["env"],
            common_overrides=run_config["config_overrides"],
            algorithm_overrides={},
            demand_history_window=int(run_config["demand_history_window"]),
        )
        env = build_env(config, seed=60)
        agent = get_agent_class(algorithm)(
            env.observation_size,
            env.action_size,
            config,
        )
        state = env.reset(seed=60)
        action = agent.select_action(state, explore=False, env=env)
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Stage F1 MPS preflight action is non-finite")
        if str(agent.device) != "mps":
            raise RuntimeError("Stage F1 agent did not initialize on MPS")
        structured = agent.structured_exploration_summary()
        if not bool(structured["enabled"]):
            raise RuntimeError("Stage F1 shared explorer is not enabled")
        if set(structured["option_labels"]) != EXPECTED_OPTION_LABELS:
            raise RuntimeError("Stage F1 structured option labels changed")
        if not bool(agent.online_paired_advantage.enabled):
            raise RuntimeError("Stage F1 candidate paired target is not enabled")
        results.append(
            {
                "algorithm": algorithm,
                "device": str(agent.device),
                "action_dim": int(action.size),
                "finite_action": True,
                "environment_steps": 0,
                "structured_options": structured["option_labels"],
                "paired_advantage_enabled": True,
            }
        )
    return {"runs": results, "trajectory_steps": 0}


def validate_lock_manifest(spec_path: Path, spec: dict[str, Any]) -> None:
    locked_paths = {
        str(entry["path"]) for entry in spec.get("locked_files", ())
    }
    if len(locked_paths) < 40:
        raise ValueError("Stage F1 lock manifest is incomplete")
    if str(spec_path) in locked_paths:
        raise ValueError("Stage F1 execution spec must not hash-reference itself")
    required = {
        str(spec["plan"]),
        str(spec["f0_summary"]),
        str(spec["control_training_config"]),
        str(spec["candidate_training_config"]),
        str(spec["control_evaluation_config"]),
        str(spec["candidate_evaluation_config"]),
        str(spec["control_smoke_config"]),
        str(spec["candidate_smoke_config"]),
        "evaluation/compare_ddpg_online_paired_advantage.py",
        "evaluation/launch_patient_indexed_specimen_routing_ddpg_online_paired_advantage.py",
        "evaluation/prepare_ddpg_online_paired_advantage_preonline.py",
        "evaluation/run_patient_indexed_specimen_routing_ddpg_online_paired_advantage.py",
        "src/rl/paired_advantage.py",
        "src/rl/replay_buffer.py",
        "src/rl/training_state.py",
        "src/models/gcn_ddpg.py",
        "src/baselines/flat_ddpg.py",
    }
    missing = sorted(required - locked_paths)
    if missing:
        raise ValueError(
            "Stage F1 lock manifest misses required assets: "
            + ", ".join(missing)
        )


def preflight(spec_path: Path, expected_commit: str) -> dict[str, Any]:
    spec = read_json(spec_path)
    actual_commit = git_output("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if git_output("status", "--short"):
        raise ValueError("Stage F1 requires a completely clean worktree")
    validate_scientific_contract(spec)
    validate_lock_manifest(spec_path, spec)
    verified = verify_locked_assets(spec)
    for raw in spec["fresh_output_roots"]:
        path = Path(str(raw))
        if path.exists():
            raise FileExistsError(f"Stage F1 output already exists: {path}")
    environment = environment_fingerprint()
    tests = run_focused_tests(spec)
    cpu_smoke = run_cpu_smoke(spec)
    mps_probe = zero_trajectory_mps_probe(spec)
    return {
        "commit": actual_commit,
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "verified_hashes": verified,
        "environment": environment,
        "preflight_tests": tests,
        "cpu_smoke": cpu_smoke,
        "zero_trajectory_mps_probe": mps_probe,
    }


def audit_f0_identifiability(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    decision = dict(summary["decision"])
    if decision.get("classification") != (
        "stage_f1_single_candidate_design_justified"
    ):
        raise ValueError("Stage F0 did not justify the locked Stage F1 design")
    if not bool(decision.get("gate_passed", False)):
        raise ValueError("Stage F0 identifiability gate did not pass")
    if bool(summary.get("formal_holdout_reused", True)):
        raise ValueError("Stage F0 reused the formal holdout")
    if decision.get("prespecified_single_correction") != (
        "Train the DDPG critic on a directly paired, finite-horizon "
        "counterfactual advantage between executed legal specimen actions "
        "and MDL-2, without changing actor, exploration, gate, scale, "
        "scenario distribution, or episode budget."
    ):
        raise ValueError("Stage F0 prespecified correction changed")
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "manifold_rows": str(summary["manifold_rows"]),
        "manifold_rows_sha256": str(summary["manifold_rows_sha256"]),
        "classification": str(decision["classification"]),
        "gate_passed": True,
        "formal_holdout_reused": False,
    }


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
        raise ValueError("Stage F1 pre-online manifest is incomplete")
    audits = []
    for algorithm, seed in sorted(expected):
        run = runs[(algorithm, seed)]
        if int(run["online_episodes"]) != 0:
            raise ValueError("Stage F1 pre-online source consumed a trajectory")
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
            raise ValueError("Stage F1 pre-online state is not episode-0 atomic")
        contract = dict(checkpoint["training_contract"])
        if int(contract.get("num_episodes", -1)) != 0:
            raise ValueError("Stage F1 source contract is not zero-episode")
        calibration = dict(contract["critic_teacher_advantage_calibration"])
        if int(calibration.get("updates", -1)) != 0:
            raise ValueError("Stage F1 source consumed boundary calibration")
        structured_contract = dict(
            contract["residual_action"]["structured_exploration"]
        )
        validate_structured_settings(
            structured_contract,
            expected_enabled=True,
        )
        validate_paired_advantage_settings(
            dict(contract["online_paired_advantage_critic"]),
            expected_enabled=False,
        )
        agent_state = dict(checkpoint["agent"])
        structured_state = dict(agent_state["structured_specimen_explorer"])
        if not bool(structured_state["enabled"]):
            raise ValueError("Stage F1 source explorer is not enabled")
        if any(
            int(structured_state[key]) != 0
            for key in (
                "total_decisions",
                "total_selections",
                "total_behaviorally_distinct",
            )
        ):
            raise ValueError("Stage F1 source explorer consumed a decision")
        if int(agent_state.get("online_updates_since_prepare", -1)) != 0:
            raise ValueError("Stage F1 source has online updates")
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


def clone_paired_states(
    preonline_manifest_path: Path,
    *,
    control_config_path: Path,
    candidate_config_path: Path,
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
        "candidate": read_json(candidate_config_path),
    }
    provenance: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experimental_role": "metadata-only paired Stage F1 episode-0 forks",
        "model_optimizer_replay_environment_rng_payload_modified": False,
        "candidate_enabled_by_declared_preonline_fork": True,
        "candidate_fork_path": "online_paired_advantage_critic.enabled",
        "changed_contract_paths": [
            "num_episodes",
            "history_screen.online_episodes",
            "history_screen.pretrain_only",
            "critic_teacher_advantage_calibration.updates",
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
                # Keep the paired target disabled in both saved payloads. The
                # candidate enables it through the declared episode-0 fork,
                # preserving every model, replay, environment, and RNG value.
                contract["online_paired_advantage_critic"]["enabled"] = False
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
    minimum_behavior_delta: float,
) -> dict[str, Any]:
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
                    "structured_specimen_episode_decisions",
                    "structured_specimen_episode_selections",
                    "structured_specimen_episode_behaviorally_distinct",
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
                        "structured_specimen_episode_option_counts_json",
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
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        state_explorer = dict(
            state["agent"]["structured_specimen_explorer"]
        )
        structured = dict(run["structured_specimen_exploration"])
        if not bool(structured["enabled"]):
            raise ValueError(f"{role} explorer enabled state changed")
        if not bool(state_explorer["enabled"]):
            raise ValueError(f"{role} saved explorer state changed")
        row_decisions = sum(
            int(float(row["structured_specimen_episode_decisions"]))
            for row in rows
        )
        row_selections = sum(
            int(float(row["structured_specimen_episode_selections"]))
            for row in rows
        )
        row_distinct = sum(
            int(
                float(
                    row[
                        "structured_specimen_episode_behaviorally_distinct"
                    ]
                )
            )
            for row in rows
        )
        if row_decisions != int(structured["total_decisions"]):
            raise ValueError(f"{role} structured decision audit mismatch")
        if row_selections != int(structured["total_selections"]):
            raise ValueError(f"{role} structured selection audit mismatch")
        if row_distinct != int(structured["total_behaviorally_distinct"]):
            raise ValueError(f"{role} structured behavior audit mismatch")
        option_counts = {
            str(key): int(value)
            for key, value in structured["total_option_counts"].items()
        }
        if set(option_counts) != EXPECTED_OPTION_LABELS:
            raise ValueError(f"{role} structured option labels changed")
        rate = float(structured["total_selection_rate"])
        if not 0.12 <= rate <= 0.28:
            raise ValueError(f"{role} structured selection rate is implausible")
        if any(value <= 0 for value in option_counts.values()):
            raise ValueError(f"{role} did not cover every structured option")
        if int(structured["total_correction_selections"]) <= 0:
            raise ValueError(f"{role} used no correction option")
        if row_distinct <= 0:
            raise ValueError(f"{role} produced no distinct route action")
        if float(structured["max_specimen_linf_delta"]) < float(
            minimum_behavior_delta
        ):
            raise ValueError(f"{role} route exploration stayed sub-grid")

        contract_paired = dict(
            state["training_contract"]["online_paired_advantage_critic"]
        )
        validate_paired_advantage_settings(
            contract_paired,
            expected_enabled=(role == "candidate"),
        )
        replay = dict(state["agent"]["replay_buffer"])
        replay_masks = np.asarray(
            replay["paired_advantage_mask"],
            dtype=bool,
        )[: int(replay["size"])]
        replay_targets = np.asarray(
            replay["paired_advantages"],
            dtype=float,
        )[: int(replay["size"]), 0]
        paired_replay_samples = int(replay_masks.sum())
        if not np.all(np.isfinite(replay_targets[replay_masks])):
            raise ValueError(f"{role} persisted non-finite paired targets")
        fork = dict(run["pretrain"].get("preonline_fork_overrides", {}))
        if role == "control":
            if paired_replay_samples:
                raise ValueError("Control unexpectedly stored paired targets")
            if fork:
                raise ValueError("Control unexpectedly used an episode-0 fork")
            paired_contexts = 0
            paired_distinct = 0
            paired_reward_error_max = 0.0
            paired_loss_batches = 0
        else:
            if set(fork) != {"online_paired_advantage_critic.enabled"}:
                raise ValueError("Candidate pre-online fork was not single-factor")
            last = rows[-1]
            paired_contexts = int(
                float(
                    last[
                        "online_rl_online_paired_advantage_context_count_final"
                    ]
                )
            )
            paired_distinct = int(
                float(
                    last[
                        "online_rl_online_paired_advantage_distinct_count_final"
                    ]
                )
            )
            paired_reward_error_max = float(
                last[
                    "online_rl_online_paired_advantage_reward_error_max_final"
                ]
            )
            if paired_contexts != row_decisions:
                raise ValueError("Candidate paired context count mismatch")
            if paired_replay_samples != paired_distinct:
                raise ValueError("Candidate paired replay count mismatch")
            if paired_reward_error_max > float(
                contract_paired["reward_consistency_atol"]
            ):
                raise ValueError("Candidate cloned reward reproduction failed")
            paired_loss_batches = sum(
                float(
                    row.get(
                        "online_rl_critic_online_paired_advantage_samples_mean",
                        0.0,
                    )
                    or 0.0
                )
                > 0.0
                for row in rows
            )
            if paired_loss_batches <= 0:
                raise ValueError("Candidate critic never sampled a paired target")
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
                "structured_specimen_exploration": structured,
                "paired_advantage": {
                    "enabled": role == "candidate",
                    "context_count": paired_contexts,
                    "distinct_count": paired_distinct,
                    "structured_explorer_distinct_count": row_distinct,
                    "replay_samples": paired_replay_samples,
                    "loss_positive_episode_count": paired_loss_batches,
                    "reward_error_max": paired_reward_error_max,
                },
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


def audit_evaluation_curve(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    root = Path(config["output_root"])
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config[
            "scenario_by_training_seed"
        ].items()
    }
    evaluation_seed = int(config["holdout_seed"])
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
                (evaluation_seed, assignment[seed], replication)
                for replication in range(50)
            }
            for rows in (learned, anchor):
                pairs = {
                    (
                        int(row["evaluation_seed"]),
                        str(row["scenario"]),
                        int(row["replication"]),
                    )
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
                    raise ValueError("Stage F1 evaluation has zero routing")
            corrected = _nested_metric_total(run, "corrected_decisions")
            if corrected <= 0.0:
                raise ValueError("Stage F1 learned policy has zero correction")
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
        status["f0_identifiability_audit"] = audit_f0_identifiability(
            Path(spec["f0_summary"])
        )
        atomic_write_json(launcher_root / "status.json", status)

        run_phase(
            launcher_root,
            "prepare_preonline",
            [
                sys.executable,
                "-m",
                "evaluation.prepare_ddpg_online_paired_advantage_preonline",
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
            candidate_config_path=Path(spec["candidate_training_config"]),
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
                minimum_behavior_delta=float(
                    spec["minimum_structured_behavior_linf_delta"]
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
