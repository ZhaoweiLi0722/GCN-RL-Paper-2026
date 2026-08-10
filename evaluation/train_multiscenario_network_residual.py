"""Train matched graph/flat network residual agents across scenario episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_full_benchmark import (
    advantage_distillation_settings,
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    run_advantage_distillation_pretrain,
    select_scenarios,
)
from evaluation.run_gcn_residual_sweep import (
    balance_demonstration_label_weights,
    load_local_search_demonstrations,
)
from evaluation.train_network_residual_history_screen import (
    make_history_screen_config,
)
from src.env.multi_scenario import EpisodeScenarioEnv
from src.rl.agents import get_agent_class
from src.rl.config import load_config, save_config_snapshot
from src.rl.experiment import (
    build_env,
    train_off_policy_agent,
    train_offline_replay_updates,
    write_rows,
)


def verify_file_sha256(path: Path, expected_sha256: str | None = None) -> str:
    """Return a file digest and reject a mismatched optional lock."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if actual_sha256 != expected:
            raise ValueError(
                "File SHA256 mismatch for "
                f"{path}: expected {expected}, got {actual_sha256}"
            )
    return actual_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-training-state")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.resume_training_state and (
        args.algorithm is None or args.seed is None
    ):
        parser.error(
            "--resume-training-state requires --algorithm and --seed"
        )

    run_config = load_config(args.config)
    result = train_multiscenario_agents(
        run_config,
        force=bool(args.force),
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
        resume_training_state=args.resume_training_state,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def train_multiscenario_agents(
    run_config: dict[str, Any],
    *,
    force: bool = False,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
    resume_training_state: str | Path | None = None,
) -> dict[str, Any]:
    if resume_training_state is not None and (
        algorithm_filter is None or seed_filter is None
    ):
        raise ValueError(
            "Resuming training requires one algorithm and one seed"
        )
    plan = load_benchmark_plan(
        run_config.get(
            "plan",
            "experiments/configs/residual_policy_benchmark.json",
        )
    )
    budget_name = str(run_config.get("budget", "diagnostic_pretrain"))
    budget = resolve_budget(plan, budget_name)
    scenario_names = tuple(
        str(value) for value in run_config.get("scenarios", ())
    )
    if not scenario_names:
        raise ValueError("Residual training requires at least one scenario")
    selected_scenarios = select_scenarios(plan, scenario_names)
    scenarios_by_name = {
        str(scenario["name"]): scenario
        for scenario in selected_scenarios
    }
    scenarios = [
        scenarios_by_name[name]
        for name in scenario_names
    ]
    reference_name = str(
        run_config.get("reference_scenario", scenario_names[0])
    )
    reference_scenario = select_scenarios(plan, (reference_name,))[0]
    teacher_cache = Path(run_config["teacher_cache"])
    if not teacher_cache.is_file():
        raise FileNotFoundError(teacher_cache)
    expected_teacher_sha256 = run_config.get("teacher_cache_sha256")
    teacher_cache_sha256 = verify_file_sha256(
        teacher_cache,
        None
        if expected_teacher_sha256 is None
        else str(expected_teacher_sha256),
    )
    demand_history_window = int(
        run_config.get("demand_history_window", 12)
    )
    online_episodes = int(run_config.get("online_episodes", 0))
    pretrain_epochs = run_config.get("pretrain_epochs")
    offline_updates = int(run_config.get("offline_updates", 0))
    if online_episodes < 0 or offline_updates < 0:
        raise ValueError("Training episode and update counts cannot be negative")
    output_root = Path(
        run_config.get(
            "output_root",
            "results/multiscenario_network_afr",
        )
    )
    run_name = str(run_config.get("name", "multiscenario_network_afr"))
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    algorithm_entries = tuple(run_config.get("algorithms", ()))
    if not algorithm_entries:
        raise ValueError("At least one learned algorithm is required")
    results: list[dict[str, Any]] = []
    for raw_entry in algorithm_entries:
        entry = dict(raw_entry)
        algorithm = str(entry["name"])
        if algorithm_filter is not None and algorithm != algorithm_filter:
            continue
        seeds = tuple(
            int(value)
            for value in entry.get("seeds", run_config.get("seeds", (0,)))
        )
        if not seeds:
            raise ValueError(f"{algorithm} requires at least one seed")
        if seed_filter is not None:
            if int(seed_filter) not in seeds:
                continue
            seeds = (int(seed_filter),)
        for seed in seeds:
            result = train_one_multiscenario_agent(
                plan=plan,
                budget_name=budget_name,
                budget=budget,
                algorithm=algorithm,
                scenarios=scenarios,
                reference_scenario=reference_scenario,
                scenario_names=scenario_names,
                seed=seed,
                teacher_cache=teacher_cache,
                demand_history_window=demand_history_window,
                online_episodes=online_episodes,
                pretrain_epochs=(
                    None
                    if pretrain_epochs is None
                    else int(pretrain_epochs)
                ),
                offline_updates=offline_updates,
                output_root=run_root,
                common_overrides=dict(
                    run_config.get("config_overrides", {})
                ),
                algorithm_overrides=dict(
                    entry.get("config_overrides", {})
                ),
                resume_training_state=(
                    None
                    if resume_training_state is None
                    else Path(resume_training_state)
                ),
                force=force,
            )
            results.append(result)

    if not results:
        raise ValueError(
            "No training run matched the requested algorithm/seed filter"
        )

    payload = {
        "name": run_name,
        "budget": budget_name,
        "scenarios": list(scenario_names),
        "reference_scenario": reference_name,
        "teacher_cache": str(teacher_cache),
        "demand_history_window": demand_history_window,
        "online_episodes": online_episodes,
        "pretrain_epochs": pretrain_epochs,
        "offline_updates": offline_updates,
        "runs": results,
    }
    if expected_teacher_sha256 is not None:
        payload["teacher_cache_sha256"] = teacher_cache_sha256
    manifest_path = run_root / "training_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if key != "runs" and existing.get(key) != value:
                raise ValueError(
                    f"Incremental manifest contract mismatch: {key}"
                )
        merged_runs = {
            (str(run["algorithm"]), int(run["seed"])): dict(run)
            for run in existing.get("runs", ())
        }
        for run in results:
            merged_runs[(str(run["algorithm"]), int(run["seed"]))] = run
        payload["runs"] = [
            merged_runs[key]
            for key in sorted(merged_runs)
        ]
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return payload


def train_one_multiscenario_agent(
    *,
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenarios: list[dict[str, Any]],
    reference_scenario: dict[str, Any],
    scenario_names: tuple[str, ...],
    seed: int,
    teacher_cache: Path,
    demand_history_window: int,
    online_episodes: int,
    pretrain_epochs: int | None,
    offline_updates: int,
    output_root: Path,
    common_overrides: dict[str, Any],
    algorithm_overrides: dict[str, Any],
    resume_training_state: Path | None,
    force: bool,
) -> dict[str, Any]:
    config = make_history_screen_config(
        plan,
        budget_name=f"multiscenario_{budget_name}",
        budget=budget,
        algorithm=algorithm,
        scenario=reference_scenario,
        seed=seed,
        teacher_cache=teacher_cache,
        demand_history_window=demand_history_window,
        online_episodes=online_episodes,
        pretrain_epochs=pretrain_epochs,
    )
    config = deep_update(config, common_overrides)
    config = deep_update(config, algorithm_overrides)
    config["algorithm"] = algorithm
    config["seed"] = int(seed)
    config["num_episodes"] = int(online_episodes)
    config.setdefault(
        "checkpoint_interval",
        max(int(online_episodes), 1),
    )
    config["save_pretrain_checkpoint"] = True
    config.setdefault("elite_imitation", {})["enabled"] = False
    config.setdefault("advantage_distillation_pretrain", {})[
        "demonstration_path"
    ] = str(teacher_cache)
    config["multi_scenario_training"] = {
        "scenarios": list(scenario_names),
        "scenario_schedule": "episode_round_robin",
        "scenario_start_index": int(seed) % len(scenarios),
        "scenario_label_in_observation": False,
        "teacher_cache": str(teacher_cache),
        "env_overrides": multiscenario_env_overrides(
            common_overrides=common_overrides,
            algorithm_overrides=algorithm_overrides,
            demand_history_window=demand_history_window,
        ),
    }

    run_dir = output_root / algorithm / f"seed{int(seed)}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["result_csv_path"] = str(run_dir / "training.csv")
    config["config_snapshot_path"] = str(run_dir / "config.json")
    training_state_checkpoint = (
        checkpoint_dir
        / f"{algorithm}_seed{int(seed)}_training_state.pt"
    )
    config["training_state_checkpoint_path"] = str(
        training_state_checkpoint
    )
    if resume_training_state is not None:
        config["resume_training_state_path"] = str(
            resume_training_state
        )
    final_label = (
        f"{algorithm}_seed{int(seed)}_episode{online_episodes}.pt"
        if online_episodes > 0
        else f"{algorithm}_seed{int(seed)}_pretrain.pt"
    )
    final_checkpoint = checkpoint_dir / final_label
    summary_path = run_dir / "summary.json"
    if final_checkpoint.is_file() and summary_path.is_file() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        training_state_checkpoint.is_file()
        and resume_training_state is None
        and not force
    ):
        raise FileExistsError(
            "Partial training state exists; resume it explicitly with "
            "--resume-training-state"
        )

    environments = []
    environment_summaries = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_config = make_training_config(
            plan,
            budget_name,
            budget,
            algorithm,
            scenario,
            seed,
        )
        scenario_config["env"] = merge_multiscenario_env_overrides(
            scenario_config["env"],
            common_overrides=common_overrides,
            algorithm_overrides=algorithm_overrides,
            demand_history_window=demand_history_window,
        )
        environment = build_env(
            scenario_config,
            seed=int(seed) + scenario_index * 10_000,
        )
        environments.append(environment)
        environment_summaries.append(
            {
                "scenario": str(
                    getattr(environment, "scenario_name", "default")
                ),
                "observation_size": int(environment.observation_size),
                "action_size": int(environment.action_size),
            }
        )
    env = EpisodeScenarioEnv(
        environments,
        start_index=int(seed) % len(environments),
    )
    if env.observation_size != int(
        environments[0].observation_size
    ):
        raise RuntimeError("Multi-scenario observation validation failed")
    config["env"] = merge_multiscenario_env_overrides(
        make_training_config(
            plan,
            budget_name,
            budget,
            algorithm,
            reference_scenario,
            seed,
        )["env"],
        common_overrides=common_overrides,
        algorithm_overrides=algorithm_overrides,
        demand_history_window=demand_history_window,
    )

    agent = get_agent_class(algorithm)(
        env.observation_size,
        env.action_size,
        config,
    )
    critic_calibration = dict(
        config.get("critic_teacher_advantage_calibration", {})
    )
    critic_calibration_setup_summary: dict[str, Any] = {}
    if bool(critic_calibration.get("enabled", False)):
        critic_teacher_cache = Path(
            critic_calibration.get("demonstration_path", teacher_cache)
        )
        if not critic_teacher_cache.is_file():
            raise FileNotFoundError(critic_teacher_cache)
        configure_calibration = getattr(
            agent,
            "configure_critic_teacher_advantage_calibration",
            None,
        )
        if not callable(configure_calibration):
            raise ValueError(
                f"{algorithm} does not support teacher-advantage critic "
                "calibration"
            )
        critic_calibration_setup_summary.update(
            configure_calibration(
                load_local_search_demonstrations(critic_teacher_cache)
            )
        )
        critic_calibration_setup_summary[
            "critic_teacher_advantage_demonstration_path"
        ] = str(critic_teacher_cache)
    initial_checkpoint = maybe_load_initial_checkpoint(agent, config)
    settings = advantage_distillation_settings(
        config,
        budget,
        algorithm,
    )
    pretrain_report: dict[str, Any] = dict(
        critic_calibration_setup_summary
    )

    def post_imitation_pretrain(current_agent, current_env):
        summary = run_advantage_distillation_pretrain(
            settings,
            algorithm=algorithm,
            seed=int(seed),
            agent=current_agent,
            env=current_env,
            config=config,
            budget=budget,
        )
        capture_reference = getattr(
            current_agent,
            "capture_pretrain_reference_policy",
            None,
        )
        summary["pretrain_reference_actor_captured"] = bool(
            capture_reference()
            if callable(capture_reference)
            else False
        )
        summary.update(
            train_offline_replay_updates(
                current_agent,
                updates=int(offline_updates),
                progress_interval=max(
                    int(offline_updates) // 4,
                    1,
                )
                if offline_updates
                else 0,
            )
        )
        pretrain_report.update(summary)
        return summary

    def preonline_setup(current_agent, _current_env):
        return configure_online_imitation_regularization(
            current_agent,
            config,
        )

    training_rows = train_off_policy_agent(
        agent,
        env,
        config,
        post_imitation_pretrain=post_imitation_pretrain,
        preonline_setup=preonline_setup,
        pretrain_report_out=pretrain_report,
    )
    if not final_checkpoint.is_file():
        agent.save(final_checkpoint)
    if training_rows:
        write_rows(training_rows, config["result_csv_path"])
    save_config_snapshot(config, config["config_snapshot_path"])

    scenario_episode_counts: dict[str, int] = {}
    for row in training_rows:
        scenario = str(row["scenario"])
        scenario_episode_counts[scenario] = (
            scenario_episode_counts.get(scenario, 0) + 1
        )
    summary = {
        "algorithm": algorithm,
        "seed": int(seed),
        "checkpoint": str(final_checkpoint),
        "config": str(config["config_snapshot_path"]),
        "pretrain_checkpoint": str(
            checkpoint_dir
            / f"{algorithm}_seed{int(seed)}_pretrain.pt"
        ),
        "preonline_training_state": (
            None
            if config.get("preonline_training_state_path") in (None, "")
            else str(config["preonline_training_state_path"])
        ),
        "initial_checkpoint": initial_checkpoint,
        "resumed_training_state": (
            None
            if resume_training_state is None
            else str(resume_training_state)
        ),
        "training_state_checkpoint": str(training_state_checkpoint),
        "online_episodes": int(online_episodes),
        "scenario_episode_counts": scenario_episode_counts,
        "environments": environment_summaries,
        "pretrain": pretrain_report,
        "parameter_count": agent_parameter_count(agent),
    }
    pretrain_checkpoint = Path(summary["pretrain_checkpoint"])
    if pretrain_checkpoint.is_file():
        summary["actor_drift_from_pretrain"] = actor_checkpoint_drift(
            agent,
            pretrain_checkpoint,
        )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def configure_online_imitation_regularization(
    agent: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load a dedicated online imitation batch without fitting the actor."""

    settings = dict(config.get("online_imitation_regularization", {}))
    if not bool(settings.get("enabled", False)):
        return {}
    regularization_weight = float(
        config.get("imitation_pretrain", {}).get(
            "regularization_weight",
            0.0,
        )
    )
    if regularization_weight <= 0.0:
        raise ValueError(
            "online_imitation_regularization requires a positive "
            "imitation_pretrain.regularization_weight"
        )
    raw_path = settings.get("demonstration_path")
    if raw_path in (None, ""):
        raise ValueError(
            "online_imitation_regularization.demonstration_path is required"
        )
    demonstration_path = Path(str(raw_path))
    if not demonstration_path.is_file():
        raise FileNotFoundError(demonstration_path)
    demonstrations = load_local_search_demonstrations(
        demonstration_path
    )
    balance_weights = bool(
        settings.get("balance_label_weights", False)
    )
    if balance_weights:
        demonstrations = balance_demonstration_label_weights(
            demonstrations
        )
    configure = getattr(
        agent,
        "configure_online_imitation_regularization",
        None,
    )
    if not callable(configure):
        raise ValueError(
            f"{getattr(agent, 'algorithm', type(agent).__name__)} does not "
            "support online imitation regularization"
        )
    seed = int(
        settings.get("seed", int(config.get("seed", 0)) + 1_500_000)
    )
    setup = dict(
        configure(
            demonstrations["states"],
            demonstrations["actions"],
            weights=demonstrations.get("weights"),
            seed=seed,
        )
    )
    weights = np.asarray(
        demonstrations.get("weights", ()),
        dtype=np.float64,
    )
    summary = {
        "online_imitation_regularization_enabled": True,
        "online_imitation_regularization_weight": regularization_weight,
        "online_imitation_demonstration_path": str(demonstration_path),
        "online_imitation_demonstration_sha256": hashlib.sha256(
            demonstration_path.read_bytes()
        ).hexdigest(),
        "online_imitation_balance_label_weights": balance_weights,
        "online_imitation_samples": int(setup["samples"]),
        "online_imitation_weighted": bool(setup["weighted"]),
        "online_imitation_seed": int(setup["seed"]),
        "online_imitation_improved_steps": int(
            demonstrations.get("improved_steps", 0)
        ),
        "online_imitation_anchor_keep_steps": int(
            demonstrations.get("anchor_keep_steps", 0)
        ),
        "online_imitation_improved_weight_fraction": float(
            demonstrations.get("improved_weight_fraction", 0.0)
        ),
        "online_imitation_weight_mean": (
            float(weights.mean()) if weights.size else ""
        ),
        "online_imitation_weight_min": (
            float(weights.min()) if weights.size else ""
        ),
        "online_imitation_weight_max": (
            float(weights.max()) if weights.size else ""
        ),
    }
    print(
        "online_imitation_regularization "
        f"samples={summary['online_imitation_samples']} "
        f"improved_steps={summary['online_imitation_improved_steps']} "
        f"path={demonstration_path}",
        flush=True,
    )
    return summary


def maybe_load_initial_checkpoint(
    agent: Any,
    config: dict[str, Any],
) -> str | None:
    """Warm-start a multi-scenario run without restoring optimizer state."""

    raw_path = config.get("initial_checkpoint")
    if raw_path in (None, ""):
        return None
    checkpoint = Path(
        str(raw_path).format(
            algorithm=str(config.get("algorithm", "")),
            seed=int(config.get("seed", 0)),
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    agent.load_actor(checkpoint)
    return str(checkpoint)


def deep_update(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_multiscenario_env_overrides(
    scenario_env: dict[str, Any],
    *,
    common_overrides: dict[str, Any],
    algorithm_overrides: dict[str, Any],
    demand_history_window: int,
) -> dict[str, Any]:
    """Apply run-level env overrides without erasing scenario parameters."""

    return deep_update(
        scenario_env,
        multiscenario_env_overrides(
            common_overrides=common_overrides,
            algorithm_overrides=algorithm_overrides,
            demand_history_window=demand_history_window,
        ),
    )


def multiscenario_env_overrides(
    *,
    common_overrides: dict[str, Any],
    algorithm_overrides: dict[str, Any],
    demand_history_window: int,
) -> dict[str, Any]:
    """Return the env contract shared by every training scenario."""

    merged: dict[str, Any] = {}
    for overrides in (common_overrides, algorithm_overrides):
        env_overrides = overrides.get("env", {})
        if not isinstance(env_overrides, dict):
            raise TypeError("config_overrides.env must be a mapping")
        merged = deep_update(merged, env_overrides)
    merged["demand_history_window"] = int(demand_history_window)
    merged["include_demand_history_state"] = True
    return merged


def agent_parameter_count(agent: Any) -> int:
    modules = [
        getattr(agent, name, None)
        for name in (
            "actor",
            "critic",
            "correction_gate",
            "q_network",
        )
    ]
    unique_modules = {
        id(module): module
        for module in modules
        if module is not None
    }
    return int(
        sum(
            parameter.numel()
            for module in unique_modules.values()
            for parameter in module.parameters()
        )
    )


def actor_checkpoint_drift(agent: Any, checkpoint_path: str | Path) -> dict[str, float]:
    """Measure final actor movement from the frozen pretrain checkpoint."""

    from src.rl.networks import require_torch, torch

    require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    reference = dict(checkpoint["actor"])
    squared_sum = 0.0
    maximum = 0.0
    parameter_count = 0
    for name, parameter in agent.actor.named_parameters():
        if name not in reference:
            raise ValueError(f"Pretrain actor is missing parameter {name!r}")
        current = parameter.detach().cpu()
        expected = reference[name].detach().cpu()
        if current.shape != expected.shape:
            raise ValueError(f"Pretrain actor shape mismatch for {name!r}")
        difference = current - expected
        squared_sum += float(torch.sum(difference * difference).item())
        maximum = max(maximum, float(torch.max(torch.abs(difference)).item()))
        parameter_count += int(difference.numel())
    return {
        "rms": float((squared_sum / parameter_count) ** 0.5 if parameter_count else 0.0),
        "max_abs": float(maximum),
        "parameter_count": float(parameter_count),
    }


if __name__ == "__main__":
    main()
