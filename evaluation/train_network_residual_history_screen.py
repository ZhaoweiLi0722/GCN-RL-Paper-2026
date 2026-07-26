"""Train pretrain-only network residual checkpoints for history-window screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.option_teacher_config import configure_options_from_teacher_cache
from evaluation.run_full_benchmark import (
    advantage_distillation_settings,
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    run_advantage_distillation_pretrain,
    select_scenarios,
)
from src.rl.agents import get_agent_class
from src.rl.config import save_config_snapshot
from src.rl.experiment import (
    build_env,
    train_off_policy_agent,
    train_offline_replay_updates,
    write_rows,
)


DEFAULT_PLAN = "experiments/configs/residual_policy_benchmark.json"
DEFAULT_ALGORITHM = "gcn_residual_mdl2_network_ddpg_afd"
DEFAULT_SCENARIO = "patient_condition_geo_demand_drift"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--budget", default="diagnostic_pretrain")
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--base-policy", default="mdl2")
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--demand-history-window", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--gcn-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--actor-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--gate-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--option-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--online-episodes", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--offline-updates", type=int, default=0)
    parser.add_argument("--offline-progress-interval", type=int, default=250)
    parser.add_argument(
        "--teacher-advantage-weight-scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--teacher-advantage-weight-cap",
        type=float,
        default=None,
    )
    parser.add_argument("--option-advantage-scale", type=float, default=None)
    parser.add_argument("--option-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--correction-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--online-reward-mode",
        choices=("environment", "one_step_anchor_relative"),
        default=None,
    )
    parser.add_argument(
        "--imitation-regularization-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--correction-gate-target-mode",
        choices=("any_improving", "teacher_best"),
        default=None,
    )
    parser.add_argument(
        "--correction-selection-mode",
        choices=("q_value", "gate_probability"),
        default=None,
    )
    parser.add_argument(
        "--distillation-objective",
        choices=("advantage_regression", "cost_sensitive_classification"),
        default=None,
    )
    parser.add_argument(
        "--use-all-teacher-options",
        action="store_true",
        help="Use every ordered correction option stored in the teacher cache.",
    )
    parser.add_argument(
        "--output-root",
        default="results/network_residual_history_screen",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory; defaults to history<window>_pretrain.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    plan = load_benchmark_plan(args.plan)
    plan["output_root"] = str(args.output_root)
    budget = resolve_budget(plan, args.budget)
    scenario = select_scenarios(plan, (args.scenario,))[0]
    run_name = str(
        args.run_name
        or f"history{int(args.demand_history_window)}_pretrain"
    )

    for seed in args.seeds:
        config = make_history_screen_config(
            plan,
            budget_name=run_name,
            budget=budget,
            algorithm=args.algorithm,
            scenario=scenario,
            seed=int(seed),
            teacher_cache=args.teacher_cache,
            demand_history_window=int(args.demand_history_window),
            base_policy=str(args.base_policy),
            gcn_hidden_sizes=args.gcn_hidden_sizes,
            actor_hidden_sizes=args.actor_hidden_sizes,
            gate_hidden_sizes=args.gate_hidden_sizes,
            option_hidden_sizes=args.option_hidden_sizes,
            use_all_teacher_options=args.use_all_teacher_options,
            online_episodes=args.online_episodes,
            pretrain_epochs=args.pretrain_epochs,
            offline_updates=args.offline_updates,
            teacher_advantage_weight_scale=args.teacher_advantage_weight_scale,
            teacher_advantage_weight_cap=args.teacher_advantage_weight_cap,
            option_advantage_scale=args.option_advantage_scale,
            option_ranking_loss_weight=args.option_ranking_loss_weight,
            correction_gate_threshold=args.correction_gate_threshold,
            distillation_objective=args.distillation_objective,
            online_reward_mode=args.online_reward_mode,
            imitation_regularization_weight=args.imitation_regularization_weight,
            correction_gate_target_mode=args.correction_gate_target_mode,
            correction_selection_mode=args.correction_selection_mode,
        )
        checkpoint = (
            Path(config["checkpoint_dir"])
            / f"{args.algorithm}_seed{int(seed)}_pretrain.pt"
        )
        completion_checkpoint = (
            Path(config["checkpoint_dir"])
            / f"{args.algorithm}_seed{int(seed)}_episode"
            f"{int(config['num_episodes'])}.pt"
            if int(config["num_episodes"]) > 0
            else checkpoint
        )
        if completion_checkpoint.is_file() and not args.force:
            print(
                f"skip existing checkpoint {completion_checkpoint}",
                flush=True,
            )
            continue

        env = build_env(config, seed=int(seed))
        agent = get_agent_class(args.algorithm)(
            env.observation_size,
            env.action_size,
            config,
        )
        settings = advantage_distillation_settings(
            config,
            budget,
            args.algorithm,
        )

        pretrain_report: dict[str, Any] = {}

        def post_imitation_pretrain(current_agent, current_env):
            summary = run_advantage_distillation_pretrain(
                settings,
                algorithm=args.algorithm,
                seed=int(seed),
                agent=current_agent,
                env=current_env,
                config=config,
                budget=budget,
            )
            summary.update(
                train_offline_replay_updates(
                    current_agent,
                    updates=int(args.offline_updates),
                    progress_interval=int(args.offline_progress_interval),
                )
            )
            pretrain_report.update(summary)
            return summary

        training_rows = train_off_policy_agent(
            agent,
            env,
            config,
            post_imitation_pretrain=post_imitation_pretrain,
        )
        if int(config["num_episodes"]) > 0 and not completion_checkpoint.is_file():
            agent.save(completion_checkpoint)
        if training_rows:
            write_rows(training_rows, config["result_csv_path"])
        pretrain_report_path = checkpoint.with_name(
            f"{args.algorithm}_seed{int(seed)}_pretrain_summary.json"
        )
        pretrain_report_path.write_text(
            json.dumps(pretrain_report, indent=2, sort_keys=True) + "\n"
        )
        save_config_snapshot(config, config["config_snapshot_path"])
        if not checkpoint.is_file():
            raise RuntimeError(f"Expected pretrain checkpoint was not written: {checkpoint}")
        if not completion_checkpoint.is_file():
            raise RuntimeError(
                f"Expected training checkpoint was not written: "
                f"{completion_checkpoint}"
            )
        print(f"wrote {completion_checkpoint}", flush=True)


def make_history_screen_config(
    plan: dict[str, Any],
    *,
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
    teacher_cache: str | Path,
    demand_history_window: int,
    base_policy: str = "mdl2",
    gcn_hidden_sizes: list[int] | tuple[int, ...] | None = None,
    actor_hidden_sizes: list[int] | tuple[int, ...] | None = None,
    gate_hidden_sizes: list[int] | tuple[int, ...] | None = None,
    option_hidden_sizes: list[int] | tuple[int, ...] | None = None,
    use_all_teacher_options: bool = False,
    online_episodes: int = 0,
    pretrain_epochs: int | None = None,
    offline_updates: int = 0,
    teacher_advantage_weight_scale: float | None = None,
    teacher_advantage_weight_cap: float | None = None,
    option_advantage_scale: float | None = None,
    option_ranking_loss_weight: float | None = None,
    correction_gate_threshold: float | None = None,
    distillation_objective: str | None = None,
    online_reward_mode: str | None = None,
    imitation_regularization_weight: float | None = None,
    correction_gate_target_mode: str | None = None,
    correction_selection_mode: str | None = None,
) -> dict[str, Any]:
    """Build a pretrain-only config whose cache and online history agree."""

    if demand_history_window < 1:
        raise ValueError("demand_history_window must be positive")
    cache_path = Path(teacher_cache)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        cache_window = (
            int(payload["demand_history_window"])
            if "demand_history_window" in payload.files
            else None
        )
    if cache_window is not None and cache_window != demand_history_window:
        raise ValueError(
            f"Teacher cache history window {cache_window} does not match "
            f"requested window {demand_history_window}"
        )

    config = make_training_config(
        plan,
        budget_name,
        budget,
        algorithm,
        scenario,
        seed,
    )
    if online_episodes < 0:
        raise ValueError("online_episodes must be non-negative")
    if offline_updates < 0:
        raise ValueError("offline_updates must be non-negative")
    config["num_episodes"] = int(online_episodes)
    config["env"]["demand_history_window"] = int(demand_history_window)
    config.setdefault("residual_action", {})["base_policy"] = str(base_policy)
    config.setdefault("imitation_pretrain", {})["policy"] = str(base_policy)
    if gcn_hidden_sizes:
        config["gcn_hidden_sizes"] = [
            int(value) for value in gcn_hidden_sizes
        ]
    if actor_hidden_sizes:
        sizes = [int(value) for value in actor_hidden_sizes]
        if str(algorithm).startswith("flat_"):
            config["hidden_sizes"] = sizes
        else:
            config["actor_hidden_sizes"] = sizes
    if gate_hidden_sizes:
        config.setdefault("residual_action", {}).setdefault(
            "correction_gate",
            {},
        )["hidden_sizes"] = [
            int(value) for value in gate_hidden_sizes
        ]
    if option_hidden_sizes:
        config.setdefault("residual_option", {})["hidden_sizes"] = [
            int(value) for value in option_hidden_sizes
        ]
    if option_advantage_scale is not None:
        if option_advantage_scale <= 0.0:
            raise ValueError("option_advantage_scale must be positive")
        config.setdefault("residual_option", {})[
            "option_advantage_scale"
        ] = float(option_advantage_scale)
    if option_ranking_loss_weight is not None:
        if option_ranking_loss_weight < 0.0:
            raise ValueError("option_ranking_loss_weight must be non-negative")
        config.setdefault("residual_option", {})[
            "ranking_loss_weight"
        ] = float(option_ranking_loss_weight)
    if correction_gate_threshold is not None:
        if not 0.0 <= correction_gate_threshold <= 1.0:
            raise ValueError("correction_gate_threshold must lie in [0, 1]")
        config.setdefault("residual_option", {})[
            "correction_gate_threshold"
        ] = float(correction_gate_threshold)
    if distillation_objective is not None:
        config.setdefault("residual_option", {})[
            "distillation_objective"
        ] = str(distillation_objective)
    if online_reward_mode is not None:
        config.setdefault("residual_option", {})[
            "online_reward_mode"
        ] = str(online_reward_mode)
    if imitation_regularization_weight is not None:
        if imitation_regularization_weight < 0.0:
            raise ValueError(
                "imitation_regularization_weight must be non-negative"
            )
        config.setdefault("residual_option", {})[
            "imitation_regularization_weight"
        ] = float(imitation_regularization_weight)
    if correction_gate_target_mode is not None:
        config.setdefault("residual_option", {})[
            "correction_gate_target_mode"
        ] = str(correction_gate_target_mode)
    if correction_selection_mode is not None:
        config.setdefault("residual_option", {})[
            "correction_selection_mode"
        ] = str(correction_selection_mode)
    explicit_options = None
    if use_all_teacher_options:
        explicit_options = configure_options_from_teacher_cache(
            config,
            cache_path,
        )
    config.setdefault("advantage_distillation_pretrain", {})[
        "demonstration_path"
    ] = str(cache_path)
    if pretrain_epochs is not None:
        if pretrain_epochs < 1:
            raise ValueError("pretrain_epochs must be positive")
        config["advantage_distillation_pretrain"]["epochs"] = int(
            pretrain_epochs
        )
    if teacher_advantage_weight_scale is not None:
        if teacher_advantage_weight_scale <= 0.0:
            raise ValueError("teacher_advantage_weight_scale must be positive")
        config["advantage_distillation_pretrain"][
            "dense_advantage_weight_scale"
        ] = float(teacher_advantage_weight_scale)
    if teacher_advantage_weight_cap is not None:
        if teacher_advantage_weight_cap < 1.0:
            raise ValueError("teacher_advantage_weight_cap must be at least 1")
        config["advantage_distillation_pretrain"][
            "dense_advantage_weight_cap"
        ] = float(teacher_advantage_weight_cap)
    config["advantage_distillation_pretrain"]["baseline_policy"] = str(
        base_policy
    )
    config["history_screen"] = {
        "teacher_cache": str(cache_path),
        "demand_history_window": int(demand_history_window),
        "pretrain_only": int(online_episodes) == 0,
        "online_episodes": int(online_episodes),
        "offline_updates": int(offline_updates),
        "base_policy": str(base_policy),
        "imitation_policy": str(
            config.get("imitation_pretrain", {}).get("policy", "")
        ),
        "gcn_hidden_sizes": list(config.get("gcn_hidden_sizes", ())),
        "hidden_sizes": list(config.get("hidden_sizes", ())),
        "actor_hidden_sizes": list(config.get("actor_hidden_sizes", ())),
        "gate_hidden_sizes": list(
            config.get("residual_action", {})
            .get("correction_gate", {})
            .get("hidden_sizes", ())
        ),
        "option_hidden_sizes": list(
            config.get("residual_option", {}).get("hidden_sizes", ())
        ),
        "teacher_option_count": (
            len(explicit_options) + 1 if explicit_options is not None else None
        ),
        "pretrain_epochs": int(
            config.get("advantage_distillation_pretrain", {}).get("epochs", 0)
        ),
        "teacher_advantage_weight_scale": (
            float(
                config.get("advantage_distillation_pretrain", {}).get(
                    "dense_advantage_weight_scale"
                )
            )
            if config.get("advantage_distillation_pretrain", {}).get(
                "dense_advantage_weight_scale"
            )
            is not None
            else None
        ),
        "teacher_advantage_weight_cap": float(
            config.get("advantage_distillation_pretrain", {}).get(
                "dense_advantage_weight_cap",
                10.0,
            )
        ),
        "option_advantage_scale": float(
            config.get("residual_option", {}).get(
                "option_advantage_scale",
                0.0,
            )
        ),
        "option_ranking_loss_weight": float(
            config.get("residual_option", {}).get(
                "ranking_loss_weight",
                0.0,
            )
        ),
        "correction_gate_threshold": float(
            config.get("residual_option", {}).get(
                "correction_gate_threshold",
                0.0,
            )
        ),
        "distillation_objective": str(
            config.get("residual_option", {}).get(
                "distillation_objective",
                "advantage_regression",
            )
        ),
        "online_reward_mode": str(
            config.get("residual_option", {}).get(
                "online_reward_mode",
                "environment",
            )
        ),
        "imitation_regularization_weight": float(
            config.get("residual_option", {}).get(
                "imitation_regularization_weight",
                0.0,
            )
        ),
        "correction_gate_target_mode": str(
            config.get("residual_option", {}).get(
                "correction_gate_target_mode",
                "any_improving",
            )
        ),
        "correction_selection_mode": str(
            config.get("residual_option", {}).get(
                "correction_selection_mode",
                "q_value",
            )
        ),
    }
    return config


if __name__ == "__main__":
    main()
