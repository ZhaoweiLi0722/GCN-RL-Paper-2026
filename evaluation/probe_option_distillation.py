"""Gate a graph option policy after teacher distillation and before RL training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from evaluation.option_teacher_config import configure_options_from_teacher_cache
from evaluation.run_full_benchmark import (
    LEARNED_EVALUATION_SEED_STRIDE,
    advantage_distillation_settings,
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    run_advantage_distillation_pretrain,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, write_rows


DEFAULT_PLAN = "experiments/configs/residual_policy_benchmark.json"
DEFAULT_ALGORITHM = "gcn_residual_mdl2_option_dqn_afd"
DEFAULT_SCENARIO = "patient_condition_geo_demand_drift"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--budget", default="network_targeted_300")
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--evaluation-seed", type=int, default=97000)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--gcn-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--option-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--use-all-teacher-options",
        action="store_true",
        help="Use every ordered correction option stored in the teacher cache.",
    )
    parser.add_argument(
        "--common-evaluation-seed",
        action="store_true",
        help="Do not offset the evaluation seed by training seed.",
    )
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--anchor-q-margin", type=float, default=None)
    parser.add_argument("--correction-gate-threshold", type=float, default=None)
    parser.add_argument("--correction-gate-loss-weight", type=float, default=None)
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
        "--correction-gate-positive-weight-power",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--correction-gate-positive-weight-cap",
        type=float,
        default=None,
    )
    parser.add_argument("--disable-correction-gate", action="store_true")
    parser.add_argument("--option-advantage-scale", type=float, default=None)
    parser.add_argument("--option-advantage-clip", type=float, default=None)
    parser.add_argument("--option-positive-advantage-weight", type=float, default=None)
    parser.add_argument("--option-infeasible-penalty", type=float, default=None)
    parser.add_argument("--option-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--distillation-objective", default=None)
    parser.add_argument(
        "--output-root",
        default="results/option_distillation_probe",
    )
    args = parser.parse_args()

    plan = load_benchmark_plan(args.plan)
    budget = resolve_budget(plan, args.budget)
    scenario = select_scenarios(plan, (args.scenario,))[0]
    config = make_training_config(
        plan,
        args.budget,
        budget,
        args.algorithm,
        scenario,
        args.seed,
    )
    if args.anchor_q_margin is not None:
        config.setdefault("residual_option", {})["anchor_q_margin"] = float(
            args.anchor_q_margin
        )
    if args.correction_gate_threshold is not None:
        config.setdefault("residual_option", {})[
            "correction_gate_threshold"
        ] = float(args.correction_gate_threshold)
    if args.correction_gate_loss_weight is not None:
        config.setdefault("residual_option", {})[
            "correction_gate_loss_weight"
        ] = float(args.correction_gate_loss_weight)
    if args.correction_gate_target_mode is not None:
        config.setdefault("residual_option", {})[
            "correction_gate_target_mode"
        ] = str(args.correction_gate_target_mode)
    if args.correction_selection_mode is not None:
        config.setdefault("residual_option", {})[
            "correction_selection_mode"
        ] = str(args.correction_selection_mode)
    if args.correction_gate_positive_weight_power is not None:
        config.setdefault("residual_option", {})[
            "correction_gate_positive_weight_power"
        ] = float(args.correction_gate_positive_weight_power)
    if args.correction_gate_positive_weight_cap is not None:
        config.setdefault("residual_option", {})[
            "correction_gate_positive_weight_cap"
        ] = float(args.correction_gate_positive_weight_cap)
    if args.disable_correction_gate:
        config.setdefault("residual_option", {})[
            "correction_gate_enabled"
        ] = False
    if args.option_advantage_scale is not None:
        config.setdefault("residual_option", {})[
            "option_advantage_scale"
        ] = float(args.option_advantage_scale)
    if args.option_advantage_clip is not None:
        config.setdefault("residual_option", {})[
            "option_advantage_clip"
        ] = float(args.option_advantage_clip)
    if args.option_positive_advantage_weight is not None:
        config.setdefault("residual_option", {})[
            "positive_advantage_weight"
        ] = float(args.option_positive_advantage_weight)
    if args.option_infeasible_penalty is not None:
        config.setdefault("residual_option", {})[
            "option_infeasible_penalty"
        ] = float(args.option_infeasible_penalty)
    if args.option_ranking_loss_weight is not None:
        config.setdefault("residual_option", {})[
            "ranking_loss_weight"
        ] = float(args.option_ranking_loss_weight)
    if args.distillation_objective is not None:
        config.setdefault("residual_option", {})[
            "distillation_objective"
        ] = str(args.distillation_objective)
    settings = advantage_distillation_settings(
        config,
        budget,
        args.algorithm,
    )
    if args.teacher_cache is not None:
        settings["demonstration_path"] = args.teacher_cache
    if args.epochs is not None:
        settings["epochs"] = int(args.epochs)
    raw_demonstration_path = settings.get("demonstration_path")
    if not raw_demonstration_path:
        raise ValueError("Advantage distillation requires a teacher cache path")
    demonstration_path = Path(str(raw_demonstration_path))
    if not demonstration_path.is_file():
        raise FileNotFoundError(
            f"Teacher cache is not ready: {demonstration_path}"
        )
    if args.option_hidden_sizes:
        config.setdefault("residual_option", {})["hidden_sizes"] = [
            int(value) for value in args.option_hidden_sizes
        ]
    if args.gcn_hidden_sizes:
        config["gcn_hidden_sizes"] = [
            int(value) for value in args.gcn_hidden_sizes
        ]
    if args.use_all_teacher_options:
        configure_options_from_teacher_cache(config, demonstration_path)

    env = build_env(config, seed=args.seed)
    agent = get_agent_class(args.algorithm)(
        env.observation_size,
        env.action_size,
        config,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.load_checkpoint is not None:
        checkpoint_path = Path(args.load_checkpoint)
        agent.load_actor(checkpoint_path)
        distillation = {
            "advantage_distillation_source": "loaded_checkpoint",
            "advantage_distillation_checkpoint": str(checkpoint_path),
        }
    else:
        distillation = run_advantage_distillation_pretrain(
            settings,
            algorithm=args.algorithm,
            seed=args.seed,
            agent=agent,
            env=env,
            config=config,
            budget=budget,
        )
        checkpoint_path = (
            output_root / f"{args.algorithm}_seed{args.seed}_pretrain.pt"
        )
        agent.save(checkpoint_path)
    if hasattr(agent, "reset_option_diagnostics"):
        agent.reset_option_diagnostics()

    evaluation_seed = int(args.evaluation_seed)
    if not args.common_evaluation_seed:
        evaluation_seed += int(args.seed) * LEARNED_EVALUATION_SEED_STRIDE
    candidate_env = build_env(config, seed=evaluation_seed)
    candidate_rows = evaluate_agent(
        agent,
        candidate_env,
        algorithm=args.algorithm,
        seed=evaluation_seed,
        replications=args.replications,
        max_steps=int(budget["max_steps_per_episode"]),
    )
    anchor_env = build_env(config, seed=evaluation_seed)
    anchor = get_heuristic_class("mdl2")(
        anchor_env.observation_size,
        anchor_env.action_size,
        {},
    )
    anchor_rows = evaluate_agent(
        anchor,
        anchor_env,
        algorithm="mdl2",
        seed=evaluation_seed,
        replications=args.replications,
        max_steps=int(budget["max_steps_per_episode"]),
    )
    annotate_pairing(candidate_rows, args.seed, evaluation_seed)
    annotate_pairing(anchor_rows, args.seed, evaluation_seed)
    write_rows(candidate_rows, output_root / "candidate.csv")
    write_rows(anchor_rows, output_root / "anchor.csv")

    paired = {
        metric: paired_two_level_summary(
            candidate_rows,
            anchor_rows,
            metric=metric,
            resamples=20000,
            seed=evaluation_seed,
        )
        for metric in (
            "total_cost",
            "completion_service_level",
            "patients_lost",
            "patient_ineligibility_during_manufacturing_rate",
            "average_waiting_time",
        )
    }
    result = {
        "algorithm": args.algorithm,
        "scenario": args.scenario,
        "seed": int(args.seed),
        "replications": int(args.replications),
        "evaluation_seed": evaluation_seed,
        "teacher_cache": str(demonstration_path),
        "checkpoint_path": str(checkpoint_path),
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in agent.q_network.parameters())
        ),
        "anchor_q_margin": float(agent.anchor_q_margin),
        "correction_gate_threshold": float(agent.correction_gate_threshold),
        "distillation": distillation,
        "candidate_summary": summarize_rows(candidate_rows),
        "anchor_summary": summarize_rows(anchor_rows),
        "paired": paired,
        "option_diagnostics": (
            agent.option_diagnostics()
            if hasattr(agent, "option_diagnostics")
            else {}
        ),
    }
    result["decision"] = pretrain_gate_decision(result)
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote option distillation probe to {output_root}", flush=True)


def annotate_pairing(
    rows: list[dict[str, Any]],
    training_seed: int,
    evaluation_seed: int,
) -> None:
    for row in rows:
        row["training_seed"] = int(training_seed)
        row["evaluation_seed"] = int(evaluation_seed)


def pretrain_gate_decision(result: dict[str, Any]) -> dict[str, Any]:
    paired = result["paired"]
    cost = paired["total_cost"]
    completion = paired["completion_service_level"]
    patients_lost = paired["patients_lost"]
    manufacturing = paired[
        "patient_ineligibility_during_manufacturing_rate"
    ]
    correction_rate = float(
        result.get("option_diagnostics", {}).get("correction_rate", 0.0)
    )
    clinically_noninferior = bool(
        float(completion["mean_difference"]) >= 0.0
        and float(patients_lost["mean_difference"]) <= 0.0
        and float(manufacturing["mean_difference"]) <= 0.0
    )
    cost_superior = bool(
        float(cost["mean_difference"]) < 0.0
        and float(cost["ci_high"]) < 0.0
    )
    correction_used = correction_rate > 0.0
    return {
        "advance_to_online_rl": bool(
            cost_superior and clinically_noninferior and correction_used
        ),
        "cost_superior": cost_superior,
        "clinically_noninferior": clinically_noninferior,
        "correction_used": correction_used,
        "cost_gap_pct": float(cost["mean_gap_pct"]),
        "cost_ci_high": float(cost["ci_high"]),
        "completion_service_level_difference": float(
            completion["mean_difference"]
        ),
        "patients_lost_difference": float(patients_lost["mean_difference"]),
        "manufacturing_ineligibility_rate_difference": float(
            manufacturing["mean_difference"]
        ),
        "correction_rate": correction_rate,
    }


if __name__ == "__main__":
    main()
