"""Calibrate an AFR-GCN-DDPG gate on actor-specific CRN lookahead returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from evaluation.network_residual_headroom import (
    evaluate_candidate_specs,
    lookahead_rollout_seeds,
    select_clinical_candidate,
)
from evaluation.run_full_benchmark import (
    checkpoint_dir_path,
    load_benchmark_plan,
    make_evaluation_config,
    resolve_budget,
    select_anchor_fallback_policy,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, write_rows
from src.rl.networks import torch


DEFAULT_PLAN = "experiments/configs/residual_policy_benchmark.json"
DEFAULT_BUDGET = "diagnostic_pretrain"
DEFAULT_ALGORITHM = "gcn_residual_mdl2_network_ddpg_afd"
DEFAULT_SCENARIO = "patient_condition_geo_demand_drift"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--budget", default=DEFAULT_BUDGET)
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output-root",
        default="results/actor_correction_gate_calibration",
    )
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument("--lookahead", type=int, default=12)
    parser.add_argument("--lookahead-replications", type=int, default=3)
    parser.add_argument("--actor-scale", type=float, default=0.25)
    parser.add_argument("--min-improvement", type=float, default=500_000.0)
    parser.add_argument("--gate-epochs", type=int, default=200)
    parser.add_argument("--gate-batch-size", type=int, default=128)
    parser.add_argument("--validation-replications", type=int, default=20)
    parser.add_argument("--evaluation-replications", type=int, default=50)
    parser.add_argument(
        "--scale-candidates",
        nargs="+",
        type=float,
        default=(0.0, 0.025, 0.05, 0.1, 0.25, 0.5),
    )
    parser.add_argument(
        "--threshold-candidates",
        nargs="+",
        type=float,
        default=(1.0, 1.5, 2.0, 2.25, 2.5),
    )
    parser.add_argument(
        "--reuse-examples",
        action="store_true",
        help="Reuse actor_gate_examples.npz instead of rerunning CRN lookahead.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rollouts = 1
        args.max_steps = 2
        args.lookahead = 2
        args.lookahead_replications = 1
        args.gate_epochs = 2
        args.validation_replications = 2
        args.evaluation_replications = 2
        args.scale_candidates = (0.0, 0.25)
        args.threshold_candidates = (0.5,)

    plan = load_benchmark_plan(args.plan)
    budget = resolve_budget(plan, args.budget)
    scenario = select_scenarios(plan, (args.scenario,))[0]
    config = make_evaluation_config(
        plan,
        args.budget,
        budget,
        args.algorithm,
        scenario,
        args.seed,
    )
    source_checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else checkpoint_dir_path(
            plan,
            args.budget,
            args.algorithm,
            scenario,
            args.seed,
        )
        / f"{args.algorithm}_seed{args.seed}_pretrain.pt"
    )
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    output_root = Path(args.output_root)
    if args.smoke:
        output_root = output_root / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    env = build_env(config, seed=args.seed)
    agent = get_agent_class(args.algorithm)(
        env.observation_size,
        env.action_size,
        config,
    )
    agent.load_actor(source_checkpoint)
    if not hasattr(agent, "select_ungated_residual_action"):
        raise ValueError(f"{args.algorithm} does not expose ungated residual actions")
    if not hasattr(agent, "fit_correction_gate_batch"):
        raise ValueError(f"{args.algorithm} does not expose correction-gate fitting")

    example_path = output_root / "actor_gate_examples.npz"
    if args.reuse_examples and example_path.exists():
        cached = np.load(example_path)
        collection = {
            "states": np.asarray(cached["states"], dtype=np.float32),
            "labels": np.asarray(cached["labels"], dtype=np.float32),
            "advantages": np.asarray(cached["advantages"], dtype=np.float32),
            "feasible": np.asarray(cached["feasible"], dtype=bool),
            "rows": [],
        }
    else:
        collection = collect_actor_gate_examples(
            agent,
            config,
            rollouts=args.rollouts,
            max_steps=args.max_steps,
            lookahead=args.lookahead,
            lookahead_replications=args.lookahead_replications,
            actor_scale=args.actor_scale,
            min_improvement=args.min_improvement,
            seed=args.seed + 410_000,
        )
    weights = balanced_gate_weights(
        collection["labels"],
        collection["advantages"],
    )
    gate_summary = agent.fit_correction_gate_batch(
        collection["states"],
        collection["labels"],
        {
            "epochs": args.gate_epochs,
            "batch_size": args.gate_batch_size,
            "seed": args.seed + 420_000,
            "mode": "advantage",
            "advantages": collection["advantages"],
            "feasible": collection["feasible"],
        },
        weights=weights,
    )
    np.savez_compressed(
        output_root / "actor_gate_examples.npz",
        states=collection["states"],
        labels=collection["labels"],
        advantages=collection["advantages"],
        feasible=collection["feasible"],
        weights=weights,
    )
    if collection["rows"]:
        write_rows(collection["rows"], output_root / "actor_gate_examples.csv")

    calibrated_checkpoint = output_root / "gcn_ddpg_actor_gate_calibrated.pt"
    checkpoint = torch.load(source_checkpoint, map_location=agent.device)
    checkpoint["correction_gate"] = agent.correction_gate.state_dict()
    checkpoint["correction_gate_mode"] = agent.correction_gate_mode
    checkpoint["correction_gate_threshold"] = agent.correction_gate_threshold
    checkpoint["actor_gate_calibration"] = {
        "lookahead": int(args.lookahead),
        "lookahead_replications": int(args.lookahead_replications),
        "actor_scale": float(args.actor_scale),
        "min_improvement": float(args.min_improvement),
        **gate_summary,
    }
    torch.save(checkpoint, calibrated_checkpoint)

    validation = select_deployment_scale(
        agent,
        config,
        args.algorithm,
        scales=tuple(args.scale_candidates),
        thresholds=tuple(args.threshold_candidates),
        replications=args.validation_replications,
        seed=args.seed + 510_000,
    )
    agent.correction_gate_threshold = float(validation["selected_threshold"])
    checkpoint["correction_gate_threshold"] = agent.correction_gate_threshold
    checkpoint["actor_gate_calibration"]["selected_threshold"] = (
        agent.correction_gate_threshold
    )
    checkpoint["actor_gate_calibration"]["selected_scale"] = float(
        validation["selected_scale"]
    )
    torch.save(checkpoint, calibrated_checkpoint)
    final = evaluate_selected_policy(
        agent,
        config,
        args.algorithm,
        scale=float(validation["selected_scale"]),
        threshold=float(validation["selected_threshold"]),
        replications=args.evaluation_replications,
        seed=args.seed + 610_000,
        output_root=output_root,
    )
    summary = {
        "source_checkpoint": str(source_checkpoint),
        "calibrated_checkpoint": str(calibrated_checkpoint),
        "collection": {
            "states": int(collection["states"].shape[0]),
            "positive_labels": int(collection["labels"].sum()),
            "positive_rate": float(collection["labels"].mean()),
            "mean_advantage": float(collection["advantages"].mean()),
            "mean_positive_advantage": float(
                collection["advantages"][collection["labels"] > 0.5].mean()
            )
            if np.any(collection["labels"] > 0.5)
            else 0.0,
        },
        "gate_fit": gate_summary,
        "validation": validation,
        "evaluation": final,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def collect_actor_gate_examples(
    agent,
    config: dict[str, Any],
    *,
    rollouts: int,
    max_steps: int,
    lookahead: int,
    lookahead_replications: int,
    actor_scale: float,
    min_improvement: float,
    seed: int,
) -> dict[str, Any]:
    states: list[np.ndarray] = []
    labels: list[float] = []
    advantages: list[float] = []
    feasible_rows: list[bool] = []
    rows: list[dict[str, Any]] = []
    decision_index = 0
    rollout_config = {
        "seed": int(seed),
        "lookahead_seed": int(seed) + 1_000_000,
        "lookahead_replications": int(lookahead_replications),
    }
    for rollout in range(int(rollouts)):
        env = build_env(config, seed=seed + rollout)
        anchor = get_heuristic_class("mdl2")(
            env.observation_size,
            env.action_size,
            {},
        )
        state = env.reset(seed=seed + rollout)
        anchor.reset()
        done = False
        step = 0
        while not done and step < int(max_steps):
            anchor_action = anchor.select_action(state, explore=False, env=env)
            actor_action = agent.select_ungated_residual_action(
                state,
                env=env,
                residual_scale=actor_scale,
            )
            evaluated = evaluate_candidate_specs(
                [
                    {
                        "group": "anchor",
                        "epsilon": 0.0,
                        "variant": 0,
                        "action": anchor_action,
                    },
                    {
                        "group": "actor",
                        "epsilon": float(actor_scale),
                        "variant": 1,
                        "action": actor_action,
                    },
                ],
                env,
                anchor,
                lookahead=int(lookahead),
                rollout_seeds=lookahead_rollout_seeds(
                    rollout_config,
                    decision_index,
                ),
            )
            best_index, scores, feasible = select_clinical_candidate(
                evaluated,
                score_weights={},
                guardrails={
                    "min_score_improvement": float(min_improvement),
                    "min_completion_service_level_delta": 0.0,
                    "max_patients_lost_delta": 0.0,
                    "max_patient_ineligibility_during_manufacturing_rate_delta": 0.0,
                },
            )
            advantage = float(scores[0] - scores[1])
            accepted = bool(best_index == 1 and feasible[1])
            anchor_metrics = evaluated[0]["metrics"]
            actor_metrics = evaluated[1]["metrics"]
            states.append(np.asarray(state, dtype=np.float32))
            labels.append(float(accepted))
            advantages.append(advantage)
            feasible_rows.append(bool(feasible[1]))
            rows.append(
                {
                    "rollout": rollout,
                    "step": step,
                    "label": int(accepted),
                    "advantage": advantage,
                    "feasible": int(bool(feasible[1])),
                    "anchor_cost": float(anchor_metrics["total_cost"]),
                    "actor_cost": float(actor_metrics["total_cost"]),
                    "completion_service_level_delta": float(
                        actor_metrics["completion_service_level"]
                        - anchor_metrics["completion_service_level"]
                    ),
                    "patients_lost_delta": float(
                        actor_metrics["patients_lost"]
                        - anchor_metrics["patients_lost"]
                    ),
                    "manufacturing_ineligibility_rate_delta": float(
                        actor_metrics[
                            "patient_ineligibility_during_manufacturing_rate"
                        ]
                        - anchor_metrics[
                            "patient_ineligibility_during_manufacturing_rate"
                        ]
                    ),
                }
            )
            decision_index += 1
            if decision_index % 50 == 0:
                print(
                    "actor_gate_collection "
                    f"states={decision_index} positives={int(sum(labels))} "
                    f"rate={float(np.mean(labels)):.3f}",
                    flush=True,
                )
            state, _reward, done, _info = env.step(anchor_action)
            step += 1
    return {
        "states": np.asarray(states, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "advantages": np.asarray(advantages, dtype=np.float32),
        "feasible": np.asarray(feasible_rows, dtype=bool),
        "rows": rows,
    }


def balanced_gate_weights(
    labels: np.ndarray,
    advantages: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    advantages = np.asarray(advantages, dtype=np.float32)
    weights = 1.0 + np.clip(np.maximum(advantages, 0.0) / 1_000_000.0, 0.0, 5.0)
    positives = labels > 0.5
    for mask in (positives, ~positives):
        if np.any(mask):
            weights[mask] *= 0.5 / float(weights[mask].sum())
    weights *= float(weights.size) / max(float(weights.sum()), 1e-8)
    return weights.astype(np.float32)


def select_deployment_scale(
    agent,
    config: dict[str, Any],
    algorithm: str,
    *,
    scales: tuple[float, ...],
    thresholds: tuple[float, ...],
    replications: int,
    seed: int,
) -> dict[str, Any]:
    anchor_env = build_env(config, seed=seed)
    anchor = get_heuristic_class("mdl2")(
        anchor_env.observation_size,
        anchor_env.action_size,
        {},
    )
    anchor_rows = evaluate_agent(
        anchor,
        anchor_env,
        algorithm="mdl2",
        seed=seed,
        replications=replications,
        max_steps=int(config["max_steps_per_episode"]),
    )
    anchor_summary = summarize_rows(anchor_rows)
    original_scale = np.asarray(agent.residual_scale_vector, dtype=np.float32).copy()
    candidates = []
    for threshold in thresholds:
        agent.correction_gate_threshold = float(threshold)
        for scale in scales:
            agent.residual_scale_vector = original_scale * float(scale)
            env = build_env(config, seed=seed)
            rows = evaluate_agent(
                agent,
                env,
                algorithm=algorithm,
                seed=seed,
                replications=replications,
                max_steps=int(config["max_steps_per_episode"]),
            )
            candidate_summary = summarize_rows(rows)
            candidate_cost = float(candidate_summary["total_cost_mean"])
            anchor_cost = float(anchor_summary["total_cost_mean"])
            decision = select_anchor_fallback_policy(
                candidate_cost,
                anchor_cost,
                learned_completion_service_level=float(
                    candidate_summary["completion_service_level_mean"]
                ),
                anchor_completion_service_level=float(
                    anchor_summary["completion_service_level_mean"]
                ),
                min_completion_service_level_delta=0.0,
                learned_patients_lost=float(candidate_summary["patients_lost_mean"]),
                anchor_patients_lost=float(anchor_summary["patients_lost_mean"]),
                max_patients_lost_delta=0.0,
                learned_patient_ineligibility_during_manufacturing_rate=float(
                    candidate_summary[
                        "patient_ineligibility_during_manufacturing_rate_mean"
                    ]
                ),
                anchor_patient_ineligibility_during_manufacturing_rate=float(
                    anchor_summary[
                        "patient_ineligibility_during_manufacturing_rate_mean"
                    ]
                ),
                max_patient_ineligibility_during_manufacturing_rate_delta=0.0,
            )
            if candidate_cost >= anchor_cost - 1e-6:
                decision = "anchor"
            candidates.append(
                {
                    "threshold": float(threshold),
                    "scale": float(scale),
                    "decision": decision,
                    "cost": candidate_cost,
                    "cost_gap_pct": 100.0
                    * (candidate_cost - anchor_cost)
                    / anchor_cost,
                    "completion_delta": float(
                        candidate_summary["completion_service_level_mean"]
                    )
                    - float(anchor_summary["completion_service_level_mean"]),
                    "patients_lost_delta": float(
                        candidate_summary["patients_lost_mean"]
                    )
                    - float(anchor_summary["patients_lost_mean"]),
                }
            )
    eligible = [
        row
        for row in candidates
        if row["decision"] == "learned" and row["scale"] > 0.0
    ]
    selected = min(
        eligible or [row for row in candidates if row["scale"] == 0.0],
        key=lambda row: row["cost"],
    )
    agent.residual_scale_vector = original_scale
    return {
        "selected_threshold": selected["threshold"],
        "selected_scale": selected["scale"],
        "selected_decision": (
            "learned" if selected["scale"] > 0.0 else "anchor"
        ),
        "anchor_cost": float(anchor_summary["total_cost_mean"]),
        "candidates": candidates,
    }


def evaluate_selected_policy(
    agent,
    config: dict[str, Any],
    algorithm: str,
    *,
    scale: float,
    threshold: float,
    replications: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    original_scale = np.asarray(agent.residual_scale_vector, dtype=np.float32).copy()
    original_threshold = float(agent.correction_gate_threshold)
    agent.correction_gate_threshold = float(threshold)
    agent.residual_scale_vector = original_scale * float(scale)
    learned_env = build_env(config, seed=seed)
    learned_rows = evaluate_agent(
        agent,
        learned_env,
        algorithm=algorithm,
        seed=seed,
        replications=replications,
        max_steps=int(config["max_steps_per_episode"]),
    )
    anchor_env = build_env(config, seed=seed)
    anchor = get_heuristic_class("mdl2")(
        anchor_env.observation_size,
        anchor_env.action_size,
        {},
    )
    anchor_rows = evaluate_agent(
        anchor,
        anchor_env,
        algorithm="mdl2",
        seed=seed,
        replications=replications,
        max_steps=int(config["max_steps_per_episode"]),
    )
    agent.residual_scale_vector = original_scale
    agent.correction_gate_threshold = original_threshold
    write_rows(learned_rows, output_root / "learned_evaluation.csv")
    write_rows(anchor_rows, output_root / "anchor_evaluation.csv")
    learned_summary = summarize_rows(learned_rows)
    anchor_summary = summarize_rows(anchor_rows)
    learned_cost = float(learned_summary["total_cost_mean"])
    anchor_cost = float(anchor_summary["total_cost_mean"])
    return {
        "scale": float(scale),
        "threshold": float(threshold),
        "replications": int(replications),
        "learned_cost": learned_cost,
        "anchor_cost": anchor_cost,
        "cost_gap_pct": 100.0 * (learned_cost - anchor_cost) / anchor_cost,
        "completion_delta": float(
            learned_summary["completion_service_level_mean"]
        )
        - float(anchor_summary["completion_service_level_mean"]),
        "patients_lost_delta": float(learned_summary["patients_lost_mean"])
        - float(anchor_summary["patients_lost_mean"]),
        "manufacturing_ineligibility_rate_delta": float(
            learned_summary[
                "patient_ineligibility_during_manufacturing_rate_mean"
            ]
        )
        - float(
            anchor_summary[
                "patient_ineligibility_during_manufacturing_rate_mean"
            ]
        ),
    }


if __name__ == "__main__":
    main()
