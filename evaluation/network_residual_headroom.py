"""Measure learnable action headroom around the calibrated MDL-2 anchor."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from evaluation.run_gcn_residual_sweep import (
    local_search_candidate_actions,
    local_search_metric_score,
    mean_rollout_metrics_after_action,
    save_local_search_demonstrations,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    make_residual_option_specs,
    residual_option_actions_from_env,
)


DEFAULT_CONFIG = "experiments/configs/network_residual_headroom.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-state-probe", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--teacher-shard-index", type=int, default=None)
    parser.add_argument("--teacher-shard-count", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    if (
        args.teacher_shard_index is not None
        or args.teacher_shard_count is not None
    ):
        if (
            args.teacher_shard_index is None
            or args.teacher_shard_count is None
        ):
            raise SystemExit(
                "Use --teacher-shard-index and --teacher-shard-count together"
            )
        config = teacher_shard_config(
            config,
            args.teacher_shard_index,
            args.teacher_shard_count,
        )
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "name": config.get("name", "network_residual_headroom"),
        "config": config,
    }
    previous_path = output_root / "summary.json"
    previous = (
        json.loads(previous_path.read_text())
        if previous_path.exists()
        else {}
    )
    if not args.skip_state_probe:
        probe_rows = run_state_probe(config)
        write_rows(probe_rows, output_root / "state_probe.csv")
        result["state_probe"] = summarize_state_probe(probe_rows, config)
    elif "state_probe" in previous:
        result["state_probe"] = previous["state_probe"]
    if not args.skip_teacher:
        teacher_result = run_online_teacher(config)
        result["online_teacher"] = teacher_result
    elif "online_teacher" in previous:
        result["online_teacher"] = previous["online_teacher"]

    result["decision"] = headroom_decision(result, config)
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote headroom outputs to {output_root}")


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    smoke["max_steps"] = 3
    smoke["state_probe_rollouts"] = 1
    smoke["teacher_replications"] = 2
    smoke["lookahead"] = 2
    if not smoke.get("store_option_advantages", False):
        smoke["epsilons"] = [0.01]
        smoke["candidate_groups"] = [
            "replenishment_pressure",
            "reagent_transfer",
            "capacity_transfer",
        ]
    smoke["output_root"] = f"{config['output_root']}/smoke"
    if config.get("demonstration_path"):
        smoke["demonstration_path"] = (
            f"{smoke['output_root']}/network_teacher_smoke.npz"
        )
    return smoke


def teacher_shard_config(
    config: dict[str, Any],
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Partition independent teacher replications without changing RNG streams."""

    total = int(config["teacher_replications"])
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("Teacher shard index/count are invalid")
    if shard_count > total:
        raise ValueError("Teacher shard count cannot exceed replications")
    base_size, remainder = divmod(total, shard_count)
    replication_count = base_size + int(shard_index < remainder)
    replication_start = (
        shard_index * base_size + min(shard_index, remainder)
    )
    shard = copy.deepcopy(config)
    shard["teacher_replications"] = replication_count
    shard["teacher_replication_start"] = replication_start
    shard["lookahead_decision_offset"] = (
        replication_start * int(config["max_steps"])
    )
    shard_root = (
        Path(config["output_root"])
        / "shards"
        / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    )
    shard["output_root"] = str(shard_root)
    shard["demonstration_path"] = str(shard_root / "teacher_cache.npz")
    shard["name"] = (
        f"{config.get('name', 'network_residual_headroom')}"
        f"_shard_{shard_index:02d}_of_{shard_count:02d}"
    )
    return shard


def load_env_config(config: dict[str, Any]) -> dict[str, Any]:
    env_config = load_config(config["env_config"])
    return {
        "algorithm": str(config.get("anchor_policy", "mdl2")),
        "env": env_config,
    }


def make_anchor(config: dict[str, Any], env):
    algorithm = str(config.get("anchor_policy", "mdl2"))
    return get_heuristic_class(algorithm)(
        env.observation_size,
        env.action_size,
        {},
    )


def candidate_action_specs(
    state: np.ndarray,
    env,
    anchor,
    *,
    epsilons: Sequence[float],
    candidate_groups: Sequence[str],
    candidate_signs: Sequence[float],
) -> list[dict[str, Any]]:
    anchor_action = anchor.select_action(state, explore=False, env=env)
    specs: list[dict[str, Any]] = [
        {
            "group": "anchor",
            "epsilon": 0.0,
            "variant": 0,
            "action": np.asarray(anchor_action, dtype=np.float32),
        }
    ]
    seen = {specs[0]["action"].tobytes()}
    for group in candidate_groups:
        for epsilon in epsilons:
            actions = local_search_candidate_actions(
                state,
                env,
                anchor,
                epsilons=(float(epsilon),),
                candidate_groups=(str(group),),
                candidate_signs=tuple(float(sign) for sign in candidate_signs),
            )
            variant = 0
            for action in actions[1:]:
                normalized = np.asarray(action, dtype=np.float32)
                key = normalized.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                variant += 1
                specs.append(
                    {
                        "group": str(group),
                        "epsilon": float(epsilon),
                        "variant": variant,
                        "action": normalized,
                    }
                )
    return specs


def lookahead_rollout_seeds(
    config: dict[str, Any],
    decision_index: int,
) -> tuple[int, ...]:
    """Return common random numbers that are independent of the live episode."""

    replications = max(int(config.get("lookahead_replications", 1)), 1)
    base_seed = int(config.get("lookahead_seed", int(config["seed"]) + 1770000))
    decision_offset = int(config.get("lookahead_decision_offset", 0))
    start = base_seed + (decision_offset + int(decision_index)) * replications
    return tuple(start + replication for replication in range(replications))


def evaluate_candidate_specs(
    specs: list[dict[str, Any]],
    env,
    anchor,
    *,
    lookahead: int,
    rollout_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    evaluated = []
    for spec in specs:
        row = dict(spec)
        row["metrics"] = mean_rollout_metrics_after_action(
            env,
            anchor,
            spec["action"],
            horizon=int(lookahead),
            rollout_seeds=rollout_seeds,
        )
        evaluated.append(row)
    return evaluated


def select_clinical_candidate(
    evaluated: list[dict[str, Any]],
    *,
    score_weights: dict[str, float],
    guardrails: dict[str, float],
) -> tuple[int, list[float], list[bool]]:
    if not evaluated:
        raise ValueError("At least one candidate is required")
    anchor_metrics = evaluated[0]["metrics"]
    anchor_completion = float(
        anchor_metrics.get("completion_service_level", float("nan"))
    )
    anchor_lost = float(anchor_metrics.get("patients_lost", float("nan")))
    anchor_mfg_ineligibility = float(
        anchor_metrics.get(
            "patient_ineligibility_during_manufacturing_rate",
            float("nan"),
        )
    )
    scores = [
        local_search_metric_score(candidate["metrics"], score_weights)
        for candidate in evaluated
    ]
    feasible = [True]
    for candidate in evaluated[1:]:
        metrics = candidate["metrics"]
        completion = float(
            metrics.get("completion_service_level", float("nan"))
        )
        patients_lost = float(metrics.get("patients_lost", float("nan")))
        mfg_ineligibility = float(
            metrics.get(
                "patient_ineligibility_during_manufacturing_rate",
                float("nan"),
            )
        )
        feasible.append(
            bool(
                np.isfinite(completion)
                and np.isfinite(anchor_completion)
                and completion
                >= anchor_completion
                + float(
                    guardrails.get("min_completion_service_level_delta", 0.0)
                )
                and np.isfinite(patients_lost)
                and np.isfinite(anchor_lost)
                and patients_lost
                <= anchor_lost
                + float(guardrails.get("max_patients_lost_delta", 0.0))
                and np.isfinite(mfg_ineligibility)
                and np.isfinite(anchor_mfg_ineligibility)
                and mfg_ineligibility
                <= anchor_mfg_ineligibility
                + float(
                    guardrails.get(
                        "max_patient_ineligibility_during_manufacturing_rate_delta",
                        0.0,
                    )
                )
            )
        )
    feasible_indices = [index for index, allowed in enumerate(feasible) if allowed]
    best_index = min(feasible_indices, key=lambda index: scores[index])
    min_improvement = float(guardrails.get("min_score_improvement", 0.0))
    if scores[0] - scores[best_index] <= min_improvement:
        best_index = 0
    return best_index, scores, feasible


def run_state_probe(config: dict[str, Any]) -> list[dict[str, Any]]:
    env_config = load_env_config(config)
    env = build_env(env_config, seed=int(config["seed"]))
    anchor = make_anchor(config, env)
    rows: list[dict[str, Any]] = []
    for rollout in range(int(config["state_probe_rollouts"])):
        state = env.reset(seed=int(config["seed"]) + rollout)
        anchor.reset()
        done = False
        step = 0
        while not done and step < int(config["max_steps"]):
            specs = candidate_action_specs(
                state,
                env,
                anchor,
                epsilons=config["epsilons"],
                candidate_groups=config["candidate_groups"],
                candidate_signs=config.get("candidate_signs", (-1.0, 1.0)),
            )
            evaluated = evaluate_candidate_specs(
                specs,
                env,
                anchor,
                lookahead=int(config["lookahead"]),
                rollout_seeds=lookahead_rollout_seeds(
                    config,
                    rollout * int(config["max_steps"]) + step,
                ),
            )
            best_index, scores, feasible = select_clinical_candidate(
                evaluated,
                score_weights=config["score_weights"],
                guardrails=config["guardrails"],
            )
            baseline = evaluated[0]
            selected = evaluated[best_index]
            baseline_metrics = baseline["metrics"]
            selected_metrics = selected["metrics"]
            rows.append(
                {
                    "rollout": rollout,
                    "step": step,
                    "candidate_count": len(evaluated),
                    "feasible_candidate_count": int(sum(feasible)),
                    "selected_group": selected["group"],
                    "selected_epsilon": selected["epsilon"],
                    "selected_variant": selected["variant"],
                    "score_improvement": float(scores[0] - scores[best_index]),
                    "lookahead_cost_improvement": float(
                        baseline_metrics["total_cost"]
                        - selected_metrics["total_cost"]
                    ),
                    "completion_service_level_delta": float(
                        selected_metrics["completion_service_level"]
                        - baseline_metrics["completion_service_level"]
                    ),
                    "patients_lost_delta": float(
                        selected_metrics["patients_lost"]
                        - baseline_metrics["patients_lost"]
                    ),
                    "patient_ineligibility_during_manufacturing_rate_delta": float(
                        selected_metrics[
                            "patient_ineligibility_during_manufacturing_rate"
                        ]
                        - baseline_metrics[
                            "patient_ineligibility_during_manufacturing_rate"
                        ]
                    ),
                    "anchor_lookahead_cost": float(
                        baseline_metrics["total_cost"]
                    ),
                }
            )
            progress_interval = int(config.get("progress_interval_states", 50))
            if progress_interval > 0 and len(rows) % progress_interval == 0:
                improved = sum(
                    row["selected_group"] != "anchor"
                    for row in rows
                )
                print(
                    "headroom_state_probe "
                    f"states={len(rows)} "
                    f"improved={improved} "
                    f"rate={improved / len(rows):.3f}",
                    flush=True,
                )
            anchor_action = baseline["action"]
            state, _reward, done, _info = env.step(anchor_action)
            step += 1
    return rows


def summarize_state_probe(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    improved = [row for row in rows if row["selected_group"] != "anchor"]
    group_counts = Counter(str(row["selected_group"]) for row in improved)
    rollout_headroom = {}
    for rollout in sorted({int(row["rollout"]) for row in rows}):
        rollout_rows = [row for row in rows if int(row["rollout"]) == rollout]
        rollout_headroom[str(rollout)] = float(
            sum(max(float(row["lookahead_cost_improvement"]), 0.0) for row in rollout_rows)
        )
    return {
        "states": len(rows),
        "improved_states": len(improved),
        "opportunity_rate": len(improved) / len(rows),
        "mean_score_improvement_when_selected": float(
            np.mean([float(row["score_improvement"]) for row in improved])
        )
        if improved
        else 0.0,
        "mean_lookahead_cost_improvement_when_selected": float(
            np.mean(
                [float(row["lookahead_cost_improvement"]) for row in improved]
            )
        )
        if improved
        else 0.0,
        "selected_group_counts": dict(sorted(group_counts.items())),
        "approximate_positive_headroom_by_rollout": rollout_headroom,
        "lookahead": int(config["lookahead"]),
        "lookahead_replications": int(config.get("lookahead_replications", 1)),
    }


class ClinicalLookaheadTeacher:
    algorithm = "mdl2_headroom_teacher"

    def __init__(self, config: dict[str, Any], env) -> None:
        self.config = config
        self.anchor = make_anchor(config, env)
        self.demonstration_advantage_weight_scale = max(
            float(config.get("demonstration_advantage_weight_scale", 0.0)),
            0.0,
        )
        self.demonstration_advantage_weight_cap = max(
            float(config.get("demonstration_advantage_weight_cap", float("inf"))),
            1.0,
        )
        self.store_option_advantages = bool(
            config.get("store_option_advantages", False)
        )
        if self.store_option_advantages:
            explicit_options = config.get("explicit_options")
            self.option_specs = (
                make_explicit_residual_option_specs(explicit_options)
                if explicit_options is not None
                else make_residual_option_specs(
                    config["epsilons"],
                    config["candidate_groups"],
                    config.get("candidate_signs", (-1.0, 1.0)),
                )
            )
        else:
            self.option_specs = ()
        self.total_decisions = 0
        self.corrected_decisions = 0
        self.selected_groups: Counter[str] = Counter()
        self.demonstration_states: list[np.ndarray] = []
        self.demonstration_actions: list[np.ndarray] = []
        self.demonstration_weights: list[float] = []
        self.demonstration_improved: list[bool] = []
        self.demonstration_improvements: list[float] = []
        self.demonstration_option_advantages: list[np.ndarray] = []
        self.demonstration_option_feasible: list[np.ndarray] = []
        self.transition_states: list[np.ndarray] = []
        self.transition_actions: list[np.ndarray] = []
        self.transition_rewards: list[float] = []
        self.transition_next_states: list[np.ndarray] = []
        self.transition_dones: list[bool] = []

    def reset(self) -> None:
        self.anchor.reset()

    def select_action(self, state, explore: bool = False, env=None):
        del explore
        if env is None:
            raise ValueError("ClinicalLookaheadTeacher requires env=...")
        if self.option_specs:
            anchor_action = self.anchor.select_action(state, explore=False, env=env)
            option_actions = residual_option_actions_from_env(
                anchor_action,
                env,
                self.option_specs,
            )
            specs = [
                {
                    "group": option.group,
                    "epsilon": option.epsilon,
                    "sign": option.sign,
                    "variant": index,
                    "action": action,
                }
                for index, (option, action) in enumerate(
                    zip(self.option_specs, option_actions)
                )
            ]
        else:
            specs = candidate_action_specs(
                state,
                env,
                self.anchor,
                epsilons=self.config["epsilons"],
                candidate_groups=self.config["candidate_groups"],
                candidate_signs=self.config.get("candidate_signs", (-1.0, 1.0)),
            )
        evaluated = evaluate_candidate_specs(
            specs,
            env,
            self.anchor,
            lookahead=int(self.config["lookahead"]),
            rollout_seeds=lookahead_rollout_seeds(
                self.config,
                self.total_decisions,
            ),
        )
        best_index, scores, feasible = select_clinical_candidate(
            evaluated,
            score_weights=self.config["score_weights"],
            guardrails=self.config["guardrails"],
        )
        selected = evaluated[best_index]
        selected_action = np.asarray(selected["action"], dtype=np.float32)
        score_improvement = max(float(scores[0] - scores[best_index]), 0.0)
        improved = best_index != 0
        transition_env = copy.deepcopy(env)
        next_state, reward, done, _info = transition_env.step(selected_action)
        self.demonstration_states.append(np.asarray(state, dtype=np.float32))
        self.demonstration_actions.append(selected_action)
        if improved and self.demonstration_advantage_weight_scale > 0.0:
            demonstration_weight = 1.0 + (
                score_improvement / self.demonstration_advantage_weight_scale
            )
        else:
            demonstration_weight = max(score_improvement, 1.0)
        self.demonstration_weights.append(
            min(
                demonstration_weight,
                self.demonstration_advantage_weight_cap,
            )
        )
        self.demonstration_improved.append(improved)
        if self.option_specs:
            self.demonstration_option_advantages.append(
                np.asarray(
                    [float(scores[0] - score) for score in scores],
                    dtype=np.float32,
                )
            )
            self.demonstration_option_feasible.append(
                np.asarray(feasible, dtype=bool)
            )
        if improved:
            self.demonstration_improvements.append(score_improvement)
        self.transition_states.append(np.asarray(state, dtype=np.float32))
        self.transition_actions.append(selected_action)
        self.transition_rewards.append(float(reward))
        self.transition_next_states.append(np.asarray(next_state, dtype=np.float32))
        self.transition_dones.append(bool(done))
        self.total_decisions += 1
        self.selected_groups[str(selected["group"])] += 1
        if improved:
            self.corrected_decisions += 1
        progress_interval = int(
            self.config.get("progress_interval_decisions", 100)
        )
        if (
            progress_interval > 0
            and self.total_decisions % progress_interval == 0
        ):
            print(
                "headroom_teacher "
                f"decisions={self.total_decisions} "
                f"corrections={self.corrected_decisions} "
                f"rate={self.corrected_decisions / self.total_decisions:.3f}",
                flush=True,
            )
        return selected_action

    def demonstrations(self) -> dict[str, Any]:
        weights = np.asarray(self.demonstration_weights, dtype=np.float32)
        improved = np.asarray(self.demonstration_improved, dtype=bool)
        total_weight = float(weights.sum())
        result = {
            "states": np.asarray(self.demonstration_states, dtype=np.float32),
            "actions": np.asarray(self.demonstration_actions, dtype=np.float32),
            "weights": weights,
            "improved_mask": improved,
            "transition_states": np.asarray(self.transition_states, dtype=np.float32),
            "transition_actions": np.asarray(self.transition_actions, dtype=np.float32),
            "transition_rewards": np.asarray(self.transition_rewards, dtype=np.float32),
            "transition_next_states": np.asarray(
                self.transition_next_states,
                dtype=np.float32,
            ),
            "transition_dones": np.asarray(self.transition_dones, dtype=bool),
            "improved_steps": int(improved.sum()),
            "anchor_keep_steps": int((~improved).sum()),
            "service_rejected_steps": 0,
            "mean_step_improvement": (
                float(np.mean(self.demonstration_improvements))
                if self.demonstration_improvements
                else 0.0
            ),
            "improved_weight_fraction": (
                float(weights[improved].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
        }
        if self.option_specs:
            result.update(
                {
                    "option_advantages": np.asarray(
                        self.demonstration_option_advantages,
                        dtype=np.float32,
                    ),
                    "option_feasible": np.asarray(
                        self.demonstration_option_feasible,
                        dtype=bool,
                    ),
                    "option_groups": np.asarray(
                        [option.group for option in self.option_specs],
                        dtype="U32",
                    ),
                    "option_epsilons": np.asarray(
                        [option.epsilon for option in self.option_specs],
                        dtype=np.float32,
                    ),
                    "option_signs": np.asarray(
                        [option.sign for option in self.option_specs],
                        dtype=np.float32,
                    ),
                }
            )
        return result


def run_online_teacher(config: dict[str, Any]) -> dict[str, Any]:
    env_config = load_env_config(config)
    replication_start = int(config.get("teacher_replication_start", 0))
    seed = int(config["seed"]) + 10000 + replication_start
    replications = int(config["teacher_replications"])
    max_steps = int(config["max_steps"])

    anchor_env = build_env(env_config, seed=seed)
    anchor = make_anchor(config, anchor_env)
    anchor_rows = evaluate_agent(
        anchor,
        anchor_env,
        algorithm=str(config.get("anchor_policy", "mdl2")),
        seed=seed,
        replications=replications,
        max_steps=max_steps,
    )

    teacher_env = build_env(env_config, seed=seed)
    teacher = ClinicalLookaheadTeacher(config, teacher_env)
    teacher_rows = evaluate_agent(
        teacher,
        teacher_env,
        algorithm=teacher.algorithm,
        seed=seed,
        replications=replications,
        max_steps=max_steps,
    )
    for rows in (anchor_rows, teacher_rows):
        for row in rows:
            row["training_seed"] = 0
            row["evaluation_seed"] = seed

    output_root = Path(config["output_root"])
    write_rows(anchor_rows, output_root / "anchor.csv")
    write_rows(teacher_rows, output_root / "teacher.csv")
    demonstration_path = config.get("demonstration_path")
    if demonstration_path:
        save_local_search_demonstrations(demonstration_path, teacher.demonstrations())
    paired = {}
    for metric in (
        "total_cost",
        "completion_service_level",
        "patients_lost",
        "patient_ineligibility_during_manufacturing_rate",
    ):
        paired[metric] = paired_two_level_summary(
            teacher_rows,
            anchor_rows,
            metric=metric,
            resamples=20000,
            seed=seed,
        )
    return {
        "anchor_summary": summarize_rows(anchor_rows),
        "teacher_summary": summarize_rows(teacher_rows),
        "paired": paired,
        "teacher_total_decisions": teacher.total_decisions,
        "teacher_corrected_decisions": teacher.corrected_decisions,
        "teacher_correction_rate": (
            teacher.corrected_decisions / teacher.total_decisions
            if teacher.total_decisions
            else 0.0
        ),
        "teacher_selected_group_counts": dict(
            sorted(teacher.selected_groups.items())
        ),
        "teacher_demonstration_path": (
            "" if demonstration_path is None else str(demonstration_path)
        ),
        "teacher_demonstration_samples": len(teacher.demonstration_states),
        "teacher_replication_start": replication_start,
        "lookahead_decision_offset": int(
            config.get("lookahead_decision_offset", 0)
        ),
    }


def headroom_decision(
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    materiality = config.get("materiality", {})
    minimum_gap = float(
        materiality.get("minimum_episode_cost_improvement_pct", 1.0)
    )
    minimum_opportunity = float(
        materiality.get("minimum_opportunity_rate", 0.05)
    )
    state_probe = result.get("state_probe", {})
    teacher = result.get("online_teacher", {})
    cost_comparison = teacher.get("paired", {}).get("total_cost", {})
    mean_gap = float(cost_comparison.get("mean_gap_pct", 0.0))
    ci_high = float(cost_comparison.get("ci_high", float("inf")))
    if "opportunity_rate" in state_probe:
        opportunity_rate = float(state_probe["opportunity_rate"])
        opportunity_rate_source = "state_probe"
    else:
        opportunity_rate = float(teacher.get("teacher_correction_rate", 0.0))
        opportunity_rate_source = "online_teacher"
    require_clinical_noninferiority = bool(
        materiality.get("require_episode_clinical_noninferiority", False)
    )
    completion_difference = float(
        teacher.get("paired", {})
        .get("completion_service_level", {})
        .get("mean_difference", float("nan"))
    )
    patients_lost_difference = float(
        teacher.get("paired", {})
        .get("patients_lost", {})
        .get("mean_difference", float("nan"))
    )
    manufacturing_ineligibility_difference = float(
        teacher.get("paired", {})
        .get("patient_ineligibility_during_manufacturing_rate", {})
        .get("mean_difference", float("nan"))
    )
    clinical_noninferiority = bool(
        not require_clinical_noninferiority
        or (
            np.isfinite(completion_difference)
            and completion_difference
            >= float(
                materiality.get(
                    "minimum_completion_service_level_difference",
                    0.0,
                )
            )
            and np.isfinite(patients_lost_difference)
            and patients_lost_difference
            <= float(materiality.get("maximum_patients_lost_difference", 0.0))
            and np.isfinite(manufacturing_ineligibility_difference)
            and manufacturing_ineligibility_difference
            <= float(
                materiality.get(
                    "maximum_patient_ineligibility_during_manufacturing_rate_difference",
                    0.0,
                )
            )
        )
    )
    advance = bool(
        mean_gap <= -minimum_gap
        and ci_high < 0.0
        and opportunity_rate >= minimum_opportunity
        and clinical_noninferiority
    )
    return {
        "advance_to_network_residual_training": advance,
        "teacher_cost_gap_pct": mean_gap,
        "teacher_cost_ci_high": ci_high,
        "state_opportunity_rate": opportunity_rate,
        "opportunity_rate_source": opportunity_rate_source,
        "required_cost_improvement_pct": minimum_gap,
        "required_opportunity_rate": minimum_opportunity,
        "require_episode_clinical_noninferiority": require_clinical_noninferiority,
        "episode_clinical_noninferiority": clinical_noninferiority,
        "completion_service_level_difference": completion_difference,
        "patients_lost_difference": patients_lost_difference,
        "patient_ineligibility_during_manufacturing_rate_difference": (
            manufacturing_ineligibility_difference
        ),
        "reason": (
            "material clinically guarded headroom detected"
            if advance
            else "headroom gate not yet cleared"
        ),
    }


if __name__ == "__main__":
    main()
