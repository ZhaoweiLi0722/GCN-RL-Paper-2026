"""Validation-selected, CRN holdout evaluation for multi-scenario residual RL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.evaluate_formal import evaluate_agent
from evaluation.probe_continuous_residual_checkpoints import (
    add_training_seed,
    paired_candidate_summary,
)
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows


PAIRING_KEYS = (
    "training_seed",
    "scenario",
    "evaluation_seed",
    "replication",
)


class ResidualUsageMonitor:
    """Count attempted non-zero residual decisions without recomputing actions."""

    def __init__(self, agent: Any, num_facilities: int) -> None:
        self.agent = agent
        self.num_facilities = int(num_facilities)
        self.total_decisions = 0
        self.corrected_decisions = 0
        self.group_corrected_decisions = {
            "specimen_transfer": 0,
            "reagent_transfer": 0,
            "capacity_transfer": 0,
            "replenishment": 0,
        }

    def reset(self) -> None:
        self.agent.reset()

    def select_action(self, state, explore=False, env=None):
        action = self.agent.select_action(
            state,
            explore=explore,
            env=env,
        )
        residual = np.asarray(
            getattr(
                self.agent,
                "last_residual_action",
                np.zeros_like(action),
            ),
            dtype=np.float32,
        )
        active = np.abs(residual) > 1e-6
        self.total_decisions += 1
        self.corrected_decisions += int(np.any(active))
        for index, group in enumerate(self.group_corrected_decisions):
            start = index * self.num_facilities
            stop = start + self.num_facilities
            self.group_corrected_decisions[group] += int(
                np.any(active[start:stop])
            )
        return action

    def summary(self) -> dict[str, Any]:
        denominator = max(self.total_decisions, 1)
        return {
            "total_decisions": self.total_decisions,
            "corrected_decisions": self.corrected_decisions,
            "correction_rate": self.corrected_decisions / denominator,
            "group_correction_rates": {
                group: count / denominator
                for group, count in self.group_corrected_decisions.items()
            },
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = evaluate_multiscenario_agents(config)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_multiscenario_agents(
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    plan = load_benchmark_plan(
        evaluation_config.get(
            "plan",
            "experiments/configs/residual_policy_benchmark.json",
        )
    )
    training_manifest_path = Path(
        evaluation_config["training_manifest"]
    )
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    scenario_names = tuple(
        str(value)
        for value in evaluation_config.get(
            "scenarios",
            training_manifest["scenarios"],
        )
    )
    selected_scenarios = select_scenarios(plan, scenario_names)
    scenarios_by_name = {
        str(scenario["name"]): scenario
        for scenario in selected_scenarios
    }
    scenarios = [
        scenarios_by_name[name]
        for name in scenario_names
    ]
    candidates = tuple(
        normalized_deployment_candidate(value)
        for value in evaluation_config.get(
            "deployment_candidates",
            (
                {"scale": 0.0, "group_thresholds": [1.0, 1.0, 1.0]},
                {"scale": 0.25, "group_thresholds": [0.7, 0.8, 1.0]},
                {"scale": 0.5, "group_thresholds": [0.7, 0.8, 1.0]},
            ),
        )
    )
    if not any(np.isclose(candidate["scale"], 0.0) for candidate in candidates):
        raise ValueError("Deployment candidates must include scale=0 fallback")

    validation_replications = int(
        evaluation_config.get("validation_replications", 10)
    )
    holdout_replications = int(
        evaluation_config.get("holdout_replications", 30)
    )
    validation_seed = int(
        evaluation_config.get("validation_seed", 8_100_000)
    )
    holdout_seed = int(
        evaluation_config.get("holdout_seed", 8_200_000)
    )
    max_steps = int(evaluation_config.get("max_steps", 52))
    strict_scenario_guardrail = bool(
        evaluation_config.get(
            "strict_scenario_clinical_noninferiority",
            False,
        )
    )
    output_root = Path(
        evaluation_config.get(
            "output_root",
            "results/multiscenario_network_afr_evaluation",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    requested_algorithms = {
        str(value)
        for value in evaluation_config.get("algorithms", ())
    }
    requested_training_seeds = {
        int(value)
        for value in evaluation_config.get("training_seeds", ())
    }
    selected_runs = [
        run
        for run in training_manifest["runs"]
        if (
            not requested_algorithms
            or str(run["algorithm"]) in requested_algorithms
        )
        and (
            not requested_training_seeds
            or int(run["seed"]) in requested_training_seeds
        )
    ]
    if not selected_runs:
        raise ValueError(
            "Evaluation algorithm/seed filters selected no training runs"
        )

    run_results = []
    holdout_rows_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    anchor_rows_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    requested_checkpoint_variants = tuple(
        str(value)
        for value in evaluation_config.get(
            "checkpoint_variants",
            ("final",),
        )
    )
    for run in selected_runs:
        algorithm = str(run["algorithm"])
        training_seed = int(run["seed"])
        checkpoint_variants = resolve_checkpoint_variants(
            run,
            requested_checkpoint_variants,
        )
        config_snapshot = load_config(run["config"])
        run_root = output_root / algorithm / f"seed{training_seed}"
        run_root.mkdir(parents=True, exist_ok=True)

        validation_results = []
        validation_anchor_rows_by_scenario = None
        for checkpoint_variant, checkpoint in checkpoint_variants.items():
            for candidate in candidates:
                result, candidate_rows, anchor_rows = evaluate_deployment_candidate(
                    plan=plan,
                    scenarios=scenarios,
                    scenario_names=scenario_names,
                    algorithm=algorithm,
                    training_seed=training_seed,
                    checkpoint=checkpoint,
                    config_snapshot=config_snapshot,
                    candidate=candidate,
                    evaluation_seed=validation_seed,
                    replications=validation_replications,
                    max_steps=max_steps,
                    precomputed_anchor_rows=(
                        validation_anchor_rows_by_scenario
                    ),
                )
                if validation_anchor_rows_by_scenario is None:
                    validation_anchor_rows_by_scenario = rows_by_scenario(
                        anchor_rows
                )
                result["checkpoint_variant"] = checkpoint_variant
                validation_results.append(result)
                aggregate_summary = result["aggregate"]
                print(
                    "multiscenario_validation "
                    f"algorithm={algorithm} seed={training_seed} "
                    f"checkpoint={checkpoint_variant} "
                    f"scale={float(candidate['scale']):g} "
                    f"cost_gap_pct="
                    f"{float(aggregate_summary['cost_gap_pct']):.6f} "
                    f"clinically_noninferior="
                    f"{bool(aggregate_summary['clinically_noninferior'])}",
                    flush=True,
                )
                label = deployment_candidate_label(candidate)
                write_rows(
                    candidate_rows,
                    run_root
                    / f"validation_{checkpoint_variant}_{label}_rows.csv",
                )
                if not (run_root / "validation_anchor_rows.csv").exists():
                    write_rows(
                        anchor_rows,
                        run_root / "validation_anchor_rows.csv",
                    )
        selected = select_deployment_candidate(
            validation_results,
            strict_scenario_guardrail=strict_scenario_guardrail,
        )
        selected_candidate = normalized_deployment_candidate(
            selected["candidate"]
        )
        selected_checkpoint_variant = str(
            selected["checkpoint_variant"]
        )
        selected_checkpoint = checkpoint_variants[
            selected_checkpoint_variant
        ]
        holdout_result, holdout_rows, holdout_anchor_rows = (
            evaluate_deployment_candidate(
                plan=plan,
                scenarios=scenarios,
                scenario_names=scenario_names,
                algorithm=algorithm,
                training_seed=training_seed,
                checkpoint=selected_checkpoint,
                config_snapshot=config_snapshot,
                candidate=selected_candidate,
                evaluation_seed=holdout_seed,
                replications=holdout_replications,
                max_steps=max_steps,
            )
        )
        print(
            "multiscenario_holdout "
            f"algorithm={algorithm} seed={training_seed} "
            f"checkpoint={selected_checkpoint_variant} "
            f"scale={float(selected_candidate['scale']):g} "
            f"cost_gap_pct="
            f"{float(holdout_result['aggregate']['cost_gap_pct']):.6f}",
            flush=True,
        )
        write_rows(holdout_rows, run_root / "holdout_rows.csv")
        write_rows(
            holdout_anchor_rows,
            run_root / "holdout_anchor_rows.csv",
        )
        key = (algorithm, training_seed)
        holdout_rows_by_run[key] = holdout_rows
        anchor_rows_by_run[key] = holdout_anchor_rows
        run_result = {
            "algorithm": algorithm,
            "training_seed": training_seed,
            "checkpoint": str(selected_checkpoint),
            "checkpoint_variants": {
                name: str(path)
                for name, path in checkpoint_variants.items()
            },
            "validation": validation_results,
            "selected_deployment": selected,
            "holdout": holdout_result,
        }
        (run_root / "summary.json").write_text(
            json.dumps(run_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_results.append(run_result)

    aggregate = aggregate_holdout_results(
        holdout_rows_by_run,
        anchor_rows_by_run,
        bootstrap_seed=holdout_seed + 500_000,
    )
    payload = {
        "training_manifest": str(training_manifest_path),
        "scenarios": list(scenario_names),
        "validation_replications": validation_replications,
        "validation_seed": validation_seed,
        "holdout_replications": holdout_replications,
        "holdout_seed": holdout_seed,
        "strict_scenario_clinical_noninferiority": strict_scenario_guardrail,
        "algorithms": sorted(requested_algorithms),
        "training_seeds": sorted(requested_training_seeds),
        "runs": run_results,
        "aggregate": aggregate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def evaluate_deployment_candidate(
    *,
    plan: dict[str, Any],
    scenarios: list[dict[str, Any]],
    scenario_names: tuple[str, ...],
    algorithm: str,
    training_seed: int,
    checkpoint: Path,
    config_snapshot: dict[str, Any],
    candidate: dict[str, Any],
    evaluation_seed: int,
    replications: int,
    max_steps: int,
    precomputed_anchor_rows: dict[
        str,
        list[dict[str, Any]],
    ]
    | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    per_scenario: dict[str, Any] = {}
    for scenario_index, (scenario_name, scenario) in enumerate(
        zip(scenario_names, scenarios)
    ):
        seed = int(evaluation_seed) + scenario_index * 100_000
        config = dict(config_snapshot)
        config["env"] = make_scenario_env_config(
            plan,
            algorithm,
            scenario,
        )
        if precomputed_anchor_rows is None:
            env = build_env(config, seed=seed)
            anchor_name = str(
                config.get("residual_action", {}).get(
                    "base_policy",
                    "mdl2",
                )
            )
            anchor = get_heuristic_class(anchor_name)(
                env.observation_size,
                env.action_size,
                dict(
                    config.get("residual_action", {}).get(
                        "base_policy_config",
                        {},
                    )
                ),
            )
            scenario_anchor_rows = evaluate_agent(
                anchor,
                env,
                algorithm=anchor_name,
                seed=seed,
                replications=replications,
                max_steps=max_steps,
            )
            add_training_seed(scenario_anchor_rows, training_seed)
        else:
            if scenario_name not in precomputed_anchor_rows:
                raise ValueError(
                    f"Precomputed anchor rows are missing {scenario_name}"
                )
            scenario_anchor_rows = [
                dict(row)
                for row in precomputed_anchor_rows[scenario_name]
            ]

        learned_env = build_env(config, seed=seed)
        agent = get_agent_class(algorithm)(
            learned_env.observation_size,
            learned_env.action_size,
            config,
        )
        agent.load_actor(checkpoint)
        configure_deployment(agent, candidate)
        monitored_agent = ResidualUsageMonitor(
            agent,
            learned_env.config.num_facilities,
        )
        scenario_candidate_rows = evaluate_agent(
            monitored_agent,
            learned_env,
            algorithm=algorithm,
            seed=seed,
            replications=replications,
            max_steps=max_steps,
        )
        add_training_seed(scenario_candidate_rows, training_seed)
        per_scenario[scenario_name] = paired_candidate_summary(
            scenario_candidate_rows,
            scenario_anchor_rows,
        )
        per_scenario[scenario_name]["residual_usage"] = (
            monitored_agent.summary()
        )
        candidate_rows.extend(scenario_candidate_rows)
        anchor_rows.extend(scenario_anchor_rows)

    aggregate = paired_candidate_summary(candidate_rows, anchor_rows)
    aggregate["residual_usage"] = aggregate_residual_usage(
        per_scenario
    )
    return (
        {
            "candidate": candidate,
            "aggregate": aggregate,
            "per_scenario": per_scenario,
            "all_scenarios_clinically_noninferior": all(
                bool(summary["clinically_noninferior"])
                for summary in per_scenario.values()
            ),
        },
        candidate_rows,
        anchor_rows,
    )


def rows_by_scenario(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        grouped.setdefault(scenario, []).append(dict(row))
    return grouped


def configure_deployment(
    agent: Any,
    candidate: dict[str, Any],
) -> None:
    scale = float(candidate["scale"])
    agent.residual_scale_vector = (
        np.asarray(agent.residual_scale_vector, dtype=np.float32)
        * scale
    )
    thresholds = tuple(
        float(value) for value in candidate["group_thresholds"]
    )
    if getattr(agent, "correction_gate_groups", ()):
        if len(thresholds) != len(agent.correction_gate_groups):
            raise ValueError(
                "Deployment group thresholds must match correction groups"
            )
        agent.correction_gate_group_thresholds = thresholds
    elif thresholds:
        agent.correction_gate_threshold = thresholds[0]


def aggregate_residual_usage(
    per_scenario: dict[str, Any],
) -> dict[str, Any]:
    summaries = [
        value["residual_usage"]
        for value in per_scenario.values()
    ]
    total = int(sum(value["total_decisions"] for value in summaries))
    corrected = int(
        sum(value["corrected_decisions"] for value in summaries)
    )
    groups = tuple(
        summaries[0]["group_correction_rates"]
        if summaries
        else ()
    )
    group_corrected = {
        group: int(
            round(
                sum(
                    value["group_correction_rates"][group]
                    * value["total_decisions"]
                    for value in summaries
                )
            )
        )
        for group in groups
    }
    denominator = max(total, 1)
    return {
        "total_decisions": total,
        "corrected_decisions": corrected,
        "correction_rate": corrected / denominator,
        "group_correction_rates": {
            group: count / denominator
            for group, count in group_corrected.items()
        },
    }


def normalized_deployment_candidate(
    value: dict[str, Any],
) -> dict[str, Any]:
    candidate = dict(value)
    scale = float(candidate.get("scale", 0.0))
    if scale < 0.0:
        raise ValueError("Deployment residual scale cannot be negative")
    thresholds = tuple(
        float(item)
        for item in candidate.get(
            "group_thresholds",
            (0.8, 0.8, 0.8),
        )
    )
    if any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("Deployment gate thresholds must lie in [0, 1]")
    return {
        "scale": scale,
        "group_thresholds": list(thresholds),
    }


def deployment_candidate_label(candidate: dict[str, Any]) -> str:
    thresholds = "-".join(
        f"{float(value):g}"
        for value in candidate["group_thresholds"]
    )
    return f"scale{float(candidate['scale']):g}_gate{thresholds}"


def select_deployment_candidate(
    results: list[dict[str, Any]],
    *,
    strict_scenario_guardrail: bool,
) -> dict[str, Any]:
    eligible = [
        result
        for result in results
        if bool(result["aggregate"]["clinically_noninferior"])
        and (
            not strict_scenario_guardrail
            or bool(result["all_scenarios_clinically_noninferior"])
        )
    ]
    if not eligible:
        eligible = [
            result
            for result in results
            if np.isclose(float(result["candidate"]["scale"]), 0.0)
        ]
    selected = min(
        eligible,
        key=lambda result: float(
            result["aggregate"]["cost_difference_mean"]
        ),
    )
    return {
        "candidate": selected["candidate"],
        "checkpoint_variant": str(
            selected.get("checkpoint_variant", "final")
        ),
        "selection_source": "validation_only",
        "validation_aggregate": selected["aggregate"],
        "all_scenarios_clinically_noninferior": bool(
            selected["all_scenarios_clinically_noninferior"]
        ),
    }


def resolve_checkpoint_variants(
    run: dict[str, Any],
    requested_variants: tuple[str, ...],
) -> dict[str, Path]:
    if not requested_variants:
        raise ValueError("At least one checkpoint variant is required")
    supported = {
        "final": "checkpoint",
        "pretrain": "pretrain_checkpoint",
    }
    unknown = tuple(
        value for value in requested_variants
        if value not in supported
    )
    if unknown:
        raise ValueError(
            "Unsupported checkpoint variants: " + ", ".join(unknown)
        )

    resolved: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for variant in requested_variants:
        manifest_key = supported[variant]
        raw_path = run.get(manifest_key)
        if not raw_path:
            raise ValueError(
                f"Training manifest is missing {manifest_key!r}"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{variant} checkpoint does not exist: {path}"
            )
        canonical_path = path.resolve()
        if canonical_path in seen_paths:
            continue
        resolved[variant] = path
        seen_paths.add(canonical_path)
    if not resolved:
        raise ValueError("Checkpoint variants resolved to an empty set")
    return resolved


def aggregate_holdout_results(
    holdout_rows_by_run: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ],
    anchor_rows_by_run: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    algorithms = sorted(
        {algorithm for algorithm, _seed in holdout_rows_by_run}
    )
    result: dict[str, Any] = {}
    for algorithm in algorithms:
        candidate_rows = [
            row
            for (name, _seed), rows in holdout_rows_by_run.items()
            if name == algorithm
            for row in rows
        ]
        anchor_rows = [
            row
            for (name, _seed), rows in anchor_rows_by_run.items()
            if name == algorithm
            for row in rows
        ]
        result[f"{algorithm}_vs_anchor"] = metric_bootstrap_bundle(
            candidate_rows,
            anchor_rows,
            seed=bootstrap_seed,
        )

    graph = "gcn_residual_mdl2_network_ddpg_afd"
    flat = "flat_residual_mdl2_network_ddpg_afd"
    graph_seeds = {
        seed for algorithm, seed in holdout_rows_by_run
        if algorithm == graph
    }
    flat_seeds = {
        seed for algorithm, seed in holdout_rows_by_run
        if algorithm == flat
    }
    common_seeds = sorted(graph_seeds & flat_seeds)
    if common_seeds:
        graph_rows = [
            row
            for seed in common_seeds
            for row in holdout_rows_by_run[(graph, seed)]
        ]
        flat_rows = [
            row
            for seed in common_seeds
            for row in holdout_rows_by_run[(flat, seed)]
        ]
        result["graph_vs_flat"] = metric_bootstrap_bundle(
            graph_rows,
            flat_rows,
            seed=bootstrap_seed + 1,
        )
    return result


def metric_bootstrap_bundle(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: paired_two_level_summary(
            candidate_rows,
            baseline_rows,
            metric=metric,
            pairing_keys=PAIRING_KEYS,
            resamples=20_000,
            seed=seed + index,
        )
        for index, metric in enumerate(
            (
                "total_cost",
                "completion_service_level",
                "patients_lost",
                "patient_ineligibility_during_manufacturing_rate",
            )
        )
    }


if __name__ == "__main__":
    main()
