"""Validation-selected, CRN holdout evaluation for multi-scenario residual RL."""

from __future__ import annotations

import argparse
import hashlib
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
CLINICAL_METRICS = {
    "completion_service_level": {
        "direction": "higher",
        "delta_key": "completion_service_level_delta",
    },
    "patients_lost": {
        "direction": "lower",
        "delta_key": "patients_lost_delta",
    },
    "patient_ineligibility_during_manufacturing_rate": {
        "direction": "lower",
        "delta_key": "manufacturing_ineligibility_rate_delta",
    },
}


class ResidualUsageMonitor:
    """Count attempted non-zero residual decisions without recomputing actions."""

    def __init__(self, agent: Any, num_facilities: int) -> None:
        self.agent = agent
        self.num_facilities = int(num_facilities)
        self.total_decisions = 0
        self.corrected_decisions = 0
        self.temporally_suppressed_decisions = 0
        self.raw_residual_l1 = 0.0
        self.applied_residual_l1 = 0.0
        self.temporally_suppressed_l1 = 0.0
        self.group_corrected_decisions = {
            "specimen_transfer": 0,
            "reagent_transfer": 0,
            "capacity_transfer": 0,
            "replenishment": 0,
        }
        self.capacity_action_effect = {
            "anchor_l1": 0.0,
            "final_l1": 0.0,
            "delta_l1": 0.0,
            "amplification_l1": 0.0,
            "damping_l1": 0.0,
            "reversal_l1": 0.0,
            "new_flow_l1": 0.0,
        }

    def reset(self) -> None:
        self.agent.reset()

    def select_action(self, state, explore=False, env=None):
        action = self.agent.select_action(
            state,
            explore=explore,
            env=env,
        )
        if hasattr(self.agent, "last_residual_action"):
            residual = np.asarray(
                self.agent.last_residual_action,
                dtype=np.float32,
            )
        elif getattr(self.agent, "_pending_anchor_action", None) is not None:
            residual = (
                np.asarray(action, dtype=np.float32)
                - np.asarray(
                    self.agent._pending_anchor_action,
                    dtype=np.float32,
                )
            )
        else:
            residual = np.zeros_like(action, dtype=np.float32)
        active = np.abs(residual) > 1e-6
        guard_info = dict(
            getattr(
                self.agent,
                "last_residual_guard_info",
                {},
            )
        )
        raw_l1 = float(
            guard_info.get(
                "raw_l1",
                np.abs(residual).sum(),
            )
        )
        applied_l1 = float(
            guard_info.get(
                "applied_l1",
                np.abs(residual).sum(),
            )
        )
        suppressed_l1 = max(raw_l1 - applied_l1, 0.0)
        self.total_decisions += 1
        self.corrected_decisions += int(np.any(active))
        self.temporally_suppressed_decisions += int(
            suppressed_l1 > 1e-8
        )
        self.raw_residual_l1 += raw_l1
        self.applied_residual_l1 += applied_l1
        self.temporally_suppressed_l1 += suppressed_l1
        for index, group in enumerate(self.group_corrected_decisions):
            start = index * self.num_facilities
            stop = start + self.num_facilities
            self.group_corrected_decisions[group] += int(
                np.any(active[start:stop])
            )
        self._record_capacity_action_effect(state, action)
        return action

    def _record_capacity_action_effect(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> None:
        anchor = None
        base_action = getattr(
            self.agent,
            "_base_action_from_state_np",
            None,
        )
        if callable(base_action):
            anchor = np.asarray(
                base_action(state),
                dtype=np.float32,
            )
        elif getattr(self.agent, "_pending_anchor_action", None) is not None:
            anchor = np.asarray(
                self.agent._pending_anchor_action,
                dtype=np.float32,
            )
        final = np.asarray(action, dtype=np.float32)
        if anchor is None or anchor.shape != final.shape:
            return

        start = 2 * self.num_facilities
        stop = 3 * self.num_facilities
        anchor_capacity = anchor[start:stop]
        final_capacity = final[start:stop]
        anchor_abs = np.abs(anchor_capacity)
        final_abs = np.abs(final_capacity)
        epsilon = 1e-6
        anchor_active = anchor_abs > epsilon
        final_active = final_abs > epsilon
        same_direction = (
            anchor_active
            & final_active
            & (anchor_capacity * final_capacity > 0.0)
        )
        reversed_direction = (
            anchor_active
            & final_active
            & (anchor_capacity * final_capacity < 0.0)
        )
        new_flow = (~anchor_active) & final_active

        self.capacity_action_effect["anchor_l1"] += float(
            anchor_abs.sum()
        )
        self.capacity_action_effect["final_l1"] += float(
            final_abs.sum()
        )
        self.capacity_action_effect["delta_l1"] += float(
            np.abs(final_capacity - anchor_capacity).sum()
        )
        self.capacity_action_effect["amplification_l1"] += float(
            np.maximum(final_abs - anchor_abs, 0.0)[
                same_direction
            ].sum()
        )
        self.capacity_action_effect["damping_l1"] += float(
            np.maximum(anchor_abs - final_abs, 0.0)[
                same_direction
            ].sum()
            + anchor_abs[reversed_direction].sum()
            + anchor_abs[anchor_active & ~final_active].sum()
        )
        self.capacity_action_effect["reversal_l1"] += float(
            final_abs[reversed_direction].sum()
        )
        self.capacity_action_effect["new_flow_l1"] += float(
            final_abs[new_flow].sum()
        )

    def summary(self) -> dict[str, Any]:
        denominator = max(self.total_decisions, 1)
        return {
            "total_decisions": self.total_decisions,
            "corrected_decisions": self.corrected_decisions,
            "correction_rate": self.corrected_decisions / denominator,
            "temporally_suppressed_decisions": (
                self.temporally_suppressed_decisions
            ),
            "temporal_suppression_rate": (
                self.temporally_suppressed_decisions / denominator
            ),
            "raw_residual_l1": self.raw_residual_l1,
            "applied_residual_l1": self.applied_residual_l1,
            "temporally_suppressed_l1": self.temporally_suppressed_l1,
            "group_correction_rates": {
                group: count / denominator
                for group, count in self.group_corrected_decisions.items()
            },
            "capacity_action_effect": dict(
                self.capacity_action_effect
            ),
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
    fixed_candidate = (
        None
        if evaluation_config.get("fixed_deployment_candidate") is None
        else normalized_deployment_candidate(
            dict(evaluation_config["fixed_deployment_candidate"])
        )
    )
    fixed_checkpoint_variant = str(
        evaluation_config.get(
            "fixed_checkpoint_variant",
            "final",
        )
    )
    if fixed_candidate is not None and fixed_candidate not in candidates:
        raise ValueError(
            "fixed_deployment_candidate must also appear in "
            "deployment_candidates"
        )

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
    selection_guardrails = dict(
        evaluation_config.get("selection_guardrails", {})
    )
    clinical_noninferiority = normalized_clinical_noninferiority(
        evaluation_config.get("clinical_noninferiority", {})
    )
    evaluation_config_overrides = dict(
        evaluation_config.get("config_overrides", {})
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
        config_snapshot = deep_update_dict(
            load_config(resolve_manifest_artifact_path(run["config"])),
            evaluation_config_overrides,
        )
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
                    clinical_noninferiority=clinical_noninferiority,
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
        selected = (
            select_fixed_deployment_candidate(
                validation_results,
                candidate=fixed_candidate,
                checkpoint_variant=fixed_checkpoint_variant,
            )
            if fixed_candidate is not None
            else select_deployment_candidate(
                validation_results,
                strict_scenario_guardrail=strict_scenario_guardrail,
                selection_guardrails=selection_guardrails,
            )
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
                clinical_noninferiority=clinical_noninferiority,
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
        "selection_guardrails": selection_guardrails,
        "clinical_noninferiority": clinical_noninferiority,
        "fixed_deployment_candidate": fixed_candidate,
        "fixed_checkpoint_variant": (
            fixed_checkpoint_variant
            if fixed_candidate is not None
            else None
        ),
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


def apply_multiscenario_env_contract(
    scenario_env: dict[str, Any],
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Restore shared observation settings when rebuilding an eval scenario."""

    shared = (
        config_snapshot.get("multi_scenario_training", {})
        .get("env_overrides", {})
    )
    if not isinstance(shared, dict):
        raise TypeError("multi_scenario_training.env_overrides must be a mapping")
    return deep_update_dict(scenario_env, shared)


def deep_update_dict(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    clinical_noninferiority: dict[str, Any] | None = None,
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
        config["env"] = apply_multiscenario_env_contract(
            make_scenario_env_config(
                plan,
                algorithm,
                scenario,
            ),
            config_snapshot,
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
        scenario_summary = paired_candidate_summary(
            scenario_candidate_rows,
            scenario_anchor_rows,
        )
        apply_clinical_noninferiority(
            scenario_summary,
            scenario_candidate_rows,
            scenario_anchor_rows,
            clinical_noninferiority,
        )
        per_scenario[scenario_name] = scenario_summary
        per_scenario[scenario_name]["residual_usage"] = (
            monitored_agent.summary()
        )
        candidate_rows.extend(scenario_candidate_rows)
        anchor_rows.extend(scenario_anchor_rows)

    aggregate = paired_candidate_summary(candidate_rows, anchor_rows)
    apply_clinical_noninferiority(
        aggregate,
        candidate_rows,
        anchor_rows,
        clinical_noninferiority,
    )
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


def normalized_clinical_noninferiority(
    value: Any,
) -> dict[str, Any]:
    raw = dict(value or {})
    mode = str(raw.get("mode", "point"))
    if mode not in ("point", "paired_ci"):
        raise ValueError(
            "clinical_noninferiority.mode must be 'point' or 'paired_ci'"
        )
    margins = {
        metric: float(dict(raw.get("margins", {})).get(metric, 0.0))
        for metric in CLINICAL_METRICS
    }
    if any(margin < 0.0 for margin in margins.values()):
        raise ValueError(
            "Clinical noninferiority margins cannot be negative"
        )
    z_value = float(raw.get("z_value", 1.96))
    if z_value < 0.0:
        raise ValueError(
            "clinical_noninferiority.z_value cannot be negative"
        )
    return {
        "mode": mode,
        "z_value": z_value,
        "margins": margins,
    }


def apply_clinical_noninferiority(
    summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    settings: dict[str, Any] | None,
) -> None:
    """Apply point or paired-CI clinical noninferiority in place."""

    config = normalized_clinical_noninferiority(settings)
    intervals: dict[str, Any] = {}
    passed = []
    for metric, spec in CLINICAL_METRICS.items():
        differences = np.asarray(
            [
                float(candidate[metric]) - float(anchor[metric])
                for candidate, anchor in zip(
                    candidate_rows,
                    anchor_rows,
                )
            ],
            dtype=np.float64,
        )
        mean = float(differences.mean())
        sem = (
            float(
                differences.std(ddof=1)
                / np.sqrt(differences.size)
            )
            if differences.size > 1
            else 0.0
        )
        radius = float(config["z_value"]) * sem
        ci_low = mean - radius
        ci_high = mean + radius
        margin = float(config["margins"][metric])
        bound = (
            mean
            if config["mode"] == "point"
            else (
                ci_low
                if spec["direction"] == "higher"
                else ci_high
            )
        )
        metric_passed = bool(
            bound >= -margin
            if spec["direction"] == "higher"
            else bound <= margin
        )
        passed.append(metric_passed)
        intervals[metric] = {
            "mean_difference": mean,
            "sem": sem,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "margin": margin,
            "direction": spec["direction"],
            "noninferiority_bound": float(bound),
            "noninferior": metric_passed,
        }
        summary[spec["delta_key"]] = mean

    summary["clinical_noninferiority_mode"] = config["mode"]
    summary["clinical_noninferiority_z_value"] = config["z_value"]
    summary["clinical_noninferiority_margins"] = config["margins"]
    summary["clinical_metric_intervals"] = intervals
    summary["clinically_noninferior"] = bool(all(passed))


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
    if hasattr(agent, "residual_scale_vector"):
        agent.residual_scale_vector = (
            np.asarray(agent.residual_scale_vector, dtype=np.float32)
            * scale
        )
    elif hasattr(agent, "correction_gate_threshold"):
        if not (np.isclose(scale, 0.0) or np.isclose(scale, 1.0)):
            raise ValueError(
                "Structured-option deployment scale must be 0 or 1"
            )
        if np.isclose(scale, 0.0):
            agent.correction_gate_threshold = 1.0
    else:
        raise ValueError(
            "Agent does not expose a residual or structured-option "
            "deployment control"
        )
    if not bool(candidate.get("use_checkpoint_group_thresholds", False)):
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
    if "anchor_q_margin" in candidate:
        if not hasattr(agent, "anchor_q_margin"):
            raise ValueError(
                "Deployment candidate requests an anchor Q margin, but the "
                "agent does not support it"
            )
        agent.anchor_q_margin = float(candidate["anchor_q_margin"])
    if "temporal_guard" in candidate:
        configure_guard = getattr(
            agent,
            "configure_residual_temporal_guard",
            None,
        )
        if configure_guard is None:
            raise ValueError(
                "Deployment candidate requests a temporal guard, but the "
                "agent does not support it"
            )
        configure_guard(dict(candidate["temporal_guard"]))
    if "endpoint_projection" in candidate:
        configure_projection = getattr(
            agent,
            "configure_residual_endpoint_projection",
            None,
        )
        if configure_projection is None:
            raise ValueError(
                "Deployment candidate requests endpoint projection, but the "
                "agent does not support it"
            )
        configure_projection(dict(candidate["endpoint_projection"]))


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
        "temporally_suppressed_decisions": int(
            sum(
                value.get("temporally_suppressed_decisions", 0)
                for value in summaries
            )
        ),
        "temporal_suppression_rate": (
            sum(
                value.get("temporally_suppressed_decisions", 0)
                for value in summaries
            )
            / denominator
        ),
        "raw_residual_l1": float(
            sum(value.get("raw_residual_l1", 0.0) for value in summaries)
        ),
        "applied_residual_l1": float(
            sum(
                value.get("applied_residual_l1", 0.0)
                for value in summaries
            )
        ),
        "temporally_suppressed_l1": float(
            sum(
                value.get("temporally_suppressed_l1", 0.0)
                for value in summaries
            )
        ),
        "group_correction_rates": {
            group: count / denominator
            for group, count in group_corrected.items()
        },
        "capacity_action_effect": {
            metric: float(
                sum(
                    value.get("capacity_action_effect", {}).get(
                        metric,
                        0.0,
                    )
                    for value in summaries
                )
            )
            for metric in (
                "anchor_l1",
                "final_l1",
                "delta_l1",
                "amplification_l1",
                "damping_l1",
                "reversal_l1",
                "new_flow_l1",
            )
        },
    }


def normalized_deployment_candidate(
    value: dict[str, Any],
) -> dict[str, Any]:
    candidate = dict(value)
    scale = float(candidate.get("scale", 0.0))
    if scale < 0.0:
        raise ValueError("Deployment residual scale cannot be negative")
    use_checkpoint_thresholds = bool(
        candidate.get("use_checkpoint_group_thresholds", False)
    )
    thresholds = (
        ()
        if use_checkpoint_thresholds
        else tuple(
            float(item)
            for item in candidate.get(
                "group_thresholds",
                (0.8, 0.8, 0.8),
            )
        )
    )
    if any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("Deployment gate thresholds must lie in [0, 1]")
    normalized = {
        "scale": scale,
        "group_thresholds": list(thresholds),
    }
    if use_checkpoint_thresholds:
        normalized["use_checkpoint_group_thresholds"] = True
    if "anchor_q_margin" in candidate:
        margin = float(candidate["anchor_q_margin"])
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(
                "Deployment anchor_q_margin must be finite and non-negative"
            )
        normalized["anchor_q_margin"] = margin
    if "temporal_guard" in candidate:
        temporal_guard = candidate["temporal_guard"]
        if temporal_guard is None:
            temporal_guard = {}
        if not isinstance(temporal_guard, dict):
            raise ValueError("Deployment temporal_guard must be a mapping")
        normalized["temporal_guard"] = dict(temporal_guard)
    if "endpoint_projection" in candidate:
        endpoint_projection = candidate["endpoint_projection"]
        if endpoint_projection is None:
            endpoint_projection = {}
        if not isinstance(endpoint_projection, dict):
            raise ValueError(
                "Deployment endpoint_projection must be a mapping"
            )
        normalized["endpoint_projection"] = dict(endpoint_projection)
    return normalized


def deployment_candidate_label(candidate: dict[str, Any]) -> str:
    thresholds = (
        "checkpoint"
        if bool(candidate.get("use_checkpoint_group_thresholds", False))
        else "-".join(
            f"{float(value):g}"
            for value in candidate["group_thresholds"]
        )
    )
    label = f"scale{float(candidate['scale']):g}_gate{thresholds}"
    if "anchor_q_margin" in candidate:
        label = (
            f"{label}_qmargin"
            f"{float(candidate['anchor_q_margin']):g}"
        )
    if "temporal_guard" in candidate:
        guard_token = hashlib.sha1(
            json.dumps(
                candidate["temporal_guard"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:8]
        label = f"{label}_guard{guard_token}"
    if "endpoint_projection" in candidate:
        projection_token = hashlib.sha1(
            json.dumps(
                candidate["endpoint_projection"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:8]
        label = f"{label}_projection{projection_token}"
    return label


def select_deployment_candidate(
    results: list[dict[str, Any]],
    *,
    strict_scenario_guardrail: bool,
    selection_guardrails: dict[str, Any] | None = None,
) -> dict[str, Any]:
    guardrails = dict(selection_guardrails or {})
    metric_limits = (
        ("max_cost_gap_pct", "cost_gap_pct", float("-inf")),
        (
            "max_cost_difference_ci_high",
            "cost_difference_ci_high",
            float("-inf"),
        ),
    )
    metric_floors = (
        ("min_paired_win_rate", "paired_win_rate", float("inf")),
    )

    def passes_statistical_guardrails(
        result: dict[str, Any],
    ) -> bool:
        aggregate = result["aggregate"]
        max_scenario_cost_gap_pct = guardrails.get(
            "max_scenario_cost_gap_pct"
        )
        if max_scenario_cost_gap_pct is not None:
            scenario_summaries = result.get("per_scenario", {})
            if not scenario_summaries or any(
                float(summary.get("cost_gap_pct", float("inf")))
                > float(max_scenario_cost_gap_pct)
                for summary in scenario_summaries.values()
            ):
                return False
        for setting, metric, missing_default in metric_limits:
            if setting not in guardrails:
                continue
            value = float(aggregate.get(metric, missing_default))
            if value > float(guardrails[setting]):
                return False
        for setting, metric, missing_default in metric_floors:
            if setting not in guardrails:
                continue
            value = float(aggregate.get(metric, missing_default))
            if value < float(guardrails[setting]):
                return False
        return True

    eligible = [
        result
        for result in results
        if bool(result["aggregate"]["clinically_noninferior"])
        and passes_statistical_guardrails(result)
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
        "selection_guardrails": guardrails,
    }


def select_fixed_deployment_candidate(
    results: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    checkpoint_variant: str,
) -> dict[str, Any]:
    """Return a deployment frozen before validation outcomes are observed."""

    matches = [
        result
        for result in results
        if result["candidate"] == candidate
        and str(result.get("checkpoint_variant", "final"))
        == str(checkpoint_variant)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Pre-registered deployment must match exactly one validation "
            f"result, found {len(matches)}"
        )
    selected = matches[0]
    return {
        "candidate": selected["candidate"],
        "checkpoint_variant": str(checkpoint_variant),
        "selection_source": "pre_registered_fixed",
        "validation_aggregate": selected["aggregate"],
        "all_scenarios_clinically_noninferior": bool(
            selected["all_scenarios_clinically_noninferior"]
        ),
        "selection_guardrails": {},
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
        if value not in supported and _checkpoint_variant_episode(value) is None
    )
    if unknown:
        raise ValueError(
            "Unsupported checkpoint variants: " + ", ".join(unknown)
        )

    resolved: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for variant in requested_variants:
        if variant in supported:
            manifest_key = supported[variant]
            raw_path = run.get(manifest_key)
            if not raw_path:
                raise ValueError(
                    f"Training manifest is missing {manifest_key!r}"
                )
            path = resolve_manifest_artifact_path(raw_path)
        else:
            episode = _checkpoint_variant_episode(variant)
            algorithm = str(run.get("algorithm", "")).strip()
            seed = run.get("seed")
            final_checkpoint = run.get("checkpoint")
            if episode is None or not algorithm or seed is None or not final_checkpoint:
                raise ValueError(
                    "Intermediate checkpoint variants require algorithm, seed, "
                    "and checkpoint in the training manifest"
                )
            checkpoint_dir = resolve_manifest_artifact_path(final_checkpoint).parent
            path = checkpoint_dir / (
                f"{algorithm}_seed{int(seed)}_episode{episode}.pt"
            )
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


def _checkpoint_variant_episode(value: str) -> int | None:
    prefix = "episode"
    suffix = value[len(prefix):] if value.startswith(prefix) else ""
    if not suffix.isdigit():
        return None
    episode = int(suffix)
    return episode if episode > 0 else None


def resolve_manifest_artifact_path(raw_path: str | Path) -> Path:
    """Resolve relative manifest paths produced on Windows or POSIX."""

    path = Path(raw_path)
    if path.is_file():
        return path
    raw_text = str(raw_path)
    if "\\" in raw_text:
        portable_path = Path(raw_text.replace("\\", "/"))
        if portable_path.is_file():
            return portable_path
        return portable_path
    return path


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

    matched_pairs = [
        (graph, f"flat_{graph[4:]}")
        for graph in algorithms
        if graph.startswith("gcn_") and f"flat_{graph[4:]}" in algorithms
    ]
    for pair_index, (graph, flat) in enumerate(matched_pairs):
        graph_seeds = {
            seed for algorithm, seed in holdout_rows_by_run
            if algorithm == graph
        }
        flat_seeds = {
            seed for algorithm, seed in holdout_rows_by_run
            if algorithm == flat
        }
        common_seeds = sorted(graph_seeds & flat_seeds)
        if not common_seeds:
            continue
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
        bundle = metric_bootstrap_bundle(
            graph_rows,
            flat_rows,
            seed=bootstrap_seed + pair_index + 1,
        )
        result[f"{graph}_vs_{flat}"] = bundle
        if "graph_vs_flat" not in result:
            result["graph_vs_flat"] = bundle
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
