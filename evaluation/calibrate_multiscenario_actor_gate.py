"""Calibrate grouped residual gates against the actor's own CRN advantages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.collect_multiscenario_dagger import (
    causal_query_start_step,
    selected_deployments,
)
from evaluation.evaluate_multiscenario_network_residual import (
    configure_deployment,
)
from evaluation.network_residual_headroom import (
    evaluate_candidate_specs,
    lookahead_rollout_seeds,
    select_clinical_candidate,
)
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.action_projection import project_action
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.networks import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = calibrate_multiscenario_actor_gates(
        load_config(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def calibrate_multiscenario_actor_gates(
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit one grouped gate per actor using causal, actor-specific labels."""

    plan = load_benchmark_plan(
        run_config.get(
            "plan",
            "experiments/configs/residual_policy_benchmark.json",
        )
    )
    training_manifest_path = Path(run_config["training_manifest"])
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    selection_summary_path = Path(run_config["selection_summary"])
    selection_summary = json.loads(
        selection_summary_path.read_text(encoding="utf-8")
    )
    algorithm = str(
        run_config.get(
            "algorithm",
            "gcn_residual_mdl2_network_ddpg_afd",
        )
    )
    training_seeds = tuple(
        int(seed)
        for seed in run_config.get("training_seeds", (0, 1, 2))
    )
    if not training_seeds:
        raise ValueError("At least one actor seed is required")
    checkpoint_variant = str(
        run_config.get("checkpoint_variant", "pretrain")
    )
    checkpoint_key = (
        "pretrain_checkpoint"
        if checkpoint_variant == "pretrain"
        else "checkpoint"
    )
    if checkpoint_variant not in ("pretrain", "final"):
        raise ValueError("checkpoint_variant must be pretrain or final")

    scenario_names = tuple(
        str(value) for value in run_config.get("scenarios", ())
    )
    if len(scenario_names) < 2:
        raise ValueError("Actor-gate calibration requires multiple scenarios")
    selected_scenarios = select_scenarios(plan, scenario_names)
    scenarios_by_name = {
        str(scenario["name"]): scenario
        for scenario in selected_scenarios
    }
    scenarios = [scenarios_by_name[name] for name in scenario_names]
    deployments = selected_deployments(
        selection_summary,
        algorithm=algorithm,
    )
    runs_by_key = {
        (str(run["algorithm"]), int(run["seed"])): run
        for run in training_manifest["runs"]
    }

    rollouts_per_scenario = int(
        run_config.get("rollouts_per_scenario", 2)
    )
    if rollouts_per_scenario < 1:
        raise ValueError("rollouts_per_scenario must be positive")
    max_steps = int(run_config.get("max_steps", 52))
    lookahead = int(run_config.get("lookahead", 12))
    lookahead_replications = int(
        run_config.get("lookahead_replications", 1)
    )
    actor_scale = float(run_config.get("actor_scale", 0.1))
    if actor_scale <= 0.0:
        raise ValueError("actor_scale must be positive")
    min_improvement = float(
        run_config.get("min_improvement", 500_000.0)
    )
    detection_delay = int(run_config.get("detection_delay", 4))
    collection_seed = int(run_config.get("collection_seed", 8_900_000))
    gate_epochs = int(run_config.get("gate_epochs", 200))
    gate_batch_size = int(run_config.get("gate_batch_size", 128))
    gate_mode = str(run_config.get("gate_mode", "classification"))
    if gate_mode not in ("classification", "advantage"):
        raise ValueError(
            "gate_mode must be classification or advantage"
        )
    reinitialize_gate = bool(run_config.get("reinitialize_gate", True))
    positive_mass = run_config.get("positive_weight_mass")
    example_paths = {
        int(seed): Path(path)
        for seed, path in dict(
            run_config.get("example_paths", {})
        ).items()
    }
    output_root = Path(
        run_config.get(
            "output_root",
            "results/multiscenario_actor_gate_calibration",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    calibrated_runs: list[dict[str, Any]] = []
    for policy_index, training_seed in enumerate(training_seeds):
        run_key = (algorithm, training_seed)
        if run_key not in runs_by_key:
            raise ValueError(f"Training manifest is missing {run_key}")
        if training_seed not in deployments:
            raise ValueError(
                f"Selection summary is missing {algorithm} seed {training_seed}"
            )
        run = runs_by_key[run_key]
        source_checkpoint = Path(run[checkpoint_key])
        if not source_checkpoint.is_file():
            raise FileNotFoundError(source_checkpoint)
        config_snapshot = load_config(run["config"])
        source_examples = example_paths.get(training_seed)
        collection = (
            load_actor_gate_examples(
                source_examples,
                scenario_names=scenario_names,
                min_improvement=min_improvement,
            )
            if source_examples is not None
            else collect_group_actor_examples(
                plan=plan,
                scenarios=scenarios,
                scenario_names=scenario_names,
                algorithm=algorithm,
                training_seed=training_seed,
                policy_index=policy_index,
                config_snapshot=config_snapshot,
                checkpoint=source_checkpoint,
                deployment=deployments[training_seed],
                rollouts_per_scenario=rollouts_per_scenario,
                max_steps=max_steps,
                lookahead=lookahead,
                lookahead_replications=lookahead_replications,
                actor_scale=actor_scale,
                min_improvement=min_improvement,
                detection_delay=detection_delay,
                collection_seed=collection_seed,
            )
        )

        reference_scenario = scenarios[0]
        reference_config = dict(config_snapshot)
        reference_config["env"] = make_scenario_env_config(
            plan,
            algorithm,
            reference_scenario,
        )
        reference_env = build_env(
            reference_config,
            seed=collection_seed + policy_index * 1_000_000,
        )
        agent = get_agent_class(algorithm)(
            reference_env.observation_size,
            reference_env.action_size,
            reference_config,
        )
        agent.load_actor(source_checkpoint)
        if reinitialize_gate:
            reset_module_parameters(
                agent.correction_gate,
                seed=collection_seed + policy_index + 31_000,
            )
        weights = actor_gate_weights(
            collection["labels"],
            collection["advantages"],
            positive_mass=(
                None if positive_mass is None else float(positive_mass)
            ),
        )
        fit_summary = agent.fit_correction_gate_batch(
            collection["states"],
            collection["labels"],
            {
                "epochs": gate_epochs,
                "batch_size": gate_batch_size,
                "seed": collection_seed + policy_index + 41_000,
                "mode": gate_mode,
                "advantages": collection["advantages"],
                "feasible": collection["feasible"],
            },
            weights=weights,
        )

        run_root = output_root / algorithm / f"seed{training_seed}"
        run_root.mkdir(parents=True, exist_ok=True)
        example_path = run_root / "actor_gate_examples.npz"
        np.savez_compressed(
            example_path,
            states=collection["states"],
            labels=collection["labels"],
            advantages=collection["advantages"],
            feasible=collection["feasible"],
            weights=weights,
            scenario_ids=collection["scenario_ids"],
            trajectory_ids=collection["trajectory_ids"],
            trajectory_steps=collection["trajectory_steps"],
            scenario_names=np.asarray(scenario_names, dtype="U96"),
            group_names=np.asarray(collection["group_names"], dtype="U32"),
        )
        write_rows(
            collection["rows"],
            run_root / "actor_gate_examples.csv",
        )

        checkpoint_payload = torch.load(
            source_checkpoint,
            map_location=agent.device,
        )
        checkpoint_payload["correction_gate"] = (
            agent.correction_gate.state_dict()
        )
        checkpoint_payload["correction_gate_mode"] = gate_mode
        checkpoint_payload["actor_gate_calibration"] = {
            "source_checkpoint": str(source_checkpoint),
            "source_examples": (
                ""
                if source_examples is None
                else str(source_examples)
            ),
            "scenario_names": list(scenario_names),
            "rollouts_per_scenario": rollouts_per_scenario,
            "lookahead": lookahead,
            "lookahead_replications": lookahead_replications,
            "actor_scale": actor_scale,
            "min_improvement": min_improvement,
            "detection_delay": detection_delay,
            "collection_seed": collection_seed,
            "positive_weight_mass": positive_mass,
            "gate_mode": gate_mode,
            "reinitialize_gate": reinitialize_gate,
            **fit_summary,
        }
        calibrated_checkpoint = (
            run_root / "actor_gate_calibrated.pt"
        )
        torch.save(checkpoint_payload, calibrated_checkpoint)
        run_summary = {
            "algorithm": algorithm,
            "seed": training_seed,
            "checkpoint": str(calibrated_checkpoint),
            "pretrain_checkpoint": str(calibrated_checkpoint),
            "source_checkpoint": str(source_checkpoint),
            "config": str(run["config"]),
            "examples": str(example_path),
            "collection": collection_summary(collection),
            "gate_fit": fit_summary,
            "parameter_count": run.get("parameter_count"),
        }
        (run_root / "summary.json").write_text(
            json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        calibrated_runs.append(run_summary)
        print(
            "multiscenario_actor_gate "
            f"seed={training_seed} samples={collection['states'].shape[0]} "
            f"label_rates={fit_summary.get('group_label_rates', '')} "
            f"precision={float(fit_summary.get('precision', 0.0)):.3f} "
            f"recall={float(fit_summary.get('recall', 0.0)):.3f}",
            flush=True,
        )

    manifest = {
        "name": str(
            run_config.get(
                "name",
                "multiscenario_actor_gate_calibration",
            )
        ),
        "source_training_manifest": str(training_manifest_path),
        "selection_summary": str(selection_summary_path),
        "scenarios": list(scenario_names),
        "algorithm": algorithm,
        "checkpoint_variant": checkpoint_variant,
        "gate_mode": gate_mode,
        "example_paths": {
            str(seed): str(path)
            for seed, path in example_paths.items()
        },
        "runs": calibrated_runs,
    }
    (output_root / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_actor_gate_examples(
    path: str | Path,
    *,
    scenario_names: tuple[str, ...],
    min_improvement: float,
) -> dict[str, Any]:
    """Reuse actor-specific CRN outcomes with a new fixed margin."""

    source_path = Path(path)
    with np.load(source_path, allow_pickle=False) as payload:
        required = (
            "states",
            "advantages",
            "feasible",
            "scenario_ids",
            "trajectory_ids",
            "trajectory_steps",
            "scenario_names",
            "group_names",
        )
        missing = tuple(
            key for key in required if key not in payload.files
        )
        if missing:
            raise ValueError(
                "Actor-gate example cache is missing: "
                + ", ".join(missing)
            )
        cached_scenarios = tuple(
            str(value) for value in payload["scenario_names"]
        )
        if cached_scenarios != tuple(scenario_names):
            raise ValueError(
                "Actor-gate example scenarios do not match calibration"
            )
        states = np.asarray(payload["states"], dtype=np.float32)
        advantages = np.asarray(
            payload["advantages"],
            dtype=np.float32,
        )
        feasible = np.asarray(payload["feasible"], dtype=bool)
        row_count = int(states.shape[0])
        if (
            states.ndim != 2
            or advantages.ndim != 2
            or feasible.shape != advantages.shape
            or advantages.shape[0] != row_count
        ):
            raise ValueError(
                "Actor-gate states, advantages, and feasibility must align"
            )
        labels = (
            feasible
            & (advantages >= float(min_improvement))
        ).astype(np.float32)
        result = {
            "states": states,
            "labels": labels,
            "advantages": advantages,
            "feasible": feasible,
            "scenario_ids": np.asarray(
                payload["scenario_ids"],
                dtype=np.int64,
            ),
            "trajectory_ids": np.asarray(
                payload["trajectory_ids"],
                dtype=np.int64,
            ),
            "trajectory_steps": np.asarray(
                payload["trajectory_steps"],
                dtype=np.int64,
            ),
            "group_names": tuple(
                str(value) for value in payload["group_names"]
            ),
            "rows": [],
        }
    for key in (
        "scenario_ids",
        "trajectory_ids",
        "trajectory_steps",
    ):
        if result[key].shape != (row_count,):
            raise ValueError(
                f"Actor-gate cached {key} must align with states"
            )
    return result


def collect_group_actor_examples(
    *,
    plan: dict[str, Any],
    scenarios: list[dict[str, Any]],
    scenario_names: tuple[str, ...],
    algorithm: str,
    training_seed: int,
    policy_index: int,
    config_snapshot: dict[str, Any],
    checkpoint: Path,
    deployment: dict[str, Any],
    rollouts_per_scenario: int,
    max_steps: int,
    lookahead: int,
    lookahead_replications: int,
    actor_scale: float,
    min_improvement: float,
    detection_delay: int,
    collection_seed: int,
) -> dict[str, Any]:
    states: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    feasible_rows: list[np.ndarray] = []
    scenario_ids: list[int] = []
    trajectory_ids: list[int] = []
    trajectory_steps: list[int] = []
    rows: list[dict[str, Any]] = []
    group_names: tuple[str, ...] | None = None
    trajectory_id = 0
    decision_index = 0
    rollout_config = {
        "seed": collection_seed,
        "lookahead_seed": collection_seed + 1_000_000,
        "lookahead_replications": lookahead_replications,
    }

    for scenario_index, (scenario_name, scenario) in enumerate(
        zip(scenario_names, scenarios)
    ):
        config = dict(config_snapshot)
        env_config = make_scenario_env_config(
            plan,
            algorithm,
            scenario,
        )
        config["env"] = env_config
        env = build_env(
            config,
            seed=collection_seed
            + policy_index * 1_000_000
            + scenario_index * 100_000,
        )
        agent = get_agent_class(algorithm)(
            env.observation_size,
            env.action_size,
            config,
        )
        agent.load_actor(checkpoint)
        original_scale_vector = np.asarray(
            agent.residual_scale_vector,
            dtype=np.float32,
        ).copy()
        configure_deployment(agent, deployment)
        behavior_scale_vector = np.asarray(
            agent.residual_scale_vector,
            dtype=np.float32,
        ).copy()
        current_group_names = tuple(agent.correction_gate_groups)
        if not current_group_names:
            raise ValueError("Actor-specific calibration requires grouped gates")
        if group_names is None:
            group_names = current_group_names
        elif group_names != current_group_names:
            raise ValueError("Scenario agents use different gate groups")
        query_start = causal_query_start_step(
            env_config,
            detection_delay=detection_delay,
        )

        for rollout in range(rollouts_per_scenario):
            rollout_seed = (
                collection_seed
                + policy_index * 1_000_000
                + scenario_index * 100_000
                + rollout
            )
            state = env.reset(seed=rollout_seed)
            agent.reset()
            anchor = get_heuristic_class("mdl2")(
                env.observation_size,
                env.action_size,
                {},
            )
            anchor.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                agent.residual_scale_vector = original_scale_vector
                actor_action = agent.select_ungated_residual_action(
                    state,
                    env=env,
                    residual_scale=actor_scale,
                )
                anchor_action = anchor.select_action(
                    state,
                    explore=False,
                    env=env,
                )
                agent.residual_scale_vector = behavior_scale_vector
                behavior_action = agent.select_action(
                    state,
                    explore=False,
                    env=env,
                )

                if step >= query_start:
                    candidate_actions = grouped_actor_actions(
                        anchor_action,
                        actor_action,
                        group_names=current_group_names,
                        num_facilities=env.config.num_facilities,
                        env=env,
                    )
                    specs = [
                        {
                            "group": "anchor",
                            "epsilon": 0.0,
                            "variant": 0,
                            "action": anchor_action,
                        },
                        *[
                            {
                                "group": group,
                                "epsilon": actor_scale,
                                "variant": index + 1,
                                "action": action,
                            }
                            for index, (group, action) in enumerate(
                                zip(current_group_names, candidate_actions)
                            )
                        ],
                    ]
                    evaluated = evaluate_candidate_specs(
                        specs,
                        env,
                        anchor,
                        lookahead=lookahead,
                        rollout_seeds=lookahead_rollout_seeds(
                            rollout_config,
                            decision_index,
                        ),
                    )
                    group_labels = []
                    group_advantages = []
                    group_feasible = []
                    for group_index, group in enumerate(
                        current_group_names,
                        start=1,
                    ):
                        best_index, scores, feasible = (
                            select_clinical_candidate(
                                [evaluated[0], evaluated[group_index]],
                                score_weights={},
                                guardrails={
                                    "min_score_improvement": min_improvement,
                                    "min_completion_service_level_delta": 0.0,
                                    "max_patients_lost_delta": 0.0,
                                    "max_patient_ineligibility_during_manufacturing_rate_delta": 0.0,
                                },
                            )
                        )
                        accepted = bool(best_index == 1 and feasible[1])
                        advantage = float(scores[0] - scores[1])
                        group_labels.append(float(accepted))
                        group_advantages.append(advantage)
                        group_feasible.append(bool(feasible[1]))
                        actor_metrics = evaluated[group_index]["metrics"]
                        anchor_metrics = evaluated[0]["metrics"]
                        rows.append(
                            {
                                "training_seed": training_seed,
                                "scenario": scenario_name,
                                "rollout": rollout,
                                "step": step,
                                "group": group,
                                "label": int(accepted),
                                "advantage": advantage,
                                "feasible": int(bool(feasible[1])),
                                "anchor_cost": float(
                                    anchor_metrics["total_cost"]
                                ),
                                "actor_cost": float(
                                    actor_metrics["total_cost"]
                                ),
                                "completion_service_level_delta": float(
                                    actor_metrics[
                                        "completion_service_level"
                                    ]
                                    - anchor_metrics[
                                        "completion_service_level"
                                    ]
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
                    states.append(np.asarray(state, dtype=np.float32))
                    labels.append(
                        np.asarray(group_labels, dtype=np.float32)
                    )
                    advantages.append(
                        np.asarray(group_advantages, dtype=np.float32)
                    )
                    feasible_rows.append(
                        np.asarray(group_feasible, dtype=bool)
                    )
                    scenario_ids.append(scenario_index)
                    trajectory_ids.append(trajectory_id)
                    trajectory_steps.append(step)
                    decision_index += 1
                    if decision_index % 100 == 0:
                        label_array = np.asarray(labels)
                        print(
                            "actor_gate_collection "
                            f"seed={training_seed} states={decision_index} "
                            f"rates="
                            + "|".join(
                                f"{rate:.3f}"
                                for rate in label_array.mean(axis=0)
                            ),
                            flush=True,
                        )
                state, _reward, done, _info = env.step(behavior_action)
                step += 1
            trajectory_id += 1
        agent.residual_scale_vector = original_scale_vector

    if not states or group_names is None:
        raise RuntimeError("Actor-gate collection produced no causal examples")
    return {
        "states": np.asarray(states, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "advantages": np.asarray(advantages, dtype=np.float32),
        "feasible": np.asarray(feasible_rows, dtype=bool),
        "scenario_ids": np.asarray(scenario_ids, dtype=np.int64),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=np.int64),
        "trajectory_steps": np.asarray(trajectory_steps, dtype=np.int64),
        "group_names": group_names,
        "rows": rows,
    }


def grouped_actor_actions(
    anchor_action: np.ndarray,
    actor_action: np.ndarray,
    *,
    group_names: tuple[str, ...],
    num_facilities: int,
    env: Any | None = None,
) -> tuple[np.ndarray, ...]:
    anchor = np.asarray(anchor_action, dtype=np.float32)
    actor = np.asarray(actor_action, dtype=np.float32)
    if anchor.shape != actor.shape:
        raise ValueError("Anchor and actor actions must have matching shapes")
    group_slices = facility_net_group_slices(num_facilities)
    actions = []
    for group in group_names:
        if group not in group_slices:
            raise ValueError(f"Unsupported actor action group: {group}")
        candidate = anchor.copy()
        group_slice = group_slices[group]
        candidate[group_slice] = actor[group_slice]
        actions.append(
            project_action(
                candidate,
                env_state=env,
                action_space_info=anchor.size,
            ).action
        )
    return tuple(actions)


def facility_net_group_slices(
    num_facilities: int,
) -> dict[str, slice]:
    n = int(num_facilities)
    return {
        "specimen_transfer": slice(0, n),
        "reagent_transfer": slice(n, 2 * n),
        "capacity_transfer": slice(2 * n, 3 * n),
        "replenishment": slice(3 * n, 4 * n),
    }


def actor_gate_weights(
    labels: np.ndarray,
    advantages: np.ndarray,
    *,
    positive_mass: float | None,
) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.float32)
    advantage_array = np.asarray(advantages, dtype=np.float32)
    if label_array.shape != advantage_array.shape or label_array.ndim != 2:
        raise ValueError("Grouped gate labels and advantages must align")
    sample_magnitude = np.max(
        np.abs(advantage_array),
        axis=1,
    )
    weights = 1.0 + np.clip(
        sample_magnitude / 1_000_000.0,
        0.0,
        5.0,
    )
    if positive_mass is not None:
        if not 0.0 < positive_mass < 1.0:
            raise ValueError("positive_mass must lie strictly between 0 and 1")
        positive = np.any(label_array > 0.5, axis=1)
        if np.any(positive) and np.any(~positive):
            weights[positive] *= (
                positive_mass / float(weights[positive].sum())
            )
            weights[~positive] *= (
                (1.0 - positive_mass)
                / float(weights[~positive].sum())
            )
    weights *= float(weights.size) / max(float(weights.sum()), 1e-8)
    return weights.astype(np.float32)


def reset_module_parameters(module: Any, *, seed: int) -> None:
    if module is None:
        raise ValueError("Cannot reset a disabled correction gate")
    torch.manual_seed(int(seed))
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def collection_summary(collection: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(collection["labels"], dtype=np.float32)
    advantages = np.asarray(
        collection["advantages"],
        dtype=np.float32,
    )
    return {
        "states": int(labels.shape[0]),
        "groups": list(collection["group_names"]),
        "positive_rates": labels.mean(axis=0).tolist(),
        "mean_advantages": advantages.mean(axis=0).tolist(),
        "mean_positive_advantages": [
            (
                float(advantages[:, index][labels[:, index] > 0.5].mean())
                if np.any(labels[:, index] > 0.5)
                else 0.0
            )
            for index in range(labels.shape[1])
        ],
        "trajectories": int(
            np.unique(collection["trajectory_ids"]).size
        ),
    }


if __name__ == "__main__":
    main()
