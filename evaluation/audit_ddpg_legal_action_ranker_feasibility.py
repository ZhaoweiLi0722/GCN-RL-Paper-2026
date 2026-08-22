"""Stage G1 offline legal-action ranker feasibility audit.

The audit never updates an actor or an immutable checkpoint. It builds a fresh
paired-CRN legal-action dataset, fits one fixed critic-only objective in three
leave-one-seed-out folds, and evaluates each fold on an independent replication
stream from the unseen seed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.audit_ddpg_actor_projection_transfer import (
    checkpoint_paths,
    manifest_runs,
    verify_locked_inputs,
)
from evaluation.audit_ddpg_online_identifiability import (
    collect_frozen_trajectory_snapshots,
    sha256_file,
)
from evaluation.run_gcn_residual_sweep import mean_rollout_metrics_after_action
from src.baselines.heuristics import get_heuristic_class
from src.models.graph_features import flat_state_to_node_features
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.networks import torch


DEFAULT_CONFIG = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_legal_action_ranker_g1.json"
)
FORMAL_HOLDOUT_SEED = 91_100_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.smoke:
        config = smoke_config(config)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    result = run_audit(
        config,
        config_path=config_path,
        smoke=bool(args.smoke),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    smoke["dataset"]["decision_steps"] = [0, 25, 51]
    smoke["dataset"]["discovery_replications"] = 1
    smoke["dataset"]["validation_replications"] = 1
    smoke["ranker"]["epochs"] = 2
    smoke["name"] = f"{config['name']}_smoke"
    smoke["output_root"] = str(config["smoke_output_root"])
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    if str(config["primary_algorithm"]) != "gcn_residual_mdl2_network_ddpg_afd":
        raise ValueError("Stage G1 is locked to the primary GCN-DDPG algorithm")
    seeds = tuple(int(value) for value in config["training_seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage G1 requires exactly three unique training seeds")
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config["scenario_by_training_seed"].items()
    }
    if set(assignment) != set(seeds):
        raise ValueError("Stage G1 scenario assignment must match training seeds")

    dataset = dict(config["dataset"])
    steps = tuple(int(value) for value in dataset["decision_steps"])
    max_steps = int(dataset["max_steps_per_episode"])
    if not steps or len(steps) != len(set(steps)):
        raise ValueError("Stage G1 decision steps must be unique and non-empty")
    if any(step < 0 or step >= max_steps for step in steps):
        raise ValueError("Stage G1 decision step is outside the episode")
    horizons = tuple(dataset["horizons"])
    primary = str(dataset["primary_horizon"])
    if primary not in {str(value) for value in horizons}:
        raise ValueError("Stage G1 primary horizon must be present")
    if "4" not in {str(value) for value in horizons} or primary != "remaining":
        raise ValueError("Stage G1 requires four-step and remaining horizons")
    if int(dataset["discovery_replications"]) <= 0:
        raise ValueError("Stage G1 discovery replications must be positive")
    if int(dataset["validation_replications"]) <= 0:
        raise ValueError("Stage G1 validation replications must be positive")
    options = tuple(dataset["explicit_options"])
    if len(options) != 4 or any(
        str(option["group"]) != "specimen_transfer" for option in options
    ):
        raise ValueError("Stage G1 requires the four locked specimen options")

    ranker = dict(config["ranker"])
    if int(ranker["epochs"]) <= 0 or float(ranker["critic_lr"]) <= 0.0:
        raise ValueError("Stage G1 ranker epochs and learning rate must be positive")
    if str(ranker["target_scale_statistic"]) != "median_nonanchor_absolute":
        raise ValueError("Stage G1 target scaling is locked")
    for key in (
        "regression_weight",
        "pairwise_ranking_weight",
        "pairwise_margin",
        "pairwise_minimum_cost_gap",
        "target_scale_floor",
    ):
        value = float(ranker[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Stage G1 ranker {key} must be positive and finite")

    gates = dict(config["stage_g1_gate"])
    for key, raw in gates.items():
        value = float(raw)
        if key == "minimum_improved_seed_count":
            if int(raw) < 1 or int(raw) > len(seeds):
                raise ValueError("Stage G1 improved-seed gate is invalid")
        elif not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Stage G1 gate {key} must be in [0, 1]")

    ranges = tuple(config["forbidden_crn_ranges"])
    if not any(
        int(entry["start"]) <= FORMAL_HOLDOUT_SEED <= int(entry["end"])
        for entry in ranges
    ):
        raise ValueError("Stage G1 must explicitly forbid the formal holdout")
    used = diagnostic_crn_seed_sequence(config)
    if len(used) != len(set(used)):
        raise ValueError("Stage G1 diagnostic CRN streams overlap")
    overlap = [
        seed
        for seed in used
        if any(
            int(entry["start"]) <= seed <= int(entry["end"])
            for entry in ranges
        )
    ]
    if overlap:
        raise ValueError(f"Stage G1 uses forbidden CRN seeds: {overlap[:10]}")


def diagnostic_crn_seed_sequence(config: dict[str, Any]) -> list[int]:
    dataset = dict(config["dataset"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    steps = tuple(int(value) for value in dataset["decision_steps"])
    horizons = tuple(dataset["horizons"])
    result = [
        int(dataset["live_seed"]) + index * 100_000
        for index, _seed in enumerate(seeds)
    ]
    for base_key, replication_key in (
        ("discovery_rollout_seed", "discovery_replications"),
        ("validation_rollout_seed", "validation_replications"),
    ):
        for scenario_index, _seed in enumerate(seeds):
            for step in steps:
                for horizon_index, _horizon in enumerate(horizons):
                    start = (
                        int(dataset[base_key])
                        + scenario_index * 1_000_000
                        + step * 100
                        + horizon_index * 10
                    )
                    result.extend(
                        start + replication
                        for replication in range(
                            int(dataset[replication_key])
                        )
                    )
    return result


def run_audit(
    config: dict[str, Any],
    *,
    config_path: Path | None,
    smoke: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    output_root = Path(str(config["output_root"]))
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    locked_before = verify_locked_inputs(config)
    manifest = _load_json(Path(str(config["training_manifest"])))
    seeds = tuple(int(value) for value in config["training_seeds"])
    algorithm = str(config["primary_algorithm"])
    runs = manifest_runs(manifest, algorithm=algorithm, seeds=seeds)

    records: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for scenario_index, seed in enumerate(seeds):
        seed_records, seed_rows = collect_seed_dataset(
            config,
            run=runs[seed],
            training_seed=seed,
            scenario_index=scenario_index,
        )
        records.extend(seed_records)
        dataset_rows.extend(seed_rows)
    validate_dataset(config, records=records, rows=dataset_rows)

    fold_results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    fold_arrays = []
    for held_out_seed in seeds:
        result = run_fold(
            config,
            run=runs[held_out_seed],
            records=records,
            held_out_seed=held_out_seed,
        )
        fold_results[str(held_out_seed)] = result["summary"]
        prediction_rows.extend(result["rows"])
        fold_arrays.append(result["arrays"])

    locked_after = verify_locked_inputs(config)
    if locked_before != locked_after:
        raise RuntimeError("Stage G1 changed an immutable input")

    aggregate = aggregate_fold_metrics(
        config,
        records=records,
        folds=fold_arrays,
    )
    label_stability = discovery_validation_label_stability(
        records,
        horizon=str(config["dataset"]["primary_horizon"]),
        minimum_pairwise_cost_gap=float(
            config["ranker"]["pairwise_minimum_cost_gap"]
        ),
    )
    decision = stage_g1_decision(
        config,
        aggregate=aggregate,
        folds=fold_results,
        label_stability=label_stability,
        smoke=smoke,
    )

    dataset_path = output_root / "legal_action_rows.csv"
    prediction_path = output_root / "fold_prediction_rows.csv"
    arrays_path = output_root / "legal_action_dataset.npz"
    write_rows(dataset_rows, dataset_path)
    write_rows(prediction_rows, prediction_path)
    write_dataset_arrays(config, records=records, path=arrays_path)

    result = {
        "name": str(config["name"]),
        "experimental_role": str(config["experimental_role"]),
        "smoke": bool(smoke),
        "config": None if config_path is None else str(config_path),
        "config_sha256": (
            None
            if config_path is None or not config_path.is_file()
            else sha256_file(config_path)
        ),
        "formal_holdout_reused": False,
        "actor_updated": False,
        "fitted_checkpoint_saved": False,
        "locked_inputs": locked_before,
        "fresh_crn_seed_count": len(diagnostic_crn_seed_sequence(config)),
        "dataset": {
            "states": len(records),
            "rows": len(dataset_rows),
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "arrays_path": str(arrays_path),
            "arrays_sha256": sha256_file(arrays_path),
        },
        "prediction_rows": {
            "rows": len(prediction_rows),
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "label_stability": label_stability,
        "folds": fold_results,
        "aggregate": aggregate,
        "decision": decision,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary"] = str(summary_path)
    result["summary_sha256"] = sha256_file(summary_path)
    return result


def collect_seed_dataset(
    config: dict[str, Any],
    *,
    run: dict[str, Any],
    training_seed: int,
    scenario_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = dict(config["dataset"])
    algorithm = str(config["primary_algorithm"])
    runtime = load_config(Path(str(run["config"])))
    runtime["device"] = str(dataset["device"])
    runtime["replay_buffer_size"] = 1
    env = build_env(
        runtime,
        seed=int(dataset["live_seed"]) + scenario_index * 100_000,
    )
    agent, _checkpoint_path = load_pretrain_agent(
        algorithm,
        run=run,
        runtime=runtime,
        state_dim=env.observation_size,
        action_dim=env.action_size,
    )
    anchor_name = str(runtime["residual_action"].get("base_policy", "mdl2"))
    anchor = get_heuristic_class(anchor_name)(
        env.observation_size,
        env.action_size,
        dict(runtime["residual_action"].get("base_policy_config", {})),
    )
    snapshot_config = {
        "replay": {"max_steps_per_episode": int(dataset["max_steps_per_episode"])},
        "manifold": {
            "decision_steps": list(dataset["decision_steps"]),
            "explicit_options": list(dataset["explicit_options"]),
            "live_seed": int(dataset["live_seed"]),
        },
    }
    snapshots = collect_frozen_trajectory_snapshots(
        snapshot_config,
        env=env,
        frozen=agent,
        anchor=anchor,
        scenario_index=scenario_index,
    )
    records = []
    rows = []
    for snapshot in snapshots:
        actions = np.asarray(snapshot["actions"], dtype=np.float32)
        identifiers = executed_action_identifiers(agent, actions)
        if len(set(identifiers)) != actions.shape[0]:
            raise ValueError("Stage G1 legal actions collapsed after quantization")
        record = {
            "training_seed": int(training_seed),
            "scenario": str(
                config["scenario_by_training_seed"][str(training_seed)]
            ),
            "step": int(snapshot["step"]),
            "state": np.asarray(snapshot["state"], dtype=np.float32),
            "actions": actions,
            "labels": tuple(str(value) for value in snapshot["labels"]),
            "executed_action_ids": tuple(identifiers),
            "advantages": {"discovery": {}, "validation": {}},
            "costs": {"discovery": {}, "validation": {}},
        }
        for horizon_index, configured in enumerate(dataset["horizons"]):
            horizon_name = str(configured)
            horizon = (
                int(snapshot["remaining_horizon"])
                if configured == "remaining"
                else min(int(configured), int(snapshot["remaining_horizon"]))
            )
            stream_values = {}
            for stream, base_key, replication_key in (
                (
                    "discovery",
                    "discovery_rollout_seed",
                    "discovery_replications",
                ),
                (
                    "validation",
                    "validation_rollout_seed",
                    "validation_replications",
                ),
            ):
                seed_start = (
                    int(dataset[base_key])
                    + scenario_index * 1_000_000
                    + int(snapshot["step"]) * 100
                    + horizon_index * 10
                )
                rollout_seeds = tuple(
                    seed_start + index
                    for index in range(int(dataset[replication_key]))
                )
                metrics = []
                for action in actions:
                    anchor.reset()
                    metrics.append(
                        mean_rollout_metrics_after_action(
                            snapshot["env"],
                            anchor,
                            action,
                            horizon=horizon,
                            rollout_seeds=rollout_seeds,
                        )
                    )
                costs = np.asarray(
                    [float(value["total_cost"]) for value in metrics],
                    dtype=np.float64,
                )
                advantages = costs[0] - costs
                record["costs"][stream][horizon_name] = costs
                record["advantages"][stream][horizon_name] = advantages
                stream_values[stream] = (
                    seed_start,
                    len(rollout_seeds),
                    metrics,
                    advantages,
                )
            for action_index, label in enumerate(record["labels"]):
                discovery = stream_values["discovery"]
                validation = stream_values["validation"]
                rows.append(
                    {
                        "training_seed": int(training_seed),
                        "scenario": record["scenario"],
                        "step": int(snapshot["step"]),
                        "configured_horizon": horizon_name,
                        "horizon": int(horizon),
                        "candidate_index": int(action_index),
                        "candidate_label": str(label),
                        "executed_action_id": identifiers[action_index],
                        "discovery_seed_start": int(discovery[0]),
                        "discovery_replications": int(discovery[1]),
                        "discovery_total_cost": float(
                            discovery[2][action_index]["total_cost"]
                        ),
                        "discovery_cost_advantage": float(
                            discovery[3][action_index]
                        ),
                        "validation_seed_start": int(validation[0]),
                        "validation_replications": int(validation[1]),
                        "validation_total_cost": float(
                            validation[2][action_index]["total_cost"]
                        ),
                        "validation_cost_advantage": float(
                            validation[3][action_index]
                        ),
                    }
                )
        records.append(record)
    return records, rows


def load_pretrain_agent(
    algorithm: str,
    *,
    run: dict[str, Any],
    runtime: dict[str, Any],
    state_dim: int,
    action_dim: int,
) -> tuple[Any, Path]:
    paths = checkpoint_paths(
        run,
        variants=("pretrain",),
        suffixes={"pretrain": "pretrain"},
    )
    path = paths["pretrain"]
    agent = get_agent_class(algorithm)(state_dim, action_dim, runtime)
    checkpoint = torch.load(
        path,
        map_location=agent.device,
        weights_only=False,
    )
    agent.load_actor(path)
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor.eval()
    agent.critic.eval()
    return agent, path


def executed_action_identifiers(agent: Any, actions: np.ndarray) -> list[str]:
    tensor = torch.as_tensor(
        np.asarray(actions, dtype=np.float32),
        dtype=torch.float32,
        device=agent.device,
    )
    with torch.no_grad():
        executed = agent._critic_actions_tensor(tensor)
    return [
        hashlib.sha256(
            np.round(value.astype(np.float64), decimals=7).tobytes()
        ).hexdigest()[:16]
        for value in executed.detach().cpu().numpy()
    ]


def validate_dataset(
    config: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    expected_states = len(config["training_seeds"]) * len(
        config["dataset"]["decision_steps"]
    )
    expected_rows = (
        expected_states
        * len(config["dataset"]["horizons"])
        * (len(config["dataset"]["explicit_options"]) + 1)
    )
    if len(records) != expected_states or len(rows) != expected_rows:
        raise ValueError("Stage G1 dataset row count mismatch")
    keys = {
        (
            int(row["training_seed"]),
            int(row["step"]),
            str(row["configured_horizon"]),
            int(row["candidate_index"]),
        )
        for row in rows
    }
    if len(keys) != expected_rows:
        raise ValueError("Stage G1 dataset contains duplicate rows")
    for row in rows:
        for key in (
            "discovery_total_cost",
            "discovery_cost_advantage",
            "validation_total_cost",
            "validation_cost_advantage",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Stage G1 dataset has non-finite {key}")
    if any(len(set(record["executed_action_ids"])) != 5 for record in records):
        raise ValueError("Stage G1 requires five distinct legal actions")


def fold_seed_assignments(
    seeds: tuple[int, ...],
) -> dict[int, tuple[int, ...]]:
    return {
        held_out: tuple(seed for seed in seeds if seed != held_out)
        for held_out in seeds
    }


def run_fold(
    config: dict[str, Any],
    *,
    run: dict[str, Any],
    records: list[dict[str, Any]],
    held_out_seed: int,
) -> dict[str, Any]:
    algorithm = str(config["primary_algorithm"])
    runtime = load_config(Path(str(run["config"])))
    runtime["device"] = str(config["dataset"]["device"])
    runtime["replay_buffer_size"] = 1
    environment = run["environments"][0]
    agent, checkpoint_path = load_pretrain_agent(
        algorithm,
        run=run,
        runtime=runtime,
        state_dim=int(environment["observation_size"]),
        action_dim=int(environment["action_size"]),
    )
    actor_before = module_state_sha256(agent.actor)
    critic_before = {
        key: value.detach().cpu().clone()
        for key, value in agent.critic.state_dict().items()
    }
    train_records = [
        record
        for record in records
        if int(record["training_seed"]) != int(held_out_seed)
    ]
    test_records = [
        record
        for record in records
        if int(record["training_seed"]) == int(held_out_seed)
    ]
    expected_train_seeds = set(config["training_seeds"]) - {held_out_seed}
    if {int(record["training_seed"]) for record in train_records} != expected_train_seeds:
        raise ValueError("Stage G1 fold training seeds are incomplete")
    if {int(record["training_seed"]) for record in test_records} != {held_out_seed}:
        raise ValueError("Stage G1 held-out fold leaked or is incomplete")

    baseline_predictions = critic_predictions(agent, test_records)
    training = train_legal_action_ranker(
        config,
        agent=agent,
        records=train_records,
        held_out_seed=held_out_seed,
    )
    fitted_predictions = critic_predictions(agent, test_records)
    actor_after = module_state_sha256(agent.actor)
    if actor_before != actor_after:
        raise RuntimeError("Stage G1 modified the actor")

    primary = str(config["dataset"]["primary_horizon"])
    minimum_gap = float(config["ranker"]["pairwise_minimum_cost_gap"])
    material = float(config["dataset"]["material_improvement"])
    stream_metrics = {}
    rows = []
    for stream in ("discovery", "validation"):
        advantages = record_advantages(
            test_records,
            stream=stream,
            horizon=primary,
        )
        stream_metrics[stream] = {
            "baseline": ranking_metrics(
                baseline_predictions,
                advantages,
                minimum_pairwise_cost_gap=minimum_gap,
                material_improvement=material,
            ),
            "fitted": ranking_metrics(
                fitted_predictions,
                advantages,
                minimum_pairwise_cost_gap=minimum_gap,
                material_improvement=material,
            ),
        }
        for record_index, record in enumerate(test_records):
            for action_index, label in enumerate(record["labels"]):
                rows.append(
                    {
                        "held_out_seed": int(held_out_seed),
                        "training_seeds": "|".join(
                            str(seed) for seed in sorted(expected_train_seeds)
                        ),
                        "scenario": str(record["scenario"]),
                        "step": int(record["step"]),
                        "stream": stream,
                        "configured_horizon": primary,
                        "candidate_index": int(action_index),
                        "candidate_label": str(label),
                        "true_cost_advantage": float(
                            advantages[record_index, action_index]
                        ),
                        "baseline_q_advantage": float(
                            baseline_predictions[record_index, action_index]
                        ),
                        "fitted_q_advantage": float(
                            fitted_predictions[record_index, action_index]
                        ),
                    }
                )

    drift = parameter_drift(critic_before, agent.critic.state_dict())
    summary = {
        "held_out_seed": int(held_out_seed),
        "training_seeds": sorted(expected_train_seeds),
        "checkpoint": str(checkpoint_path),
        "actor_sha256_before": actor_before,
        "actor_sha256_after": actor_after,
        "actor_unchanged": actor_before == actor_after,
        "critic_parameter_drift": drift,
        "training": training,
        "metrics": stream_metrics,
    }
    return {
        "summary": summary,
        "rows": rows,
        "arrays": {
            "held_out_seed": int(held_out_seed),
            "records": test_records,
            "baseline_predictions": baseline_predictions,
            "fitted_predictions": fitted_predictions,
        },
    }


def train_legal_action_ranker(
    config: dict[str, Any],
    *,
    agent: Any,
    records: list[dict[str, Any]],
    held_out_seed: int,
) -> dict[str, Any]:
    ranker = dict(config["ranker"])
    primary = str(config["dataset"]["primary_horizon"])
    states, actions = record_state_action_arrays(records)
    raw_advantages = record_advantages(
        records,
        stream="discovery",
        horizon=primary,
    )
    scales = target_scales(
        raw_advantages,
        floor=float(ranker["target_scale_floor"]),
    )
    targets = raw_advantages / scales[:, None]
    state_tensor, action_tensor = critic_training_tensors(
        agent,
        states=states,
        actions=actions,
    )
    target_tensor = torch.as_tensor(
        targets,
        dtype=torch.float32,
        device=agent.device,
    )
    raw_advantage_tensor = torch.as_tensor(
        raw_advantages,
        dtype=torch.float32,
        device=agent.device,
    )
    torch.manual_seed(int(ranker["torch_seed"]) + int(held_out_seed))
    optimizer = torch.optim.Adam(
        agent.critic.parameters(),
        lr=float(ranker["critic_lr"]),
    )
    agent.critic.train()
    initial = legal_action_ranker_loss(
        agent,
        state_tensor=state_tensor,
        action_tensor=action_tensor,
        target_tensor=target_tensor,
        raw_advantage_tensor=raw_advantage_tensor,
        ranker=ranker,
    )
    final = initial
    for _epoch in range(int(ranker["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        losses = legal_action_ranker_loss(
            agent,
            state_tensor=state_tensor,
            action_tensor=action_tensor,
            target_tensor=target_tensor,
            raw_advantage_tensor=raw_advantage_tensor,
            ranker=ranker,
        )
        losses["total"].backward()
        optimizer.step()
    final = legal_action_ranker_loss(
        agent,
        state_tensor=state_tensor,
        action_tensor=action_tensor,
        target_tensor=target_tensor,
        raw_advantage_tensor=raw_advantage_tensor,
        ranker=ranker,
    )
    agent.critic.eval()
    result = {
        "epochs": int(ranker["epochs"]),
        "states": len(records),
        "initial": detached_loss_summary(initial),
        "final": detached_loss_summary(final),
        "target_scale": numeric_summary(scales),
    }
    if not all(
        math.isfinite(float(value))
        for phase in ("initial", "final")
        for value in result[phase].values()
    ):
        raise ValueError("Stage G1 ranker loss is non-finite")
    return result


def critic_training_tensors(
    agent: Any,
    *,
    states: np.ndarray,
    actions: np.ndarray,
) -> tuple[Any, Any]:
    count, option_count, action_dim = actions.shape
    repeated_states = np.repeat(states, option_count, axis=0)
    raw_states = torch.as_tensor(
        repeated_states,
        dtype=torch.float32,
        device=agent.device,
    )
    node_features = flat_state_to_node_features(raw_states, agent.graph_spec)
    action_tensor = torch.as_tensor(
        actions.reshape(count * option_count, action_dim),
        dtype=torch.float32,
        device=agent.device,
    )
    with torch.no_grad():
        executed_actions = agent._critic_actions_tensor(action_tensor)
    return node_features, executed_actions


def legal_action_ranker_loss(
    agent: Any,
    *,
    state_tensor: Any,
    action_tensor: Any,
    target_tensor: Any,
    raw_advantage_tensor: Any,
    ranker: dict[str, Any],
) -> dict[str, Any]:
    state_count, option_count = target_tensor.shape
    q_values = agent.critic(state_tensor, action_tensor).reshape(
        state_count,
        option_count,
    )
    predicted = q_values - q_values[:, :1]
    regression = torch.nn.functional.smooth_l1_loss(
        predicted,
        target_tensor,
    )
    pair_losses = []
    pair_count = 0
    for left, right in combinations(range(option_count), 2):
        raw_delta = (
            raw_advantage_tensor[:, left] - raw_advantage_tensor[:, right]
        )
        mask = torch.abs(raw_delta) >= float(
            ranker["pairwise_minimum_cost_gap"]
        )
        if not bool(mask.any()):
            continue
        target_sign = torch.sign(
            target_tensor[:, left] - target_tensor[:, right]
        )
        predicted_delta = predicted[:, left] - predicted[:, right]
        pair_losses.append(
            torch.nn.functional.softplus(
                float(ranker["pairwise_margin"])
                - target_sign[mask] * predicted_delta[mask]
            )
        )
        pair_count += int(mask.sum().detach().cpu().item())
    if not pair_losses:
        raise ValueError("Stage G1 has no material legal-action pairs")
    ranking = torch.cat(pair_losses).mean()
    total = (
        float(ranker["regression_weight"]) * regression
        + float(ranker["pairwise_ranking_weight"]) * ranking
    )
    return {
        "total": total,
        "regression": regression,
        "ranking": ranking,
        "pair_count": torch.as_tensor(
            float(pair_count),
            dtype=torch.float32,
            device=agent.device,
        ),
    }


def detached_loss_summary(losses: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu().item())
        for key, value in losses.items()
    }


def critic_predictions(agent: Any, records: list[dict[str, Any]]) -> np.ndarray:
    states, actions = record_state_action_arrays(records)
    state_tensor, action_tensor = critic_training_tensors(
        agent,
        states=states,
        actions=actions,
    )
    agent.critic.eval()
    with torch.no_grad():
        q_values = agent.critic(state_tensor, action_tensor).reshape(
            len(records),
            actions.shape[1],
        )
        advantages = q_values - q_values[:, :1]
    return advantages.detach().cpu().numpy().astype(np.float64)


def record_state_action_arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    states = np.stack(
        [np.asarray(record["state"], dtype=np.float32) for record in records]
    )
    actions = np.stack(
        [np.asarray(record["actions"], dtype=np.float32) for record in records]
    )
    return states, actions


def record_advantages(
    records: list[dict[str, Any]],
    *,
    stream: str,
    horizon: str,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                record["advantages"][stream][horizon],
                dtype=np.float64,
            )
            for record in records
        ]
    )


def target_scales(advantages: np.ndarray, *, floor: float) -> np.ndarray:
    values = np.asarray(advantages, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Stage G1 target scale requires non-anchor actions")
    medians = np.median(np.abs(values[:, 1:]), axis=1)
    return np.maximum(medians, float(floor))


def ranking_metrics(
    predictions: np.ndarray,
    advantages: np.ndarray,
    *,
    minimum_pairwise_cost_gap: float,
    material_improvement: float,
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(advantages, dtype=np.float64)
    if predicted.shape != true.shape or predicted.ndim != 2:
        raise ValueError("Stage G1 ranking arrays must align")
    top1 = []
    pairwise = []
    selected_material = []
    headroom = []
    for predicted_row, true_row in zip(predicted, true):
        choice = int(np.argmax(predicted_row))
        best = float(np.max(true_row))
        best_indices = set(
            np.flatnonzero(np.isclose(true_row, best, atol=1e-9, rtol=0.0))
        )
        top1.append(float(choice in best_indices))
        selected_material.append(
            float(true_row[choice] >= float(material_improvement))
        )
        headroom.append(float(best >= float(material_improvement)))
        for left, right in combinations(range(true.shape[1]), 2):
            true_delta = float(true_row[left] - true_row[right])
            if abs(true_delta) < float(minimum_pairwise_cost_gap):
                continue
            predicted_delta = float(predicted_row[left] - predicted_row[right])
            product = true_delta * predicted_delta
            pairwise.append(
                1.0 if product > 0.0 else 0.5 if product == 0.0 else 0.0
            )
    return {
        "states": int(true.shape[0]),
        "top1_accuracy": float(np.mean(top1)),
        "pairwise_accuracy": (
            None if not pairwise else float(np.mean(pairwise))
        ),
        "pairwise_comparisons": len(pairwise),
        "selected_material_improvement_fraction": float(
            np.mean(selected_material)
        ),
        "material_headroom_fraction": float(np.mean(headroom)),
    }


def discovery_validation_label_stability(
    records: list[dict[str, Any]],
    *,
    horizon: str,
    minimum_pairwise_cost_gap: float,
    include_per_seed: bool = True,
) -> dict[str, Any]:
    discovery = record_advantages(records, stream="discovery", horizon=horizon)
    validation = record_advantages(records, stream="validation", horizon=horizon)
    best_agreement = []
    pairwise = []
    for discovery_row, validation_row in zip(discovery, validation):
        discovery_best = set(np.flatnonzero(discovery_row == discovery_row.max()))
        validation_best = set(np.flatnonzero(validation_row == validation_row.max()))
        best_agreement.append(float(bool(discovery_best & validation_best)))
        for left, right in combinations(range(discovery.shape[1]), 2):
            discovery_delta = float(discovery_row[left] - discovery_row[right])
            validation_delta = float(validation_row[left] - validation_row[right])
            if min(abs(discovery_delta), abs(validation_delta)) < float(
                minimum_pairwise_cost_gap
            ):
                continue
            pairwise.append(float(discovery_delta * validation_delta > 0.0))
    per_seed = {}
    if include_per_seed:
        for seed in sorted(
            {int(record["training_seed"]) for record in records}
        ):
            subset = [
                record
                for record in records
                if int(record["training_seed"]) == seed
            ]
            per_seed[str(seed)] = discovery_validation_label_stability(
                subset,
                horizon=horizon,
                minimum_pairwise_cost_gap=minimum_pairwise_cost_gap,
                include_per_seed=False,
            )
    return {
        "states": len(records),
        "best_action_agreement": float(np.mean(best_agreement)),
        "pairwise_sign_agreement": (
            None if not pairwise else float(np.mean(pairwise))
        ),
        "pairwise_comparisons": len(pairwise),
        "per_seed": per_seed,
    }


def aggregate_fold_metrics(
    config: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    del records
    baseline = np.concatenate(
        [np.asarray(fold["baseline_predictions"]) for fold in folds],
        axis=0,
    )
    fitted = np.concatenate(
        [np.asarray(fold["fitted_predictions"]) for fold in folds],
        axis=0,
    )
    fold_records = [record for fold in folds for record in fold["records"]]
    primary = str(config["dataset"]["primary_horizon"])
    minimum_gap = float(config["ranker"]["pairwise_minimum_cost_gap"])
    material = float(config["dataset"]["material_improvement"])
    result = {}
    for stream in ("discovery", "validation"):
        advantages = record_advantages(
            fold_records,
            stream=stream,
            horizon=primary,
        )
        result[stream] = {
            "baseline": ranking_metrics(
                baseline,
                advantages,
                minimum_pairwise_cost_gap=minimum_gap,
                material_improvement=material,
            ),
            "fitted": ranking_metrics(
                fitted,
                advantages,
                minimum_pairwise_cost_gap=minimum_gap,
                material_improvement=material,
            ),
        }
    return result


def stage_g1_decision(
    config: dict[str, Any],
    *,
    aggregate: dict[str, Any],
    folds: dict[str, Any],
    label_stability: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    gates = dict(config["stage_g1_gate"])
    validation = aggregate["validation"]["fitted"]
    discovery = aggregate["discovery"]["fitted"]
    seed_gains = {
        seed: (
            float(fold["metrics"]["validation"]["fitted"]["top1_accuracy"])
            - float(fold["metrics"]["validation"]["baseline"]["top1_accuracy"])
        )
        for seed, fold in folds.items()
    }
    improved = [
        int(seed)
        for seed, gain in seed_gains.items()
        if gain >= float(gates["minimum_per_seed_top1_gain"])
    ]
    per_seed_top1 = {
        seed: float(
            fold["metrics"]["validation"]["fitted"]["top1_accuracy"]
        )
        for seed, fold in folds.items()
    }
    discovery_to_validation_drop = (
        float(discovery["top1_accuracy"])
        - float(validation["top1_accuracy"])
    )
    checks = {
        "stable_counterfactual_labels": (
            float(label_stability["best_action_agreement"])
            >= float(
                gates["minimum_discovery_validation_best_action_agreement"]
            )
        ),
        "aggregate_validation_top1": (
            float(validation["top1_accuracy"])
            >= float(gates["minimum_validation_top1_accuracy"])
        ),
        "every_seed_validation_top1": all(
            value
            >= float(gates["minimum_per_seed_validation_top1_accuracy"])
            for value in per_seed_top1.values()
        ),
        "aggregate_validation_pairwise": (
            validation["pairwise_accuracy"] is not None
            and float(validation["pairwise_accuracy"])
            >= float(gates["minimum_validation_pairwise_accuracy"])
        ),
        "improved_seed_count": (
            len(improved) >= int(gates["minimum_improved_seed_count"])
        ),
        "replication_stream_stability": (
            discovery_to_validation_drop
            <= float(gates["maximum_discovery_to_validation_top1_drop"])
        ),
        "actors_unchanged": all(
            bool(fold["actor_unchanged"]) for fold in folds.values()
        ),
    }
    if smoke:
        classification = "smoke_only_no_scientific_decision"
        passed = False
    elif not checks["stable_counterfactual_labels"]:
        classification = "unstable_counterfactual_labels_close_extension"
        passed = False
    else:
        passed = all(checks.values())
        classification = (
            "authorize_locked_actor_transfer_smoke_design"
            if passed
            else "close_legal_action_aligned_ddpg_extension"
        )
    return {
        "classification": classification,
        "legal_action_ranker_feasibility_passed": bool(passed),
        "actor_transfer_smoke_design_authorized": bool(passed),
        "online_training_authorized": False,
        "formal_confirmation_authorized": False,
        "checks": checks,
        "observed": {
            "discovery_validation_best_action_agreement": float(
                label_stability["best_action_agreement"]
            ),
            "validation_top1_accuracy": float(validation["top1_accuracy"]),
            "validation_pairwise_accuracy": validation["pairwise_accuracy"],
            "per_seed_validation_top1_accuracy": per_seed_top1,
            "per_seed_validation_top1_gain": seed_gains,
            "improved_seeds": improved,
            "discovery_to_validation_top1_drop": float(
                discovery_to_validation_drop
            ),
        },
    }


def write_dataset_arrays(
    config: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    horizons = tuple(str(value) for value in config["dataset"]["horizons"])
    states, actions = record_state_action_arrays(records)
    discovery = np.stack(
        [
            np.stack(
                [record["advantages"]["discovery"][horizon] for horizon in horizons]
            )
            for record in records
        ]
    )
    validation = np.stack(
        [
            np.stack(
                [record["advantages"]["validation"][horizon] for horizon in horizons]
            )
            for record in records
        ]
    )
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        discovery_advantages=discovery,
        validation_advantages=validation,
        training_seeds=np.asarray(
            [int(record["training_seed"]) for record in records],
            dtype=np.int64,
        ),
        steps=np.asarray([int(record["step"]) for record in records], dtype=np.int64),
        horizons=np.asarray(horizons),
        labels=np.asarray(records[0]["labels"]),
    )


def module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_drift(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float]:
    squared_sum = 0.0
    maximum = 0.0
    count = 0
    for key, original in before.items():
        current = after[key].detach().cpu()
        delta = current - original
        squared_sum += float(torch.sum(delta * delta).item())
        maximum = max(maximum, float(torch.max(torch.abs(delta)).item()))
        count += int(delta.numel())
    return {
        "parameter_count": float(count),
        "rms": float(math.sqrt(squared_sum / max(count, 1))),
        "max_abs": float(maximum),
    }


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()
