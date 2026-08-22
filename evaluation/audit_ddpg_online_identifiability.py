"""Read-only Stage F0 audit of online DDPG targets and legal action ranking.

The audit never trains an agent or reads the formal holdout. It has two parts:

1. Reconstruct one-step, configured n-step, and behavior-trajectory remainder
   returns from immutable online replay buffers.
2. Compare checkpoint critics and their straight-through action gradients with
   fresh paired-CRN rollouts on a small, prespecified legal specimen-action
   manifold.

The behavior-trajectory remainder is diagnostic rather than a causal action
value: future one-step terms are evaluated on the behavior trajectory. Causal
finite-horizon action values are measured only by the manifold rollouts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.run_gcn_residual_sweep import (
    mean_rollout_metrics_after_action,
)
from src.baselines.heuristics import get_heuristic_class
from src.models.graph_features import flat_state_to_node_features
from src.rl.action_projection import project_action
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.networks import torch
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    residual_option_actions_from_env,
)


DEFAULT_CONFIG = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_online_identifiability_f0.json"
)
FORMAL_HOLDOUT_SEED = 91_100_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--artifact-root",
        default=".",
        help="Repository root containing the immutable full Stage C3 results",
    )
    parser.add_argument(
        "--component",
        choices=("all", "replay", "manifold"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--reclassify-summary",
        help="Recompute only the gate decision in an existing Stage F0 summary",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.reclassify_summary:
        result = reclassify_summary(
            config,
            summary_path=Path(args.reclassify_summary),
        )
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
        return
    if args.smoke:
        config = smoke_config(config)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    result = run_audit(
        config,
        source_root=Path.cwd(),
        artifact_root=Path(args.artifact_root),
        config_path=config_path,
        component=str(args.component),
        smoke=bool(args.smoke),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LockedArtifactVerifier:
    """Verify selected full artifacts against the immutable C3 inventory."""

    def __init__(
        self,
        *,
        source_root: Path,
        artifact_root: Path,
        inventory_path: Path,
        expected_inventory_sha256: str,
    ) -> None:
        self.source_root = source_root.resolve()
        self.artifact_root = artifact_root.resolve()
        resolved_inventory = _resolve(self.source_root, inventory_path)
        observed = sha256_file(resolved_inventory)
        if observed != str(expected_inventory_sha256).lower():
            raise ValueError(
                "Stage C3 inventory SHA256 mismatch: "
                f"expected {expected_inventory_sha256}, got {observed}"
            )
        self.inventory_path = resolved_inventory
        self.inventory_sha256 = observed
        self.inventory = _load_json(resolved_inventory)
        if not isinstance(self.inventory, dict) or not self.inventory:
            raise ValueError("Stage C3 inventory must be a non-empty mapping")
        self.observed: dict[str, str] = {}

    def verify(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError("Locked artifact paths must be repository-relative")
        key = relative.as_posix()
        expected = self.inventory.get(key)
        if expected in (None, ""):
            raise ValueError(f"Stage C3 inventory does not lock {key}")
        path = self.artifact_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != str(expected).lower():
            raise ValueError(
                f"Locked artifact SHA256 mismatch for {key}: "
                f"expected {expected}, got {observed}"
            )
        self.observed[key] = observed
        return path

    def reverify(self) -> None:
        before = dict(self.observed)
        for relative, expected in before.items():
            observed = sha256_file(self.artifact_root / relative)
            if observed != expected:
                raise RuntimeError(
                    f"Read-only Stage F0 audit modified locked input {relative}"
                )


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    primary = str(config["primary_algorithm"])
    first_seed = int(config["training_seeds"][0])
    smoke["algorithms"] = [primary]
    smoke["training_seeds"] = [first_seed]
    smoke["scenario_by_training_seed"] = {
        str(first_seed): str(
            config["scenario_by_training_seed"][str(first_seed)]
        )
    }
    smoke["manifold"]["checkpoint_variants"] = ["pretrain"]
    smoke["manifold"]["decision_steps"] = [
        int(config["manifold"]["decision_steps"][0])
    ]
    smoke["manifold"]["horizons"] = [1, 4]
    smoke["manifold"]["rollout_replications"] = 1
    smoke["output_root"] = str(config["smoke_output_root"])
    smoke["name"] = f"{config['name']}_smoke"
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    algorithms = tuple(str(value) for value in config["algorithms"])
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("Stage F0 algorithms must be unique and non-empty")
    primary = str(config["primary_algorithm"])
    if primary not in algorithms:
        raise ValueError("Stage F0 primary algorithm must be audited")
    seeds = tuple(int(value) for value in config["training_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage F0 training seeds must be unique and non-empty")
    assignment = {
        int(seed): str(name)
        for seed, name in config["scenario_by_training_seed"].items()
    }
    if set(assignment) != set(seeds):
        raise ValueError("Stage F0 scenario assignment must match training seeds")

    replay = dict(config["replay"])
    if int(replay["max_steps_per_episode"]) <= 0:
        raise ValueError("Replay episode length must be positive")
    if int(replay["n_step_horizon"]) <= 0:
        raise ValueError("Replay n-step horizon must be positive")

    manifold = dict(config["manifold"])
    variants = tuple(str(value) for value in manifold["checkpoint_variants"])
    checkpoint_suffixes = dict(manifold["checkpoint_suffixes"])
    if not variants or any(value not in checkpoint_suffixes for value in variants):
        raise ValueError("Every Stage F0 checkpoint variant needs a suffix")
    steps = tuple(int(value) for value in manifold["decision_steps"])
    max_steps = int(replay["max_steps_per_episode"])
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("Stage F0 decision steps must be unique and non-empty")
    if any(step < 0 or step >= max_steps for step in steps):
        raise ValueError("Stage F0 decision step is outside the episode")
    if int(manifold["rollout_replications"]) <= 0:
        raise ValueError("Stage F0 rollout replications must be positive")
    horizons = tuple(manifold["horizons"])
    if not horizons:
        raise ValueError("Stage F0 requires at least one rollout horizon")
    for horizon in horizons:
        if horizon != "remaining" and int(horizon) <= 0:
            raise ValueError("Stage F0 rollout horizons must be positive")

    option_specs = make_explicit_residual_option_specs(
        manifold["explicit_options"]
    )
    if any(
        not option.is_anchor and option.group != "specimen_transfer"
        for option in option_specs
    ):
        raise ValueError("Stage F0 legal options must be specimen-transfer only")

    forbidden = {int(value) for value in config["forbidden_crn_seeds"]}
    if FORMAL_HOLDOUT_SEED not in forbidden:
        raise ValueError("Stage F0 must explicitly forbid the formal holdout")
    used = manifold_crn_seeds(config)
    overlap = sorted(used & forbidden)
    if overlap:
        raise ValueError(f"Stage F0 uses forbidden CRN seeds: {overlap}")
    if len(used) != len(manifold_crn_seed_sequence(config)):
        raise ValueError("Stage F0 manifold CRN streams are not disjoint")


def manifold_crn_seed_sequence(config: dict[str, Any]) -> list[int]:
    manifold = dict(config["manifold"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    variants = tuple(str(value) for value in manifold["checkpoint_variants"])
    steps = tuple(int(value) for value in manifold["decision_steps"])
    horizons = tuple(manifold["horizons"])
    replications = int(manifold["rollout_replications"])
    result = [
        int(manifold["live_seed"]) + scenario_index * 100_000
        for scenario_index, _seed in enumerate(seeds)
    ]
    for scenario_index, _seed in enumerate(seeds):
        for checkpoint_index, _variant in enumerate(variants):
            for step in steps:
                for horizon_index, _horizon in enumerate(horizons):
                    start = (
                        int(manifold["rollout_seed"])
                        + scenario_index * 1_000_000
                        + checkpoint_index * 100_000
                        + step * 100
                        + horizon_index * 10
                    )
                    result.extend(start + index for index in range(replications))
    return result


def manifold_crn_seeds(config: dict[str, Any]) -> set[int]:
    return set(manifold_crn_seed_sequence(config))


def run_audit(
    config: dict[str, Any],
    *,
    source_root: Path,
    artifact_root: Path,
    config_path: Path | None,
    component: str = "all",
    smoke: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    if component not in {"all", "replay", "manifold"}:
        raise ValueError(f"Unsupported Stage F0 component: {component}")
    output_root = _resolve(source_root, Path(config["output_root"]))
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    verifier = LockedArtifactVerifier(
        source_root=source_root,
        artifact_root=artifact_root,
        inventory_path=Path(config["artifact_inventory"]),
        expected_inventory_sha256=str(config["artifact_inventory_sha256"]),
    )
    manifest_path = verifier.verify(config["training_manifest"])
    manifest = _load_json(manifest_path)
    runs = _manifest_runs(
        manifest,
        algorithms=tuple(str(value) for value in config["algorithms"]),
        seeds=tuple(int(value) for value in config["training_seeds"]),
    )
    comparison = _load_locked_compact_json(
        source_root,
        Path(config["comparison_evidence"]),
        str(config["comparison_evidence_sha256"]),
    )

    replay_result = None
    manifold_result = None
    manifold_rows: list[dict[str, Any]] = []
    if component in {"all", "replay"}:
        replay_result = run_replay_audit(
            config,
            verifier=verifier,
            runs=runs,
            comparison=comparison,
        )
    if component in {"all", "manifold"}:
        manifold_result, manifold_rows = run_manifold_audit(
            config,
            verifier=verifier,
            runs=runs,
        )
        rows_path = output_root / "manifold_rows.csv"
        write_rows(manifold_rows, rows_path)
    verifier.reverify()

    decision = stage_f0_decision(
        config,
        replay_result=replay_result,
        manifold_result=manifold_result,
        smoke=smoke,
    )
    result = {
        "name": str(config["name"]),
        "experimental_role": str(config["experimental_role"]),
        "component": component,
        "smoke": bool(smoke),
        "config": None if config_path is None else str(config_path),
        "config_sha256": (
            None
            if config_path is None or not config_path.is_file()
            else sha256_file(config_path)
        ),
        "artifact_root": str(artifact_root.resolve()),
        "artifact_inventory": str(verifier.inventory_path),
        "artifact_inventory_sha256": verifier.inventory_sha256,
        "locked_input_sha256": dict(sorted(verifier.observed.items())),
        "formal_holdout_reused": False,
        "causal_interpretation_limit": (
            "Replay remainder returns follow the behavior trajectory and are "
            "not single-action causal values. Only fresh paired-CRN manifold "
            "rollouts estimate finite-horizon action effects."
        ),
        "replay": replay_result,
        "manifold": manifold_result,
        "decision": decision,
    }
    if manifold_rows:
        result["manifold_rows"] = str(output_root / "manifold_rows.csv")
        result["manifold_rows_sha256"] = sha256_file(
            output_root / "manifold_rows.csv"
        )
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary"] = str(summary_path)
    result["summary_sha256"] = sha256_file(summary_path)
    return result


def run_replay_audit(
    config: dict[str, Any],
    *,
    verifier: LockedArtifactVerifier,
    runs: dict[tuple[str, int], dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    algorithms = tuple(str(value) for value in config["algorithms"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    replay_config = dict(config["replay"])
    per_seed: dict[str, dict[str, Any]] = defaultdict(dict)
    pooled_arrays: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for algorithm in algorithms:
        for seed in seeds:
            run = runs[(algorithm, seed)]
            runtime_config_path = verifier.verify(run["config"])
            runtime_config = load_config(runtime_config_path)
            summary_path = verifier.verify(
                Path(run["config"]).parent / "summary.json"
            )
            run_summary = _load_json(summary_path)
            state_path = verifier.verify(run_summary["training_state_checkpoint"])
            checkpoint = torch.load(
                state_path,
                map_location="cpu",
                weights_only=False,
            )
            contract_algorithm = str(
                checkpoint.get("training_contract", {}).get("algorithm", "")
            )
            if contract_algorithm != algorithm:
                raise ValueError("Replay training-state contract mismatch")
            internal_algorithm = str(checkpoint.get("algorithm"))
            expected_internal = (
                "flat_ddpg" if algorithm.startswith("flat_") else algorithm
            )
            if internal_algorithm != expected_internal:
                raise ValueError("Replay training-state agent mismatch")
            if int(checkpoint.get("seed", -1)) != seed:
                raise ValueError("Replay training-state seed mismatch")
            arrays = ordered_online_replay_arrays(
                checkpoint["agent"]["replay_buffer"]
            )
            expected = int(runtime_config["num_episodes"]) * int(
                runtime_config["max_steps_per_episode"]
            )
            if arrays["rewards"].size != expected:
                raise ValueError(
                    f"Online replay count mismatch for {algorithm} seed {seed}: "
                    f"{arrays['rewards'].size} != {expected}"
                )
            result, derived = replay_target_metrics(
                arrays,
                gamma=float(runtime_config["gamma"]),
                episode_length=int(replay_config["max_steps_per_episode"]),
                n_step_horizon=int(replay_config["n_step_horizon"]),
                self_imitation_minimum_return=float(
                    replay_config["self_imitation_minimum_return"]
                ),
                sign_tolerance=float(replay_config["sign_tolerance"]),
            )
            result["training_state"] = str(state_path)
            result["training_state_sha256"] = sha256_file(state_path)
            result["runtime_config"] = str(runtime_config_path)
            result["runtime_config_sha256"] = sha256_file(runtime_config_path)
            result["reward_scale"] = float(runtime_config["reward_scale"])
            result["final_vs_frozen_total_cost_difference"] = (
                final_vs_frozen_cost_difference(comparison, algorithm, seed)
            )
            per_seed[algorithm][str(seed)] = result
            for name, values in derived.items():
                pooled_arrays[algorithm][name].append(values)
            del checkpoint

    pooled = {}
    for algorithm, named_arrays in pooled_arrays.items():
        one = np.concatenate(named_arrays["one_step"])
        n_step = np.concatenate(named_arrays["n_step"])
        remainder = np.concatenate(named_arrays["remainder"])
        active = np.concatenate(named_arrays["self_imitation_active"]).astype(bool)
        pooled[algorithm] = target_relationship_metrics(
            one,
            n_step,
            remainder,
            active,
            sign_tolerance=float(replay_config["sign_tolerance"]),
        )
        pooled[algorithm]["transitions"] = int(one.size)
    return {
        "interpretation": (
            "The remainder target is a discounted sum of exact one-step "
            "behavior-versus-anchor rewards along the realized behavior "
            "trajectory; it diagnoses temporal sign consistency but is not a "
            "counterfactual value of the first action."
        ),
        "per_seed": dict(per_seed),
        "pooled": pooled,
    }


def ordered_online_replay_arrays(replay: dict[str, Any]) -> dict[str, np.ndarray]:
    size = int(replay["size"])
    capacity = int(replay["capacity"])
    position = int(replay["position"])
    if size <= 0 or size > capacity:
        raise ValueError("Replay size is invalid")
    if size < capacity:
        order = np.arange(size, dtype=np.int64)
    else:
        order = np.concatenate(
            (
                np.arange(position, size, dtype=np.int64),
                np.arange(0, position, dtype=np.int64),
            )
        )
    online = np.asarray(replay["online_mask"], dtype=bool).reshape(-1)[order]
    if not np.any(online):
        raise ValueError("Replay contains no online transitions")
    result = {}
    for name in (
        "rewards",
        "one_step_rewards",
        "dones",
        "discount_multipliers",
    ):
        values = np.asarray(replay[name], dtype=np.float64).reshape(-1)[order]
        result[name] = values[online]
    if not all(np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError("Replay audit requires finite numeric arrays")
    return result


def discounted_episode_remainders(
    one_step_rewards: np.ndarray,
    *,
    gamma: float,
    episode_length: int,
) -> np.ndarray:
    rewards = np.asarray(one_step_rewards, dtype=np.float64).reshape(-1)
    if episode_length <= 0 or rewards.size % episode_length:
        raise ValueError("Replay transitions do not form fixed-length episodes")
    result = np.zeros_like(rewards)
    for start in range(0, rewards.size, episode_length):
        running = 0.0
        for index in range(start + episode_length - 1, start - 1, -1):
            running = float(rewards[index]) + float(gamma) * running
            result[index] = running
    return result


def recompute_n_step_targets(
    one_step_rewards: np.ndarray,
    *,
    gamma: float,
    episode_length: int,
    n_step_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(one_step_rewards, dtype=np.float64).reshape(-1)
    if n_step_horizon <= 0:
        raise ValueError("n_step_horizon must be positive")
    if episode_length <= 0 or rewards.size % episode_length:
        raise ValueError("Replay transitions do not form fixed-length episodes")
    targets = np.zeros_like(rewards)
    multipliers = np.ones_like(rewards)
    for start in range(0, rewards.size, episode_length):
        end = start + episode_length
        for index in range(start, end):
            horizon = min(n_step_horizon, end - index)
            targets[index] = sum(
                (float(gamma) ** offset) * rewards[index + offset]
                for offset in range(horizon)
            )
            multipliers[index] = float(gamma) ** (horizon - 1)
    return targets, multipliers


def replay_target_metrics(
    arrays: dict[str, np.ndarray],
    *,
    gamma: float,
    episode_length: int,
    n_step_horizon: int,
    self_imitation_minimum_return: float,
    sign_tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    one = np.asarray(arrays["one_step_rewards"], dtype=np.float64).reshape(-1)
    n_step = np.asarray(arrays["rewards"], dtype=np.float64).reshape(-1)
    multipliers = np.asarray(
        arrays["discount_multipliers"], dtype=np.float64
    ).reshape(-1)
    expected_targets, expected_multipliers = recompute_n_step_targets(
        one,
        gamma=gamma,
        episode_length=episode_length,
        n_step_horizon=n_step_horizon,
    )
    target_error = np.abs(n_step - expected_targets)
    multiplier_error = np.abs(multipliers - expected_multipliers)
    if float(target_error.max(initial=0.0)) > 1e-6:
        raise ValueError("Persisted n-step targets do not match one-step rewards")
    if float(multiplier_error.max(initial=0.0)) > 1e-6:
        raise ValueError("Persisted n-step discount multipliers do not match")
    remainder = discounted_episode_remainders(
        one,
        gamma=gamma,
        episode_length=episode_length,
    )
    full_horizon = np.isclose(
        multipliers,
        float(gamma) ** (n_step_horizon - 1),
        rtol=1e-5,
        atol=1e-7,
    )
    active = (
        full_horizon
        & (n_step > self_imitation_minimum_return)
        & (one > self_imitation_minimum_return)
    )
    relationships = target_relationship_metrics(
        one,
        n_step,
        remainder,
        active,
        sign_tolerance=sign_tolerance,
    )
    relationships.update(
        {
            "episodes": int(one.size // episode_length),
            "transitions": int(one.size),
            "full_n_step_horizon_fraction": float(full_horizon.mean()),
            "self_imitation_active_count": int(active.sum()),
            "self_imitation_active_fraction": float(active.mean()),
            "n_step_target_reconstruction_max_abs_error": float(
                target_error.max(initial=0.0)
            ),
            "discount_multiplier_reconstruction_max_abs_error": float(
                multiplier_error.max(initial=0.0)
            ),
        }
    )
    return relationships, {
        "one_step": one,
        "n_step": n_step,
        "remainder": remainder,
        "self_imitation_active": active,
    }


def target_relationship_metrics(
    one_step: np.ndarray,
    n_step: np.ndarray,
    remainder: np.ndarray,
    self_imitation_active: np.ndarray,
    *,
    sign_tolerance: float,
) -> dict[str, Any]:
    one = np.asarray(one_step, dtype=np.float64).reshape(-1)
    n_value = np.asarray(n_step, dtype=np.float64).reshape(-1)
    long_value = np.asarray(remainder, dtype=np.float64).reshape(-1)
    active = np.asarray(self_imitation_active, dtype=bool).reshape(-1)
    if not (one.shape == n_value.shape == long_value.shape == active.shape):
        raise ValueError("Replay target arrays must align")
    active_long = long_value[active]
    return {
        "one_step": numeric_summary(one),
        "n_step": numeric_summary(n_value),
        "behavior_trajectory_remainder": numeric_summary(long_value),
        "one_step_vs_n_step": paired_target_metrics(
            one, n_value, sign_tolerance=sign_tolerance
        ),
        "one_step_vs_remainder": paired_target_metrics(
            one, long_value, sign_tolerance=sign_tolerance
        ),
        "n_step_vs_remainder": paired_target_metrics(
            n_value, long_value, sign_tolerance=sign_tolerance
        ),
        "self_imitation_active_remainder": {
            "samples": int(active_long.size),
            "positive_fraction": _fraction(active_long > sign_tolerance),
            "negative_fraction": _fraction(active_long < -sign_tolerance),
            "zero_fraction": _fraction(np.abs(active_long) <= sign_tolerance),
            "summary": numeric_summary(active_long),
        },
    }


def paired_target_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sign_tolerance: float,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise ValueError("Paired target arrays must align and be non-empty")
    left_sign = signed_classes(left_array, tolerance=sign_tolerance)
    right_sign = signed_classes(right_array, tolerance=sign_tolerance)
    both_nonzero = (left_sign != 0) & (right_sign != 0)
    opposite = both_nonzero & (left_sign != right_sign)
    return {
        "samples": int(left_array.size),
        "pearson": correlation(left_array, right_array),
        "spearman": correlation(
            average_ranks(left_array), average_ranks(right_array)
        ),
        "exact_sign_agreement_fraction": float(
            np.mean(left_sign == right_sign)
        ),
        "nonzero_sign_agreement_fraction": (
            None
            if not np.any(both_nonzero)
            else float(np.mean(left_sign[both_nonzero] == right_sign[both_nonzero]))
        ),
        "opposite_sign_fraction": float(np.mean(opposite)),
        "left_positive_right_negative_fraction": float(
            np.mean((left_sign > 0) & (right_sign < 0))
        ),
        "left_negative_right_positive_fraction": float(
            np.mean((left_sign < 0) & (right_sign > 0))
        ),
    }


def run_manifold_audit(
    config: dict[str, Any],
    *,
    verifier: LockedArtifactVerifier,
    runs: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifold = dict(config["manifold"])
    primary = str(config["primary_algorithm"])
    if primary not in config["algorithms"]:
        raise ValueError("Primary algorithm is not available for manifold audit")
    seeds = tuple(int(value) for value in config["training_seeds"])
    all_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for scenario_index, seed in enumerate(seeds):
        run = runs[(primary, seed)]
        config_path = verifier.verify(run["config"])
        runtime_config = load_config(config_path)
        runtime_config["device"] = str(manifold.get("device", "cpu"))
        runtime_config["replay_buffer_size"] = 1
        summary_path = verifier.verify(Path(run["config"]).parent / "summary.json")
        run_summary = _load_json(summary_path)
        checkpoints = checkpoint_paths(
            run,
            run_summary,
            variants=tuple(
                str(value) for value in manifold["checkpoint_variants"]
            ),
            suffixes=dict(manifold["checkpoint_suffixes"]),
            verifier=verifier,
        )
        env = build_env(
            runtime_config,
            seed=int(manifold["live_seed"]) + scenario_index * 100_000,
        )
        frozen = get_agent_class(primary)(
            env.observation_size,
            env.action_size,
            runtime_config,
        )
        frozen.load_actor(checkpoints["pretrain"])
        if frozen.residual_temporal_guard.enabled:
            raise ValueError("Stage F0 requires a stateless residual guard")
        anchor_name = str(
            runtime_config["residual_action"].get("base_policy", "mdl2")
        )
        anchor = get_heuristic_class(anchor_name)(
            env.observation_size,
            env.action_size,
            dict(
                runtime_config["residual_action"].get(
                    "base_policy_config", {}
                )
            ),
        )
        snapshots = collect_frozen_trajectory_snapshots(
            config,
            env=env,
            frozen=frozen,
            anchor=anchor,
            scenario_index=scenario_index,
        )
        for checkpoint_index, variant in enumerate(
            manifold["checkpoint_variants"]
        ):
            agent = get_agent_class(primary)(
                env.observation_size,
                env.action_size,
                runtime_config,
            )
            checkpoint_path = checkpoints[str(variant)]
            checkpoint = torch.load(
                checkpoint_path,
                map_location=agent.device,
                weights_only=False,
            )
            agent.load_actor(checkpoint_path)
            agent.critic.load_state_dict(checkpoint["critic"])
            agent.critic.eval()
            for snapshot in snapshots:
                rows = audit_snapshot_checkpoint(
                    config,
                    agent=agent,
                    snapshot=snapshot,
                    algorithm=primary,
                    training_seed=seed,
                    scenario=str(
                        config["scenario_by_training_seed"][str(seed)]
                    ),
                    scenario_index=scenario_index,
                    checkpoint_variant=str(variant),
                    checkpoint_index=checkpoint_index,
                )
                all_rows.extend(rows)
            del checkpoint
        provenance.append(
            {
                "algorithm": primary,
                "training_seed": seed,
                "scenario": str(
                    config["scenario_by_training_seed"][str(seed)]
                ),
                "runtime_config": str(config_path),
                "runtime_config_sha256": sha256_file(config_path),
                "checkpoints": {
                    key: {
                        "path": str(value),
                        "sha256": sha256_file(value),
                    }
                    for key, value in checkpoints.items()
                },
                "states": len(snapshots),
            }
        )
    return {
        "followup_policy": (
            "The same checkpoint actor being audited follows the first legal "
            "candidate action for each finite-horizon rollout."
        ),
        "state_distribution": (
            "Prespecified states on a fresh deterministic frozen-pretrain "
            "trajectory in each seed's locked persistent-hotspot scenario."
        ),
        "fresh_crn_seeds": sorted(manifold_crn_seeds(config)),
        "rows": len(all_rows),
        "provenance": provenance,
        "metrics": summarize_manifold_rows(
            all_rows,
            material_improvement=float(manifold["material_improvement"]),
        ),
    }, all_rows


def checkpoint_paths(
    run: dict[str, Any],
    summary: dict[str, Any],
    *,
    variants: tuple[str, ...],
    suffixes: dict[str, str],
    verifier: LockedArtifactVerifier,
) -> dict[str, Path]:
    checkpoint_dir = Path(run["checkpoint"]).parent
    algorithm = str(run["algorithm"])
    seed = int(run["seed"])
    result = {}
    for variant in variants:
        if variant == "pretrain":
            relative = Path(summary["pretrain_checkpoint"])
        elif variant == "final":
            relative = Path(run["checkpoint"])
        else:
            suffix = str(suffixes[variant])
            relative = checkpoint_dir / f"{algorithm}_seed{seed}_{suffix}.pt"
        result[variant] = verifier.verify(relative)
    if "pretrain" not in result:
        result["pretrain"] = verifier.verify(summary["pretrain_checkpoint"])
    return result


def collect_frozen_trajectory_snapshots(
    config: dict[str, Any],
    *,
    env: Any,
    frozen: Any,
    anchor: Any,
    scenario_index: int,
) -> list[dict[str, Any]]:
    manifold = dict(config["manifold"])
    selected_steps = {int(value) for value in manifold["decision_steps"]}
    max_steps = int(config["replay"]["max_steps_per_episode"])
    option_specs = make_explicit_residual_option_specs(
        manifold["explicit_options"]
    )
    live_seed = int(manifold["live_seed"]) + scenario_index * 100_000
    state = env.reset(seed=live_seed)
    frozen.reset()
    anchor.reset()
    snapshots = []
    done = False
    step = 0
    while not done and step < max_steps:
        if step in selected_steps:
            anchor_action = np.asarray(
                anchor.select_action(state, explore=False, env=env),
                dtype=np.float32,
            )
            raw_actions = residual_option_actions_from_env(
                anchor_action,
                env,
                option_specs,
            )
            actions = [
                np.asarray(
                    project_action(
                        action,
                        env_state=env,
                        action_space_info=env.action_size,
                    ).action,
                    dtype=np.float32,
                )
                for action in raw_actions
            ]
            snapshots.append(
                {
                    "step": step,
                    "remaining_horizon": max_steps - step,
                    "state": np.asarray(state, dtype=np.float32).copy(),
                    "env": copy.deepcopy(env),
                    "labels": [option_label(option) for option in option_specs],
                    "actions": actions,
                }
            )
        action = frozen.select_action(state, explore=False, env=env)
        state, _reward, done, _info = env.step(action)
        step += 1
    observed = {int(snapshot["step"]) for snapshot in snapshots}
    if observed != selected_steps:
        raise ValueError(
            f"Frozen trajectory missed prespecified steps {sorted(selected_steps - observed)}"
        )
    return snapshots


def option_label(option: Any) -> str:
    if bool(option.is_anchor):
        return "mdl2"
    return f"{option.group}:{float(option.sign):+g}x{float(option.epsilon):g}"


def audit_snapshot_checkpoint(
    config: dict[str, Any],
    *,
    agent: Any,
    snapshot: dict[str, Any],
    algorithm: str,
    training_seed: int,
    scenario: str,
    scenario_index: int,
    checkpoint_variant: str,
    checkpoint_index: int,
) -> list[dict[str, Any]]:
    state = np.asarray(snapshot["state"], dtype=np.float32)
    actions = np.asarray(snapshot["actions"], dtype=np.float32)
    policy_action = np.asarray(
        agent.select_action(
            state,
            explore=False,
            env=snapshot["env"],
        ),
        dtype=np.float32,
    )
    critic = critic_legal_action_diagnostics(
        agent,
        state=state,
        actions=actions,
        policy_action=policy_action,
    )
    manifold = dict(config["manifold"])
    rows = []
    for horizon_index, configured_horizon in enumerate(manifold["horizons"]):
        horizon = (
            int(snapshot["remaining_horizon"])
            if configured_horizon == "remaining"
            else min(
                int(configured_horizon),
                int(snapshot["remaining_horizon"]),
            )
        )
        seed_start = (
            int(manifold["rollout_seed"])
            + scenario_index * 1_000_000
            + checkpoint_index * 100_000
            + int(snapshot["step"]) * 100
            + horizon_index * 10
        )
        rollout_seeds = tuple(
            seed_start + index
            for index in range(int(manifold["rollout_replications"]))
        )
        metrics = [
            mean_rollout_metrics_after_action(
                snapshot["env"],
                agent,
                action,
                horizon=horizon,
                rollout_seeds=rollout_seeds,
            )
            for action in actions
        ]
        anchor_cost = float(metrics[0]["total_cost"])
        for index, (label, action_metrics) in enumerate(
            zip(snapshot["labels"], metrics)
        ):
            rows.append(
                {
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "scenario": scenario,
                    "checkpoint_variant": checkpoint_variant,
                    "step": int(snapshot["step"]),
                    "configured_horizon": str(configured_horizon),
                    "horizon": horizon,
                    "rollout_seed_start": seed_start,
                    "rollout_replications": len(rollout_seeds),
                    "candidate_index": index,
                    "candidate_label": label,
                    "executed_action_id": critic["executed_action_ids"][index],
                    "executed_action_distinct": int(
                        critic["first_indices"][index] == index
                    ),
                    "total_cost": float(action_metrics["total_cost"]),
                    "true_cost_advantage": (
                        anchor_cost - float(action_metrics["total_cost"])
                    ),
                    "critic_q": float(critic["q_values"][index]),
                    "critic_advantage": float(
                        critic["q_values"][index] - critic["q_values"][0]
                    ),
                    "gradient_score": float(critic["gradient_scores"][index]),
                    "policy_action_linf_to_candidate": float(
                        np.max(np.abs(actions[index] - policy_action))
                    ),
                    "completion_service_level": float(
                        action_metrics["completion_service_level"]
                    ),
                    "patients_lost": float(action_metrics["patients_lost"]),
                    "patient_ineligibility_during_manufacturing_rate": float(
                        action_metrics[
                            "patient_ineligibility_during_manufacturing_rate"
                        ]
                    ),
                }
            )
    return rows


def critic_legal_action_diagnostics(
    agent: Any,
    *,
    state: np.ndarray,
    actions: np.ndarray,
    policy_action: np.ndarray,
) -> dict[str, Any]:
    action_array = np.asarray(actions, dtype=np.float32)
    raw_state = torch.as_tensor(
        np.asarray(state, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=agent.device,
    )
    action_tensor = torch.as_tensor(
        action_array,
        dtype=torch.float32,
        device=agent.device,
    )
    critic_states = _critic_state_batch(agent, raw_state, action_array.shape[0])
    with torch.no_grad():
        executed = agent._critic_actions_tensor(action_tensor)
        q_values = agent.critic(critic_states, executed).reshape(-1)

    policy_tensor = torch.as_tensor(
        np.asarray(policy_action, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=agent.device,
    ).requires_grad_(True)
    policy_state = _critic_state_batch(agent, raw_state, 1)
    policy_q = agent.critic(
        policy_state,
        agent._critic_actions_tensor(policy_tensor, straight_through=True),
    ).sum()
    gradient = torch.autograd.grad(policy_q, policy_tensor)[0]
    gradient_array = gradient.detach().cpu().numpy().reshape(-1)
    gradient_scores = (
        (action_array - np.asarray(policy_action, dtype=np.float32))
        @ gradient_array
    )
    executed_array = executed.detach().cpu().numpy()
    identifiers = [
        hashlib.sha256(
            np.round(value.astype(np.float64), decimals=7).tobytes()
        ).hexdigest()[:16]
        for value in executed_array
    ]
    first_indices: dict[str, int] = {}
    first_by_row = []
    for index, identifier in enumerate(identifiers):
        first_indices.setdefault(identifier, index)
        first_by_row.append(first_indices[identifier])
    return {
        "q_values": q_values.detach().cpu().numpy(),
        "gradient": gradient_array,
        "gradient_scores": np.asarray(gradient_scores, dtype=np.float64),
        "executed_action_ids": identifiers,
        "first_indices": first_by_row,
    }


def _critic_state_batch(agent: Any, raw_state: Any, count: int) -> Any:
    states = raw_state.repeat(int(count), 1)
    if hasattr(agent, "graph_spec"):
        return flat_state_to_node_features(states, agent.graph_spec)
    if agent.temporal_demand_encoder_enabled:
        return states
    return agent.observation_scaler.normalize_tensor(states)


def summarize_manifold_rows(
    rows: list[dict[str, Any]],
    *,
    material_improvement: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["algorithm"]),
            int(row["training_seed"]),
            str(row["checkpoint_variant"]),
            str(row["configured_horizon"]),
        )
        groups[key].append(row)
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for (algorithm, seed, checkpoint, horizon), group_rows in sorted(groups.items()):
        result[algorithm][str(seed)][checkpoint][horizon] = legal_ranking_metrics(
            group_rows,
            material_improvement=material_improvement,
        )
    return {
        algorithm: {
            seed: {checkpoint: dict(horizons) for checkpoint, horizons in checkpoints.items()}
            for seed, checkpoints in seeds.items()
        }
        for algorithm, seeds in result.items()
    }


def legal_ranking_metrics(
    rows: list[dict[str, Any]],
    *,
    material_improvement: float,
) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    pairwise_scores = []
    gradient_pairwise_scores = []
    top1 = []
    gradient_top1 = []
    headroom = []
    critic_selected_improvement = []
    gradient_selected_improvement = []
    distinct_counts = []
    true_values = []
    predicted_values = []
    gradient_values = []
    for step_rows in by_step.values():
        ordered = sorted(step_rows, key=lambda row: int(row["candidate_index"]))
        distinct = [row for row in ordered if int(row["executed_action_distinct"])]
        true = np.asarray(
            [float(row["true_cost_advantage"]) for row in distinct],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [float(row["critic_advantage"]) for row in distinct],
            dtype=np.float64,
        )
        gradient = np.asarray(
            [float(row["gradient_score"]) for row in distinct],
            dtype=np.float64,
        )
        distinct_counts.append(len(distinct))
        true_values.extend(true[1:])
        predicted_values.extend(predicted[1:])
        gradient_values.extend(gradient[1:])
        pairwise_scores.extend(pairwise_ranking_scores(true, predicted))
        gradient_pairwise_scores.extend(pairwise_ranking_scores(true, gradient))
        best_true = float(np.max(true))
        predicted_choice = int(np.argmax(predicted))
        gradient_choice = int(np.argmax(gradient))
        top1.append(
            float(true[predicted_choice] >= best_true - material_improvement)
        )
        gradient_top1.append(
            float(true[gradient_choice] >= best_true - material_improvement)
        )
        headroom.append(float(best_true > material_improvement))
        critic_selected_improvement.append(
            float(true[predicted_choice] > material_improvement)
        )
        gradient_selected_improvement.append(
            float(true[gradient_choice] > material_improvement)
        )
    true_array = np.asarray(true_values, dtype=np.float64)
    predicted_array = np.asarray(predicted_values, dtype=np.float64)
    gradient_array = np.asarray(gradient_values, dtype=np.float64)
    return {
        "states": len(by_step),
        "nonanchor_action_values": int(true_array.size),
        "mean_distinct_executed_actions": float(np.mean(distinct_counts)),
        "headroom_state_fraction": float(np.mean(headroom)),
        "critic_pairwise_accuracy": _mean_or_none(pairwise_scores),
        "critic_top1_accuracy": _mean_or_none(top1),
        "critic_selected_improving_action_fraction": _mean_or_none(
            critic_selected_improvement
        ),
        "critic_pearson": correlation(true_array, predicted_array),
        "critic_spearman": correlation(
            average_ranks(true_array), average_ranks(predicted_array)
        ),
        "gradient_pairwise_accuracy": _mean_or_none(gradient_pairwise_scores),
        "gradient_top1_accuracy": _mean_or_none(gradient_top1),
        "gradient_selected_improving_action_fraction": _mean_or_none(
            gradient_selected_improvement
        ),
        "gradient_pearson": correlation(true_array, gradient_array),
        "gradient_spearman": correlation(
            average_ranks(true_array), average_ranks(gradient_array)
        ),
        "true_advantage": numeric_summary(true_array),
        "critic_advantage": numeric_summary(predicted_array),
    }


def pairwise_ranking_scores(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> list[float]:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if target.shape != predicted.shape:
        raise ValueError("Pairwise ranking arrays must align")
    scores = []
    for left in range(target.size):
        for right in range(left + 1, target.size):
            target_delta = float(target[left] - target[right])
            if abs(target_delta) <= tolerance:
                continue
            prediction_delta = float(predicted[left] - predicted[right])
            product = target_delta * prediction_delta
            scores.append(1.0 if product > 0.0 else 0.5 if product == 0.0 else 0.0)
    return scores


def stage_f0_decision(
    config: dict[str, Any],
    *,
    replay_result: dict[str, Any] | None,
    manifold_result: dict[str, Any] | None,
    smoke: bool,
) -> dict[str, Any]:
    if smoke or replay_result is None or manifold_result is None:
        return {
            "classification": "incomplete_diagnostic_only",
            "f1_training_authorized": False,
            "reason": (
                "The full replay and manifold components are both required; "
                "a smoke run cannot pass the Stage F0 gate."
            ),
        }
    primary = str(config["primary_algorithm"])
    thresholds = dict(config["stage_f0_gate"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    replay_mismatch = []
    critic_mismatch = []
    headroom = []
    degraded = []
    for seed in seeds:
        replay_seed = replay_result["per_seed"][primary][str(seed)]
        negative_rate = replay_seed["self_imitation_active_remainder"][
            "negative_fraction"
        ]
        if (
            negative_rate is not None
            and negative_rate
            >= float(thresholds["minimum_active_remainder_conflict_rate"])
        ):
            replay_mismatch.append(seed)
        final_difference = replay_seed["final_vs_frozen_total_cost_difference"]
        if final_difference is not None and final_difference > 0.0:
            degraded.append(seed)
        final_remaining = manifold_result["metrics"][primary][str(seed)][
            "final"
        ]["remaining"]
        if (
            float(final_remaining["critic_pairwise_accuracy"] or 0.0)
            <= float(thresholds["maximum_critic_pairwise_accuracy"])
            or float(final_remaining["gradient_top1_accuracy"] or 0.0)
            <= float(thresholds["maximum_gradient_top1_accuracy"])
        ):
            critic_mismatch.append(seed)
        pretrain_remaining = manifold_result["metrics"][primary][str(seed)][
            "pretrain"
        ]["remaining"]
        if float(pretrain_remaining["headroom_state_fraction"]) >= float(
            thresholds["minimum_headroom_state_fraction"]
        ):
            headroom.append(seed)
    replay_branch = sorted(set(replay_mismatch) & set(headroom))
    critic_branch = sorted(set(critic_mismatch) & set(headroom))
    actionable = sorted(set(replay_branch) | set(critic_branch))
    conjunctive = sorted(set(replay_branch) & set(critic_branch))
    linked = sorted(set(actionable) & set(degraded))
    minimum_seeds = int(thresholds["minimum_reproducible_seed_count"])
    gate_passed = len(actionable) >= minimum_seeds and bool(linked)
    return {
        "gate_logic": (
            "Alternative mechanism branches: a reproducible temporal-target "
            "mismatch or legal-action ranking mismatch may pass when headroom "
            "and a link to final-versus-frozen degradation are also present."
        ),
        "classification": (
            "stage_f1_single_candidate_design_justified"
            if gate_passed
            else "close_online_ddpg_extension"
        ),
        "replay_mismatch_seeds": replay_mismatch,
        "critic_or_gradient_mismatch_seeds": critic_mismatch,
        "headroom_seeds": headroom,
        "final_vs_frozen_degraded_seeds": degraded,
        "replay_target_branch_seeds": replay_branch,
        "legal_action_ranking_branch_seeds": critic_branch,
        "actionable_reproducible_mismatch_seeds": actionable,
        "conjunctive_mismatch_seeds": conjunctive,
        "mismatch_linked_to_degradation_seeds": linked,
        "gate_passed": gate_passed,
        "prespecified_single_correction": (
            "Train the DDPG critic on a directly paired, finite-horizon "
            "counterfactual advantage between executed legal specimen actions "
            "and MDL-2, without changing actor, exploration, gate, scale, "
            "scenario distribution, or episode budget."
        ),
        "f1_training_authorized": False,
        "next_action": (
            "Freeze and review one paired Stage F1 candidate protocol."
            if gate_passed
            else "Do not run another online DDPG training experiment."
        ),
        "authorization_limit": (
            "A passing F0 audit justifies protocol design only. It does not "
            "launch training or open a formal holdout."
        ),
    }


def reclassify_summary(
    config: dict[str, Any],
    *,
    summary_path: Path,
) -> dict[str, Any]:
    """Recompute a reporting-only decision without rerunning any rollout."""

    result = _load_json(summary_path)
    result["decision"] = stage_f0_decision(
        config,
        replay_result=result.get("replay"),
        manifold_result=result.get("manifold"),
        smoke=bool(result.get("smoke", False)),
    )
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def final_vs_frozen_cost_difference(
    comparison: dict[str, Any],
    algorithm: str,
    seed: int,
) -> float | None:
    try:
        value = comparison["roles"]["control"][algorithm]["final_vs_frozen"][
            "total_cost"
        ]["per_seed_mean_difference"][str(seed)]
    except KeyError:
        return None
    return float(value)


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"samples": 0}
    if not np.all(np.isfinite(array)):
        raise ValueError("Numeric summary requires finite values")
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def signed_classes(values: np.ndarray, *, tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(array > tolerance, 1, np.where(array < -tolerance, -1, 0))


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.shape != right_array.shape or left_array.size < 2:
        return None
    if float(left_array.std()) <= 1e-15 or float(right_array.std()) <= 1e-15:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _manifest_runs(
    manifest: dict[str, Any],
    *,
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[str, int], dict[str, Any]]:
    result = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    expected = {(algorithm, seed) for algorithm in algorithms for seed in seeds}
    missing = sorted(expected - set(result))
    if missing:
        raise ValueError(f"Stage F0 training manifest is missing {missing}")
    return {key: result[key] for key in expected}


def _load_locked_compact_json(
    source_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    resolved = _resolve(source_root, path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != str(expected_sha256).lower():
        raise ValueError(
            f"Compact evidence SHA256 mismatch for {resolved}: "
            f"expected {expected_sha256}, got {observed}"
        )
    return _load_json(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _fraction(values: np.ndarray) -> float | None:
    array = np.asarray(values)
    return None if array.size == 0 else float(np.mean(array))


def _mean_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    return None if array.size == 0 else float(array.mean())


if __name__ == "__main__":
    main()
