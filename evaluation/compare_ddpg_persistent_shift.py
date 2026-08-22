"""Paired Stage C2 comparison for persistent-shift DDPG attribution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate_stats import paired_two_level_summary
from src.rl.config import load_config


PAIRING_KEYS = (
    "training_seed",
    "scenario",
    "evaluation_seed",
    "replication",
)
METRICS = (
    "total_cost",
    "completion_service_level",
    "patients_lost",
    "patient_ineligibility_during_manufacturing_rate",
)
VARIANT_EPISODES = {
    "pretrain": 0,
    "episode10": 10,
    "episode25": 25,
    "episode50": 50,
    "episode75": 75,
    "final": 100,
}


def _enable_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_rows(path: Path) -> list[dict[str, str]]:
    _enable_large_csv_fields()
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_key(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in PAIRING_KEYS)
        if key in result:
            raise ValueError(f"Duplicate Stage C2 CRN key: {key}")
        result[key] = row
    return result


def _metric_bundle(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: paired_two_level_summary(
            candidate,
            baseline,
            metric=metric,
            pairing_keys=PAIRING_KEYS,
            resamples=resamples,
            seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }


def _clinical_noninferiority(
    bundle: dict[str, Any],
    margins: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "completion_service_level": float(
            bundle["completion_service_level"]["ci_low"]
        )
        >= -float(margins.get("completion_service_level", 0.001)),
        "patients_lost": float(bundle["patients_lost"]["ci_high"])
        <= float(margins.get("patients_lost", 1.0)),
        "patient_ineligibility_during_manufacturing_rate": float(
            bundle[
                "patient_ineligibility_during_manufacturing_rate"
            ]["ci_high"]
        )
        <= float(
            margins.get(
                "patient_ineligibility_during_manufacturing_rate",
                0.001,
            )
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _curve_auc_rows(
    rows_by_variant: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keyed = {
        variant: _rows_by_key(rows)
        for variant, rows in rows_by_variant.items()
    }
    reference_keys = keyed["pretrain"].keys()
    if any(rows.keys() != reference_keys for rows in keyed.values()):
        raise ValueError("Checkpoint-curve CRN keys differ")
    variants = sorted(
        rows_by_variant,
        key=lambda value: VARIANT_EPISODES[value],
    )
    candidate_rows = []
    baseline_rows = []
    for key in sorted(reference_keys):
        candidate = {
            field: value
            for field, value in zip(PAIRING_KEYS, key)
        }
        baseline = dict(candidate)
        costs = [
            float(keyed[variant][key]["total_cost"])
            for variant in variants
        ]
        episodes = [VARIANT_EPISODES[variant] for variant in variants]
        area = sum(
            (episodes[index + 1] - episodes[index])
            * (costs[index + 1] + costs[index])
            / 2.0
            for index in range(len(episodes) - 1)
        )
        candidate["total_cost"] = area / float(episodes[-1])
        baseline["total_cost"] = float(
            keyed["pretrain"][key]["total_cost"]
        )
        candidate_rows.append(candidate)
        baseline_rows.append(baseline)
    return candidate_rows, baseline_rows


def _load_role_rows(
    config: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    root = Path(config["output_root"])
    algorithms = tuple(str(value) for value in config["algorithms"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config["scenario_by_training_seed"].items()
    }
    replications = int(config["holdout_replications"])
    holdout_seed = int(config["holdout_seed"])
    result = {
        variant: {algorithm: [] for algorithm in algorithms}
        for variant in config["checkpoint_variants"]
    }
    for variant in result:
        for algorithm in algorithms:
            for seed in seeds:
                path = (
                    root
                    / variant
                    / algorithm
                    / f"seed{seed}"
                    / "holdout_rows.csv"
                )
                rows = _read_rows(path)
                if len(rows) != replications:
                    raise ValueError(
                        f"{variant} {algorithm} seed {seed} has "
                        f"{len(rows)} rows, expected {replications}"
                    )
                keys = {
                    (
                        int(row["evaluation_seed"]),
                        int(row["replication"]),
                        str(row["scenario"]),
                    )
                    for row in rows
                }
                expected = {
                    (holdout_seed, replication, assignment[seed])
                    for replication in range(replications)
                }
                if keys != expected:
                    raise ValueError(
                        f"{variant} {algorithm} seed {seed} CRN/scenario "
                        "contract mismatch"
                    )
                result[variant][algorithm].extend(rows)
    return result


def _load_role_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    algorithms = tuple(str(value) for value in config["algorithms"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    expected = {
        (algorithm, seed)
        for algorithm in algorithms
        for seed in seeds
    }
    manifest = load_config(config["training_manifest"])
    manifest_runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    if set(manifest_runs) != expected:
        raise ValueError("Stage C2 diagnostic training manifest is incomplete")
    actor_drift = {
        algorithm: {
            str(seed): dict(
                manifest_runs[(algorithm, seed)]["actor_drift_from_pretrain"]
            )
            for seed in seeds
        }
        for algorithm in algorithms
    }

    output_root = Path(config["output_root"])
    residual_usage = {}
    for variant in config["checkpoint_variants"]:
        residual_usage[str(variant)] = {}
        for algorithm in algorithms:
            residual_usage[str(variant)][algorithm] = {}
            for seed in seeds:
                summary = load_config(
                    output_root
                    / str(variant)
                    / algorithm
                    / f"seed{seed}"
                    / "summary.json"
                )
                residual_usage[str(variant)][algorithm][str(seed)] = dict(
                    summary["holdout"]["aggregate"]["residual_usage"]
                )
    return {
        "actor_drift_from_pretrain": actor_drift,
        "residual_usage": residual_usage,
    }


def _validate_matching_configs(
    control: dict[str, Any],
    realigned: dict[str, Any],
) -> None:
    matched_keys = (
        "algorithms",
        "training_seeds",
        "checkpoint_variants",
        "scenarios",
        "scenario_by_training_seed",
        "validation_seed",
        "holdout_seed",
        "holdout_replications",
        "max_steps",
        "fixed_deployment_candidate",
        "clinical_noninferiority",
    )
    mismatched = [
        key
        for key in matched_keys
        if control.get(key) != realigned.get(key)
    ]
    if mismatched:
        raise ValueError(
            "Control/realigned evaluation contracts differ: "
            + ", ".join(mismatched)
        )
    variants = list(control["checkpoint_variants"])
    if variants != list(VARIANT_EPISODES):
        raise ValueError("Stage C2 checkpoint variants do not match the lock")


def _advancement_gate(
    bundle: dict[str, Any],
    clinical: dict[str, Any],
    *,
    minimum_improvement_pct: float,
) -> dict[str, Any]:
    cost = bundle["total_cost"]
    per_seed = {
        str(seed): float(value)
        for seed, value in cost["per_seed_mean_difference"].items()
    }
    favorable_seeds = sum(value < 0.0 for value in per_seed.values())
    effect_passed = (
        float(cost["mean_gap_pct"]) <= -minimum_improvement_pct
        or float(cost["ci_high"]) < 0.0
    )
    seed_passed = favorable_seeds >= 2
    passed = bool(effect_passed and seed_passed and clinical["passed"])
    return {
        "passed": passed,
        "effect_passed": effect_passed,
        "seed_direction_passed": seed_passed,
        "clinical_noninferiority_passed": bool(clinical["passed"]),
        "minimum_improvement_pct": float(minimum_improvement_pct),
        "favorable_seed_count": favorable_seeds,
        "required_favorable_seed_count": 2,
        "per_seed_mean_cost_difference": per_seed,
    }


def compare_persistent_shift(
    control_config_path: Path,
    realigned_config_path: Path,
    *,
    output_path: Path,
    resamples: int = 20_000,
    bootstrap_seed: int = 95_350_000,
    minimum_improvement_pct: float = 0.02,
) -> dict[str, Any]:
    control_config = load_config(control_config_path)
    realigned_config = load_config(realigned_config_path)
    _validate_matching_configs(control_config, realigned_config)
    role_rows = {
        "control": _load_role_rows(control_config),
        "realigned": _load_role_rows(realigned_config),
    }
    diagnostics = {
        "control": _load_role_diagnostics(control_config),
        "realigned": _load_role_diagnostics(realigned_config),
    }
    algorithms = tuple(str(value) for value in control_config["algorithms"])
    margins = dict(
        control_config["clinical_noninferiority"]["margins"]
    )
    roles: dict[str, Any] = {}
    for role_index, (role, rows) in enumerate(role_rows.items()):
        roles[role] = {}
        for algorithm_index, algorithm in enumerate(algorithms):
            frozen = rows["pretrain"][algorithm]
            checkpoint_vs_frozen = {
                variant: _metric_bundle(
                    rows[variant][algorithm],
                    frozen,
                    resamples=resamples,
                    seed=(
                        bootstrap_seed
                        + role_index * 100_000
                        + algorithm_index * 10_000
                        + variant_index * 100
                    ),
                )
                for variant_index, variant in enumerate(VARIANT_EPISODES)
            }
            final_bundle = checkpoint_vs_frozen["final"]
            clinical = _clinical_noninferiority(final_bundle, margins)
            auc_candidate, auc_baseline = _curve_auc_rows(
                {
                    variant: rows[variant][algorithm]
                    for variant in VARIANT_EPISODES
                }
            )
            auc = paired_two_level_summary(
                auc_candidate,
                auc_baseline,
                metric="total_cost",
                pairing_keys=PAIRING_KEYS,
                resamples=resamples,
                seed=(
                    bootstrap_seed
                    + role_index * 100_000
                    + algorithm_index * 10_000
                    + 9_000
                ),
            )
            roles[role][algorithm] = {
                "checkpoint_vs_frozen": checkpoint_vs_frozen,
                "checkpoint_curve_adaptation_auc_vs_frozen": auc,
                "final_vs_frozen": final_bundle,
                "clinical_noninferiority": clinical,
                "advancement_gate": _advancement_gate(
                    final_bundle,
                    clinical,
                    minimum_improvement_pct=minimum_improvement_pct,
                ),
            }

    realigned_vs_control = {}
    for algorithm_index, algorithm in enumerate(algorithms):
        bundle = _metric_bundle(
            role_rows["realigned"]["final"][algorithm],
            role_rows["control"]["final"][algorithm],
            resamples=resamples,
            seed=bootstrap_seed + 500_000 + algorithm_index * 10_000,
        )
        realigned_vs_control[algorithm] = {
            "final": bundle,
            "clinical_noninferiority": _clinical_noninferiority(
                bundle,
                margins,
            ),
        }

    primary_algorithm = "gcn_residual_mdl2_network_ddpg_afd"
    control_passed = bool(
        roles["control"][primary_algorithm]["advancement_gate"]["passed"]
    )
    realigned_base_passed = bool(
        roles["realigned"][primary_algorithm]["advancement_gate"]["passed"]
    )
    realigned_no_worse = float(
        realigned_vs_control[primary_algorithm]["final"]["total_cost"][
            "mean_difference"
        ]
    ) <= 0.0
    realigned_passed = bool(realigned_base_passed and realigned_no_worse)
    if realigned_passed:
        decision = "advance_realigned_ddpg_to_fresh_confirmation"
    elif control_passed:
        decision = "advance_standard_ddpg_to_fresh_confirmation"
    else:
        decision = "close_persistent_shift_ddpg_candidate"

    payload = {
        "experimental_role": "Stage C2 development evidence only",
        "control_config": str(control_config_path),
        "realigned_config": str(realigned_config_path),
        "resamples": int(resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "roles": roles,
        "diagnostics": diagnostics,
        "realigned_vs_control": realigned_vs_control,
        "decision": {
            "classification": decision,
            "primary_algorithm": primary_algorithm,
            "control_gate_passed": control_passed,
            "realigned_final_vs_frozen_gate_passed": realigned_base_passed,
            "realigned_no_worse_than_control": realigned_no_worse,
            "realigned_gate_passed": realigned_passed,
            "formal_confirmation_launched": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--realigned-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=95_350_000)
    parser.add_argument("--minimum-improvement-pct", type=float, default=0.02)
    args = parser.parse_args()
    result = compare_persistent_shift(
        Path(args.control_config),
        Path(args.realigned_config),
        output_path=Path(args.output),
        resamples=int(args.resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        minimum_improvement_pct=float(args.minimum_improvement_pct),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
