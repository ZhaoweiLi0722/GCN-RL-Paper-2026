"""Prospective paired comparison for the Stage F1 online DDPG experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.compare_ddpg_persistent_shift import (
    METRICS,
    PAIRING_KEYS,
    VARIANT_EPISODES,
    _clinical_noninferiority,
    _curve_auc_rows,
    _load_role_diagnostics,
    _load_role_rows,
    _metric_bundle,
    _rows_by_key,
)
from src.rl.config import load_config


PRIMARY_ALGORITHM = "gcn_residual_mdl2_network_ddpg_afd"


def _strict_online_gate(
    bundle: dict[str, Any],
    clinical: dict[str, Any],
) -> dict[str, Any]:
    cost = dict(bundle["total_cost"])
    per_seed = {
        str(seed): float(value)
        for seed, value in cost["per_seed_mean_difference"].items()
    }
    favorable_seeds = sum(value < 0.0 for value in per_seed.values())
    interval_passed = float(cost["ci_high"]) < 0.0
    seed_passed = favorable_seeds >= 2
    passed = bool(interval_passed and seed_passed and clinical["passed"])
    return {
        "passed": passed,
        "paired_cost_interval_wholly_below_zero": interval_passed,
        "seed_direction_passed": seed_passed,
        "clinical_noninferiority_passed": bool(clinical["passed"]),
        "favorable_seed_count": favorable_seeds,
        "required_favorable_seed_count": 2,
        "per_seed_mean_cost_difference": per_seed,
    }


def _assert_matched_frozen_rows(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    control_by_key = _rows_by_key(control)
    candidate_by_key = _rows_by_key(candidate)
    if control_by_key.keys() != candidate_by_key.keys():
        raise ValueError("Control/candidate frozen CRN keys differ")
    maximum_gap = 0.0
    for key in control_by_key:
        for metric in METRICS:
            gap = abs(
                float(control_by_key[key][metric])
                - float(candidate_by_key[key][metric])
            )
            maximum_gap = max(maximum_gap, gap)
            if gap != 0.0:
                raise ValueError(
                    "Control/candidate frozen evaluations are not identical: "
                    f"{key} {metric} gap={gap}"
                )
    return {
        "rows": len(control_by_key),
        "metrics": list(METRICS),
        "maximum_absolute_gap": maximum_gap,
        "exact": True,
    }


def compare_online_paired_advantage(
    control_config_path: Path,
    candidate_config_path: Path,
    *,
    output_path: Path,
    resamples: int = 20_000,
    bootstrap_seed: int = 96_750_000,
) -> dict[str, Any]:
    control_config = load_config(control_config_path)
    candidate_config = load_config(candidate_config_path)
    validate_matching_evaluation_contracts(control_config, candidate_config)
    role_rows = {
        "control": _load_role_rows(control_config),
        "candidate": _load_role_rows(candidate_config),
    }
    diagnostics = {
        "control": _load_role_diagnostics(control_config),
        "candidate": _load_role_diagnostics(candidate_config),
    }
    algorithms = tuple(str(value) for value in control_config["algorithms"])
    margins = dict(control_config["clinical_noninferiority"]["margins"])
    frozen_identity = {
        algorithm: _assert_matched_frozen_rows(
            role_rows["control"]["pretrain"][algorithm],
            role_rows["candidate"]["pretrain"][algorithm],
        )
        for algorithm in algorithms
    }

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
                "strict_online_gate": _strict_online_gate(
                    final_bundle,
                    clinical,
                ),
            }

    candidate_vs_control = {}
    for algorithm_index, algorithm in enumerate(algorithms):
        bundle = _metric_bundle(
            role_rows["candidate"]["final"][algorithm],
            role_rows["control"]["final"][algorithm],
            resamples=resamples,
            seed=bootstrap_seed + 500_000 + algorithm_index * 10_000,
        )
        clinical = _clinical_noninferiority(bundle, margins)
        mean_cost_difference = float(
            bundle["total_cost"]["mean_difference"]
        )
        candidate_vs_control[algorithm] = {
            "final": bundle,
            "clinical_noninferiority": clinical,
            "descriptive_cost_no_regression": mean_cost_difference <= 0.0,
            "no_regression": bool(
                mean_cost_difference <= 0.0 and clinical["passed"]
            ),
        }

    primary_gate = bool(
        roles["candidate"][PRIMARY_ALGORITHM]["strict_online_gate"]["passed"]
    )
    no_regression = bool(
        candidate_vs_control[PRIMARY_ALGORITHM]["no_regression"]
    )
    passed = bool(primary_gate and no_regression)
    payload = {
        "experimental_role": "Stage F1 development evidence only",
        "control_config": str(control_config_path),
        "candidate_config": str(candidate_config_path),
        "resamples": int(resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "frozen_identity": frozen_identity,
        "roles": roles,
        "diagnostics": diagnostics,
        "candidate_vs_control": candidate_vs_control,
        "decision": {
            "classification": (
                "advance_paired_online_ddpg_to_fresh_confirmation"
                if passed
                else "close_online_ddpg_attribution_extension"
            ),
            "primary_algorithm": PRIMARY_ALGORITHM,
            "candidate_final_vs_frozen_gate_passed": primary_gate,
            "candidate_no_regression_vs_control": no_regression,
            "candidate_gate_passed": passed,
            "formal_confirmation_launched": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_matching_evaluation_contracts(
    control: dict[str, Any],
    candidate: dict[str, Any],
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
        key for key in matched_keys if control.get(key) != candidate.get(key)
    ]
    if mismatched:
        raise ValueError(
            "Control/candidate evaluation contracts differ: "
            + ", ".join(mismatched)
        )
    if list(control["checkpoint_variants"]) != list(VARIANT_EPISODES):
        raise ValueError("Stage F1 checkpoint variants do not match the lock")
    if int(control["validation_seed"]) != 96_600_000:
        raise ValueError("Stage F1 execution-only evaluation seed changed")
    if int(control["holdout_seed"]) != 96_700_000:
        raise ValueError("Stage F1 development evaluation seed changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=96_750_000)
    args = parser.parse_args()
    result = compare_online_paired_advantage(
        Path(args.control_config),
        Path(args.candidate_config),
        output_path=Path(args.output),
        resamples=int(args.resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
