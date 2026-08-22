"""Paired Stage C3 comparison for structured DDPG behavior exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.compare_ddpg_persistent_shift import (
    PAIRING_KEYS,
    VARIANT_EPISODES,
    _advancement_gate,
    _clinical_noninferiority,
    _curve_auc_rows,
    _load_role_diagnostics,
    _load_role_rows,
    _metric_bundle,
)
from src.rl.config import load_config


PRIMARY_ALGORITHM = "gcn_residual_mdl2_network_ddpg_afd"


def compare_structured_exploration(
    control_config_path: Path,
    candidate_config_path: Path,
    *,
    output_path: Path,
    resamples: int = 20_000,
    bootstrap_seed: int = 95_850_000,
    minimum_improvement_pct: float = 0.02,
) -> dict[str, Any]:
    control_config = load_config(control_config_path)
    candidate_config = load_config(candidate_config_path)
    validate_matching_evaluation_contracts(
        control_config,
        candidate_config,
    )
    role_rows = {
        "control": _load_role_rows(control_config),
        "candidate": _load_role_rows(candidate_config),
    }
    diagnostics = {
        "control": _structured_role_diagnostics(control_config),
        "candidate": _structured_role_diagnostics(candidate_config),
    }
    algorithms = tuple(str(value) for value in control_config["algorithms"])
    margins = dict(control_config["clinical_noninferiority"]["margins"])
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
            "cost_no_regression": mean_cost_difference <= 0.0,
            "no_regression": bool(
                mean_cost_difference <= 0.0 and clinical["passed"]
            ),
        }

    control_passed = bool(
        roles["control"][PRIMARY_ALGORITHM]["advancement_gate"]["passed"]
    )
    candidate_base_passed = bool(
        roles["candidate"][PRIMARY_ALGORITHM]["advancement_gate"]["passed"]
    )
    candidate_no_regression = bool(
        candidate_vs_control[PRIMARY_ALGORITHM]["no_regression"]
    )
    candidate_passed = bool(
        candidate_base_passed and candidate_no_regression
    )
    classification = (
        "advance_structured_exploration_ddpg_to_fresh_confirmation"
        if candidate_passed
        else "close_structured_exploration_ddpg_candidate"
    )
    payload = {
        "experimental_role": "Stage C3 development evidence only",
        "control_config": str(control_config_path),
        "candidate_config": str(candidate_config_path),
        "resamples": int(resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "minimum_improvement_pct": float(minimum_improvement_pct),
        "roles": roles,
        "diagnostics": diagnostics,
        "candidate_vs_control": candidate_vs_control,
        "decision": {
            "classification": classification,
            "primary_algorithm": PRIMARY_ALGORITHM,
            "control_gate_passed": control_passed,
            "candidate_final_vs_frozen_gate_passed": (
                candidate_base_passed
            ),
            "candidate_no_regression_vs_control": (
                candidate_no_regression
            ),
            "candidate_gate_passed": candidate_passed,
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
        raise ValueError("Stage C3 checkpoint variants do not match the lock")


def _structured_role_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _load_role_diagnostics(config)
    manifest = load_config(config["training_manifest"])
    diagnostics["structured_specimen_exploration"] = {
        str(run["algorithm"]): {
            str(seed_run["seed"]): dict(
                seed_run.get("structured_specimen_exploration", {})
            )
            for seed_run in manifest["runs"]
            if str(seed_run["algorithm"]) == str(run["algorithm"])
        }
        for run in manifest["runs"]
    }
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=95_850_000)
    parser.add_argument("--minimum-improvement-pct", type=float, default=0.02)
    args = parser.parse_args()
    result = compare_structured_exploration(
        Path(args.control_config),
        Path(args.candidate_config),
        output_path=Path(args.output),
        resamples=int(args.resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        minimum_improvement_pct=float(args.minimum_improvement_pct),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
