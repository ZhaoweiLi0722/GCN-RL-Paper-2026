"""Paired development comparison for the DDPG support-alignment experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate_stats import paired_two_level_summary


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-final", required=True)
    parser.add_argument("--control-pretrain", required=True)
    parser.add_argument("--candidate-final", required=True)
    parser.add_argument("--candidate-pretrain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=95_200_000)
    args = parser.parse_args()

    result = compare_support_alignment(
        control_final_path=Path(args.control_final),
        control_pretrain_path=Path(args.control_pretrain),
        candidate_final_path=Path(args.candidate_final),
        candidate_pretrain_path=Path(args.candidate_pretrain),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def compare_support_alignment(
    *,
    control_final_path: Path,
    control_pretrain_path: Path,
    candidate_final_path: Path,
    candidate_pretrain_path: Path,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 95_200_000,
) -> dict[str, Any]:
    """Compare matched control/candidate policies on one fresh CRN stream."""

    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    paths = {
        "control_final": control_final_path,
        "control_pretrain": control_pretrain_path,
        "candidate_final": candidate_final_path,
        "candidate_pretrain": candidate_pretrain_path,
    }
    summaries = {name: _read_json(path) for name, path in paths.items()}
    protocol = _validate_protocol(summaries)
    rows = {
        name: _load_learned_rows(paths[name], summaries[name])
        for name in paths
    }
    expected_keys = set(rows["control_final"])
    if any(set(stage_rows) != expected_keys for stage_rows in rows.values()):
        raise ValueError("Control/candidate evaluation runs are not matched")

    margins = dict(
        summaries["control_final"]
        .get("clinical_noninferiority", {})
        .get("margins", {})
    )
    comparisons: dict[str, Any] = {}
    classifications: dict[str, Any] = {}
    algorithms = sorted({algorithm for algorithm, _seed in expected_keys})
    for algorithm_index, algorithm in enumerate(algorithms):
        selected = {
            name: _combine_algorithm(stage_rows, algorithm)
            for name, stage_rows in rows.items()
        }
        offset = bootstrap_seed + algorithm_index * 100_000
        candidate_vs_control = _metric_bundle(
            selected["candidate_final"],
            selected["control_final"],
            resamples=bootstrap_resamples,
            seed=offset,
        )
        candidate_online_gain = _metric_bundle(
            selected["candidate_final"],
            selected["candidate_pretrain"],
            resamples=bootstrap_resamples,
            seed=offset + 10_000,
        )
        control_online_gain = _metric_bundle(
            selected["control_final"],
            selected["control_pretrain"],
            resamples=bootstrap_resamples,
            seed=offset + 20_000,
        )
        pretrain_balance = _metric_bundle(
            selected["candidate_pretrain"],
            selected["control_pretrain"],
            resamples=bootstrap_resamples,
            seed=offset + 30_000,
        )
        difference_in_differences = _difference_in_differences(
            candidate_final=selected["candidate_final"],
            candidate_pretrain=selected["candidate_pretrain"],
            control_final=selected["control_final"],
            control_pretrain=selected["control_pretrain"],
            resamples=bootstrap_resamples,
            seed=offset + 40_000,
        )
        candidate_clinical = _clinical_noninferiority(
            candidate_vs_control,
            margins,
        )
        online_clinical = _clinical_noninferiority(
            candidate_online_gain,
            margins,
        )
        comparisons[algorithm] = {
            "candidate_final_vs_control_final": candidate_vs_control,
            "candidate_final_vs_candidate_pretrain": candidate_online_gain,
            "control_final_vs_control_pretrain": control_online_gain,
            "candidate_pretrain_vs_control_pretrain": pretrain_balance,
            "online_gain_difference_in_differences": (
                difference_in_differences
            ),
            "candidate_final_vs_control_clinical_noninferiority": (
                candidate_clinical
            ),
            "candidate_online_gain_clinical_noninferiority": online_clinical,
            "candidate_final_vs_control_per_scenario": _per_scenario(
                selected["candidate_final"],
                selected["control_final"],
                resamples=bootstrap_resamples,
                seed=offset + 50_000,
            ),
        }
        classifications[algorithm] = _classify(
            candidate_vs_control=candidate_vs_control,
            candidate_online_gain=candidate_online_gain,
            candidate_clinical=candidate_clinical,
            online_clinical=online_clinical,
        )

    return {
        "experimental_role": (
            "fresh-seed development evidence; not formal confirmation"
        ),
        "input_summaries": {name: str(path) for name, path in paths.items()},
        "protocol": protocol,
        "pairing_keys": list(PAIRING_KEYS),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "comparisons": comparisons,
        "classifications": classifications,
        "decision": {
            "advance_to_fresh_formal_confirmation": all(
                result["classification"] == "pass"
                for result in classifications.values()
            ),
            "requires_human_review": True,
            "no_automatic_checkpoint_or_hyperparameter_selection": True,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_protocol(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference = summaries["control_final"]
    equal_fields = (
        "scenarios",
        "holdout_replications",
        "holdout_seed",
        "algorithms",
        "training_seeds",
        "fixed_deployment_candidate",
        "clinical_noninferiority",
    )
    for name, summary in summaries.items():
        for field in equal_fields:
            if summary.get(field) != reference.get(field):
                raise ValueError(f"Protocol mismatch for {name}: {field}")
    for name in ("control_final", "candidate_final"):
        if summaries[name].get("fixed_checkpoint_variant") != "final":
            raise ValueError(f"{name} must evaluate final checkpoints")
    for name in ("control_pretrain", "candidate_pretrain"):
        if summaries[name].get("fixed_checkpoint_variant") != "pretrain":
            raise ValueError(f"{name} must evaluate pretrain checkpoints")
    return {field: reference.get(field) for field in equal_fields}


def _load_learned_rows(
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[tuple[str, int], list[dict[str, str]]]:
    result: dict[tuple[str, int], list[dict[str, str]]] = {}
    for run in summary["runs"]:
        algorithm = str(run["algorithm"])
        seed = int(run["training_seed"])
        key = (algorithm, seed)
        if key in result:
            raise ValueError(f"Duplicate evaluation run: {key}")
        path = summary_path.parent / algorithm / f"seed{seed}" / "holdout_rows.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            result[key] = list(csv.DictReader(handle))
    return result


def _combine_algorithm(
    runs: dict[tuple[str, int], list[dict[str, str]]],
    algorithm: str,
) -> list[dict[str, str]]:
    return [
        row
        for (name, _seed), run_rows in sorted(runs.items())
        if name == algorithm
        for row in run_rows
    ]


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


def _difference_in_differences(
    *,
    candidate_final: list[dict[str, Any]],
    candidate_pretrain: list[dict[str, Any]],
    control_final: list[dict[str, Any]],
    control_pretrain: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_delta = _delta_rows(candidate_final, candidate_pretrain)
    control_delta = _delta_rows(control_final, control_pretrain)
    return _metric_bundle(
        candidate_delta,
        control_delta,
        resamples=resamples,
        seed=seed,
    )


def _delta_rows(
    final_rows: Iterable[dict[str, Any]],
    pretrain_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    final = _rows_by_key(final_rows)
    pretrain = _rows_by_key(pretrain_rows)
    if final.keys() != pretrain.keys():
        raise ValueError("Final/pretrain CRN keys differ")
    result = []
    for key in sorted(final):
        row = {name: value for name, value in zip(PAIRING_KEYS, key)}
        for metric in METRICS:
            row[metric] = float(final[key][metric]) - float(
                pretrain[key][metric]
            )
        result.append(row)
    return result


def _rows_by_key(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in PAIRING_KEYS)
        if key in result:
            raise ValueError(f"Duplicate CRN key: {key}")
        result[key] = row
    return result


def _clinical_noninferiority(
    bundle: dict[str, Any],
    margins: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "completion_service_level": float(
            bundle["completion_service_level"]["ci_low"]
        ) >= -float(margins.get("completion_service_level", 0.001)),
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


def _per_scenario(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    scenarios = sorted({str(row["scenario"]) for row in candidate})
    return {
        scenario: _metric_bundle(
            [row for row in candidate if str(row["scenario"]) == scenario],
            [row for row in baseline if str(row["scenario"]) == scenario],
            resamples=resamples,
            seed=seed + index * 1_000,
        )
        for index, scenario in enumerate(scenarios)
    }


def _classify(
    *,
    candidate_vs_control: dict[str, Any],
    candidate_online_gain: dict[str, Any],
    candidate_clinical: dict[str, Any],
    online_clinical: dict[str, Any],
) -> dict[str, Any]:
    control_cost = candidate_vs_control["total_cost"]
    online_cost = candidate_online_gain["total_cost"]
    means_improve = (
        float(control_cost["mean_difference"]) < 0.0
        and float(online_cost["mean_difference"]) < 0.0
    )
    clinically_safe = bool(candidate_clinical["passed"] and online_clinical["passed"])
    conclusive = (
        float(control_cost["ci_high"]) < 0.0
        and float(online_cost["ci_high"]) < 0.0
    )
    classification = (
        "pass"
        if conclusive and clinically_safe
        else "promising"
        if means_improve and clinically_safe
        else "fail"
    )
    return {
        "classification": classification,
        "mean_cost_improves_vs_control": (
            float(control_cost["mean_difference"]) < 0.0
        ),
        "mean_online_gain_is_positive": (
            float(online_cost["mean_difference"]) < 0.0
        ),
        "cost_cis_exclude_zero": conclusive,
        "clinical_noninferiority": clinically_safe,
    }


if __name__ == "__main__":
    main()
