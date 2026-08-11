"""Summarize a fixed-policy multi-checkpoint development curve."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.evaluate_multiscenario_network_residual import (
    normalized_deployment_candidate,
)


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
NON_SCIENTIFIC_FIELDS = {"average_inference_ms"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-spec", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    spec_path = Path(args.stage_spec)
    spec = _load_json(spec_path)
    result = summarize_checkpoint_curve(
        stage_spec_path=spec_path,
        bootstrap_resamples=int(spec["bootstrap_resamples"]),
        bootstrap_seed=int(spec["bootstrap_seed"]),
    )
    output = Path(args.output or spec["curve_summary"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def summarize_checkpoint_curve(
    *,
    stage_spec_path: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    spec = _load_json(stage_spec_path)
    variants = tuple(str(value) for value in spec["checkpoint_variants"])
    if len(variants) < 2 or variants[0] != "pretrain":
        raise ValueError("Checkpoint curve must start with pretrain")
    if len(set(variants)) != len(variants):
        raise ValueError("Checkpoint variants must be unique")

    configs: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    summary_paths: dict[str, Path] = {}
    for variant in variants:
        config_path = Path(spec["evaluation_configs"][variant])
        config = _load_json(config_path)
        summary_path = Path(config["output_root"]) / "summary.json"
        summary = _load_json(summary_path)
        _validate_evaluation_summary(
            variant=variant,
            config=config,
            summary=summary,
        )
        configs[variant] = config
        summaries[variant] = summary
        summary_paths[variant] = summary_path

    protocol = _validate_shared_protocol(configs, summaries, variants)
    runs = {
        variant: _load_runs(summary_paths[variant], summaries[variant])
        for variant in variants
    }
    crn_audit = _audit_runs(
        runs=runs,
        variants=variants,
        expected_rows=(
            int(protocol["holdout_replications"])
            * len(protocol["scenarios"])
        ),
        expected_scenarios=set(protocol["scenarios"]),
    )

    algorithms = tuple(sorted(protocol["algorithms"]))
    margins = dict(protocol["clinical_noninferiority"]["margins"])
    algorithm_results = {
        algorithm: _algorithm_curve(
            algorithm=algorithm,
            variants=variants,
            runs=runs,
            summaries=summaries,
            margins=margins,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + index * 100_000,
        )
        for index, algorithm in enumerate(algorithms)
    }
    decision = _stage_decision(
        spec=spec,
        algorithms=algorithm_results,
    )
    return {
        "stage_spec": str(stage_spec_path),
        "summary_paths": {
            variant: str(path)
            for variant, path in summary_paths.items()
        },
        "protocol": protocol,
        "crn_audit": crn_audit,
        "algorithms": algorithm_results,
        "decision": decision,
    }


def _validate_evaluation_summary(
    *,
    variant: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    if config.get("checkpoint_variants") != [variant]:
        raise ValueError(f"{variant} config must request only itself")
    if config.get("fixed_checkpoint_variant") != variant:
        raise ValueError(f"{variant} config is not fixed to its checkpoint")
    if summary.get("fixed_checkpoint_variant") != variant:
        raise ValueError(f"{variant} summary checkpoint mismatch")
    fixed_candidate = normalized_deployment_candidate(
        dict(config["fixed_deployment_candidate"])
    )
    if summary.get("fixed_deployment_candidate") != fixed_candidate:
        raise ValueError(f"{variant} fixed deployment mismatch")
    for run in summary.get("runs", []):
        selected = dict(run.get("selected_deployment", {}))
        if selected.get("selection_source") != "pre_registered_fixed":
            raise ValueError(f"{variant} run used deployment selection")
        if selected.get("checkpoint_variant") != variant:
            raise ValueError(f"{variant} run selected another checkpoint")
        if selected.get("candidate") != fixed_candidate:
            raise ValueError(f"{variant} run changed the fixed deployment")


def _validate_shared_protocol(
    configs: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    variants: tuple[str, ...],
) -> dict[str, Any]:
    fields = (
        "training_manifest",
        "algorithms",
        "training_seeds",
        "scenarios",
        "validation_replications",
        "validation_seed",
        "holdout_replications",
        "holdout_seed",
        "max_steps",
        "fixed_deployment_candidate",
        "clinical_noninferiority",
    )
    baseline = configs[variants[0]]
    for variant in variants[1:]:
        for field in fields:
            if configs[variant].get(field) != baseline.get(field):
                raise ValueError(
                    f"Checkpoint protocol mismatch for {field}: {variant}"
                )
    for variant in variants:
        summary = summaries[variant]
        for field in (
            "training_manifest",
            "algorithms",
            "training_seeds",
            "scenarios",
            "validation_replications",
            "validation_seed",
            "holdout_replications",
            "holdout_seed",
            "fixed_deployment_candidate",
            "clinical_noninferiority",
        ):
            summary_value = summary.get(field)
            baseline_value = baseline.get(field)
            if field == "algorithms":
                summary_value = sorted(summary_value or [])
                baseline_value = sorted(baseline_value or [])
            elif field == "training_seeds":
                summary_value = sorted(int(value) for value in summary_value or [])
                baseline_value = sorted(int(value) for value in baseline_value or [])
            if field == "fixed_deployment_candidate":
                summary_value = normalized_deployment_candidate(
                    dict(summary_value)
                )
                baseline_value = normalized_deployment_candidate(
                    dict(baseline_value)
                )
            if summary_value != baseline_value:
                raise ValueError(
                    f"Summary protocol mismatch for {field}: {variant}"
                )
    return {field: baseline.get(field) for field in fields}


def _load_runs(
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    root = summary_path.parent
    for run in summary["runs"]:
        algorithm = str(run["algorithm"])
        seed = int(run["training_seed"])
        key = (algorithm, seed)
        if key in result:
            raise ValueError(f"Duplicate evaluation run: {key}")
        run_root = root / algorithm / f"seed{seed}"
        result[key] = {
            "candidate_rows": read_rows([run_root / "holdout_rows.csv"]),
            "anchor_rows": read_rows(
                [run_root / "holdout_anchor_rows.csv"]
            ),
            "checkpoint": str(run["checkpoint"]),
            "residual_usage": dict(
                run["holdout"]["aggregate"]["residual_usage"]
            ),
        }
    return result


def _audit_runs(
    *,
    runs: dict[str, dict[tuple[str, int], dict[str, Any]]],
    variants: tuple[str, ...],
    expected_rows: int,
    expected_scenarios: set[str],
) -> dict[str, Any]:
    baseline = runs[variants[0]]
    candidate_keys_matched = 0
    anchor_values_matched = 0
    for variant in variants:
        if runs[variant].keys() != baseline.keys():
            raise ValueError(f"Run identities differ for {variant}")
        for key in sorted(baseline):
            current = runs[variant][key]
            reference = baseline[key]
            _audit_rows(
                current["candidate_rows"],
                expected_rows=expected_rows,
                expected_scenarios=expected_scenarios,
                label=f"{variant}:{key}:candidate",
            )
            _audit_rows(
                current["anchor_rows"],
                expected_rows=expected_rows,
                expected_scenarios=expected_scenarios,
                label=f"{variant}:{key}:anchor",
            )
            current_candidate = _rows_by_key(current["candidate_rows"])
            reference_candidate = _rows_by_key(
                reference["candidate_rows"]
            )
            if current_candidate.keys() != reference_candidate.keys():
                raise ValueError(
                    f"Candidate CRN keys differ for {variant} run {key}"
                )
            candidate_keys_matched += len(current_candidate)
            current_anchor = _rows_by_key(current["anchor_rows"])
            reference_anchor = _rows_by_key(reference["anchor_rows"])
            if current_anchor.keys() != reference_anchor.keys():
                raise ValueError(
                    f"Anchor CRN keys differ for {variant} run {key}"
                )
            for row_key in current_anchor:
                left = current_anchor[row_key]
                right = reference_anchor[row_key]
                fields = set(left) & set(right) - NON_SCIENTIFIC_FIELDS
                for field in fields:
                    if left[field] != right[field]:
                        raise ValueError(
                            "Anchor outcome differs across checkpoints for "
                            f"{variant} run={key} key={row_key} field={field}"
                        )
                anchor_values_matched += len(fields)
            usage = current["residual_usage"]
            corrected = float(usage.get("corrected_decisions", 0.0))
            applied = float(usage.get("applied_residual_l1", 0.0))
            if (
                not math.isfinite(corrected)
                or not math.isfinite(applied)
                or corrected <= 0.0
                or applied <= 0.0
            ):
                raise ValueError(
                    f"Residual usage is zero for {variant} run {key}"
                )
    return {
        "passed": True,
        "variant_count": len(variants),
        "runs_per_variant": len(baseline),
        "rows_per_run": expected_rows,
        "candidate_keys_matched": candidate_keys_matched,
        "scientific_anchor_field_values_matched": anchor_values_matched,
        "excluded_wall_clock_fields": sorted(NON_SCIENTIFIC_FIELDS),
    }


def _audit_rows(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_scenarios: set[str],
    label: str,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(
            f"{label} has {len(rows)} rows, expected {expected_rows}"
        )
    if len(_rows_by_key(rows)) != expected_rows:
        raise ValueError(f"{label} contains duplicate CRN keys")
    scenarios = {str(row.get("scenario", "")) for row in rows}
    if scenarios != expected_scenarios:
        raise ValueError(
            f"{label} scenarios differ: {sorted(scenarios)}"
        )
    if sum(float(row.get("specimen_route_count", 0.0)) for row in rows) <= 0:
        raise ValueError(f"{label} has zero specimen routing")
    for row in rows:
        for field in METRICS + (
            "specimen_route_count",
            "specimen_route_distance_miles",
            "specimen_route_time_hours",
        ):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{label} has nonfinite {field}")


def _algorithm_curve(
    *,
    algorithm: str,
    variants: tuple[str, ...],
    runs: dict[str, dict[tuple[str, int], dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    margins: dict[str, Any],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_rows = _combined_rows(runs[variants[0]], algorithm, "candidate_rows")
    checkpoints: dict[str, Any] = {}
    segments: dict[str, Any] = {}
    previous_variant = variants[0]
    for index, variant in enumerate(variants):
        candidate_rows = _combined_rows(runs[variant], algorithm, "candidate_rows")
        anchor_rows = _combined_rows(runs[variant], algorithm, "anchor_rows")
        vs_anchor = _metric_bundle(
            candidate_rows,
            anchor_rows,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + index * 1_000,
        )
        vs_pretrain = _metric_bundle(
            candidate_rows,
            baseline_rows,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + index * 1_000 + 100,
        )
        checkpoints[variant] = {
            "vs_anchor": vs_anchor,
            "vs_pretrain": vs_pretrain,
            "vs_anchor_clinical_noninferiority": (
                _clinical_noninferiority(vs_anchor, margins)
            ),
            "vs_pretrain_clinical_noninferiority": (
                _clinical_noninferiority(vs_pretrain, margins)
            ),
            "residual_usage": _residual_usage(summaries[variant], algorithm),
            "actor_drift_from_pretrain": _actor_drift(
                summaries[variant],
                summaries[variants[0]],
                algorithm,
            ),
        }
        if index > 0:
            previous_rows = _combined_rows(
                runs[previous_variant], algorithm, "candidate_rows"
            )
            bundle = _metric_bundle(
                candidate_rows,
                previous_rows,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + index * 1_000 + 200,
            )
            segments[f"{previous_variant}_to_{variant}"] = {
                "metrics": bundle,
                "clinical_noninferiority": (
                    _clinical_noninferiority(bundle, margins)
                ),
            }
        previous_variant = variant
    return {
        "checkpoints": checkpoints,
        "segments": segments,
    }


def _stage_decision(
    *,
    spec: dict[str, Any],
    algorithms: dict[str, Any],
) -> dict[str, Any]:
    rule = dict(spec["decision_rule"])
    start = str(rule["late_segment_from"])
    end = str(rule["late_segment_to"])
    algorithm = str(rule["primary_algorithm"])
    segment = algorithms[algorithm]["segments"][f"{start}_to_{end}"]
    total_cost = segment["metrics"]["total_cost"]
    per_seed = dict(total_cost["per_seed_mean_difference"])
    improving_seeds = sum(float(value) < 0.0 for value in per_seed.values())
    required_wins = int(rule["improving_requires_seed_wins_at_least"])
    require_cost_interval = bool(
        rule["improving_requires_total_cost_ci_high_below_zero"]
    )
    require_clinical = bool(
        rule["improving_requires_clinical_noninferiority"]
    )
    checks = {
        "total_cost_ci_high_below_zero": (
            float(total_cost["ci_high"]) < 0.0
            if require_cost_interval
            else True
        ),
        "minimum_improving_seeds": improving_seeds >= required_wins,
        "clinical_noninferiority": (
            bool(segment["clinical_noninferiority"]["passed"])
            if require_clinical
            else True
        ),
    }
    improving = all(checks.values())
    clearly_worse = float(total_cost["ci_low"]) > 0.0
    classification = (
        "improving"
        if improving
        else "deteriorating"
        if clearly_worse or not checks["clinical_noninferiority"]
        else "plateaued_or_inconclusive"
    )
    return {
        "primary_algorithm": algorithm,
        "late_segment": f"{start}_to_{end}",
        "checks": checks,
        "improving_seed_count": improving_seeds,
        "required_improving_seed_count": required_wins,
        "classification": classification,
        "next_step": (
            rule["if_improving"]
            if improving
            else rule["otherwise"]
        ),
    }


def _metric_bundle(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: paired_two_level_summary(
            candidate_rows,
            baseline_rows,
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
        "completion_service_level": (
            float(bundle["completion_service_level"]["ci_low"])
            >= -float(margins.get("completion_service_level", 0.001))
        ),
        "patients_lost": (
            float(bundle["patients_lost"]["ci_high"])
            <= float(margins.get("patients_lost", 1.0))
        ),
        "patient_ineligibility_during_manufacturing_rate": (
            float(
                bundle[
                    "patient_ineligibility_during_manufacturing_rate"
                ]["ci_high"]
            )
            <= float(
                margins.get(
                    "patient_ineligibility_during_manufacturing_rate",
                    0.001,
                )
            )
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _residual_usage(
    summary: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    per_seed = {
        str(int(run["training_seed"])): dict(
            run["holdout"]["aggregate"]["residual_usage"]
        )
        for run in summary["runs"]
        if str(run["algorithm"]) == algorithm
    }
    return {
        "per_seed": per_seed,
        "nonzero_residual_seeds": sum(
            float(value.get("corrected_decisions", 0.0)) > 0.0
            and float(value.get("applied_residual_l1", 0.0)) > 0.0
            for value in per_seed.values()
        ),
        "seed_count": len(per_seed),
    }


def _actor_drift(
    summary: dict[str, Any],
    pretrain_summary: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch is not installed"}

    checkpoints = {
        int(run["training_seed"]): Path(str(run["checkpoint"]))
        for run in summary["runs"]
        if str(run["algorithm"]) == algorithm
    }
    references = {
        int(run["training_seed"]): Path(str(run["checkpoint"]))
        for run in pretrain_summary["runs"]
        if str(run["algorithm"]) == algorithm
    }
    if checkpoints.keys() != references.keys():
        raise ValueError(f"Actor-drift seed mismatch for {algorithm}")
    per_seed: dict[str, Any] = {}
    for seed in sorted(checkpoints):
        current = torch.load(
            checkpoints[seed], map_location="cpu", weights_only=False
        )["actor"]
        reference = torch.load(
            references[seed], map_location="cpu", weights_only=False
        )["actor"]
        if current.keys() != reference.keys():
            raise ValueError(f"Actor parameter mismatch for {algorithm} seed {seed}")
        squared_error = 0.0
        maximum = 0.0
        parameter_count = 0
        for name in current:
            left = current[name]
            right = reference[name]
            if left.shape != right.shape:
                raise ValueError(
                    f"Actor shape mismatch for {algorithm} seed {seed}: {name}"
                )
            if not left.is_floating_point():
                continue
            difference = left.double() - right.double()
            squared_error += float((difference * difference).sum().item())
            maximum = max(maximum, float(difference.abs().max().item()))
            parameter_count += int(difference.numel())
        if parameter_count == 0:
            raise ValueError(f"No floating actor parameters for {algorithm}")
        per_seed[str(seed)] = {
            "rms": math.sqrt(squared_error / parameter_count),
            "max_abs": maximum,
            "parameter_count": parameter_count,
        }
    return {"available": True, "per_seed": per_seed}


def _combined_rows(
    runs: dict[tuple[str, int], dict[str, Any]],
    algorithm: str,
    field: str,
) -> list[dict[str, Any]]:
    return [
        row
        for (name, _seed), run in sorted(runs.items())
        if name == algorithm
        for row in run[field]
    ]


def _rows_by_key(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in PAIRING_KEYS)
        if key in result:
            raise ValueError(f"Duplicate CRN key: {key}")
        result[key] = row
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
