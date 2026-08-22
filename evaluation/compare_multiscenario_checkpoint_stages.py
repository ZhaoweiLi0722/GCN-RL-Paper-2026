"""Paired final-versus-frozen attribution for multi-scenario residual RL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate_results import read_rows
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
NON_SCIENTIFIC_FIELDS = {"average_inference_ms"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-summary", required=True)
    parser.add_argument("--frozen-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=61_000_000)
    args = parser.parse_args()

    result = compare_checkpoint_stages(
        final_summary_path=Path(args.final_summary),
        frozen_summary_path=Path(args.frozen_summary),
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


def compare_checkpoint_stages(
    *,
    final_summary_path: Path,
    frozen_summary_path: Path,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 61_000_000,
) -> dict[str, Any]:
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    final_summary = _load_json(final_summary_path)
    frozen_summary = _load_json(frozen_summary_path)
    protocol = _validate_protocol(final_summary, frozen_summary)

    final_runs = _load_stage_runs(final_summary_path, final_summary)
    frozen_runs = _load_stage_runs(frozen_summary_path, frozen_summary)
    if final_runs.keys() != frozen_runs.keys():
        raise ValueError("Final and frozen evaluations contain different runs")

    crn_audit = _audit_common_random_numbers(final_runs, frozen_runs)
    algorithms = sorted({algorithm for algorithm, _seed in final_runs})
    margins = dict(
        final_summary.get("clinical_noninferiority", {}).get("margins", {})
    )

    comparisons: dict[str, Any] = {}
    for algorithm_index, algorithm in enumerate(algorithms):
        final_candidate = _combined_rows(
            final_runs, algorithm, "candidate_rows"
        )
        frozen_candidate = _combined_rows(
            frozen_runs, algorithm, "candidate_rows"
        )
        final_anchor = _combined_rows(final_runs, algorithm, "anchor_rows")
        frozen_anchor = _combined_rows(
            frozen_runs, algorithm, "anchor_rows"
        )
        seed_offset = bootstrap_seed + algorithm_index * 10_000
        final_vs_anchor = _metric_bundle(
            final_candidate,
            final_anchor,
            resamples=bootstrap_resamples,
            seed=seed_offset,
        )
        frozen_vs_anchor = _metric_bundle(
            frozen_candidate,
            frozen_anchor,
            resamples=bootstrap_resamples,
            seed=seed_offset + 100,
        )
        final_vs_frozen = _metric_bundle(
            final_candidate,
            frozen_candidate,
            resamples=bootstrap_resamples,
            seed=seed_offset + 200,
        )
        comparisons[algorithm] = {
            "final_vs_anchor": final_vs_anchor,
            "frozen_vs_anchor": frozen_vs_anchor,
            "final_vs_frozen": final_vs_frozen,
            "final_vs_anchor_clinical_noninferiority": (
                _clinical_noninferiority(final_vs_anchor, margins)
            ),
            "final_vs_frozen_clinical_noninferiority": (
                _clinical_noninferiority(final_vs_frozen, margins)
            ),
            "per_scenario": _per_scenario_comparisons(
                final_candidate,
                frozen_candidate,
                final_anchor,
                resamples=bootstrap_resamples,
                seed=seed_offset + 1_000,
            ),
            "final_residual_usage": _residual_usage(
                final_summary, algorithm
            ),
            "frozen_residual_usage": _residual_usage(
                frozen_summary, algorithm
            ),
            "actor_drift": _actor_drift_by_seed(
                final_summary,
                frozen_summary,
                algorithm,
            ),
        }

    graph_algorithm, flat_algorithm = _matched_graph_flat_pair(algorithms)
    graph_flat = _graph_flat_attribution(
        final_runs=final_runs,
        frozen_runs=frozen_runs,
        graph_algorithm=graph_algorithm,
        flat_algorithm=flat_algorithm,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 90_000,
    )
    training_diagnostics = _training_diagnostics(
        final_summary.get("training_manifest")
    )
    gates = _progression_gates(
        comparisons=comparisons,
        graph_flat=graph_flat,
        graph_algorithm=graph_algorithm,
        minimum_nonzero_seeds=2,
    )
    return {
        "final_summary": str(final_summary_path),
        "frozen_summary": str(frozen_summary_path),
        "protocol": protocol,
        "crn_audit": crn_audit,
        "algorithms": algorithms,
        "comparisons": comparisons,
        "graph_flat_attribution": graph_flat,
        "training_diagnostics": training_diagnostics,
        "progression_gates": gates,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_protocol(
    final_summary: dict[str, Any],
    frozen_summary: dict[str, Any],
) -> dict[str, Any]:
    equal_fields = (
        "scenarios",
        "holdout_replications",
        "holdout_seed",
        "algorithms",
        "training_seeds",
        "fixed_deployment_candidate",
        "clinical_noninferiority",
    )
    for field in equal_fields:
        if final_summary.get(field) != frozen_summary.get(field):
            raise ValueError(
                f"Final and frozen protocol mismatch for {field}"
            )
    if final_summary.get("fixed_checkpoint_variant") != "final":
        raise ValueError("Final evaluation must lock checkpoint variant final")
    if frozen_summary.get("fixed_checkpoint_variant") != "pretrain":
        raise ValueError(
            "Frozen evaluation must lock checkpoint variant pretrain"
        )
    return {
        field: final_summary.get(field)
        for field in equal_fields
    } | {
        "final_checkpoint_variant": "final",
        "frozen_checkpoint_variant": "pretrain",
    }


def _load_stage_runs(
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
        candidate_path = run_root / "holdout_rows.csv"
        anchor_path = run_root / "holdout_anchor_rows.csv"
        candidate_rows = read_rows([candidate_path])
        anchor_rows = read_rows([anchor_path])
        result[key] = {
            "candidate_rows": candidate_rows,
            "anchor_rows": anchor_rows,
            "checkpoint": str(run["checkpoint"]),
            "selected_deployment": dict(run["selected_deployment"]),
        }
    return result


def _audit_common_random_numbers(
    final_runs: dict[tuple[str, int], dict[str, Any]],
    frozen_runs: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    candidate_keys = 0
    anchor_keys = 0
    scientific_anchor_fields = 0
    for key in sorted(final_runs):
        final_run = final_runs[key]
        frozen_run = frozen_runs[key]
        if (
            final_run["selected_deployment"]["candidate"]
            != frozen_run["selected_deployment"]["candidate"]
        ):
            raise ValueError(f"Deployment candidate differs for run {key}")
        final_candidate = _rows_by_key(final_run["candidate_rows"])
        frozen_candidate = _rows_by_key(frozen_run["candidate_rows"])
        if final_candidate.keys() != frozen_candidate.keys():
            raise ValueError(f"Candidate CRN keys differ for run {key}")
        candidate_keys += len(final_candidate)

        final_anchor = _rows_by_key(final_run["anchor_rows"])
        frozen_anchor = _rows_by_key(frozen_run["anchor_rows"])
        if final_anchor.keys() != frozen_anchor.keys():
            raise ValueError(f"Anchor CRN keys differ for run {key}")
        anchor_keys += len(final_anchor)
        for row_key in final_anchor:
            left = final_anchor[row_key]
            right = frozen_anchor[row_key]
            fields = (
                set(left)
                & set(right)
                - NON_SCIENTIFIC_FIELDS
            )
            for field in fields:
                if left[field] != right[field]:
                    raise ValueError(
                        "Final/frozen anchor outcomes differ for "
                        f"run={key} key={row_key} field={field}"
                    )
            scientific_anchor_fields += len(fields)
    return {
        "run_count": len(final_runs),
        "candidate_keys_matched": candidate_keys,
        "anchor_keys_matched": anchor_keys,
        "scientific_anchor_field_values_matched": (
            scientific_anchor_fields
        ),
        "excluded_wall_clock_fields": sorted(NON_SCIENTIFIC_FIELDS),
        "passed": True,
    }


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
    completion_margin = float(
        margins.get("completion_service_level", 0.001)
    )
    lost_margin = float(margins.get("patients_lost", 1.0))
    ineligibility_margin = float(
        margins.get(
            "patient_ineligibility_during_manufacturing_rate",
            0.001,
        )
    )
    checks = {
        "completion_service_level": (
            float(bundle["completion_service_level"]["ci_low"])
            >= -completion_margin
        ),
        "patients_lost": (
            float(bundle["patients_lost"]["ci_high"])
            <= lost_margin
        ),
        "patient_ineligibility_during_manufacturing_rate": (
            float(
                bundle[
                    "patient_ineligibility_during_manufacturing_rate"
                ]["ci_high"]
            )
            <= ineligibility_margin
        ),
    }
    return {
        "margins": {
            "completion_service_level": completion_margin,
            "patients_lost": lost_margin,
            "patient_ineligibility_during_manufacturing_rate": (
                ineligibility_margin
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _per_scenario_comparisons(
    final_rows: list[dict[str, Any]],
    frozen_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    scenarios = sorted({str(row["scenario"]) for row in final_rows})
    return {
        scenario: {
            "final_vs_anchor": _metric_bundle(
                _scenario_rows(final_rows, scenario),
                _scenario_rows(anchor_rows, scenario),
                resamples=resamples,
                seed=seed + index * 100,
            ),
            "final_vs_frozen": _metric_bundle(
                _scenario_rows(final_rows, scenario),
                _scenario_rows(frozen_rows, scenario),
                resamples=resamples,
                seed=seed + index * 100 + 10,
            ),
        }
        for index, scenario in enumerate(scenarios)
    }


def _scenario_rows(
    rows: list[dict[str, Any]],
    scenario: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["scenario"]) == scenario]


def _matched_graph_flat_pair(
    algorithms: list[str],
) -> tuple[str, str]:
    pairs = [
        (algorithm, f"flat_{algorithm[4:]}")
        for algorithm in algorithms
        if algorithm.startswith("gcn_")
        and f"flat_{algorithm[4:]}" in algorithms
    ]
    if len(pairs) != 1:
        raise ValueError(
            "Checkpoint-stage attribution requires exactly one matched "
            f"GCN/flat pair, found {pairs}"
        )
    return pairs[0]


def _graph_flat_attribution(
    *,
    final_runs: dict[tuple[str, int], dict[str, Any]],
    frozen_runs: dict[tuple[str, int], dict[str, Any]],
    graph_algorithm: str,
    flat_algorithm: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    final_graph = _combined_rows(
        final_runs, graph_algorithm, "candidate_rows"
    )
    final_flat = _combined_rows(
        final_runs, flat_algorithm, "candidate_rows"
    )
    frozen_graph = _combined_rows(
        frozen_runs, graph_algorithm, "candidate_rows"
    )
    frozen_flat = _combined_rows(
        frozen_runs, flat_algorithm, "candidate_rows"
    )
    graph_increment = _difference_rows(final_graph, frozen_graph)
    flat_increment = _difference_rows(final_flat, frozen_flat)
    return {
        "graph_algorithm": graph_algorithm,
        "flat_algorithm": flat_algorithm,
        "final_graph_vs_flat": _metric_bundle(
            final_graph,
            final_flat,
            resamples=resamples,
            seed=seed,
        ),
        "frozen_graph_vs_flat": _metric_bundle(
            frozen_graph,
            frozen_flat,
            resamples=resamples,
            seed=seed + 100,
        ),
        "graph_minus_flat_difference_in_differences": _metric_bundle(
            graph_increment,
            flat_increment,
            resamples=resamples,
            seed=seed + 200,
        ),
    }


def _difference_rows(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = _rows_by_key(candidate_rows)
    baseline = _rows_by_key(baseline_rows)
    if candidate.keys() != baseline.keys():
        raise ValueError("Cannot form stage differences from unmatched CRNs")
    result = []
    for key in sorted(candidate):
        row = {
            pairing_key: candidate[key][pairing_key]
            for pairing_key in PAIRING_KEYS
        }
        for metric in METRICS:
            row[metric] = (
                float(candidate[key][metric])
                - float(baseline[key][metric])
            )
        result.append(row)
    return result


def _residual_usage(
    summary: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for run in summary["runs"]:
        if str(run["algorithm"]) != algorithm:
            continue
        seed = str(int(run["training_seed"]))
        usage = dict(run["holdout"]["aggregate"]["residual_usage"])
        per_seed[seed] = usage
    nonzero = sum(
        float(usage.get("correction_rate", 0.0)) > 0.0
        for usage in per_seed.values()
    )
    return {
        "per_seed": per_seed,
        "nonzero_residual_seeds": int(nonzero),
        "seed_count": len(per_seed),
        "mean_correction_rate": (
            sum(
                float(value.get("correction_rate", 0.0))
                for value in per_seed.values()
            )
            / len(per_seed)
            if per_seed
            else 0.0
        ),
    }


def _actor_drift_by_seed(
    final_summary: dict[str, Any],
    frozen_summary: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch is not installed"}

    final_runs = {
        int(run["training_seed"]): run
        for run in final_summary["runs"]
        if str(run["algorithm"]) == algorithm
    }
    frozen_runs = {
        int(run["training_seed"]): run
        for run in frozen_summary["runs"]
        if str(run["algorithm"]) == algorithm
    }
    per_seed: dict[str, float] = {}
    for seed in sorted(final_runs.keys() & frozen_runs.keys()):
        final_checkpoint = Path(str(final_runs[seed]["checkpoint"]))
        frozen_checkpoint = Path(str(frozen_runs[seed]["checkpoint"]))
        if not final_checkpoint.is_file() or not frozen_checkpoint.is_file():
            return {
                "available": False,
                "reason": "checkpoint path is unavailable",
            }
        final_state = torch.load(
            final_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["actor"]
        frozen_state = torch.load(
            frozen_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["actor"]
        squared_error = 0.0
        element_count = 0
        for name in sorted(final_state.keys() & frozen_state.keys()):
            final_tensor = final_state[name]
            frozen_tensor = frozen_state[name]
            if (
                final_tensor.shape != frozen_tensor.shape
                or not final_tensor.is_floating_point()
            ):
                continue
            difference = final_tensor.double() - frozen_tensor.double()
            squared_error += float((difference * difference).sum().item())
            element_count += int(difference.numel())
        if element_count == 0:
            raise ValueError(
                f"No comparable floating actor parameters for {algorithm}"
            )
        per_seed[str(seed)] = math.sqrt(squared_error / element_count)
    return {
        "available": True,
        "per_seed_rms": per_seed,
        "all_nonzero": bool(per_seed) and all(
            value > 0.0 for value in per_seed.values()
        ),
    }


def _training_diagnostics(
    training_manifest_value: Any,
) -> dict[str, Any]:
    if not training_manifest_value:
        return {"available": False, "reason": "manifest not recorded"}
    manifest_path = Path(str(training_manifest_value))
    if not manifest_path.is_file():
        return {"available": False, "reason": "manifest is unavailable"}
    manifest = _load_json(manifest_path)
    per_run: dict[str, Any] = {}
    for run in manifest["runs"]:
        config_path = Path(str(run["config"]))
        if not config_path.is_file():
            return {"available": False, "reason": "config is unavailable"}
        config = _load_json(config_path)
        csv_path = Path(str(config["result_csv_path"]))
        if not csv_path.is_file():
            return {"available": False, "reason": "training CSV is unavailable"}
        rows = read_rows([csv_path])
        update_calls = sum(
            int(float(row.get("online_rl_update_calls", 0) or 0))
            for row in rows
        )
        updates = sum(
            int(float(row.get("online_rl_updates", 0) or 0))
            for row in rows
        )
        actor_updates = sum(
            float(row.get("online_rl_actor_updated_mean", 0.0) or 0.0)
            * int(float(row.get("online_rl_updates", 0) or 0))
            for row in rows
        )
        key = f"{run['algorithm']}:seed{int(run['seed'])}"
        per_run[key] = {
            "episodes": len(rows),
            "parameter_count": int(run.get("parameter_count", 0)),
            "online_rl_update_calls": update_calls,
            "online_rl_updates": updates,
            "estimated_actor_updates": actor_updates,
            "finite": _rows_are_finite(rows),
        }
    return {
        "available": True,
        "per_run": per_run,
        "parameter_counts": {
            key: value["parameter_count"]
            for key, value in per_run.items()
        },
        "all_runs_have_online_updates": all(
            value["online_rl_updates"] > 0
            for value in per_run.values()
        ),
        "all_logged_values_finite": all(
            value["finite"] for value in per_run.values()
        ),
    }


def _rows_are_finite(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for key, value in row.items():
            if not key.startswith("online_rl_") or value in ("", None):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not math.isfinite(numeric):
                return False
    return True


def _progression_gates(
    *,
    comparisons: dict[str, Any],
    graph_flat: dict[str, Any],
    graph_algorithm: str,
    minimum_nonzero_seeds: int,
) -> dict[str, Any]:
    graph = comparisons[graph_algorithm]
    checks = {
        "final_graph_beats_mdl2": (
            float(
                graph["final_vs_anchor"]["total_cost"]["ci_high"]
            )
            < 0.0
        ),
        "final_graph_beats_matched_flat": (
            float(
                graph_flat["final_graph_vs_flat"]["total_cost"][
                    "ci_high"
                ]
            )
            < 0.0
        ),
        "online_graph_increment_beats_frozen": (
            float(
                graph["final_vs_frozen"]["total_cost"]["ci_high"]
            )
            < 0.0
        ),
        "graph_specific_online_increment": (
            float(
                graph_flat[
                    "graph_minus_flat_difference_in_differences"
                ]["total_cost"]["ci_high"]
            )
            < 0.0
        ),
        "final_graph_clinically_noninferior_to_mdl2": bool(
            graph[
                "final_vs_anchor_clinical_noninferiority"
            ]["passed"]
        ),
        "final_graph_clinically_noninferior_to_frozen": bool(
            graph[
                "final_vs_frozen_clinical_noninferiority"
            ]["passed"]
        ),
        "nonzero_residual_in_at_least_two_seeds": (
            int(
                graph["final_residual_usage"][
                    "nonzero_residual_seeds"
                ]
            )
            >= int(minimum_nonzero_seeds)
        ),
    }
    return {
        "checks": checks,
        "advance_to_five_seed_confirmation": all(checks.values()),
    }


if __name__ == "__main__":
    main()
