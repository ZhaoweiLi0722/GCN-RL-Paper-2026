"""Paired attribution for the patient-indexed specimen-routing pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate_stats import paired_two_level_summary
from src.rl.config import load_config


DEFAULT_CONFIG = (
    "experiments/configs/"
    "patient_indexed_specimen_routing_attribution.json"
)
PAIRING_KEYS = (
    "training_seed",
    "scenario_family",
    "evaluation_seed",
    "replication",
)
DEFAULT_METRICS = {
    "total_cost": "lower",
    "completion_service_level": "higher",
    "patients_lost": "lower",
    "patients_lost_manufacturing": "lower",
    "patients_lost_expired": "lower",
    "patients_lost_waiting_expired": "lower",
    "patient_ineligibility_during_manufacturing_rate": "lower",
    "average_turnaround_time": "lower",
    "specimen_route_count": "descriptive",
    "specimen_route_distance_miles": "descriptive",
    "specimen_route_time_hours": "descriptive",
    "specimen_route_cost": "lower",
    "blocked_specimen_requests": "lower",
    "transit_loss": "lower",
    "transit_expiry": "lower",
    "transit_ineligible": "lower",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = compare_attribution(load_config(args.config))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote routing attribution to {result['output_path']}")


def compare_attribution(config: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(config["output_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite attribution: {output_path}")

    algorithms = tuple(str(value) for value in config["algorithms"])
    if len(algorithms) != 2:
        raise ValueError("Attribution requires exactly one GCN and one flat algorithm")
    gcn_algorithm, flat_algorithm = algorithms
    roots = {
        name: Path(config[name])
        for name in (
            "routing_final_root",
            "no_routing_final_root",
            "routing_pretrain_root",
            "no_routing_pretrain_root",
        )
    }
    for name, root in roots.items():
        if not (root / "summary.json").is_file():
            raise FileNotFoundError(f"Missing completed evaluation {name}: {root}")

    learned = {
        name: _load_learned_rows(root, algorithms)
        for name, root in roots.items()
    }
    anchors = {
        "routing_final": _load_anchor_rows(
            roots["routing_final_root"],
            source_algorithm=gcn_algorithm,
        ),
        "no_routing_final": _load_anchor_rows(
            roots["no_routing_final_root"],
            source_algorithm=gcn_algorithm,
        ),
    }
    metrics = {
        str(metric): str(direction)
        for metric, direction in config.get("metrics", DEFAULT_METRICS).items()
    }
    resamples = int(config.get("bootstrap_resamples", 20_000))
    bootstrap_seed = int(config.get("bootstrap_seed", 8_500_000))

    routing_final = learned["routing_final_root"]
    no_routing_final = learned["no_routing_final_root"]
    routing_pretrain = learned["routing_pretrain_root"]
    no_routing_pretrain = learned["no_routing_pretrain_root"]
    comparisons: dict[str, Any] = {}

    pairs = {
        "routing_gcn_vs_flat": (
            routing_final[gcn_algorithm],
            routing_final[flat_algorithm],
        ),
        "routing_gcn_vs_mdl2": (
            routing_final[gcn_algorithm],
            anchors["routing_final"],
        ),
        "no_routing_gcn_vs_flat": (
            no_routing_final[gcn_algorithm],
            no_routing_final[flat_algorithm],
        ),
        "routing_vs_no_routing_gcn": (
            routing_final[gcn_algorithm],
            no_routing_final[gcn_algorithm],
        ),
        "routing_vs_no_routing_flat": (
            routing_final[flat_algorithm],
            no_routing_final[flat_algorithm],
        ),
        "routing_vs_no_routing_mdl2": (
            anchors["routing_final"],
            anchors["no_routing_final"],
        ),
        "routing_final_vs_pretrain_gcn": (
            routing_final[gcn_algorithm],
            routing_pretrain[gcn_algorithm],
        ),
        "routing_final_vs_pretrain_flat": (
            routing_final[flat_algorithm],
            routing_pretrain[flat_algorithm],
        ),
        "no_routing_final_vs_pretrain_gcn": (
            no_routing_final[gcn_algorithm],
            no_routing_pretrain[gcn_algorithm],
        ),
        "no_routing_final_vs_pretrain_flat": (
            no_routing_final[flat_algorithm],
            no_routing_pretrain[flat_algorithm],
        ),
    }
    for index, (label, (candidate, baseline)) in enumerate(pairs.items()):
        comparisons[label] = _metric_bundle(
            candidate,
            baseline,
            metrics=metrics,
            resamples=resamples,
            seed=bootstrap_seed + index * 10_000,
        )

    interaction_candidate, interaction_baseline = _interaction_rows(
        routing_gcn=routing_final[gcn_algorithm],
        routing_flat=routing_final[flat_algorithm],
        no_routing_gcn=no_routing_final[gcn_algorithm],
        no_routing_flat=no_routing_final[flat_algorithm],
        metrics=metrics,
    )
    comparisons["graph_specific_interaction"] = _metric_bundle(
        interaction_candidate,
        interaction_baseline,
        metrics={f"interaction_{metric}": direction for metric, direction in metrics.items()},
        resamples=resamples,
        seed=bootstrap_seed + 500_000,
    )

    routing_cost = comparisons["routing_vs_no_routing_gcn"]["pooled"].get(
        "total_cost"
    )
    interaction_cost = comparisons["graph_specific_interaction"]["pooled"].get(
        "interaction_total_cost"
    )
    routing_beneficial = bool(routing_cost and float(routing_cost["ci_high"]) < 0.0)
    graph_advantage = bool(
        interaction_cost and float(interaction_cost["ci_high"]) < 0.0
    )
    if routing_beneficial and not graph_advantage:
        interpretation = (
            "Specimen routing is beneficial, but graph-specific DRL advantage "
            "is not established."
        )
    elif routing_beneficial and graph_advantage:
        interpretation = (
            "Specimen routing and a graph-specific routing interaction are "
            "supported by the preregistered total-cost comparison."
        )
    else:
        interpretation = (
            "The preregistered total-cost comparison does not establish a "
            "benefit from specimen routing."
        )

    result = {
        "name": str(config.get("name", "patient_indexed_specimen_routing_attribution")),
        "pairing_keys": list(PAIRING_KEYS),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "metric_directions": metrics,
        "comparisons": comparisons,
        "decision": {
            "routing_beneficial_total_cost": routing_beneficial,
            "graph_specific_advantage_total_cost": graph_advantage,
            "interpretation": interpretation,
        },
        "output_path": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def _load_learned_rows(
    root: Path,
    algorithms: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for algorithm in algorithms:
        paths = sorted((root / algorithm).glob("seed*/holdout_rows.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing learned holdout rows: {root / algorithm}")
        result[algorithm] = _read_rows(paths)
    return result


def _load_anchor_rows(root: Path, *, source_algorithm: str) -> list[dict[str, Any]]:
    paths = sorted((root / source_algorithm).glob("seed*/holdout_anchor_rows.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing MDL-2 holdout rows: {root / source_algorithm}")
    return _read_rows(paths)


def _read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                normalized = dict(row)
                normalized["scenario_family"] = _scenario_family(
                    str(row.get("scenario", ""))
                )
                rows.append(normalized)
    if not rows:
        raise ValueError("Attribution input contains no rows")
    return rows


def _scenario_family(name: str) -> str:
    if name.startswith("no_routing_"):
        return name[len("no_routing_") :]
    if name.startswith("routing_"):
        return name[len("routing_") :]
    return name


def _metric_bundle(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    metrics: dict[str, str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    families = sorted({str(row["scenario_family"]) for row in candidate})
    result: dict[str, Any] = {"pooled": {}, "by_scenario": {}}
    for metric_index, (metric, direction) in enumerate(metrics.items()):
        if not _rows_have_metric(candidate, metric) or not _rows_have_metric(baseline, metric):
            continue
        summary = paired_two_level_summary(
            candidate,
            baseline,
            metric=metric,
            pairing_keys=PAIRING_KEYS,
            resamples=resamples,
            seed=seed + metric_index,
        )
        result["pooled"][metric] = _with_direction(summary, direction)
        for family_index, family in enumerate(families):
            candidate_family = [
                row for row in candidate if row["scenario_family"] == family
            ]
            baseline_family = [
                row for row in baseline if row["scenario_family"] == family
            ]
            if not candidate_family or not baseline_family:
                raise ValueError(f"Missing paired scenario family {family!r}")
            scenario_summary = paired_two_level_summary(
                candidate_family,
                baseline_family,
                metric=metric,
                pairing_keys=PAIRING_KEYS,
                resamples=resamples,
                seed=seed + 1_000 + metric_index * 100 + family_index,
            )
            result["by_scenario"].setdefault(family, {})[metric] = _with_direction(
                scenario_summary,
                direction,
            )
    return result


def _rows_have_metric(rows: list[dict[str, Any]], metric: str) -> bool:
    return bool(rows) and all(row.get(metric) not in (None, "") for row in rows)


def _with_direction(summary: dict[str, Any], direction: str) -> dict[str, Any]:
    result = dict(summary)
    result["direction"] = direction
    difference = float(summary["mean_difference"])
    if direction == "higher":
        result["favorable_mean_difference"] = difference
    elif direction == "lower":
        result["favorable_mean_difference"] = -difference
    else:
        result["favorable_mean_difference"] = None
    return result


def _interaction_rows(
    *,
    routing_gcn: list[dict[str, Any]],
    routing_flat: list[dict[str, Any]],
    no_routing_gcn: list[dict[str, Any]],
    no_routing_flat: list[dict[str, Any]],
    metrics: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = [routing_gcn, routing_flat, no_routing_gcn, no_routing_flat]
    keyed = [_key_rows(rows) for rows in groups]
    keys = keyed[0].keys()
    if any(group.keys() != keys for group in keyed[1:]):
        raise ValueError("Routing interaction inputs do not share exact CRN keys")
    routing_gap_rows: list[dict[str, Any]] = []
    no_routing_gap_rows: list[dict[str, Any]] = []
    for key in sorted(keys, key=lambda value: tuple(str(item) for item in value)):
        routing_gcn_row, routing_flat_row, no_gcn_row, no_flat_row = (
            group[key] for group in keyed
        )
        common = {
            name: routing_gcn_row[name]
            for name in PAIRING_KEYS
        }
        routing_gap = dict(common)
        no_routing_gap = dict(common)
        for metric in metrics:
            if not all(
                row.get(metric) not in (None, "")
                for row in (
                    routing_gcn_row,
                    routing_flat_row,
                    no_gcn_row,
                    no_flat_row,
                )
            ):
                continue
            interaction_metric = f"interaction_{metric}"
            routing_gap[interaction_metric] = (
                float(routing_gcn_row[metric]) - float(routing_flat_row[metric])
            )
            no_routing_gap[interaction_metric] = (
                float(no_gcn_row[metric]) - float(no_flat_row[metric])
            )
        routing_gap_rows.append(routing_gap)
        no_routing_gap_rows.append(no_routing_gap)
    return routing_gap_rows, no_routing_gap_rows


def _key_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[name] for name in PAIRING_KEYS)
        if key in result:
            raise ValueError(f"Duplicate attribution key: {key}")
        result[key] = row
    return result


if __name__ == "__main__":
    main()
