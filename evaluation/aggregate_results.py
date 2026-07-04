"""Aggregate raw evaluation CSV files into summary tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_GROUP_BY = ("algorithm", "scenario", "graph_ablation")
DEFAULT_METRICS = (
    "total_reward",
    "total_cost",
    "service_level",
    "average_waiting_time",
    "reagent_shortage_frequency",
    "bioreactor_shortage_frequency",
    "bioreactor_utilization",
    "transshipment_count",
    "transshipment_cost",
    "average_inference_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="results/aggregate_summary.csv")
    parser.add_argument("--group-by", nargs="+", default=list(DEFAULT_GROUP_BY))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    summary = aggregate_rows(rows, group_by=args.group_by, metrics=args.metrics)
    write_rows(summary, args.output)
    print(f"wrote {len(summary)} summary rows to {args.output}")


def read_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: Iterable[str] = DEFAULT_GROUP_BY,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    group_by = tuple(group_by)
    metrics = tuple(metrics)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_by)].append(row)

    summary_rows = []
    for group_key, group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {key: value for key, value in zip(group_by, group_key)}
        summary["count"] = len(group_rows)
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group_rows
                if metric in row and row[metric] not in ("", None)
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            summary[f"{metric}_mean"] = float(array.mean())
            summary[f"{metric}_std"] = float(array.std(ddof=1)) if array.size > 1 else 0.0
            summary[f"{metric}_sem"] = (
                float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
            )
        summary_rows.append(summary)
    return summary_rows


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

