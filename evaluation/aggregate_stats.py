"""Robust across-seed aggregation: IQM + bootstrap CIs + divergence flags.

The Phase 3 statistical protocol requires the interquartile mean (IQM) for
DDPG-family results — a single divergent seed must not move the reported point
estimate (Phase 5 found flat DDPG ranging from near-optimal to full divergence).
This module aggregates evaluation rows in two levels: collapse replications to a
per-seed value, then compute IQM + bootstrap CIs across seeds.

Deterministic: bootstrap resampling takes an explicit RNG seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from evaluation.aggregate_results import read_rows

DEFAULT_GROUP_BY = ("algorithm", "scenario", "graph_ablation")
DEFAULT_METRICS = (
    "total_cost",
    "eligibility_rate",
    "eligibility_rate_mean",
    "patients_lost",
    "material_wasted",
    "at_risk_unserved",
    "service_level",
)


def interquartile_mean(values: Sequence[float], trim: float = 0.25) -> float:
    """Mean of the middle (1 - 2*trim) fraction after sorting (rliable-style IQM)."""

    arr = np.sort(np.asarray(list(values), dtype=float))
    n = arr.size
    if n == 0:
        return float("nan")
    if n <= 2:
        return float(arr.mean())
    k = int(np.floor(n * trim))
    trimmed = arr[k : n - k] if n - 2 * k > 0 else arr
    return float(trimmed.mean())


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = interquartile_mean,
    alpha: float = 0.05,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic`. Deterministic under `seed`."""

    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples, dtype=float)
    for i in range(resamples):
        stats[i] = statistic(arr[rng.integers(0, n, n)])
    lo = float(np.percentile(stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def divergent_seeds(values: Sequence[float], k: float = 3.0) -> list[int]:
    """Indices of seeds outside [q25 - k*IQR, q75 + k*IQR] (robust outlier rule)."""

    arr = np.asarray(list(values), dtype=float)
    if arr.size < 4:
        return []
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr = q75 - q25
    lo, hi = q25 - k * iqr, q75 + k * iqr
    return [i for i, v in enumerate(arr) if v < lo or v > hi]


def summarize_across_seeds(seed_values: Sequence[float], *, seed: int = 0) -> dict[str, Any]:
    arr = np.asarray(list(seed_values), dtype=float)
    lo, hi = bootstrap_ci(arr, seed=seed)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "iqm": interquartile_mean(arr),
        "ci_low": lo,
        "ci_high": hi,
        "n_seeds": int(arr.size),
        "divergent_seeds": divergent_seeds(arr),
        "per_seed": [float(v) for v in arr],
    }


def aggregate_iqm(
    rows: list[dict[str, Any]],
    *,
    group_by: Iterable[str] = DEFAULT_GROUP_BY,
    metrics: Iterable[str] = DEFAULT_METRICS,
    seed_key: str = "seed",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Collapse replications to per-seed means, then IQM + CI across seeds."""

    group_by = tuple(group_by)
    metrics = tuple(metrics)

    # Level 1: (group, seed) -> per-metric mean over replications.
    per_seed: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        gkey = tuple(row.get(key, "") for key in group_by)
        skey = row.get(seed_key, "")
        for metric in metrics:
            if metric in row and row[metric] not in ("", None):
                per_seed[(gkey, skey)][metric].append(float(row[metric]))
    # Level 2: group -> list of per-seed point estimates for each metric.
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (gkey, _skey), metric_values in per_seed.items():
        for metric, values in metric_values.items():
            grouped[gkey][metric].append(float(np.mean(values)))

    summary_rows: list[dict[str, Any]] = []
    for gkey in sorted(grouped, key=lambda k: tuple(str(x) for x in k)):
        summary: dict[str, Any] = {key: value for key, value in zip(group_by, gkey)}
        for metric in metrics:
            values = grouped[gkey].get(metric)
            if not values:
                continue
            stats = summarize_across_seeds(values, seed=seed)
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_iqm"] = stats["iqm"]
            summary[f"{metric}_ci_low"] = stats["ci_low"]
            summary[f"{metric}_ci_high"] = stats["ci_high"]
            summary[f"{metric}_n_seeds"] = stats["n_seeds"]
            summary[f"{metric}_divergent_seeds"] = json.dumps(stats["divergent_seeds"])
            summary[f"{metric}_per_seed"] = json.dumps(stats["per_seed"])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="results/iqm_summary.csv")
    parser.add_argument("--group-by", nargs="+", default=list(DEFAULT_GROUP_BY))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    summary = aggregate_iqm(rows, group_by=args.group_by, metrics=args.metrics, seed=args.seed)
    write_rows(summary, args.output)
    print(f"wrote {len(summary)} IQM summary rows to {args.output}")


if __name__ == "__main__":
    main()
