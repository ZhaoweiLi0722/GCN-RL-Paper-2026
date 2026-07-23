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


def paired_two_level_summary(
    candidate_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    metric: str = "total_cost",
    training_seed_key: str = "training_seed",
    pairing_keys: Sequence[str] = ("training_seed", "evaluation_seed", "replication"),
    alpha: float = 0.05,
    resamples: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison with a training-seed/replication hierarchical bootstrap."""

    def keyed(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = tuple(row.get(name, "") for name in pairing_keys)
            if key in result:
                raise ValueError(f"duplicate paired evaluation key: {key}")
            result[key] = row
        return result

    candidate_by_key = keyed(candidate_rows)
    baseline_by_key = keyed(baseline_rows)
    if candidate_by_key.keys() != baseline_by_key.keys():
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        raise ValueError(
            "paired evaluation keys differ: "
            f"missing_candidate={missing_candidate[:3]} "
            f"missing_baseline={missing_baseline[:3]}"
        )
    if not candidate_by_key:
        raise ValueError("paired comparison requires at least one evaluation row")

    differences_by_seed: dict[Any, list[float]] = defaultdict(list)
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    wins = 0
    ties = 0
    for key in sorted(candidate_by_key, key=lambda value: tuple(str(item) for item in value)):
        candidate_row = candidate_by_key[key]
        baseline_row = baseline_by_key[key]
        candidate_value = float(candidate_row[metric])
        baseline_value = float(baseline_row[metric])
        difference = candidate_value - baseline_value
        training_seed = candidate_row.get(training_seed_key, "")
        differences_by_seed[training_seed].append(difference)
        candidate_values.append(candidate_value)
        baseline_values.append(baseline_value)
        wins += int(difference < 0.0)
        ties += int(difference == 0.0)

    training_seeds = sorted(differences_by_seed, key=str)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=float)
    for sample_index in range(resamples):
        sampled_seed_indices = rng.integers(0, len(training_seeds), len(training_seeds))
        sampled_differences: list[float] = []
        for seed_index in sampled_seed_indices:
            values = np.asarray(differences_by_seed[training_seeds[int(seed_index)]], dtype=float)
            sampled_differences.extend(values[rng.integers(0, values.size, values.size)])
        bootstrap_means[sample_index] = float(np.mean(sampled_differences))

    mean_candidate = float(np.mean(candidate_values))
    mean_baseline = float(np.mean(baseline_values))
    mean_difference = mean_candidate - mean_baseline
    ci_low, ci_high = np.percentile(
        bootstrap_means,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )
    return {
        "metric": metric,
        "candidate_mean": mean_candidate,
        "baseline_mean": mean_baseline,
        "mean_difference": mean_difference,
        "mean_gap_pct": (
            100.0 * mean_difference / mean_baseline if mean_baseline != 0.0 else float("nan")
        ),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_training_seeds": len(training_seeds),
        "n_pairs": len(candidate_values),
        "wins": wins,
        "ties": ties,
        "per_seed_mean_difference": {
            str(training_seed): float(np.mean(differences_by_seed[training_seed]))
            for training_seed in training_seeds
        },
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


def stability_report(
    rows: list[dict[str, Any]],
    *,
    metric: str = "total_cost",
    group_by: Iterable[str] = ("algorithm",),
    seed_key: str = "seed",
) -> list[dict[str, Any]]:
    """Per-group cross-seed stability for one metric: spread, CV, divergence, IQM.

    Complements the point estimate: surfaces which algorithms are seed-fragile
    (the DDPG-family concern) so they can be flagged in the pilot findings.
    """

    group_by = tuple(group_by)
    per_seed: dict[tuple[Any, ...], dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if metric not in row or row[metric] in ("", None):
            continue
        gkey = tuple(row.get(key, "") for key in group_by)
        per_seed[gkey][row.get(seed_key, "")].append(float(row[metric]))

    reports: list[dict[str, Any]] = []
    for gkey in sorted(per_seed, key=lambda k: tuple(str(x) for x in k)):
        seed_values = [float(np.mean(values)) for values in per_seed[gkey].values()]
        arr = np.asarray(seed_values, dtype=float)
        mean = float(arr.mean()) if arr.size else float("nan")
        spread = float(arr.max() - arr.min()) if arr.size else float("nan")
        cv = float(arr.std(ddof=0) / abs(mean)) if arr.size and mean != 0.0 else float("nan")
        report: dict[str, Any] = {key: value for key, value in zip(group_by, gkey)}
        report.update(
            {
                "metric": metric,
                "n_seeds": int(arr.size),
                "iqm": interquartile_mean(arr),
                "mean": mean,
                "spread": spread,
                "cv": cv,
                "divergent_seeds": json.dumps(divergent_seeds(seed_values)),
                "per_seed": json.dumps([float(v) for v in seed_values]),
            }
        )
        reports.append(report)
    return reports


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
