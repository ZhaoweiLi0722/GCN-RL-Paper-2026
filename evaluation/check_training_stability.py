"""Check basic stability signals in training CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GROUP_BY = ("algorithm", "scenario", "graph_ablation", "seed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="results/training_stability_summary.csv")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-episodes", type=int, default=1)
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    summary = summarize_training_stability(rows, window=args.window, min_episodes=args.min_episodes)
    write_rows(summary, args.output)
    print(f"wrote {len(summary)} stability rows to {args.output}")


def read_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    return rows


def summarize_training_stability(
    rows: list[dict[str, Any]],
    *,
    window: int = 5,
    min_episodes: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in GROUP_BY)].append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: int(float(row.get("episode", 0))))
        rewards = [_as_float(row.get("total_reward")) for row in ordered]
        costs = [_as_float(row.get("total_cost")) for row in ordered]
        finite = all(math.isfinite(value) for value in rewards + costs)
        episode_count = len(ordered)
        status = "ok"
        issues = []
        if not finite:
            issues.append("non_finite_metric")
        if episode_count < int(min_episodes):
            issues.append("too_few_episodes")
        if issues:
            status = "check"

        first_rewards = rewards[:window]
        final_rewards = rewards[-window:]
        first_costs = costs[:window]
        final_costs = costs[-window:]

        summary: dict[str, Any] = {key: value for key, value in zip(GROUP_BY, group_key)}
        summary.update(
            {
                "episodes": episode_count,
                "status": status,
                "issues": ";".join(issues),
                "first_reward_mean": _mean(first_rewards),
                "final_reward_mean": _mean(final_rewards),
                "reward_delta": _mean(final_rewards) - _mean(first_rewards),
                "first_cost_mean": _mean(first_costs),
                "final_cost_mean": _mean(final_costs),
                "cost_delta": _mean(final_costs) - _mean(first_costs),
            }
        )
        summaries.append(summary)
    return summaries


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


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


if __name__ == "__main__":
    main()
