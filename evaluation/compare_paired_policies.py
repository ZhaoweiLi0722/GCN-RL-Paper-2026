"""Compare two policies with paired, two-level bootstrap inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import paired_two_level_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-inputs", nargs="+", required=True)
    parser.add_argument("--baseline-inputs", nargs="+", required=True)
    parser.add_argument("--metric", default="total_cost")
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/paired_policy_comparison.json")
    args = parser.parse_args()

    summary = paired_two_level_summary(
        read_rows(args.candidate_inputs),
        read_rows(args.baseline_inputs),
        metric=args.metric,
        resamples=args.resamples,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote paired comparison to {output_path}")


if __name__ == "__main__":
    main()
