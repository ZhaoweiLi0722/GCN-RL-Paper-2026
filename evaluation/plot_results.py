"""Plot aggregate evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metric", default="total_cost_mean")
    parser.add_argument("--output", default="figures/total_cost_summary.png")
    parser.add_argument("--category", default="scenario")
    parser.add_argument("--series", default="algorithm")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/gcn_rl_matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with `python -m pip install matplotlib`.") from exc

    rows = _read_rows(args.summary)
    if not rows:
        raise SystemExit(f"No rows found in {args.summary}")
    _plot_grouped_bars(
        rows,
        metric=args.metric,
        category_key=args.category,
        series_key=args.series,
        output=Path(args.output),
        title=args.title or args.metric.replace("_", " ").title(),
        plt=plt,
    )
    print(f"wrote figure to {args.output}")


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_grouped_bars(
    rows: list[dict[str, str]],
    *,
    metric: str,
    category_key: str,
    series_key: str,
    output: Path,
    title: str,
    plt,
) -> None:
    categories = sorted({row[category_key] for row in rows})
    series = sorted({row[series_key] for row in rows})
    values = {(row[category_key], row[series_key]): float(row[metric]) for row in rows if row.get(metric)}

    width = 0.8 / max(len(series), 1)
    x_positions = list(range(len(categories)))
    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.7), 4.5))
    for idx, label in enumerate(series):
        offsets = [x + (idx - (len(series) - 1) / 2) * width for x in x_positions]
        heights = [values.get((category, label), 0.0) for category in categories]
        ax.bar(offsets, heights, width=width, label=label)

    ax.set_title(title)
    ax.set_xlabel(category_key.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
