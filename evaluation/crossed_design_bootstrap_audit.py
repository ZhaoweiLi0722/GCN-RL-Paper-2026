"""Crossed-design bootstrap sensitivity audit for paired holdout rows.

The formal routing-primary holdout is a crossed design: every training seed's
policy is evaluated on the same common-random-number (CRN) worlds, indexed by
``(scenario, replication)``. The publication intervals come from
``evaluation.aggregate_stats.paired_two_level_summary``, which resamples
training seeds and then resamples rows independently inside each selected
seed. That inner step treats the shared worlds as if they were independent
draws per seed and pools scenarios instead of holding their mixture fixed.

This module recomputes the same paired contrasts under four resampling schemes
so the sensitivity of the reported intervals to the crossed dependence can be
documented:

``nested_seed_then_row``
    The existing publication method, called directly from ``aggregate_stats``
    so the reproduction is exact.
``world_cluster``
    Resample worlds with replacement, stratified by scenario, applying the same
    world draw to every training seed. Seeds are held fixed. This captures the
    world-level common uncertainty and keeps the scenario mixture fixed.
``seed_cluster``
    Resample training seeds with replacement and keep every world. This
    captures seed-to-seed policy variation only.
``two_way``
    Resample seeds and worlds (stratified by scenario) with replacement and
    average the crossed sub-array. This is the pigeonhole bootstrap of Owen
    (2007) for crossed random effects and is generally conservative.

All schemes use percentile intervals on the mean paired difference with equal
scenario weights. Nothing here trains, evaluates, or selects a policy; the
inputs are immutable row-level CSVs.

Row layout expected under ``--root``::

    <root>/<algorithm>/seed<NN>/holdout_rows.csv          (learned policy)
    <root>/<algorithm>/seed<NN>/holdout_anchor_rows.csv   (MDL-2 anchor)

``--frozen-root`` points to the matching frozen-pretrain evaluation tree and
enables the final-minus-frozen contrast.

``--compact-final-summary`` (optionally with ``--compact-pretrain-summary``)
runs only the seed-level cluster bootstrap on the per-seed, per-scenario cell
means that the compact publication summary already contains. That mode needs
no row-level files and is the only part of the audit that can run when the raw
formal root is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.aggregate_results import read_rows  # noqa: E402
from evaluation.aggregate_stats import paired_two_level_summary  # noqa: E402

DEFAULT_SCENARIOS = (
    "routing_nominal_history",
    "routing_abrupt_regime_shift",
    "routing_regional_drift",
    "routing_compound_regional_stress",
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_policy_rows(
    root: Path,
    algorithm: str,
    filename: str,
) -> list[dict[str, Any]]:
    paths = sorted((root / algorithm).glob("seed*/" + filename))
    if not paths:
        raise FileNotFoundError(f"no {filename} under {root / algorithm}")
    return read_rows(paths)


def rows_to_array(
    rows: Iterable[dict[str, Any]],
    *,
    metric: str,
    scenarios: tuple[str, ...],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Return ``values[seed, scenario, replication]`` plus the seed/rep labels."""

    table: dict[tuple[int, str, int], float] = {}
    for row in rows:
        key = (int(row["training_seed"]), str(row["scenario"]), int(row["replication"]))
        if key in table:
            raise ValueError(f"duplicate row key {key}")
        table[key] = float(row[metric])
    seeds = sorted({k[0] for k in table})
    reps = sorted({k[2] for k in table})
    found_scenarios = {k[1] for k in table}
    missing = set(scenarios) - found_scenarios
    extra = found_scenarios - set(scenarios)
    if missing or extra:
        raise ValueError(f"scenario mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    values = np.full((len(seeds), len(scenarios), len(reps)), np.nan, dtype=float)
    for (seed, scenario, rep), value in table.items():
        values[seeds.index(seed), scenarios.index(scenario), reps.index(rep)] = value
    if not np.all(np.isfinite(values)):
        raise ValueError("holdout array is not fully populated")
    return values, seeds, reps


# --------------------------------------------------------------------------- #
# Bootstrap schemes on a difference array D[seed, scenario, replication]
# --------------------------------------------------------------------------- #


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def world_cluster_bootstrap(
    diff: np.ndarray, *, resamples: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float]:
    n_seeds, n_scen, n_rep = diff.shape
    # Per-world means across seeds, per scenario: shape (scenario, rep)
    world_means = diff.mean(axis=0)
    out = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, n_rep, size=(n_scen, n_rep))
        out[b] = np.mean(np.take_along_axis(world_means, idx, axis=1))
    return _percentile_ci(out, alpha)


def seed_cluster_bootstrap(
    diff: np.ndarray, *, resamples: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float]:
    n_seeds = diff.shape[0]
    seed_means = diff.mean(axis=(1, 2))
    idx = rng.integers(0, n_seeds, size=(resamples, n_seeds))
    out = seed_means[idx].mean(axis=1)
    return _percentile_ci(out, alpha)


def two_way_bootstrap(
    diff: np.ndarray, *, resamples: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float]:
    n_seeds, n_scen, n_rep = diff.shape
    out = np.empty(resamples)
    for b in range(resamples):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        rep_idx = rng.integers(0, n_rep, size=(n_scen, n_rep))
        sub = diff[seed_idx]  # (S, K, R)
        sub = np.take_along_axis(sub, np.broadcast_to(rep_idx, sub.shape), axis=2)
        out[b] = sub.mean()
    return _percentile_ci(out, alpha)


def dependence_diagnostics(diff: np.ndarray) -> dict[str, Any]:
    """Describe how much of the difference variance is shared across seeds."""

    n_seeds, n_scen, n_rep = diff.shape
    flat = diff.reshape(n_seeds, n_scen * n_rep)
    # Mean pairwise correlation across seeds of the per-world differences.
    corrs = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            if np.std(flat[i]) > 0 and np.std(flat[j]) > 0:
                corrs.append(float(np.corrcoef(flat[i], flat[j])[0, 1]))
    # Two-way variance components by method of moments (balanced design).
    grand = flat.mean()
    seed_means = flat.mean(axis=1)
    world_means = flat.mean(axis=0)
    resid = flat - seed_means[:, None] - world_means[None, :] + grand
    n_w = n_scen * n_rep
    ms_seed = n_w * np.var(seed_means, ddof=1) if n_seeds > 1 else float("nan")
    ms_world = n_seeds * np.var(world_means, ddof=1)
    ms_resid = np.sum(resid**2) / ((n_seeds - 1) * (n_w - 1)) if n_seeds > 1 else float("nan")
    var_seed = max((ms_seed - ms_resid) / n_w, 0.0) if n_seeds > 1 else float("nan")
    var_world = max((ms_world - ms_resid) / n_seeds, 0.0) if n_seeds > 1 else float("nan")
    var_resid = ms_resid
    if n_seeds > 1:
        se_mean = float(np.sqrt(var_seed / n_seeds + var_world / n_w + var_resid / (n_seeds * n_w)))
    else:
        se_mean = float("nan")
    return {
        "mean_pairwise_seed_correlation_of_world_differences": (
            float(np.mean(corrs)) if corrs else float("nan")
        ),
        "variance_component_seed": float(var_seed),
        "variance_component_world": float(var_world),
        "variance_component_residual": float(var_resid),
        "analytic_se_of_mean": se_mean,
        "analytic_normal_ci_halfwidth": 1.959964 * se_mean,
    }


def analyze_difference(
    diff: np.ndarray,
    *,
    resamples: int,
    alpha: float,
    seed: int,
    scenarios: tuple[str, ...],
    baseline_mean: float | None,
    nested: dict[str, Any] | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mean = float(diff.mean())
    per_scenario = {s: float(diff[:, k, :].mean()) for k, s in enumerate(scenarios)}
    per_seed = [float(v) for v in diff.mean(axis=(1, 2))]
    result: dict[str, Any] = {
        "mean_difference": mean,
        "mean_gap_pct": (100.0 * mean / baseline_mean) if baseline_mean else None,
        "n_training_seeds": int(diff.shape[0]),
        "n_worlds_per_scenario": int(diff.shape[2]),
        "per_scenario_mean_difference": per_scenario,
        "per_seed_mean_difference": per_seed,
        "wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "intervals": {},
        "dependence": dependence_diagnostics(diff),
    }
    if nested is not None:
        result["intervals"]["nested_seed_then_row"] = {
            "ci_low": nested["ci_low"],
            "ci_high": nested["ci_high"],
            "source": "evaluation.aggregate_stats.paired_two_level_summary",
        }
    lo, hi = world_cluster_bootstrap(diff, resamples=resamples, alpha=alpha, rng=rng)
    result["intervals"]["world_cluster"] = {"ci_low": lo, "ci_high": hi}
    if diff.shape[0] > 1:
        lo, hi = seed_cluster_bootstrap(diff, resamples=resamples, alpha=alpha, rng=rng)
        result["intervals"]["seed_cluster"] = {"ci_low": lo, "ci_high": hi}
        lo, hi = two_way_bootstrap(diff, resamples=resamples, alpha=alpha, rng=rng)
        result["intervals"]["two_way"] = {"ci_low": lo, "ci_high": hi}
    for name, interval in result["intervals"].items():
        interval["width"] = interval["ci_high"] - interval["ci_low"]
        interval["excludes_zero"] = bool(interval["ci_high"] < 0.0 or interval["ci_low"] > 0.0)
    return result


# --------------------------------------------------------------------------- #
# Row-level audit
# --------------------------------------------------------------------------- #


def run_row_level_audit(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(args.scenarios)
    root = Path(args.root)
    algorithms = list(args.algorithms)
    report: dict[str, Any] = {
        "mode": "row_level",
        "root": str(root),
        "metric": args.metric,
        "resamples": args.resamples,
        "alpha": args.alpha,
        "scenarios": list(scenarios),
        "contrasts": {},
        "crn_checks": {},
    }

    learned: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    anchors: dict[str, np.ndarray] = {}
    anchor_rows_by_algo: dict[str, list[dict[str, Any]]] = {}
    seeds_ref: list[int] | None = None
    for algo in algorithms:
        l_rows = load_policy_rows(root, algo, "holdout_rows.csv")
        a_rows = load_policy_rows(root, algo, "holdout_anchor_rows.csv")
        l_arr, seeds, _ = rows_to_array(l_rows, metric=args.metric, scenarios=scenarios)
        a_arr, a_seeds, _ = rows_to_array(a_rows, metric=args.metric, scenarios=scenarios)
        if seeds != a_seeds:
            raise ValueError(f"{algo}: learned/anchor seed sets differ")
        if seeds_ref is None:
            seeds_ref = seeds
        elif seeds != seeds_ref:
            raise ValueError("algorithms were evaluated on different training seeds")
        learned[algo] = (l_arr, l_rows)
        anchors[algo] = a_arr
        anchor_rows_by_algo[algo] = a_rows
        # CRN check 1: anchor identical across training seeds for each world.
        spread = float(np.max(np.ptp(a_arr, axis=0)))
        report["crn_checks"][f"{algo}: anchor max spread across training seeds"] = spread

    # CRN check 2: anchor identical across algorithms.
    if len(algorithms) > 1:
        a0 = anchors[algorithms[0]]
        for algo in algorithms[1:]:
            report["crn_checks"][
                f"anchor max abs diff {algorithms[0]} vs {algo}"
            ] = float(np.max(np.abs(a0 - anchors[algo])))

    def nested(cand_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return paired_two_level_summary(
            cand_rows, base_rows, metric=args.metric, resamples=args.resamples, seed=args.seed
        )

    for algo in algorithms:
        l_arr, l_rows = learned[algo]
        diff = l_arr - anchors[algo]
        report["contrasts"][f"{algo}_vs_anchor"] = analyze_difference(
            diff,
            resamples=args.resamples,
            alpha=args.alpha,
            seed=args.seed,
            scenarios=scenarios,
            baseline_mean=float(anchors[algo].mean()),
            nested=nested(l_rows, anchor_rows_by_algo[algo]),
        )
    if len(algorithms) == 2:
        a, b = algorithms
        diff = learned[a][0] - learned[b][0]
        report["contrasts"][f"{a}_vs_{b}"] = analyze_difference(
            diff,
            resamples=args.resamples,
            alpha=args.alpha,
            seed=args.seed,
            scenarios=scenarios,
            baseline_mean=float(learned[b][0].mean()),
            nested=nested(learned[a][1], learned[b][1]),
        )

    if args.frozen_root:
        froot = Path(args.frozen_root)
        report["frozen_root"] = str(froot)
        for algo in algorithms:
            f_rows = load_policy_rows(froot, algo, "holdout_rows.csv")
            f_arr, f_seeds, _ = rows_to_array(f_rows, metric=args.metric, scenarios=scenarios)
            if f_seeds != seeds_ref:
                raise ValueError(f"{algo}: frozen seeds differ from final seeds")
            diff = learned[algo][0] - f_arr
            report["contrasts"][f"{algo}_final_vs_frozen"] = analyze_difference(
                diff,
                resamples=args.resamples,
                alpha=args.alpha,
                seed=args.seed,
                scenarios=scenarios,
                baseline_mean=float(f_arr.mean()),
                nested=nested(learned[algo][1], f_rows),
            )
    return report


# --------------------------------------------------------------------------- #
# Compact-summary (seed-level only) audit
# --------------------------------------------------------------------------- #


def cell_means_from_summary(
    summary: dict[str, Any], *, scenarios: tuple[str, ...]
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{algorithm: {"candidate": M, "anchor": M}}`` with M[seed, scenario]."""

    out: dict[str, dict[str, np.ndarray]] = {}
    by_algo: dict[str, dict[int, dict[str, Any]]] = {}
    for run in summary["runs"]:
        by_algo.setdefault(run["algorithm"], {})[int(run["training_seed"])] = run
    for algo, runs in by_algo.items():
        seeds = sorted(runs)
        cand = np.empty((len(seeds), len(scenarios)))
        anch = np.empty_like(cand)
        for i, seed in enumerate(seeds):
            per = runs[seed]["holdout"]["per_scenario"]
            for k, scenario in enumerate(scenarios):
                cand[i, k] = float(per[scenario]["candidate_cost_mean"])
                anch[i, k] = float(per[scenario]["anchor_cost_mean"])
        out[algo] = {"candidate": cand, "anchor": anch, "seeds": np.asarray(seeds)}
    return out


def run_compact_audit(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(args.scenarios)
    final = json.loads(Path(args.compact_final_summary).read_text())
    cells = cell_means_from_summary(final, scenarios=scenarios)
    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {
        "mode": "compact_seed_level_only",
        "note": (
            "Seed-cluster bootstrap on per-seed, per-scenario holdout cell means "
            "from the compact publication summary. World-level and two-way "
            "intervals require the row-level formal root and are not computed here."
        ),
        "final_summary": str(args.compact_final_summary),
        "resamples": args.resamples,
        "alpha": args.alpha,
        "contrasts": {},
        "crn_checks": {},
    }

    def seed_level(diff_cells: np.ndarray, baseline_mean: float, published: dict[str, Any] | None):
        # diff_cells: (seed, scenario) -> treat as D[seed, scenario, 1]
        diff = diff_cells[:, :, None]
        res = {
            "mean_difference": float(diff.mean()),
            "mean_gap_pct": 100.0 * float(diff.mean()) / baseline_mean,
            "per_seed_mean_difference": [float(v) for v in diff.mean(axis=(1, 2))],
            "per_scenario_mean_difference": {
                s: float(diff[:, k, :].mean()) for k, s in enumerate(scenarios)
            },
            "intervals": {},
        }
        if published is not None:
            res["intervals"]["nested_seed_then_row_published"] = {
                "ci_low": published["ci_low"],
                "ci_high": published["ci_high"],
            }
        lo, hi = seed_cluster_bootstrap(diff, resamples=args.resamples, alpha=args.alpha, rng=rng)
        res["intervals"]["seed_cluster"] = {"ci_low": lo, "ci_high": hi}
        seed_means = diff.mean(axis=(1, 2))
        n = len(seed_means)
        se = float(np.std(seed_means, ddof=1) / np.sqrt(n))
        from math import sqrt  # noqa: F401  (kept simple; t quantile below)

        t_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n - 1, 1.96)
        res["intervals"]["seed_t_interval"] = {
            "ci_low": float(seed_means.mean() - t_975 * se),
            "ci_high": float(seed_means.mean() + t_975 * se),
            "df": n - 1,
        }
        for interval in res["intervals"].values():
            interval["width"] = interval["ci_high"] - interval["ci_low"]
            interval["excludes_zero"] = bool(
                interval["ci_high"] < 0.0 or interval["ci_low"] > 0.0
            )
        return res

    algos = sorted(cells)
    published = final.get("aggregate", {})
    for algo in algos:
        c = cells[algo]
        report["crn_checks"][f"{algo}: anchor cell-mean max spread across seeds"] = float(
            np.max(np.ptp(c["anchor"], axis=0))
        )
        report["contrasts"][f"{algo}_vs_anchor"] = seed_level(
            c["candidate"] - c["anchor"],
            float(c["anchor"].mean()),
            published.get(f"{algo}_vs_anchor", {}).get("total_cost"),
        )
    if len(algos) == 2:
        a, b = [x for x in algos if "gcn" in x] + [x for x in algos if "gcn" not in x]
        report["contrasts"][f"{a}_vs_{b}"] = seed_level(
            cells[a]["candidate"] - cells[b]["candidate"],
            float(cells[b]["candidate"].mean()),
            published.get(f"{a}_vs_{b}", {}).get("total_cost"),
        )
    if args.compact_pretrain_summary:
        pre = json.loads(Path(args.compact_pretrain_summary).read_text())
        pcells = cell_means_from_summary(pre, scenarios=scenarios)
        report["pretrain_summary"] = str(args.compact_pretrain_summary)
        for algo in algos:
            if algo in pcells:
                report["contrasts"][f"{algo}_final_vs_frozen"] = seed_level(
                    cells[algo]["candidate"] - pcells[algo]["candidate"],
                    float(pcells[algo]["candidate"].mean()),
                    None,
                )
    return report


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Crossed-design bootstrap audit ({report['mode']})", ""]
    for key in ("root", "frozen_root", "final_summary", "pretrain_summary", "note"):
        if key in report:
            lines.append(f"- {key}: `{report[key]}`" if key != "note" else f"- {report[key]}")
    lines.append(f"- resamples: {report['resamples']}, alpha: {report['alpha']}")
    if report.get("crn_checks"):
        lines += ["", "## CRN sharing checks", ""]
        for k, v in report["crn_checks"].items():
            lines.append(f"- {k}: `{v:.6g}`")
    for name, c in report["contrasts"].items():
        lines += ["", f"## {name}", ""]
        gap = c.get("mean_gap_pct")
        gap_txt = f" ({gap:+.6f}%)" if gap is not None else ""
        lines.append(f"Mean difference: `{c['mean_difference']:,.1f}`{gap_txt}")
        lines.append(
            "Per-seed means: "
            + ", ".join(f"`{v:,.0f}`" for v in c["per_seed_mean_difference"])
        )
        lines += ["", "| Interval | low | high | width | excludes 0 |", "| --- | ---: | ---: | ---: | :--: |"]
        for iname, iv in c["intervals"].items():
            lines.append(
                f"| {iname} | {iv['ci_low']:,.0f} | {iv['ci_high']:,.0f} | {iv['width']:,.0f} | "
                f"{'yes' if iv['excludes_zero'] else 'no'} |"
            )
        if "dependence" in c:
            d = c["dependence"]
            lines += [
                "",
                f"Mean pairwise seed correlation of world-level differences: "
                f"`{d['mean_pairwise_seed_correlation_of_world_differences']:.3f}`; "
                f"variance components (seed / world / residual): "
                f"`{d['variance_component_seed']:.3g}` / `{d['variance_component_world']:.3g}` / "
                f"`{d['variance_component_residual']:.3g}`; analytic normal half-width "
                f"`{d['analytic_normal_ci_halfwidth']:,.0f}`.",
            ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="row-level evaluation root (final policies)")
    p.add_argument("--frozen-root", help="row-level evaluation root (frozen-pretrain policies)")
    p.add_argument("--algorithms", nargs="+", default=[], help="learned algorithm directory names")
    p.add_argument("--compact-final-summary", help="compact final_summary.json (seed-level mode)")
    p.add_argument("--compact-pretrain-summary", help="compact pretrain_summary.json")
    p.add_argument("--metric", default="total_cost")
    p.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    p.add_argument("--resamples", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compact_final_summary:
        report = run_compact_audit(args)
    else:
        if not args.root or not args.algorithms:
            raise SystemExit("--root and --algorithms are required in row-level mode")
        report = run_row_level_audit(args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(args.out_md).write_text(render_markdown(report))
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
