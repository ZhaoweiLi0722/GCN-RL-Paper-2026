"""Routing label budget study: offline analysis of the paired-CRN pool.

Answers the study's question -- was Stage G1's 54.5% label agreement a property
of the routing channel or of an eight-world budget? -- by replaying allocation
policies against a complete pool. No further simulation, so uniform and
sequential-halving allocation are scored on identical data.

The classification is computed from the reading rule frozen in the config
before collection; it is never assigned by hand.
See specs/2026-08-29-routing-label-budget-study/protocol.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.routing_label_budget_pool import DEFAULT_CONFIG, sha256_path
from evaluation.sequential_halving_labeling import sequential_halving_best_arm

G1_REPORTED_AGREEMENT = 0.545  # Stage G1 discovery/validation best-action agreement
G1_GATE = 0.70


def load_pool(rows_path: Path) -> tuple[dict, list[str], list[str], list[int]]:
    with open(rows_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pool: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        pool[(row["state_id"], row["arm"])][int(row["world_seed"])] = float(
            row["remaining_cost"]
        )
    states = sorted({s for s, _ in pool})
    arms = sorted({a for _, a in pool})
    worlds = sorted(next(iter(pool.values())))
    for key, values in pool.items():
        if sorted(values) != worlds:
            raise RuntimeError(f"incomplete pool for {key}")
    return pool, states, arms, worlds


def label(pool, state: str, arms, world_group) -> str:
    return min(arms, key=lambda a: float(np.mean([pool[(state, a)][w] for w in world_group])))


def agreement_curve(pool, states, arms, worlds, k_values, splits, rng) -> dict[str, Any]:
    curve = {}
    for k in k_values:
        if 2 * k > len(worlds):
            continue
        scores = []
        for _ in range(splits):
            order = list(rng.permutation(worlds))
            left, right = order[:k], order[k : 2 * k]
            agree = sum(
                label(pool, s, arms, left) == label(pool, s, arms, right) for s in states
            )
            scores.append(agree / len(states))
        curve[k] = {
            "mean": float(np.mean(scores)),
            "p10": float(np.percentile(scores, 10)),
            "p90": float(np.percentile(scores, 90)),
        }
    return curve


def g1_budget_point(pool, states, arms, worlds, shape, splits, rng) -> dict[str, Any]:
    """Agreement at Stage G1's actual budget shape (3 discovery vs 5 validation)."""

    a, b = int(shape[0]), int(shape[1])
    scores = []
    for _ in range(splits):
        order = list(rng.permutation(worlds))
        left, right = order[:a], order[a : a + b]
        agree = sum(
            label(pool, s, arms, left) == label(pool, s, arms, right) for s in states
        )
        scores.append(agree / len(states))
    return {
        "shape": [a, b],
        "mean": float(np.mean(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
        "g1_reported": G1_REPORTED_AGREEMENT,
    }


def allocation_comparison(pool, states, arms, worlds, budgets, splits, rng) -> dict[str, Any]:
    """Uniform versus sequential halving at matched simulator budget."""

    results = {}
    for per_arm in budgets:
        budget = per_arm * len(arms)
        uniform_scores, halving_scores = [], []
        for _ in range(splits):
            order = list(rng.permutation(worlds))
            half = len(order) // 2
            groups = (order[:half], order[half:])
            uniform_picks, halving_picks = [], []
            for group in groups:
                slots = list(group[:per_arm]) or list(group[:1])
                uniform_picks.append(
                    {s: label(pool, s, arms, slots) for s in states}
                )
                halving_picks.append(
                    {
                        s: sequential_halving_best_arm(
                            arms,
                            lambda arm, world, st=s: pool[(st, arm)][world],
                            list(group),
                            budget=budget,
                        )["best_arm"]
                        for s in states
                    }
                )
            uniform_scores.append(
                sum(uniform_picks[0][s] == uniform_picks[1][s] for s in states) / len(states)
            )
            halving_scores.append(
                sum(halving_picks[0][s] == halving_picks[1][s] for s in states) / len(states)
            )
        results[str(per_arm)] = {
            "simulator_calls_per_group": budget,
            "uniform_agreement": float(np.mean(uniform_scores)),
            "sequential_halving_agreement": float(np.mean(halving_scores)),
        }
    return results


def convergence(pool, states, arms, worlds, k_values, splits, rng) -> dict[str, Any]:
    """Agreement of small-budget labels with the full-pool label (a reference, not truth)."""

    full = {s: label(pool, s, arms, worlds) for s in states}
    out = {}
    for k in k_values:
        scores = []
        for _ in range(splits):
            group = list(rng.permutation(worlds))[:k]
            scores.append(
                sum(label(pool, s, arms, group) == full[s] for s in states) / len(states)
            )
        out[str(k)] = float(np.mean(scores))
    return out


def classify(curve: dict[int, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen reading rule. Never assign a classification by hand."""

    ks = sorted(curve)
    top = ks[-1]
    top_mean = curve[top]["mean"]
    last_gain = (
        top_mean - curve[ks[-2]]["mean"] if len(ks) >= 2 else float("inf")
    )
    underpowered = float(rule["underpowered_if_agreement_at_k32_at_least"])
    fundamental = float(rule["fundamental_if_agreement_at_k32_below"])
    gain_cap = float(rule["fundamental_also_requires_last_doubling_gain_below"])

    if top_mean >= underpowered:
        name = "g1_negative_underpowered"
        consequence = (
            "the 70% gate is reachable with budget; the Stage G1 conclusion is "
            "reclassified as a budget statement and the question it closed is REOPENED "
            "(any follow-up needs its own specification)"
        )
    elif top_mean < fundamental and last_gain < gain_cap:
        name = "g1_negative_confirmed_fundamental"
        consequence = (
            "labels do not stabilize at 8x the Stage G1 budget; the closed conclusion "
            "stands on stronger evidence"
        )
    else:
        name = "inconclusive_at_this_budget"
        consequence = (
            "recorded as inconclusive; extension to larger pools requires a new "
            "change-control entry, not a quiet re-run"
        )
    return {
        "classification": name,
        "consequence": consequence,
        "max_k": top,
        "agreement_at_max_k": top_mean,
        "last_doubling_gain": last_gain,
        "thresholds": {
            "underpowered_at_least": underpowered,
            "fundamental_below": fundamental,
            "fundamental_gain_cap": gain_cap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    rows_path = root / "pool_rows.csv"

    pool, states, arms, worlds = load_pool(rows_path)
    settings = config["analysis"]
    splits = int(settings["splits_per_k"])
    rng = np.random.default_rng(int(settings["split_seed"]))
    k_values = [int(k) for k in settings["k_values"]]

    curve = agreement_curve(pool, states, arms, worlds, k_values, splits, rng)
    g1_point = g1_budget_point(pool, states, arms, worlds, settings["g1_budget_shape"], splits, rng)
    allocation = allocation_comparison(
        pool, states, arms, worlds,
        [int(b) for b in settings["allocation_budgets_per_arm"]], max(50, splits // 4), rng,
    )
    converge = convergence(pool, states, arms, worlds, k_values, splits, rng)
    decision = classify(curve, config["reading_rule"])

    print(f"states {len(states)} | arms {len(arms)} | worlds {len(worlds)}")
    print()
    print(f"{'worlds per group (k)':>22}{'agreement':>12}{'p10':>8}{'p90':>8}")
    for k in sorted(curve):
        row = curve[k]
        print(f"{k:>22}{row['mean']:>12.3f}{row['p10']:>8.3f}{row['p90']:>8.3f}")
    print()
    print(f"Stage G1 budget shape {g1_point['shape']}: agreement {g1_point['mean']:.3f} "
          f"(G1 reported {G1_REPORTED_AGREEMENT:.3f}, gate {G1_GATE:.2f})")
    print()
    print(f"{'calls/arm':>10}{'uniform':>10}{'seq halving':>14}")
    for key in sorted(allocation, key=int):
        row = allocation[key]
        print(f"{key:>10}{row['uniform_agreement']:>10.3f}{row['sequential_halving_agreement']:>14.3f}")
    print()
    print(f"DECISION: {decision['classification']}")
    print(f"  agreement at k={decision['max_k']}: {decision['agreement_at_max_k']:.3f} "
          f"| last doubling gain {decision['last_doubling_gain']:+.3f}")
    print(f"  {decision['consequence']}")

    summary = {
        "name": "routing_label_budget_analysis",
        "experimental_role": config["experimental_role"],
        "states": len(states),
        "arms": arms,
        "worlds": len(worlds),
        "agreement_curve": {str(k): v for k, v in curve.items()},
        "g1_budget_point": g1_point,
        "allocation_comparison": allocation,
        "convergence_to_full_pool": converge,
        "decision": decision,
        "config_sha256": sha256_path(config_path),
        "rows_sha256": sha256_path(rows_path),
    }
    out = root / "budget_analysis.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
