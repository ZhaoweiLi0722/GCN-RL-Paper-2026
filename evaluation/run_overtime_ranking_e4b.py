"""Stage E4b (EXPLORATORY): does per-clinic graph structure carry the signal?

Stage E4 tested learnability with 15 network-level aggregates -- sums, maxima,
standard deviations over 20 clinics -- and failed its top-1 gate. Those
aggregates destroy exactly the per-clinic topology a GCN exists to exploit, so
the failure bounds what a linear model on aggregates can do, not what a
graph-aware model could.

This screen is an ABLATION rather than a single number, because that is what
actually answers the question. Three nested feature families:

  aggregate     the 15 Stage E4 features (the flat baseline)
  +distribution order statistics of the per-clinic quantities, preserving
                distributional shape that a single sum discards
  +graph        one-hop message passing over the resource edges, plus spatial
                autocorrelation -- quantities a GCN can compute and a flat
                encoder structurally cannot

If the graph family adds nothing, the E4 verdict stands on stronger ground and
the channel is genuinely hard to learn. If it adds a lot, E4 was measuring the
wrong representation, which is the same class of error as the E2 gate defect.

EXPLORATORY. Per specs/2026-08-29-continuous-overtime-control/held_out_reservation.md,
nothing here may touch the reserved scenario or seeds, and no result from this
screen is confirmatory. It consumes rows already collected and trains no
deployed policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.audit_overtime_headroom_e2 import (
    build_scenario_env,
    generate_states,
    load_screen_config,
    scenario_env_dict,
    sha256_path,
    validate_config,
)
from evaluation.ranking_feasibility import (
    leave_one_seed_out_ranking,
    ranking_gate,
)
from evaluation.run_full_benchmark import load_benchmark_plan, select_scenarios
from evaluation.run_overtime_ranking_e4 import decision_features, load_outcomes
from src.baselines.heuristics import get_heuristic_class

DEFAULT_CONFIG = Path("experiments/configs/continuous_overtime_critic_ranking_e4.json")

DISTRIBUTION_NAMES = tuple(
    f"{quantity}_{statistic}"
    for quantity in ("shortfall", "at_risk", "idle", "reagents")
    for statistic in ("p10", "p50", "p90")
)
GRAPH_NAMES = (
    "neighbour_slack_p50",
    "neighbour_slack_min",
    "local_imbalance_p90",
    "local_imbalance_max",
    "cluster_pressure_p90",
    "cluster_pressure_max",
    "shortfall_spatial_autocorrelation",
)


def per_clinic_quantities(env) -> dict[str, np.ndarray]:
    waiting = np.asarray(
        env.waiting_counts() if hasattr(env, "waiting_counts") else env.specimens,
        dtype=float,
    )
    idle = np.asarray(env.bioreactors[:, 0], dtype=float)
    reagents = np.asarray(env.reagents, dtype=float)
    at_risk = (
        np.asarray(env.at_risk_counts(), dtype=float)
        if hasattr(env, "at_risk_counts")
        else np.zeros_like(waiting)
    )
    return {
        "shortfall": np.maximum(np.minimum(waiting, reagents) - idle, 0.0),
        "at_risk": at_risk,
        "idle": idle,
        "reagents": reagents,
    }


def distribution_features(env) -> np.ndarray:
    """Order statistics: distributional shape a single sum throws away."""

    quantities = per_clinic_quantities(env)
    values = []
    for name in ("shortfall", "at_risk", "idle", "reagents"):
        column = quantities[name]
        values.extend(np.percentile(column, [10, 50, 90]))
    return np.asarray(values, dtype=float)


def adjacency(env) -> np.ndarray:
    """Symmetric one-hop adjacency over the resource-sharing edges."""

    n = env.config.num_facilities
    matrix = np.zeros((n, n), dtype=float)
    for edge in env.resource_edges:
        left, right = int(edge[0]), int(edge[1])
        if left < n and right < n:
            matrix[left, right] = 1.0
            matrix[right, left] = 1.0
    return matrix


def graph_features(env) -> np.ndarray:
    """One-hop message passing plus spatial autocorrelation.

    These are the quantities a graph encoder can compute and a flat encoder
    structurally cannot: whether a stressed clinic sits next to slack or next
    to more stress, and whether stress is clustered or dispersed.
    """

    quantities = per_clinic_quantities(env)
    shortfall = quantities["shortfall"]
    idle = quantities["idle"]
    matrix = adjacency(env)
    degree = matrix.sum(axis=1)
    safe_degree = np.where(degree > 0, degree, 1.0)

    neighbour_shortfall = (matrix @ shortfall) / safe_degree
    neighbour_idle = (matrix @ idle) / safe_degree
    local_imbalance = shortfall - neighbour_idle  # unmet demand a neighbour cannot absorb
    cluster_pressure = shortfall + neighbour_shortfall

    centred = shortfall - shortfall.mean()
    denominator = float((centred**2).sum())
    if denominator > 0 and matrix.sum() > 0:
        autocorrelation = float(
            (len(shortfall) / matrix.sum())
            * (centred @ (matrix @ centred))
            / denominator
        )
    else:
        autocorrelation = 0.0

    values = (
        float(np.percentile(neighbour_idle, 50)),
        float(neighbour_idle.min()),
        float(np.percentile(local_imbalance, 90)),
        float(local_imbalance.max()),
        float(np.percentile(cluster_pressure, 90)),
        float(cluster_pressure.max()),
        autocorrelation,
    )
    return np.asarray(values, dtype=float)


def collect_all_features(config: dict[str, Any]) -> tuple[dict, dict, dict]:
    plan = load_benchmark_plan(Path(config["plan"]))
    scenarios = select_scenarios(plan, config["scenarios"])
    policy = get_heuristic_class(str(config["anchor_algorithm"]))()
    epochs = [int(e) for e in config["state_generation"]["decision_epochs"]]
    aggregate: dict[str, np.ndarray] = {}
    distribution: dict[str, np.ndarray] = {}
    graph: dict[str, np.ndarray] = {}
    seed_of: dict[str, int] = {}
    for scenario in scenarios:
        env_dict = scenario_env_dict(plan, config, scenario)
        for state_seed in config["state_generation"]["seeds"]:
            policy.reset()
            states = generate_states(env_dict, policy, int(state_seed), epochs)
            env = build_scenario_env(env_dict, int(state_seed))
            for state in states:
                env.load_state_dict(state["snapshot"])
                key = state["state_id"]
                aggregate[key] = decision_features(env)
                distribution[key] = distribution_features(env)
                graph[key] = graph_features(env)
                seed_of[key] = int(state_seed)
    return {"aggregate": aggregate, "distribution": distribution, "graph": graph}, seed_of, {}


def stack(families: dict[str, dict[str, np.ndarray]], names: list[str]) -> dict[str, np.ndarray]:
    states = sorted(families[names[0]])
    return {
        state: np.concatenate([families[name][state] for name in names])
        for state in states
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--alphas", default="0.1,1.0,10.0,100.0")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_screen_config(config_path)
    validate_config(config)
    rows_path = Path(config["output_root"]) / "headroom_rows.csv"
    outcomes = load_outcomes(rows_path)
    families, seed_of, _ = collect_all_features(config)

    ablation = [
        ("aggregate", ["aggregate"]),
        ("aggregate+distribution", ["aggregate", "distribution"]),
        ("aggregate+distribution+graph", ["aggregate", "distribution", "graph"]),
        ("graph_only", ["graph"]),
    ]
    alphas = [float(a) for a in args.alphas.split(",")]
    gates = config["gates"]

    results: dict[str, Any] = {}
    print(f"{'feature set':<32}{'alpha':>8}{'top-1':>9}{'worst':>9}{'pairwise':>10}{'gain':>9}")
    for label, names in ablation:
        features = stack(families, names)
        dimension = len(next(iter(features.values())))
        results[label] = {"feature_dimension": dimension, "by_alpha": {}}
        for alpha in alphas:
            report = leave_one_seed_out_ranking(
                features, seed_of, outcomes, alpha=alpha
            )
            decision = ranking_gate(
                report,
                minimum_top1=float(gates["ranking_minimum_top1"]),
                minimum_pairwise=float(gates["ranking_minimum_pairwise"]),
                minimum_gain_over_state_blind=float(
                    gates["ranking_minimum_gain_over_state_blind"]
                ),
            )
            results[label]["by_alpha"][str(alpha)] = {
                "report": report,
                "gate": decision,
            }
            print(
                f"{label:<32}{alpha:>8.1f}{report['pooled_fitted_top1']:>9.3f}"
                f"{report['worst_fold_top1']:>9.3f}"
                f"{report['pooled_pairwise_accuracy']:>10.3f}"
                f"{report['worst_fold_gain']:>+9.3f}"
            )

    output = Path(config["output_root"]) / "ranking_ablation_e4b.json"
    output.write_text(
        json.dumps(
            {
                "name": "continuous_overtime_ranking_ablation_e4b",
                "experimental_role": "EXPLORATORY feature-representation ablation; not confirmatory",
                "feature_families": {
                    "distribution": list(DISTRIBUTION_NAMES),
                    "graph": list(GRAPH_NAMES),
                },
                "alphas": alphas,
                "results": results,
                "config_sha256": sha256_path(config_path),
                "rows_sha256": sha256_path(rows_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwritten: {output}")


if __name__ == "__main__":
    main()
