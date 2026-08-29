"""Stage E4 runner: can a model predict the per-state optimum from the state?

Consumes the rows written by the Stage E4 screen, regenerates the same decision
states deterministically, extracts features that a deployed policy could
actually observe at decision time, and runs the leave-one-generation-seed-out
ranking audit.

Trains no deployed policy and writes no checkpoint. The fitted ridge models
exist only to answer whether the signal is present in the state; they are
discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.audit_overtime_headroom_e2 import (
    anchor_action,
    build_scenario_env,
    generate_states,
    load_screen_config,
    scenario_env_dict,
    sha256_path,
    validate_config,
)
from evaluation.ranking_feasibility import leave_one_seed_out_ranking, ranking_gate
from evaluation.run_full_benchmark import load_benchmark_plan, select_scenarios
from evaluation.state_dependence_value import validation_means
from src.baselines.heuristics import get_heuristic_class

DEFAULT_CONFIG = Path("experiments/configs/continuous_overtime_critic_ranking_e4.json")

FEATURE_NAMES = (
    "waiting_sum",
    "waiting_max",
    "waiting_std",
    "idle_sum",
    "idle_min",
    "reagents_sum",
    "reagents_min",
    "bound_clinics",
    "bound_shortfall_sum",
    "bound_shortfall_max",
    "at_risk_sum",
    "near_expiry_sum",
    "in_production_sum",
    "normalized_time",
    "demand_sum",
)


def decision_features(env) -> np.ndarray:
    """Network summaries a deployed policy could read at the decision epoch.

    Deliberately excludes anything unobservable at decision time: no realized
    future demand, no outcome of the rollout, no oracle information.
    """

    waiting = np.asarray(
        env.waiting_counts() if hasattr(env, "waiting_counts") else env.specimens,
        dtype=float,
    )
    idle = np.asarray(env.bioreactors[:, 0], dtype=float)
    reagents = np.asarray(env.reagents, dtype=float)
    usable = np.minimum(waiting, reagents)
    shortfall = np.maximum(usable - idle, 0.0)
    at_risk = (
        np.asarray(env.at_risk_counts(), dtype=float)
        if hasattr(env, "at_risk_counts")
        else np.zeros_like(waiting)
    )
    near_expiry = (
        np.asarray(env.near_expiry_counts(), dtype=float)
        if hasattr(env, "near_expiry_counts")
        else np.zeros_like(waiting)
    )
    in_production = (
        np.asarray(env._in_production_counts(), dtype=float)
        if hasattr(env, "_in_production_counts")
        else env.bioreactors[:, 1:].sum(axis=1).astype(float)
    )
    # Deliberately no surge-headroom term: it is a fixed function of the
    # configuration, constant across states, so it carries no information and
    # would only add a free parameter.
    values = (
        waiting.sum(),
        waiting.max(),
        waiting.std(),
        idle.sum(),
        idle.min(),
        reagents.sum(),
        reagents.min(),
        float((shortfall > 0).sum()),
        shortfall.sum(),
        shortfall.max(),
        at_risk.sum(),
        near_expiry.sum(),
        in_production.sum(),
        float(env.t) / float(env.config.episode_horizon),
        float(np.asarray(env.demand, dtype=float).sum()),
    )
    return np.asarray(values, dtype=float)


def collect_state_features(config: dict[str, Any]) -> tuple[dict, dict]:
    """Regenerate the screen's states and extract decision-time features."""

    plan = load_benchmark_plan(Path(config["plan"]))
    scenarios = select_scenarios(plan, config["scenarios"])
    policy = get_heuristic_class(str(config["anchor_algorithm"]))()
    epochs = [int(e) for e in config["state_generation"]["decision_epochs"]]
    features: dict[str, np.ndarray] = {}
    seed_of: dict[str, int] = {}
    for scenario in scenarios:
        env_dict = scenario_env_dict(plan, config, scenario)
        for state_seed in config["state_generation"]["seeds"]:
            policy.reset()
            states = generate_states(env_dict, policy, int(state_seed), epochs)
            env = build_scenario_env(env_dict, int(state_seed))
            for state in states:
                env.load_state_dict(state["snapshot"])
                features[state["state_id"]] = decision_features(env)
                seed_of[state["state_id"]] = int(state_seed)
    return features, seed_of


def load_outcomes(rows_path: Path) -> dict[tuple[str, str], float]:
    with open(rows_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return validation_means(rows, stream="validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_screen_config(config_path)
    validate_config(config)
    rows_path = Path(config["output_root"]) / "headroom_rows.csv"
    if not rows_path.exists():
        raise SystemExit(f"missing screen rows: {rows_path}")

    outcomes = load_outcomes(rows_path)
    features, seed_of = collect_state_features(config)
    missing = sorted({state for state, _ in outcomes} - set(features))
    if missing:
        raise RuntimeError(f"states in rows but not regenerated: {missing[:3]}")

    report = leave_one_seed_out_ranking(
        features, seed_of, outcomes, alpha=float(args.alpha)
    )
    gates = config["gates"]
    decision = ranking_gate(
        report,
        minimum_top1=float(gates["ranking_minimum_top1"]),
        minimum_pairwise=float(gates["ranking_minimum_pairwise"]),
        minimum_gain_over_state_blind=float(
            gates["ranking_minimum_gain_over_state_blind"]
        ),
    )
    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "feature_names": list(FEATURE_NAMES),
        "ridge_alpha": float(args.alpha),
        "report": report,
        "gate": decision,
        "config_sha256": sha256_path(config_path),
        "rows_sha256": sha256_path(rows_path),
    }
    output = Path(config["output_root"]) / "ranking_feasibility.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"states {report['state_count']}  ladder {report['ladder_size']}  "
          f"chance top-1 {report['chance_top1']:.3f}")
    print(f"state-blind top-1 : {report['pooled_state_blind_top1']:.3f} "
          f"(arm {report['state_blind_arm']})")
    print(f"fitted top-1      : {report['pooled_fitted_top1']:.3f} "
          f"(gate >= {gates['ranking_minimum_top1']})")
    print(f"pairwise accuracy : {report['pooled_pairwise_accuracy']:.3f} "
          f"(gate >= {gates['ranking_minimum_pairwise']})")
    print(f"worst fold top-1  : {report['worst_fold_top1']:.3f}  "
          f"worst gain {report['worst_fold_gain']:+.3f}")
    for fold in report["folds"]:
        print(f"   seed {fold['held_out_seed']}: top-1 {fold['fitted_top1']:.3f} "
              f"vs blind {fold['state_blind_top1']:.3f} "
              f"(gain {fold['gain_over_state_blind']:+.3f}), "
              f"pairwise {fold['pairwise_accuracy']:.3f}")
    print(f"DECISION: {decision['classification']} | "
          f"E5 design authorized: {decision['e5_design_authorized']}")
    print(f"written: {output}")


if __name__ == "__main__":
    main()
