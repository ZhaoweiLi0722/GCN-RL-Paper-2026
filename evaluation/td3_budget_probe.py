"""Decisive GCN-TD3 budget probe for the Howard PR decision gate.

This runner answers the open question left by the DDPG budget probe:
does the stable graph backbone (`gcn_td3`, facility_action readout) improve
when trained at 300k/500k steps on the nominal 20-clinic patient-condition env?

The run is resumable at the (algorithm, steps, seed) CSV level.

Typical full run:
    PYTHONPATH=. python -m evaluation.td3_budget_probe

Quick plumbing check:
    PYTHONPATH=. python -m evaluation.td3_budget_probe --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import aggregate_iqm, write_rows
from evaluation.campaign_manifest import CAMPAIGN_ENV, campaign_config
from evaluation.evaluate_formal import evaluate_agent
from evaluation.pilot_manifest import HEURISTICS
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent

DEFAULT_STEPS = (300_000, 500_000)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_REPLICATIONS = 20
ALGORITHM = "gcn_td3"
REFERENCE_HEURISTICS = ("mdl2", "mdl1", "iso", "myo", "umyo")
OUT = Path("results/td3_budget_probe")
METRICS = (
    "total_cost",
    "eligibility_rate_mean",
    "patients_lost",
    "material_wasted",
    "at_risk_unserved",
    "service_level",
)


def _run(algorithm: str, steps: int, seed: int, replications: int) -> list[dict[str, Any]]:
    out_dir = OUT / f"steps_{steps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    row_path = out_dir / f"{algorithm}_seed{seed}.csv"
    if row_path.exists():
        return read_rows([row_path])

    config = campaign_config(algorithm, seed, steps=steps, env_config_path=CAMPAIGN_ENV)
    env = build_env(config, seed=seed)
    agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
    if algorithm not in HEURISTICS:
        train_off_policy_agent(agent, env, config)

    eval_env = build_env(config, seed=seed)
    rows = evaluate_agent(
        agent,
        eval_env,
        algorithm=algorithm,
        seed=400_000 + seed,
        replications=replications,
        max_steps=None,
    )
    for row in rows:
        row["train_steps"] = steps
        row["probe"] = "td3_budget"
    write_rows(rows, row_path)
    return rows


def _print_summary(steps: int, summary: list[dict[str, Any]]) -> None:
    print(f"\n=== GCN-TD3 budget probe: {steps} steps (IQM) ===")
    print(f'{"algorithm":<12}{"cost_iqm":>14}{"elig_iqm":>10}{"lost_iqm":>10}{"service":>10}')
    for row in sorted(summary, key=lambda item: float(item.get("total_cost_iqm", 0) or 0)):
        print(
            f'{row["algorithm"]:<12}'
            f'{float(row.get("total_cost_iqm", 0)):>14.4g}'
            f'{float(row.get("eligibility_rate_mean_iqm", 0)):>10.4f}'
            f'{float(row.get("patients_lost_iqm", 0)):>10.1f}'
            f'{float(row.get("service_level_iqm", 0)):>10.4f}'
        )

    td3 = next((row for row in summary if row.get("algorithm") == ALGORITHM), None)
    heuristics = [row for row in summary if row.get("algorithm") in REFERENCE_HEURISTICS]
    if td3 and heuristics:
        best = min(heuristics, key=lambda row: float(row.get("total_cost_iqm", "inf") or "inf"))
        td3_cost = float(td3.get("total_cost_iqm", 0) or 0)
        best_cost = float(best.get("total_cost_iqm", 0) or 0)
        gap = td3_cost / best_cost - 1.0 if best_cost else float("nan")
        print(f"\nBest heuristic: {best['algorithm']} cost={best_cost:.4g}")
        print(f"GCN-TD3 gap vs best heuristic: {gap:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny budget for a quick plumbing check.")
    args = parser.parse_args()

    steps = tuple(args.steps)
    seeds = tuple(args.seeds)
    replications = int(args.replications)
    if args.smoke:
        steps = (200,)
        seeds = (0,)
        replications = 2

    roster = (ALGORITHM,)
    if not args.no_heuristics:
        roster += REFERENCE_HEURISTICS

    for step_budget in steps:
        all_rows: list[dict[str, Any]] = []
        for algorithm in roster:
            readout = campaign_config(algorithm, 0).get("actor_readout_mode", "n/a")
            for seed in seeds:
                print(
                    f"[td3-budget] algorithm={algorithm} steps={step_budget} "
                    f"seed={seed} readout={readout}",
                    flush=True,
                )
                all_rows.extend(_run(algorithm, step_budget, seed, replications))

        summary = aggregate_iqm(all_rows, group_by=("algorithm",), metrics=METRICS)
        for row in summary:
            row["train_steps"] = step_budget
            row["probe"] = "td3_budget"
        write_rows(summary, OUT / f"steps_{step_budget}" / "summary.csv")
        _print_summary(step_budget, summary)

    print("TD3_BUDGET_PROBE_DONE")


if __name__ == "__main__":
    main()
