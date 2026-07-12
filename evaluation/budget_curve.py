"""Stage 0 diagnostic: learning-curve probe over training budget.

Discriminates undertraining from a real capability gap. Trains `gcn_ddpg`
(facility_action) on the nominal 20-clinic env at increasing step budgets and
compares the cost/eligibility gap to the (already-banked) strong heuristics.
150k is anchored by the campaign/A/B runs; this fills 300k and 500k. `flat_ddpg`
is added at the top budget to check whether it is the encoder or the RL that is
starved.

Decision gate (see budget-escalation-plan.md): relative cost gap
G(s) = cost_gcn(s)/cost_bestHeur - 1. If G(500k) < 0.20 and still descending ->
undertraining (escalate); if G(500k) >= 0.35 -> real gap (pivot).

Run: caffeinate -i env PYTHONPATH=. python -m evaluation.budget_curve
Override the ladder with e.g. BUDGET_STEPS=500000 for a single-point probe.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import aggregate_iqm, write_rows
from evaluation.campaign_manifest import CAMPAIGN_ENV
from evaluation.campaign_manifest import campaign_config
from evaluation.evaluate_formal import evaluate_agent
from evaluation.pilot_manifest import HEURISTICS
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent

# (algo, steps) cells; cheapest-decisive-point first. flat_ddpg only at the top budget.
CELLS = [
    ("gcn_ddpg", 500_000),
    ("gcn_ddpg", 300_000),
    ("flat_ddpg", 500_000),
]
# Optional single-point override for the fastest first look.
_env_steps = os.environ.get("BUDGET_STEPS")
if _env_steps:
    CELLS = [("gcn_ddpg", int(_env_steps))]

SEEDS = (0, 1, 2)
EVAL_REPLICATIONS = 20
OUT = Path("results/budget_curve")
METRICS = (
    "total_cost",
    "eligibility_rate_mean",
    "patients_lost",
    "material_wasted",
    "service_level",
)


def _run(algo: str, steps: int, seed: int) -> list[dict[str, Any]]:
    d = OUT / f"steps_{steps}"
    d.mkdir(parents=True, exist_ok=True)
    row_path = d / f"{algo}_seed{seed}.csv"
    if row_path.exists():
        return read_rows([row_path])
    config = campaign_config(algo, seed, steps=steps, env_config_path=CAMPAIGN_ENV)
    env = build_env(config, seed=seed)
    agent = get_agent_class(algo)(env.observation_size, env.action_size, config)
    if algo not in HEURISTICS:
        train_off_policy_agent(agent, env, config)
    eval_env = build_env(config, seed=seed)
    rows = evaluate_agent(
        agent, eval_env, algorithm=algo, seed=300_000 + seed,
        replications=EVAL_REPLICATIONS, max_steps=None,
    )
    for r in rows:
        r["train_steps"] = steps
    write_rows(rows, row_path)
    return rows


def main() -> None:
    by_steps: dict[int, list[dict[str, Any]]] = {}
    for algo, steps in CELLS:
        for seed in SEEDS:
            print(f"[budget] {algo} steps={steps} seed={seed}", flush=True)
            by_steps.setdefault(steps, []).extend(_run(algo, steps, seed))

    for steps in sorted(by_steps):
        summary = aggregate_iqm(by_steps[steps], group_by=("algorithm",), metrics=METRICS)
        for r in summary:
            r["train_steps"] = steps
        write_rows(summary, OUT / f"steps_{steps}" / "summary.csv")
        print(f"\n=== budget {steps} (IQM) ===")
        for r in sorted(summary, key=lambda x: float(x.get("total_cost_iqm", 0) or 0)):
            print(f'  {r["algorithm"]:<12} cost={float(r.get("total_cost_iqm",0)):.4g} '
                  f'elig={float(r.get("eligibility_rate_mean_iqm",0)):.3f}')
    print("BUDGET_CURVE_DONE")


if __name__ == "__main__":
    main()
