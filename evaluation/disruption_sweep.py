"""Experiment A: supply-disruption severity sweep.

Question: does graph-DRL's advantage relative to heuristics grow with supplier
disruption? (The prior DRL paper found DRL wins at high disruption; and network
rebalancing should matter most when supply fails.) We sweep the disruption rate
and compare the best graph backbone (gcn_ddpg, facility_action) and the flat
ablation against the heuristics.

The 0.3 midpoint is already covered by the main campaign (results/campaign/, same
env); this runner produces the 0.05 and 0.6 endpoints. 3 seeds, 150k steps.

Run: caffeinate -i env PYTHONPATH=. python -m evaluation.disruption_sweep
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import aggregate_iqm, write_rows
from evaluation.campaign_manifest import campaign_config
from evaluation.evaluate_formal import evaluate_agent
from evaluation.pilot_manifest import HEURISTICS
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent

RATES = [
    ("0.05", "experiments/configs/20_clinic_patient_condition_disruption_0_05.json"),
    ("0.6", "experiments/configs/20_clinic_patient_condition_disruption_0_6.json"),
]
LEARNED = ("gcn_ddpg", "flat_ddpg")
HEUR = ("mdl2", "iso", "mdl1", "myo")
SEEDS = (0, 1, 2)
STEPS = int(os.environ.get("CAMPAIGN_STEPS", 150_000))
EVAL_REPLICATIONS = 20
OUT = Path("results/disruption_sweep")
METRICS = ("total_cost", "eligibility_rate_mean", "patients_lost", "material_wasted", "service_level")


def _run(rate_tag: str, algo: str, seed: int, env_config: str) -> list[dict[str, Any]]:
    d = OUT / f"rate_{rate_tag}"
    d.mkdir(parents=True, exist_ok=True)
    row_path = d / f"{algo}_seed{seed}.csv"
    if row_path.exists():
        return read_rows([row_path])
    config = campaign_config(algo, seed, steps=STEPS, env_config_path=env_config)
    env = build_env(config, seed=seed)
    agent = get_agent_class(algo)(env.observation_size, env.action_size, config)
    if algo not in HEURISTICS:
        train_off_policy_agent(agent, env, config)
    eval_env = build_env(config, seed=seed)
    rows = evaluate_agent(
        agent, eval_env, algorithm=algo, seed=100_000 + seed,
        replications=EVAL_REPLICATIONS, max_steps=None,
    )
    write_rows(rows, row_path)
    return rows


def main() -> None:
    for rate_tag, env_config in RATES:
        rows: list[dict[str, Any]] = []
        for algo in LEARNED:
            for seed in SEEDS:
                print(f"[train] rate={rate_tag} {algo} seed={seed} steps={STEPS}", flush=True)
                rows.extend(_run(rate_tag, algo, seed, env_config))
        for algo in HEUR:
            for seed in SEEDS:
                print(f"[eval ] rate={rate_tag} {algo} seed={seed}", flush=True)
                rows.extend(_run(rate_tag, algo, seed, env_config))
        summary = aggregate_iqm(rows, group_by=("algorithm",), metrics=METRICS)
        write_rows(summary, OUT / f"rate_{rate_tag}" / "summary.csv")
        print(f"\n=== disruption rate {rate_tag} (IQM) ===")
        for r in sorted(summary, key=lambda x: -float(x.get("eligibility_rate_mean_iqm", 0) or 0)):
            print(f'  {r["algorithm"]:<12} elig={float(r.get("eligibility_rate_mean_iqm",0)):.3f} '
                  f'cost={float(r.get("total_cost_iqm",0)):.3g} lost={float(r.get("patients_lost_iqm",0)):.0f}')
    print("DISRUPTION_SWEEP_DONE")


if __name__ == "__main__":
    main()
