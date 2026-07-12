"""Full 20-clinic campaign runner (Lever 1 scaled budget + Lever 2 facility_action).

Trains the flagship and ablations at the campaign budget and ranks them against the
heuristics with IQM + bootstrap CIs. The flagship (gcn_td3, facility_action) is run
first so the central RQ5 answer (learned vs mdl2 at proper budget) lands early.

Scope: the deterministic graph backbones use facility_action (the validated readout;
see readout_comparison in pilot-findings.md). gcn_sac/gcn_ppo are NOT in this campaign
because their stochastic actors still use global_flat, which the readout comparison
showed collapses at 20-clinic -- including them here would confound backbone with
readout. They rejoin once facility_action is added to their actor classes.

Resumable: completed per-run CSVs are skipped. Run under caffeinate for stability.
Run:  caffeinate -i env PYTHONPATH=. python -m evaluation.campaign_runner
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import aggregate_iqm, write_rows
from evaluation.campaign_manifest import CAMPAIGN_TRAIN_STEPS, CAMPAIGN_SEEDS, campaign_config
from evaluation.evaluate_formal import evaluate_agent
from evaluation.pilot_manifest import HEURISTICS
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent

# Flagship first, then ablations; RQ1 (graph vs flat) = gcn_td3 vs flat_ddpg,
# RQ2 (backbone) = gcn_td3 vs gcn_ddpg, RQ5 (learned vs heuristic) = vs mdl2.
LEARNED = ("gcn_td3", "flat_ddpg", "gcn_ddpg")
REFERENCE_HEURISTICS = ("mdl2", "mdl1", "iso", "myo")
SEEDS = CAMPAIGN_SEEDS
# Per-seed budget. Overridable via CAMPAIGN_STEPS for robustness in environments that
# kill long background runs (each seed only banks its CSV once it fully completes, so a
# shorter budget survives sleep/kill cycles better). 300k is the manifest floor; 150k is
# still ~2x the 80k at which facility_action already reached ~0.69 eligibility.
STEPS = int(os.environ.get("CAMPAIGN_STEPS", CAMPAIGN_TRAIN_STEPS))
EVAL_REPLICATIONS = 20
OUT_DIR = Path("results/campaign")
METRICS = (
    "total_cost",
    "eligibility_rate_mean",
    "patients_lost",
    "material_wasted",
    "at_risk_unserved",
    "service_level",
)


def _run(algorithm: str, seed: int) -> list[dict[str, Any]]:
    row_path = OUT_DIR / f"{algorithm}_seed{seed}.csv"
    if row_path.exists():
        return read_rows([row_path])
    config = campaign_config(algorithm, seed, steps=STEPS)
    env = build_env(config, seed=seed)
    agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
    if algorithm not in HEURISTICS:
        train_off_policy_agent(agent, env, config)
    eval_env = build_env(config, seed=seed)
    rows = evaluate_agent(
        agent, eval_env, algorithm=algorithm, seed=100_000 + seed,
        replications=EVAL_REPLICATIONS, max_steps=None,
    )
    write_rows(rows, row_path)
    return rows


def _print_ranking(summary: list[dict[str, Any]]) -> None:
    print("\n=== 20-clinic campaign ranking (IQM) ===")
    print(f'{"algorithm":<14}{"elig_iqm":>10}{"cost_iqm":>14}{"lost_iqm":>10}{"service_iqm":>12}')
    for r in sorted(summary, key=lambda x: -float(x.get("eligibility_rate_mean_iqm", 0) or 0)):
        print(
            f'{r["algorithm"]:<14}'
            f'{float(r.get("eligibility_rate_mean_iqm", 0)):>10.4f}'
            f'{float(r.get("total_cost_iqm", 0)):>14.4g}'
            f'{float(r.get("patients_lost_iqm", 0)):>10.1f}'
            f'{float(r.get("service_level_iqm", 0)):>12.4f}'
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for algo in LEARNED:
        readout = campaign_config(algo, 0).get("actor_readout_mode", "n/a")
        for seed in SEEDS:
            print(f"[train] {algo} seed={seed} steps={STEPS} readout={readout}", flush=True)
            all_rows.extend(_run(algo, seed))
    for algo in REFERENCE_HEURISTICS:
        for seed in SEEDS:
            print(f"[eval ] {algo} seed={seed}", flush=True)
            all_rows.extend(_run(algo, seed))

    summary = aggregate_iqm(all_rows, group_by=("algorithm",), metrics=METRICS)
    write_rows(summary, OUT_DIR / "campaign_summary.csv")
    _print_ranking(summary)
    print("CAMPAIGN_DONE")


if __name__ == "__main__":
    main()
