"""Experiment B: patient-condition stress.

Question: does condition-aware graph-DRL beat condition-BLIND heuristics by a larger
margin when patient deterioration is harsh? The heuristics (MYO/MDL/ISO) plan on
demand and inventory only; the DRL policy sees the survival summary and optimizes the
blended cost including the patient-loss / urgency / expiry terms. Under nominal
condition dynamics the campaign showed heuristics still win; here we stress the
dynamics (faster deterioration, tighter eligibility) to test whether DRL's
condition-awareness pays off.

Config `20_clinic_patient_condition_stress.json` keeps disruption at the nominal 0.3
and changes only the patient block (frail_decay 0.06->0.12, weibull_scale 8->4,
post_shock 2->3, eligibility 0.75->0.80), isolating the condition effect. Compare
against the nominal campaign (results/campaign/, same arms). 3 seeds, 150k steps.

Run: caffeinate -i env PYTHONPATH=. python -m evaluation.condition_stress
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

ENV = "experiments/configs/20_clinic_patient_condition_stress.json"
LEARNED = ("gcn_ddpg", "flat_ddpg")
# Include the FAIR condition/forecast-aware baselines (umyo surges toward at-risk
# clinics; fmyo uses the demand forecast) alongside the condition-blind ones. The
# real, non-strawman claim is DRL vs umyo: does learning the condition response beat
# a hand-crafted urgency rule? Beating only the condition-blind heuristics is weak.
HEUR = ("umyo", "fmyo", "mdl2", "iso", "mdl1", "myo")
SEEDS = (0, 1, 2)
STEPS = int(os.environ.get("CAMPAIGN_STEPS", 150_000))
EVAL_REPLICATIONS = 20
OUT = Path("results/condition_stress")
METRICS = ("total_cost", "eligibility_rate_mean", "patients_lost", "material_wasted", "at_risk_unserved", "service_level")


def _run(algo: str, seed: int) -> list[dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    row_path = OUT / f"{algo}_seed{seed}.csv"
    if row_path.exists():
        return read_rows([row_path])
    config = campaign_config(algo, seed, steps=STEPS, env_config_path=ENV)
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
    rows: list[dict[str, Any]] = []
    for algo in LEARNED:
        for seed in SEEDS:
            print(f"[train] stress {algo} seed={seed} steps={STEPS}", flush=True)
            rows.extend(_run(algo, seed))
    for algo in HEUR:
        for seed in SEEDS:
            print(f"[eval ] stress {algo} seed={seed}", flush=True)
            rows.extend(_run(algo, seed))
    summary = aggregate_iqm(rows, group_by=("algorithm",), metrics=METRICS)
    write_rows(summary, OUT / "summary.csv")
    print("\n=== condition-stress ranking (IQM) ===")
    for r in sorted(summary, key=lambda x: -float(x.get("eligibility_rate_mean_iqm", 0) or 0)):
        print(f'  {r["algorithm"]:<12} elig={float(r.get("eligibility_rate_mean_iqm",0)):.3f} '
              f'cost={float(r.get("total_cost_iqm",0)):.3g} lost={float(r.get("patients_lost_iqm",0)):.0f}')
    print("CONDITION_STRESS_DONE")


if __name__ == "__main__":
    main()
