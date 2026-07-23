"""Experiment C (flagship): forecast error, redeemed.

Direct, referee-proof redo of the prior DRL paper's Case II. Both the learned
policy and the forecast-aware heuristic (`fmyo`) consume the **same** demand-
forecast signal (I1 — no fixed-wrong-point handicap given only to the baseline).
We train ONE policy per seed with the forecast error randomized over a training
support, then test it across a range of errors — including values **outside** the
training support (I2 — true OOD, not interpolation). Non-stationary demand shocks
(fixed across regimes) make the forecast genuinely worth using. IQM + bootstrap
CIs over seeds (I3).

- Train support: demand_forecast_error ~ U[0, 0.4]  (per-episode randomization hook)
- Eval regimes: {0.0, 0.2, 0.4} in-distribution, {0.6, 0.8} OOD
- Learned: gcn_ddpg (facility_action) headline + flat_ddpg contrast
- Heuristics: fmyo (fair, forecast-aware) + umyo + blind (mdl2/iso/mdl1/myo)

Claim tested: graph-DRL degrades more gracefully than the forecast-aware
heuristic as the shared forecast drifts, especially OOD.

Run: caffeinate -i env PYTHONPATH=. python -m evaluation.forecast_robustness
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

ENV_CONFIG = "experiments/configs/20_clinic_patient_condition_forecast.json"
TRAIN_ERROR_RANGE = (0.0, 0.4)          # DRL training support
EVAL_ERRORS = (0.0, 0.2, 0.4, 0.6, 0.8)  # 0.6, 0.8 are OOD (> train support)
IN_DIST_MAX = TRAIN_ERROR_RANGE[1]
LEARNED = ("gcn_ddpg", "flat_ddpg")
HEUR = ("fmyo", "umyo", "mdl2", "iso", "mdl1", "myo")
SEEDS = (0, 1, 2, 3, 4)
STEPS = int(os.environ.get("CAMPAIGN_STEPS", 150_000))
EVAL_REPLICATIONS = 20
OUT = Path("results/forecast_robustness")
METRICS = (
    "total_cost",
    "eligibility_rate_mean",
    "patients_lost",
    "material_wasted",
    "service_level",
)


def _err_tag(error: float) -> str:
    return f"{error:.2f}".replace(".", "_")


def _build_eval_env(algo: str, seed: int, error: float):
    """Fixed-forecast-error eval env (randomization off, so the override holds)."""
    eval_env = build_env(
        campaign_config(algo, seed, steps=STEPS, env_config_path=ENV_CONFIG), seed=seed
    )
    eval_env.demand_forecast_error = float(error)
    return eval_env


def _eval_at(agent: Any, eval_env, algo: str, seed: int, error: float) -> list[dict[str, Any]]:
    """Evaluate an agent on a prepared fixed-error eval env."""
    rows = evaluate_agent(
        agent, eval_env, algorithm=algo, seed=200_000 + seed,
        replications=EVAL_REPLICATIONS, max_steps=None,
    )
    for r in rows:
        r["forecast_error"] = f"{error:.2f}"
        r["regime"] = "in_dist" if error <= IN_DIST_MAX + 1e-9 else "ood"
    return rows


def _run_learned(algo: str, seed: int) -> list[dict[str, Any]]:
    """Train one policy with randomized forecast error; eval at every regime."""
    d = OUT / algo
    d.mkdir(parents=True, exist_ok=True)
    paths = {e: d / f"seed{seed}_err{_err_tag(e)}.csv" for e in EVAL_ERRORS}
    if all(p.exists() for p in paths.values()):
        return [row for p in paths.values() for row in read_rows([p])]

    config = campaign_config(algo, seed, steps=STEPS, env_config_path=ENV_CONFIG)
    train_env = build_env(config, seed=seed)
    train_env.enable_train_randomization(forecast_error_range=TRAIN_ERROR_RANGE)
    agent = get_agent_class(algo)(train_env.observation_size, train_env.action_size, config)
    print(f"[train] {algo} seed={seed} steps={STEPS} err~U{TRAIN_ERROR_RANGE}", flush=True)
    train_off_policy_agent(agent, train_env, config)

    rows: list[dict[str, Any]] = []
    for e in EVAL_ERRORS:
        eval_env = _build_eval_env(algo, seed, e)
        r = _eval_at(agent, eval_env, algo, seed, e)
        write_rows(r, paths[e])
        rows.extend(r)
    return rows


def _run_heuristic(algo: str, seed: int) -> list[dict[str, Any]]:
    """No training; evaluate the heuristic at every forecast-error regime."""
    d = OUT / algo
    d.mkdir(parents=True, exist_ok=True)
    config = campaign_config(algo, seed, steps=STEPS, env_config_path=ENV_CONFIG)
    rows: list[dict[str, Any]] = []
    for e in EVAL_ERRORS:
        p = d / f"seed{seed}_err{_err_tag(e)}.csv"
        if p.exists():
            rows.extend(read_rows([p]))
            continue
        eval_env = _build_eval_env(algo, seed, e)
        agent = get_agent_class(algo)(eval_env.observation_size, eval_env.action_size, config)
        r = _eval_at(agent, eval_env, algo, seed, e)
        write_rows(r, p)
        rows.extend(r)
    return rows


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    for algo in LEARNED:
        for seed in SEEDS:
            all_rows.extend(_run_learned(algo, seed))
    for algo in HEUR:
        for seed in SEEDS:
            print(f"[eval ] {algo} seed={seed}", flush=True)
            all_rows.extend(_run_heuristic(algo, seed))

    # Per-error IQM/CI table, in-dist vs OOD flagged.
    for e in EVAL_ERRORS:
        subset = [r for r in all_rows if r.get("forecast_error") == f"{e:.2f}"]
        summary = aggregate_iqm(subset, group_by=("algorithm",), metrics=METRICS)
        regime = "in_dist" if e <= IN_DIST_MAX + 1e-9 else "OOD"
        for r in summary:
            r["forecast_error"] = f"{e:.2f}"
            r["regime"] = regime
        write_rows(summary, OUT / f"summary_err{_err_tag(e)}.csv")
        print(f"\n=== forecast_error {e:.2f} ({regime}) IQM ===")
        for r in sorted(summary, key=lambda x: float(x.get("total_cost_iqm", 0) or 0)):
            print(f'  {r["algorithm"]:<12} cost={float(r.get("total_cost_iqm",0)):.4g} '
                  f'elig={float(r.get("eligibility_rate_mean_iqm",0)):.3f} '
                  f'lost={float(r.get("patients_lost_iqm",0)):.0f}')
    print("FORECAST_ROBUSTNESS_DONE")


if __name__ == "__main__":
    main()
