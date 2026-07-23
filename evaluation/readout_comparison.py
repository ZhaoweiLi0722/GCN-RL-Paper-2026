"""Focused 20-clinic readout comparison: facility_action vs global_flat (Lever 2).

Closes the last empirical gap before the campaign. The undertraining is a 20-clinic
phenomenon (global_flat's actor head is ~1280-dim there vs ~128-dim at 2 clinics), so
this compares the two readouts *at scale* on gcn_td3 at a moderate budget. Reference
point: the mdl2 heuristic that won the 30k pilot.

Run:  PYTHONPATH=. python -m evaluation.readout_comparison
Output: results/readout_comparison/*.csv + a printed IQM summary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.aggregate_stats import aggregate_iqm, write_rows
from evaluation.evaluate_formal import evaluate_agent
from evaluation.pilot_manifest import build_agent_config
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent

ENV = "experiments/configs/20_clinic_patient_condition.json"
STEPS = 80_000
SEEDS = (0, 1)
READOUTS = ("global_flat", "facility_action")
EVAL_REPLICATIONS = 20
OUT_DIR = Path("results/readout_comparison")
METRICS = ("total_cost", "eligibility_rate_mean", "patients_lost", "material_wasted", "service_level")


def _run(variant: str, algorithm: str, seed: int, *, readout: str | None, out_dir: Path) -> list[dict[str, Any]]:
    row_path = out_dir / f"{variant}_seed{seed}.csv"
    if row_path.exists():
        from evaluation.aggregate_results import read_rows
        return read_rows([row_path])
    config = build_agent_config(algorithm, ENV, seed=seed, target_steps=STEPS)
    if readout is not None:
        config["actor_readout_mode"] = readout
    env = build_env(config, seed=seed)
    agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
    if algorithm != "mdl2":
        train_off_policy_agent(agent, env, config)
    eval_env = build_env(config, seed=seed)
    rows = evaluate_agent(
        agent, eval_env, algorithm=variant, seed=100_000 + seed,
        replications=EVAL_REPLICATIONS, max_steps=None,
    )
    write_rows(rows, row_path)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    # gcn_td3 under each readout
    for readout in READOUTS:
        for seed in SEEDS:
            variant = f"gcn_td3_{readout}"
            print(f"[train] {variant} seed={seed} steps={STEPS}", flush=True)
            all_rows.extend(_run(variant, "gcn_td3", seed, readout=readout, out_dir=OUT_DIR))
    # mdl2 heuristic reference (no training)
    for seed in SEEDS:
        print(f"[eval ] mdl2 seed={seed}", flush=True)
        all_rows.extend(_run("mdl2", "mdl2", seed, readout=None, out_dir=OUT_DIR))

    summary = aggregate_iqm(all_rows, group_by=("algorithm",), metrics=METRICS)
    write_rows(summary, OUT_DIR / "readout_comparison_summary.csv")
    print("\n=== 20-clinic readout comparison (IQM) ===")
    print(f'{"variant":<26} {"elig_iqm":>9} {"cost_iqm":>14} {"lost_iqm":>9}')
    for r in sorted(summary, key=lambda x: -float(x.get("eligibility_rate_mean_iqm", 0))):
        print(
            f'{r["algorithm"]:<26} '
            f'{float(r.get("eligibility_rate_mean_iqm", 0)):>9.4f} '
            f'{float(r.get("total_cost_iqm", 0)):>14.4g} '
            f'{float(r.get("patients_lost_iqm", 0)):>9.1f}'
        )
    print("READOUT_COMPARISON_DONE")


if __name__ == "__main__":
    main()
