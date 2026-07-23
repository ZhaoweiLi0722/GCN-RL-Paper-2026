"""Two-stage lean patient pilot (Phase 7).

Stage A screens the full roster on the 2-clinic patient regime; results are
ranked by patient eligibility (IQM across seeds); the top graph backbone is
promoted, with fixed anchors, to a Stage B confirmation on the 20-clinic regime.

Every learned agent is trained in-process (`train_off_policy_agent`, generic over
the agent interface) then Monte-Carlo evaluated on fresh episodes; heuristics are
evaluated directly. Per-(algorithm, seed) eval rows are written so `--resume`
can skip completed work. Heavy runs are compute-bound — the flagship decision is
recorded from whatever completed (partial runs reported as partial).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.aggregate_results import read_rows
from evaluation.aggregate_stats import aggregate_iqm, write_rows
from evaluation.evaluate_formal import evaluate_agent
from evaluation.pilot_manifest import (
    DEFAULT_EVAL_REPLICATIONS,
    DEFAULT_TRAIN_STEPS,
    FLAGSHIP_CANDIDATES,
    HEURISTICS,
    SEEDS,
    STAGE_A_ENV,
    STAGE_A_ROSTER,
    STAGE_B_ENV,
    build_agent_config,
    stage_b_roster,
)
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent

RANK_METRIC = "eligibility_rate_mean"  # higher is better
TIE_BREAK_METRIC = "total_cost"        # lower is better
PILOT_METRICS = (
    "total_cost",
    "eligibility_rate",
    "eligibility_rate_mean",
    "patients_lost",
    "material_wasted",
    "at_risk_unserved",
    "service_level",
)


def run_stage(
    *,
    stage: str,
    env_config_path: str,
    roster: list[str],
    seeds: tuple[int, ...],
    target_steps: int,
    eval_replications: int,
    out_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage_dir = out_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for algorithm in roster:
        for seed in seeds:
            row_path = stage_dir / f"{algorithm}_seed{seed}.csv"
            if resume and row_path.exists():
                rows.extend(read_rows([row_path]))
                continue
            config = build_agent_config(algorithm, env_config_path, seed=seed, target_steps=target_steps)
            env = build_env(config, seed=seed)
            agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
            if algorithm not in HEURISTICS:
                train_off_policy_agent(agent, env, config)
            eval_env = build_env(config, seed=seed)
            eval_rows = evaluate_agent(
                agent,
                eval_env,
                algorithm=algorithm,
                seed=100_000 + seed,  # eval seeds disjoint from training seeds
                replications=eval_replications,
                max_steps=None,
            )
            write_rows(eval_rows, row_path)
            rows.extend(eval_rows)
    return rows


def rank_by_eligibility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """IQM summary per algorithm, sorted by eligibility IQM desc, cost IQM asc."""

    summary = aggregate_iqm(rows, group_by=("algorithm",), metrics=PILOT_METRICS)

    def sort_key(row: dict[str, Any]) -> tuple[float, float]:
        elig = row.get(f"{RANK_METRIC}_iqm", float("-inf"))
        cost = row.get(f"{TIE_BREAK_METRIC}_iqm", float("inf"))
        return (-float(elig), float(cost))

    return sorted(summary, key=sort_key)


def select_top_backbone(ranking: list[dict[str, Any]]) -> str | None:
    for row in ranking:  # ranking is already best-first
        if row["algorithm"] in FLAGSHIP_CANDIDATES:
            return row["algorithm"]
    return None


def run_two_stage_pilot(
    *,
    stage_a_roster: list[str] | None = None,
    seeds: tuple[int, ...] = SEEDS,
    target_steps: int = DEFAULT_TRAIN_STEPS,
    eval_replications: int = DEFAULT_EVAL_REPLICATIONS,
    stage_a_env: str = STAGE_A_ENV,
    stage_b_env: str = STAGE_B_ENV,
    out_dir: str | Path = "results/patient_pilot",
    resume: bool = False,
    run_stage_b: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    roster = list(stage_a_roster) if stage_a_roster is not None else list(STAGE_A_ROSTER)

    a_rows = run_stage(
        stage="A", env_config_path=stage_a_env, roster=roster, seeds=seeds,
        target_steps=target_steps, eval_replications=eval_replications,
        out_dir=out_dir, resume=resume,
    )
    a_ranking = rank_by_eligibility(a_rows)
    write_rows(a_ranking, out_dir / "stage_A_ranking.csv")
    top_backbone = select_top_backbone(a_ranking)

    result: dict[str, Any] = {
        "stage_a_ranking": a_ranking,
        "top_backbone": top_backbone,
        "stage_b_ranking": None,
        "flagship": None,
    }

    if run_stage_b and top_backbone is not None:
        b_roster = [a for a in stage_b_roster(top_backbone) if a in roster or a in HEURISTICS]
        b_rows = run_stage(
            stage="B", env_config_path=stage_b_env, roster=b_roster, seeds=seeds,
            target_steps=target_steps, eval_replications=eval_replications,
            out_dir=out_dir, resume=resume,
        )
        b_ranking = rank_by_eligibility(b_rows)
        write_rows(b_ranking, out_dir / "stage_B_ranking.csv")
        result["stage_b_ranking"] = b_ranking
        result["flagship"] = select_top_backbone(b_ranking)

    (out_dir / "pilot_result.json").write_text(
        json.dumps(
            {k: v for k, v in result.items() if k in ("top_backbone", "flagship")},
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--eval-replications", type=int, default=DEFAULT_EVAL_REPLICATIONS)
    parser.add_argument("--out-dir", default="results/patient_pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage-a-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny budgets for a plumbing check.")
    args = parser.parse_args()

    kwargs: dict[str, Any] = dict(
        seeds=tuple(args.seeds),
        target_steps=args.target_steps,
        eval_replications=args.eval_replications,
        out_dir=args.out_dir,
        resume=args.resume,
        run_stage_b=not args.stage_a_only,
    )
    if args.smoke:
        kwargs.update(
            stage_a_roster=["mdl2", "gcn_sac", "gcn_ddpg", "flat_ddpg"],
            seeds=(0, 1),
            target_steps=200,
            eval_replications=2,
        )
    result = run_two_stage_pilot(**kwargs)
    print(f"top backbone: {result['top_backbone']}  flagship: {result['flagship']}")


if __name__ == "__main__":
    main()
