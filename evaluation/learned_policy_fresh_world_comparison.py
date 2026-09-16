"""Evaluate the locally archived learned policies on the fresh-seed worlds used by the anchor screens.

The 2026-09-15/16 heuristic screens (prior-oracle, order-up-to, lookahead) ran
MDL-2 variants on seed stream 99.7M with 100 paired worlds per routing
scenario. This script evaluates the locally archived Stage C development
checkpoints (GCN-TD3 and flat-TD3 residual policies, seeds 20-22, final and
frozen-pretrain variants) on exactly those worlds through the formal
evaluator's own `evaluate_deployment_candidate`, so the learned policies, the
executed MDL-2 anchor, and the retuned heuristics can be put in one table with
world-level pairing. Optionally evaluates the same frozen actors zero-shot on
an MDL-3 anchor (`residual_action.base_policy_config = {lookahead_periods: 3}`),
which is an out-of-distribution probe, not a trained configuration.

No training. No protected seed stream. Formal DDPG checkpoints are not on this
machine; the TD3 backbone ablation is what is available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_multiscenario_network_residual import (  # noqa: E402
    deep_update_dict,
    evaluate_deployment_candidate,
    resolve_manifest_artifact_path,
)
from evaluation.run_full_benchmark import load_benchmark_plan, select_scenarios  # noqa: E402
from src.rl.config import load_config  # noqa: E402
from src.rl.experiment import write_rows  # noqa: E402

MANIFEST = "results/patient_indexed_specimen_routing_stage_c_td3_development/training/patient_indexed_specimen_routing_stage_c_td3_100/training_manifest.json"
SCENARIOS = ("routing_nominal_history", "routing_abrupt_regime_shift", "routing_regional_drift", "routing_compound_regional_stress")
CANDIDATE = {"group_thresholds": [], "scale": 1.0, "use_checkpoint_group_thresholds": True}
CLINICAL = {"mode": "paired_ci", "z_value": 1.96,
            "margins": {"completion_service_level": 0.001, "patient_ineligibility_during_manufacturing_rate": 0.001, "patients_lost": 1.0}}


def variant_checkpoint(run: dict[str, Any], variant: str) -> str:
    final = resolve_manifest_artifact_path(run["checkpoint"])
    if variant == "final":
        return str(final)
    if variant == "pretrain":
        return str(final.with_name(final.name.replace(f"_episode{run.get('episodes', 100)}", "_pretrain").replace("_episode100", "_pretrain")))
    raise ValueError(variant)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    plan = load_benchmark_plan(Path(job["plan"]))
    scenarios = select_scenarios(plan, list(job["scenarios"]))
    snapshot = load_config(resolve_manifest_artifact_path(job["config"]))
    snapshot = deep_update_dict(snapshot, {"device": "cpu"})
    if job["anchor_lookahead"] is not None:
        snapshot = deep_update_dict(snapshot, {"residual_action": {"base_policy_config": {"lookahead_periods": int(job["anchor_lookahead"])}}})
    result, cand_rows, anchor_rows = evaluate_deployment_candidate(
        plan=plan, scenarios=scenarios, scenario_names=tuple(job["scenarios"]),
        algorithm=job["algorithm"], training_seed=int(job["training_seed"]),
        checkpoint=Path(job["checkpoint"]), config_snapshot=snapshot, candidate=dict(CANDIDATE),
        evaluation_seed=int(job["evaluation_seed"]), replications=int(job["replications"]),
        max_steps=None, clinical_noninferiority=CLINICAL, precomputed_anchor_rows=None,
    )
    label = f"{job['algorithm']}:{job['variant']}" + (f":anchor_mdl{job['anchor_lookahead']}" if job["anchor_lookahead"] else "")
    for r in cand_rows:
        r["policy_label"] = label; r["anchor_lookahead"] = job["anchor_lookahead"] or 2
    for r in anchor_rows:
        r["policy_label"] = f"anchor_mdl{job['anchor_lookahead'] or 2}"; r["anchor_lookahead"] = job["anchor_lookahead"] or 2
    return {"job": {k: v for k, v in job.items()}, "label": label, "aggregate": result["aggregate"],
            "per_scenario": {k: {kk: vv for kk, vv in v.items() if kk != "residual_usage"} for k, v in result["per_scenario"].items()},
            "residual_usage": result["aggregate"].get("residual_usage"),
            "candidate_rows": cand_rows, "anchor_rows": anchor_rows, "seconds": time.perf_counter() - started}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--plan", default="experiments/configs/patient_indexed_specimen_routing_benchmark.json")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--algorithms", nargs="+", default=["gcn_residual_mdl2_network_td3_bc", "flat_residual_mdl2_network_td3_bc"])
    p.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22])
    p.add_argument("--variants", nargs="+", default=["final", "pretrain"])
    p.add_argument("--zero-shot-mdl3", action="store_true", help="also evaluate final actors on an MDL-3 anchor (out-of-distribution probe)")
    p.add_argument("--evaluation-seed", type=int, default=99_700_000)
    p.add_argument("--replications", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-root", default="results/learned_policy_fresh_world_comparison")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    runs = [r for r in manifest["runs"] if r["algorithm"] in args.algorithms and int(r["seed"]) in args.seeds]
    jobs = []
    for r in runs:
        for v in args.variants:
            jobs.append({"plan": args.plan, "scenarios": list(args.scenarios), "algorithm": r["algorithm"], "training_seed": int(r["seed"]),
                         "config": r["config"], "checkpoint": variant_checkpoint(r, v), "variant": v, "anchor_lookahead": None,
                         "evaluation_seed": args.evaluation_seed, "replications": args.replications})
        if args.zero_shot_mdl3:
            jobs.append({**jobs[-1], "checkpoint": variant_checkpoint(r, "final"), "variant": "final", "anchor_lookahead": 3})
    for j in jobs:
        if not Path(j["checkpoint"]).is_file():
            raise SystemExit(f"missing checkpoint {j['checkpoint']}")
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    started = time.perf_counter(); results = []
    if args.workers <= 1:
        for j in jobs:
            r = run_job(j); results.append(r); print(f"done {r['label']} seed{j['training_seed']} gap={r['aggregate']['cost_gap_pct']:+.4f}% in {r['seconds']:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(run_job, jobs):
                results.append(r); print(f"done {r['label']} seed{r['job']['training_seed']} gap={r['aggregate']['cost_gap_pct']:+.4f}% in {r['seconds']:.0f}s", flush=True)
    cand = [row for r in results for row in r["candidate_rows"]]
    anch = [row for r in results for row in r["anchor_rows"]]
    write_rows(cand, out / "candidate_rows.csv"); write_rows(anch, out / "anchor_rows.csv")
    summary = {"evaluation_seed": args.evaluation_seed, "replications": args.replications, "scenarios": list(args.scenarios),
               "wall_seconds": time.perf_counter() - started,
               "runs": [{"label": r["label"], "training_seed": r["job"]["training_seed"], "variant": r["job"]["variant"], "anchor_lookahead": r["job"]["anchor_lookahead"] or 2,
                         "cost_gap_pct": r["aggregate"]["cost_gap_pct"], "clinically_noninferior": r["aggregate"]["clinically_noninferior"],
                         "per_scenario_gap_pct": {k: v["cost_gap_pct"] for k, v in r["per_scenario"].items()},
                         "clinical_intervals": r["aggregate"]["clinical_metric_intervals"], "residual_usage": r["residual_usage"], "seconds": r["seconds"]} for r in results]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    for r in summary["runs"]:
        print(f"{r['label']:52s} seed{r['training_seed']} gap {r['cost_gap_pct']:+.4f}%  clin {'ok' if r['clinically_noninferior'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
