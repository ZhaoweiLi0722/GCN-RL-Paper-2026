"""Specimen-routing headroom and label-stability screen on a chosen heuristic anchor.

Question. Does specimen routing still offer state-dependent, replicable headroom
when the anchor is coverage-correct (MDL-3, lookahead = production lead time)
rather than the under-covering MDL-2 the formal study used? Run the identical
protocol on both anchors so the answer is a paired contrast, not an absolute.

Protocol (no training, no checkpoint, no protected seed stream):

1. For each anchor and each routing-primary scenario, roll the anchor on
   ``trajectories`` fresh state seeds and snapshot the environment at the
   prespecified ``decision_epochs``.
2. At each snapshot form the five legal specimen options used by Stage G1 and
   the routing-label-budget study: the anchor action and shifts
   {-0.10, -0.05, +0.05, +0.10} applied to the mean-centred specimen pressure
   pattern (a uniform shift executes nothing because specimen flow is
   conserved). Keep the state only if all five arms execute DISTINCT specimen
   flows (the quantization filter); report the drop rate as coverage.
3. Continue every arm to episode end under the anchor on two disjoint seed
   families, ``discovery`` and ``validation``, recording remaining cost and the
   clinical counters. Continuations restore the full patient registry, so the
   labels are conditional simulator diagnostics, not information-matched gains.
4. Apply the pre-written gate functions unchanged:
   ``evaluation.label_stability`` (best-action agreement, material-pair sign
   agreement) and ``evaluation.state_dependence_value`` (prospective value of
   state dependence versus the best constant option, both selected on
   discovery and scored on validation).

Thresholds are the ones written in
``specs/2026-09-15-network-input-dependence/plan.md`` Stage 1 before this run:
coverage >= 80% of sampled states, best-action agreement >= 0.70 pooled and
>= 0.60 in every trajectory, pairwise sign agreement >= 0.65 over pairs
separated by >= 250,000 cost units in both streams, prospective state-
dependence value >= 0.5% of anchor remaining cost with positive value in at
least four of six trajectories.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.label_stability import (  # noqa: E402
    best_action_agreement,
    label_stability_gate,
    pairwise_sign_agreement,
)
from evaluation.routing_label_budget_pool import (  # noqa: E402
    anchor_action,
    arm_action,
    centred_pressure,
    executed_signature,
    rollout,
)
from evaluation.run_full_benchmark import (  # noqa: E402
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from evaluation.state_dependence_value import (  # noqa: E402
    state_dependence_gate,
    state_dependence_value,
    validation_means,
)
from src.baselines.heuristics import get_heuristic_class  # noqa: E402
from src.rl.experiment import build_env  # noqa: E402

ANCHORS: dict[str, dict[str, Any]] = {
    "mdl2": {"algorithm": "mdl2", "config": {}},
    "mdl3": {"algorithm": "mdl2", "config": {"lookahead_periods": 3}},
}
SCENARIOS = (
    "routing_nominal_history",
    "routing_abrupt_regime_shift",
    "routing_regional_drift",
    "routing_compound_regional_stress",
)
SHIFTS = (0.0, -0.10, -0.05, 0.05, 0.10)
DECISION_EPOCHS = (4, 8, 12, 16, 24, 32, 40, 44)
ENV_OVERRIDES = {
    "enable_specimen_routing": True,
    "enable_overtime_control": False,
    "enable_production_throttle": False,
}
GATES = {
    "coverage_min_fraction": 0.80,
    "best_action_agreement_min_pooled": 0.70,
    "best_action_agreement_min_per_trajectory": 0.60,
    "pairwise_sign_agreement_min": 0.65,
    "material_pair_threshold": 250_000.0,
    "state_dependence_min_fraction": 0.005,
    "state_dependence_positive_trajectories_min": 4,
}
# Fresh seed families. Nothing else in the repository uses 99.95M-99.99M.
STATE_SEED_BASE = 99_950_000
DISCOVERY_SEED_BASE = 99_960_000
VALIDATION_SEED_BASE = 99_970_000

# Protected streams (from routing_label_budget_pool.validate_config plus this session's screens).
PROTECTED = [(91_100_000, 91_100_000), (94_000_000, 94_100_000), (96_000_000, 96_999_999),
             (97_100_000, 97_300_009), (98_100_000, 98_300_000), (99_100_000, 99_940_000),
             (101_000_000, 101_999_999), (103_000_000, 103_999_999), (107_000_000, 107_999_999)]


def arm_name(shift: float) -> str:
    return "anchor" if shift == 0.0 else f"shift_{shift:+.2f}"


def make_anchor(name: str):
    spec = ANCHORS[name]
    return get_heuristic_class(spec["algorithm"])(None, None, dict(spec["config"]))


def scenario_env_dict(plan, scenario) -> dict[str, Any]:
    env = dict(make_scenario_env_config(plan, "mdl2", scenario))
    env.update(ENV_OVERRIDES)
    env["scenario_name"] = scenario["name"]
    return env


def build_scenario_env(env_dict: dict[str, Any], seed: int):
    env = build_env({"env": dict(env_dict)}, seed)
    if getattr(env, "scenario_name", None) != env_dict["scenario_name"]:
        raise RuntimeError("scenario provenance mismatch at construction")
    if not env.env_config.enable_specimen_routing:
        raise RuntimeError("built environment lost specimen routing")
    return env


def check_seed(seed: int) -> None:
    for lo, hi in PROTECTED:
        if lo <= seed <= hi:
            raise ValueError(f"seed {seed} lies in a protected range [{lo}, {hi}]")


def collect_states(env_dict, policy, state_seed: int, epochs, shifts):
    env = build_scenario_env(env_dict, state_seed)
    env.reset(seed=state_seed)
    kept, dropped = [], []
    horizon = int(env.config.episode_horizon)
    for epoch in sorted(epochs):
        if epoch >= horizon:
            raise ValueError(f"decision epoch {epoch} is not before the horizon {horizon}")
        while env.t < epoch:
            env.step(anchor_action(policy, env))
        snapshot = env.state_dict()
        pattern = centred_pressure(env)
        signatures, actions = [], {}
        for shift in shifts:
            env.load_state_dict(snapshot)
            env.rng = np.random.default_rng(int(state_seed) + 1)
            action = arm_action(policy, env, shift, pattern)
            actions[shift] = action
            _, _, _, info = env.step(action)
            signatures.append(executed_signature(info))
        env.load_state_dict(snapshot)
        record = {
            "state_id": f"{env_dict['scenario_name']}:seed{state_seed}:epoch{epoch}",
            "state_seed": int(state_seed),
            "epoch": int(epoch),
            "snapshot": snapshot,
            "actions": actions,
            "distinct_executions": len(set(signatures)),
        }
        (kept if len(set(signatures)) == len(shifts) else dropped).append(record)
    return kept, dropped


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """One (anchor, scenario, state seed): generate states, roll every arm on every stream."""

    started = time.perf_counter()
    policy = make_anchor(job["anchor"])
    kept, dropped = collect_states(job["env_dict"], policy, job["state_seed"], job["epochs"], job["shifts"])
    rows: list[dict[str, Any]] = []
    env = build_scenario_env(job["env_dict"], job["state_seed"])
    for record in kept:
        for shift in job["shifts"]:
            action = record["actions"][shift]
            for stream, seeds in job["streams"].items():
                for world in seeds:
                    env.load_state_dict(record["snapshot"])
                    env.rng = np.random.default_rng(int(world))
                    metrics = rollout(env, policy, action)
                    rows.append({
                        "anchor": job["anchor"],
                        "scenario": job["env_dict"]["scenario_name"],
                        "state_id": record["state_id"],
                        "state_seed": record["state_seed"],
                        "epoch": record["epoch"],
                        "arm": arm_name(shift),
                        "shift": shift,
                        "stream": stream,
                        "world_seed": int(world),
                        **metrics,
                    })
    return {
        "anchor": job["anchor"],
        "scenario": job["env_dict"]["scenario_name"],
        "state_seed": job["state_seed"],
        "kept": [r["state_id"] for r in kept],
        "dropped": [{"state_id": r["state_id"], "distinct_executions": r["distinct_executions"]} for r in dropped],
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def analyze_group(rows: list[dict[str, Any]], *, sampled_states: int) -> dict[str, Any]:
    """Gates for one set of rows (one anchor, one or all scenarios)."""

    disc = validation_means(rows, stream="discovery")
    val = validation_means(rows, stream="validation")
    states = sorted({s for s, _ in val})
    out: dict[str, Any] = {
        "sampled_states": sampled_states,
        "actionable_states": len(states),
        "coverage_fraction": (len(states) / sampled_states) if sampled_states else 0.0,
    }
    out["coverage_passed"] = bool(out["coverage_fraction"] >= GATES["coverage_min_fraction"])
    if not states:
        out["classification"] = "no_actionable_states"
        return out

    agreement = best_action_agreement(disc, val)
    pairwise = pairwise_sign_agreement(disc, val, material_threshold=GATES["material_pair_threshold"])
    stability = label_stability_gate(
        agreement, pairwise,
        minimum_best_action_agreement=GATES["best_action_agreement_min_pooled"],
        minimum_pairwise_agreement=GATES["pairwise_sign_agreement_min"],
    )
    # Agreement including the anchor as a candidate (the plan's convention).
    ladder_all = sorted({a for _, a in val})
    incl = sum(
        1 for s in states
        if min(ladder_all, key=lambda a: disc[(s, a)]) == min(ladder_all, key=lambda a: val[(s, a)])
    ) / len(states)
    # Per-trajectory agreement and state-dependence sign.
    traj = sorted({s.split(":epoch")[0] for s in states})
    per_traj: dict[str, Any] = {}
    for t in traj:
        t_states = [s for s in states if s.startswith(t + ":epoch")]
        d = {k: v for k, v in disc.items() if k[0] in t_states}
        v = {k: vv for k, vv in val.items() if k[0] in t_states}
        agr = best_action_agreement(d, v)["best_action_agreement"]
        sdv = state_dependence_value(v, selection_means=d)
        per_traj[t] = {
            "states": len(t_states),
            "best_action_agreement": agr,
            "prospective_value_fraction": sdv["prospective_value_of_state_dependence_fraction"],
        }
    sdv_all = state_dependence_value(val, selection_means=disc)
    sd_gate = state_dependence_gate(sdv_all, minimum_fraction=GATES["state_dependence_min_fraction"])
    positive_traj = sum(1 for p in per_traj.values() if p["prospective_value_fraction"] > 0)
    min_traj_agreement = min(p["best_action_agreement"] for p in per_traj.values())

    # Clinical screening guardrails on the prospective per-state choice (validation stream).
    ladder = sorted({a for _, a in val if a != "anchor"})
    choice = {s: min(ladder, key=lambda a: disc[(s, a)]) for s in states}
    def stream_mean(metric, pick):
        vals = {}
        for r in rows:
            if r["stream"] != "validation":
                continue
            key = (r["state_id"], r["arm"])
            vals.setdefault(key, []).append(float(r[metric]))
        return float(np.mean([np.mean(vals[(s, pick(s))]) for s in states]))
    clinical = {}
    for metric in ("completion_service_level", "patients_lost", "manufacturing_ineligibility"):
        a = stream_mean(metric, lambda s: "anchor")
        c = stream_mean(metric, lambda s: choice[s])
        clinical[metric] = {"anchor": a, "prospective_choice": c, "difference": c - a}
    clinical_ok = (
        clinical["completion_service_level"]["difference"] >= 0.0
        and clinical["patients_lost"]["difference"] <= 0.0
        and clinical["manufacturing_ineligibility"]["difference"] <= 0.0
    )

    out.update({
        "best_action_agreement_excl_anchor": agreement["best_action_agreement"],
        "best_action_agreement_incl_anchor": incl,
        "pairwise_sign_agreement": pairwise["pairwise_sign_agreement"],
        "material_pairs": pairwise["material_pairs"],
        "min_trajectory_best_action_agreement": min_traj_agreement,
        "label_stability_gate": stability,
        "per_trajectory_floor_passed": bool(min_traj_agreement >= GATES["best_action_agreement_min_per_trajectory"]),
        "state_dependence": {k: v for k, v in sdv_all.items()},
        "state_dependence_gate": sd_gate,
        "positive_value_trajectories": positive_traj,
        "trajectories": len(per_traj),
        "positive_trajectories_passed": bool(positive_traj >= GATES["state_dependence_positive_trajectories_min"]),
        "per_trajectory": per_traj,
        "clinical_screen": clinical,
        "clinical_screen_passed": bool(clinical_ok),
        "value_of_channel_over_anchor_fraction": sdv_all["value_of_channel_over_anchor"] / sdv_all["anchor_total"],
        "prospective_choice_minus_anchor_fraction": (sdv_all["prospective_total"] - sdv_all["anchor_total"]) / sdv_all["anchor_total"],
    })
    all_pass = (
        out["coverage_passed"] and stability["label_stability_gate_passed"]
        and out["per_trajectory_floor_passed"] and sd_gate["state_dependence_gate_passed"]
        and out["positive_trajectories_passed"]
    )
    out["all_gates_passed"] = bool(all_pass)
    out["classification"] = "routing_headroom_established" if all_pass else (
        "unstable_labels" if not stability["label_stability_gate_passed"] else
        "captured_by_constant_or_insufficient" if not sd_gate["state_dependence_gate_passed"] else
        "coverage_or_trajectory_floor_failed"
    )
    return out


def render(summary: dict[str, Any]) -> str:
    L = ["# Routing headroom screen on MDL-2 versus MDL-3 anchors", ""]
    L.append(f"{summary['meta']['trajectories']} trajectories x {len(summary['meta']['decision_epochs'])} epochs per scenario, "
             f"{summary['meta']['n_discovery']} discovery + {summary['meta']['n_validation']} validation continuations per arm. "
             f"Wall {summary['wall_seconds']/60:.1f} min.")
    L += ["", "| anchor | scope | coverage | best-action agr (excl / incl anchor) | pairwise (n pairs) | min traj agr | prosp. state-dep value | pos. traj | channel over anchor | prosp. choice vs anchor | clinical | verdict |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--: | --- |"]
    for anchor, scopes in summary["analysis"].items():
        for scope, a in scopes.items():
            if "best_action_agreement_excl_anchor" not in a:
                L.append(f"| {anchor} | {scope} | {a['coverage_fraction']:.0%} | — | — | — | — | — | — | — | — | {a['classification']} |"); continue
            L.append(f"| {anchor} | {scope} | {a['coverage_fraction']:.0%} ({a['actionable_states']}/{a['sampled_states']}) | "
                     f"{a['best_action_agreement_excl_anchor']:.3f} / {a['best_action_agreement_incl_anchor']:.3f} | "
                     f"{a['pairwise_sign_agreement']:.3f} ({a['material_pairs']}) | {a['min_trajectory_best_action_agreement']:.2f} | "
                     f"{100*a['state_dependence']['prospective_value_of_state_dependence_fraction']:+.3f}% | "
                     f"{a['positive_value_trajectories']}/{a['trajectories']} | {100*a['value_of_channel_over_anchor_fraction']:+.3f}% | "
                     f"{100*a['prospective_choice_minus_anchor_fraction']:+.3f}% | {'ok' if a['clinical_screen_passed'] else 'x'} | {a['classification']} |")
    L += ["", "Columns: prospective state-dependence value = best constant (chosen on discovery) minus per-state choice (chosen on discovery), both scored on validation, as % of anchor remaining cost; "
          "channel over anchor = anchor minus best constant on validation (negative means every constant is worse than the anchor); "
          "prospective choice vs anchor = per-state choice minus anchor on validation (negative is better)."]
    return "\n".join(L) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="experiments/configs/patient_indexed_specimen_routing_benchmark.json")
    p.add_argument("--anchors", nargs="+", default=list(ANCHORS))
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--trajectories", type=int, default=6)
    p.add_argument("--decision-epochs", nargs="+", type=int, default=list(DECISION_EPOCHS))
    p.add_argument("--n-discovery", type=int, default=8)
    p.add_argument("--n-validation", type=int, default=8)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--output-root", default="results/routing_headroom_anchor_screen")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = load_benchmark_plan(Path(args.plan))
    scenarios = select_scenarios(plan, args.scenarios)
    out_root = Path(args.output_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"output root {out_root} is not empty; refusing to overwrite")
    out_root.mkdir(parents=True, exist_ok=True)

    disc = [DISCOVERY_SEED_BASE + i for i in range(args.n_discovery)]
    val = [VALIDATION_SEED_BASE + i for i in range(args.n_validation)]
    jobs = []
    for s_idx, scenario in enumerate(scenarios):
        env_dict = scenario_env_dict(plan, scenario)
        for anchor in args.anchors:
            for t in range(args.trajectories):
                state_seed = STATE_SEED_BASE + s_idx * 1000 + t
                for seed in [state_seed, state_seed + 1] + disc + val:
                    check_seed(seed)
                jobs.append({"anchor": anchor, "env_dict": env_dict, "state_seed": state_seed,
                             "epochs": list(args.decision_epochs), "shifts": list(SHIFTS),
                             "streams": {"discovery": disc, "validation": val}})
    meta = {
        "anchors": {a: ANCHORS[a] for a in args.anchors}, "scenarios": [s["name"] for s in scenarios],
        "trajectories": args.trajectories, "decision_epochs": list(args.decision_epochs), "shifts": list(SHIFTS),
        "n_discovery": args.n_discovery, "n_validation": args.n_validation,
        "discovery_seeds": disc, "validation_seeds": val, "state_seed_base": STATE_SEED_BASE,
        "gates": GATES, "env_overrides": ENV_OVERRIDES, "jobs": len(jobs),
        "caveat": "continuations restore the full patient registry (latent attributes already sampled); labels are conditional simulator diagnostics, not information-matched achievable gains",
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    started = time.perf_counter()
    results = []
    if args.workers <= 1:
        for j in jobs:
            r = run_job(j); results.append(r); print(f"done {r['anchor']} {r['scenario']} seed{r['state_seed']} kept {len(r['kept'])} in {r['seconds']:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(run_job, jobs):
                results.append(r); print(f"done {r['anchor']} {r['scenario']} seed{r['state_seed']} kept {len(r['kept'])} in {r['seconds']:.0f}s", flush=True)
    rows = [row for r in results for row in r["rows"]]
    cols = ["anchor", "scenario", "state_id", "state_seed", "epoch", "arm", "shift", "stream", "world_seed",
            "remaining_cost", "completion_service_level", "patients_lost", "manufacturing_ineligibility"]
    with open(out_root / "rows.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols); w.writeheader(); w.writerows({c: r[c] for c in cols} for r in rows)

    sampled = {(a, s): 0 for a in args.anchors for s in meta["scenarios"]}
    dropped = []
    for r in results:
        sampled[(r["anchor"], r["scenario"])] += len(r["kept"]) + len(r["dropped"])
        dropped.extend({"anchor": r["anchor"], **d} for d in r["dropped"])
    analysis: dict[str, Any] = {}
    for anchor in args.anchors:
        a_rows = [r for r in rows if r["anchor"] == anchor]
        analysis[anchor] = {"all_scenarios": analyze_group(a_rows, sampled_states=sum(v for (a, _), v in sampled.items() if a == anchor))}
        for sc in meta["scenarios"]:
            analysis[anchor][sc] = analyze_group([r for r in a_rows if r["scenario"] == sc], sampled_states=sampled[(anchor, sc)])
    summary = {"meta": meta, "analysis": analysis, "dropped_states": dropped, "row_count": len(rows),
               "wall_seconds": time.perf_counter() - started,
               "rows_sha256": hashlib.sha256((out_root / "rows.csv").read_bytes()).hexdigest()}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    md = render(summary); (out_root / "summary.md").write_text(md); print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
