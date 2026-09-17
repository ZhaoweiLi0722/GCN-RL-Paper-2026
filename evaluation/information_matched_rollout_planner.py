"""Measurement A: an information-matched rollout planner (achievable-improvement upper bound on J*).

At each weekly decision the planner enumerates a small set of legal candidate
actions, simulates each to episode end on K common-random-number worlds under
the MDL-2 continuation policy, executes the candidate with the lowest mean
cost, and replans next epoch. Its realized cost is a feasible-policy value, so
it bounds the optimal cost from above and, if below MDL-2 or the learned
policy, proves that improvement is achievable.

Information contract (the proposal's mandatory audit). Rollout worlds are
built from a deep copy of the live environment in which every private
quantity is replaced by a draw conditional on observables only:

* patient latent attributes (health index, deterioration epoch) are resampled
  from the enrollment prior conditional on the patient's observable age,
  current survival, and risk type (risk-type counts are part of the
  observation), then survival is recomputed from the resampled attributes;
* true demand rates, regime multipliers, and active demand shocks are replaced
  by the observable estimate: the 12-epoch arrival history mean blended with
  the prior estimate, no regime schedule, no active shock;
* remaining durations of regional supplier disruptions are cleared (current
  availability is observed; its persistence is not);
* future draws use a fresh RNG per rollout world.

`--privileged` disables the resampling and keeps true rates, which turns the
planner into a diagnostic of the value of information, not an achievable
policy. It is labelled as such in the output.

Two candidate classes, per the proposal:
  restricted: MDL-2 anchor plus the four G1 specimen options (+-0.05, +-0.10);
  full: restricted, plus the same five built on the MDL-3 anchor, plus
        +-0.05 reagent-transfer and +-0.05 capacity-transfer options on MDL-2.
Ties go to the anchor. Settings are fixed before evaluation; no tuning.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_formal import evaluate_agent  # noqa: E402
from evaluation.prior_oracle_gap_screen import compose_scenarios  # noqa: E402
from evaluation.run_full_benchmark import load_benchmark_plan  # noqa: E402
from src.baselines.heuristics import get_heuristic_class  # noqa: E402
from src.env.patient_condition import PatientStatus  # noqa: E402
from src.rl.experiment import build_env, write_rows  # noqa: E402
from src.rl.residual_options import (  # noqa: E402
    make_explicit_residual_option_specs,
    residual_option_actions_from_env,
)

SCENARIOS = ("routing_nominal_history", "routing_abrupt_regime_shift", "routing_regional_drift", "routing_compound_regional_stress")
SPECIMEN_OPTIONS = [{"group": "specimen_transfer", "epsilon": e, "sign": s} for e in (0.05, 0.10) for s in (-1.0, 1.0)]
TRANSFER_OPTIONS = [{"group": g, "epsilon": 0.05, "sign": s} for g in ("reagent_transfer", "combined_transfer") for s in (-1.0, 1.0)]
MAX_AGE = 16


class PosteriorSampler:
    """Draws latent patient attributes from the enrollment prior conditional on observables."""

    def __init__(self, model, size: int = 40000, seed: int = 12345):
        rng = np.random.default_rng(seed)
        cfg = model.config
        self.model = model
        self.h = rng.uniform(0.0, 1.0, size)
        self.det = cfg.weibull_scale * rng.weibull(cfg.weibull_shape, size)
        probs = np.asarray(model.risk_type_probabilities, float); probs = probs / probs.sum()
        self.risk = rng.choice(len(probs), size=size, p=probs)
        self.mult = np.asarray(model.risk_decay_multipliers, float)[self.risk]
        self.S = np.empty((size, MAX_AGE + 1))
        for a in range(MAX_AGE + 1):
            self.S[:, a] = [model.survival_at(a, self.h[m], self.det[m], risk_multiplier=self.mult[m]) for m in range(size)]
        self.by_risk = {k: np.flatnonzero(self.risk == k) for k in range(len(probs))}

    def draw(self, age: int, survival: float, risk_type: int, rng: np.random.Generator):
        age = int(min(max(age, 0), MAX_AGE)); pool = self.by_risk.get(int(risk_type), np.arange(len(self.h)))
        tol = 0.002
        while True:
            cand = pool[np.abs(self.S[pool, age] - survival) <= tol]
            if cand.size >= 5 or tol > 0.08: break
            tol *= 2.0
        if cand.size == 0:
            cand = pool[np.argsort(np.abs(self.S[pool, age] - survival))[:5]]
        m = int(rng.choice(cand))
        return float(self.h[m]), float(self.det[m]), int(self.risk[m]), float(self.mult[m])


def observable_rate_estimate(env) -> np.ndarray:
    prior = np.asarray(env.demand_rate_estimates, float)
    hist = list(getattr(env, "demand_history", []) or [])
    window = max(int(env.config.demand_history_window), 1)
    hist = hist[-window:]
    if not hist:
        return prior.copy()
    k = len(hist)
    return (np.sum(hist, axis=0) + (window - k) * prior) / float(window)


def make_matched_clone(env, world_seed: int, sampler: PosteriorSampler | None, *, privileged: bool) -> tuple[Any, dict[str, float]]:
    clone = copy.deepcopy(env)
    rng = np.random.default_rng(int(world_seed))
    audit = {"patients": 0, "identical_health_index": 0, "abs_survival_mismatch": 0.0}
    if not privileged:
        n = clone.config.num_facilities
        for p in clone.patient_registry.values():
            if p.status not in (PatientStatus.WAITING, PatientStatus.IN_PRODUCTION) and getattr(p.status, "value", "") != "in_transit":
                continue
            h, det, risk, mult = sampler.draw(p.age, p.survival, p.risk_type, rng)
            audit["patients"] += 1; audit["identical_health_index"] += int(abs(h - p.health_index) < 1e-12)
            p.health_index = h; p.deterioration_epoch = det; p.risk_type = risk; p.risk_multiplier = mult
            new_s = clone.patient_model.survival_at(p.age, h, det, risk_multiplier=mult)
            audit["abs_survival_mismatch"] += abs(new_s - p.survival); p.survival = new_s
        est = observable_rate_estimate(clone)
        clone.demand_rates = est.copy(); clone.base_demand_rates = est.copy()
        clone.demand_regime_multiplier = np.ones(n); clone.demand_rate_multiplier = np.ones(n)
        clone.demand_regime_initial_multipliers = np.ones(n); clone.demand_regime_final_multipliers = np.ones(n)
        clone.demand_shock_remaining = np.zeros(n, dtype=int)
        clone.regional_supplier_disruption_remaining = np.zeros(n, dtype=int)
    clone.rng = rng
    return clone, audit


def rollout_cost(clone, action: np.ndarray, continuation) -> float:
    total = 0.0
    state, reward, done, _ = clone.step(action); total -= float(reward)
    while not done:
        state, reward, done, _ = clone.step(continuation.select_action(state, explore=False, env=clone)); total -= float(reward)
    return total


class RolloutPlanner:
    algorithm = "rollout_planner"

    def __init__(self, *, candidate_class: str, worlds: int, seed_base: int, privileged: bool, sampler: PosteriorSampler | None):
        self.candidate_class = candidate_class; self.K = int(worlds); self.seed_base = int(seed_base)
        self.privileged = privileged; self.sampler = sampler
        self.mdl2 = get_heuristic_class("mdl2")(None, None, {}); self.mdl3 = get_heuristic_class("mdl2")(None, None, {"lookahead_periods": 3})
        self.continuation = get_heuristic_class("mdl2")(None, None, {})
        self.spec_specs = make_explicit_residual_option_specs(SPECIMEN_OPTIONS)
        self.transfer_specs = make_explicit_residual_option_specs(TRANSFER_OPTIONS)
        self.log: list[dict[str, Any]] = []; self.audit = {"patients": 0, "identical_health_index": 0, "abs_survival_mismatch": 0.0, "clones": 0}
        self.decision_index = 0

    def reset(self) -> None:
        self.decision_index = 0

    def candidates(self, state, env) -> tuple[list[np.ndarray], list[str]]:
        a2 = np.asarray(self.mdl2.select_action(state, explore=False, env=env), np.float32)
        acts = list(residual_option_actions_from_env(a2, env, self.spec_specs)); names = ["mdl2"] + [f"mdl2+spec{s.sign:+g}{s.epsilon:g}" for s in self.spec_specs[1:]]
        if self.candidate_class == "full":
            a3 = np.asarray(self.mdl3.select_action(state, explore=False, env=env), np.float32)
            acts += list(residual_option_actions_from_env(a3, env, self.spec_specs)); names += ["mdl3"] + [f"mdl3+spec{s.sign:+g}{s.epsilon:g}" for s in self.spec_specs[1:]]
            acts += list(residual_option_actions_from_env(a2, env, self.transfer_specs))[1:]; names += [f"mdl2+{s.group[:3]}{s.sign:+g}{s.epsilon:g}" for s in self.transfer_specs[1:]]
        return acts, names

    def select_action(self, state, explore: bool = False, env=None):
        acts, names = self.candidates(state, env)
        costs = np.zeros((len(acts), self.K))
        for k in range(self.K):
            base, audit = make_matched_clone(env, self.seed_base + 1000 * self.decision_index + k, self.sampler, privileged=self.privileged)
            for key in ("patients", "identical_health_index", "abs_survival_mismatch"): self.audit[key] += audit[key]
            self.audit["clones"] += 1
            for c, act in enumerate(acts):
                clone = copy.deepcopy(base); clone.rng = np.random.default_rng(self.seed_base + 1000 * self.decision_index + k)
                costs[c, k] = rollout_cost(clone, act, self.continuation)
        means = costs.mean(axis=1); best = int(np.argmin(means))
        if means[best] >= means[0] - 1e-6: best = 0
        self.log.append({"decision": self.decision_index, "chosen": names[best], "anchor_minus_chosen": float(means[0] - means[best]),
                         "means": {nm: float(m) for nm, m in zip(names, means)}})
        self.decision_index += 1
        return acts[best]


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    env = build_env({"env": copy.deepcopy(job["env"])}, seed=job["seed"])
    sampler = None if job["privileged"] else PosteriorSampler(env.patient_model)
    planner = RolloutPlanner(candidate_class=job["candidate_class"], worlds=job["worlds"], seed_base=job["seed"] * 7 + 11,
                             privileged=job["privileged"], sampler=sampler)
    rows = evaluate_agent(planner, env, algorithm=f"planner_{job['candidate_class']}{'_privileged' if job['privileged'] else ''}",
                          seed=job["seed"], replications=1, max_steps=job.get("max_steps"))
    for r in rows:
        r["replication"] = job["rep"]; r["evaluation_seed"] = job["seed"]; r["training_seed"] = 0; r["candidate_class"] = job["candidate_class"]; r["privileged"] = job["privileged"]
    chosen = [d["chosen"] for d in planner.log]
    return {"row": rows[0], "log": planner.log, "audit": planner.audit, "scenario": job["scenario"], "rep": job["rep"],
            "non_anchor_fraction": float(np.mean([c != "mdl2" for c in chosen])) if chosen else 0.0, "seconds": time.perf_counter() - started}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="experiments/configs/patient_indexed_specimen_routing_benchmark.json")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--candidate-class", choices=["restricted", "full"], default="restricted")
    p.add_argument("--worlds", type=int, default=6, help="rollout worlds per candidate (CRN across candidates)")
    p.add_argument("--replications", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=99_700_000)
    p.add_argument("--privileged", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="smoke only")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-root", required=True)
    p.add_argument("--resume", action="store_true", help="skip (scenario, replication) pairs already in <output-root>/rows.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = load_benchmark_plan(Path(args.plan)); scen = compose_scenarios(plan, list(args.scenarios))
    order = [s["name"] for s in plan["scenarios"] if s["name"] in args.scenarios]
    jobs = [{"scenario": name, "env": scen[name]["env"], "rep": rep, "seed": args.seed_base + s_idx * 100_000 + rep, "candidate_class": args.candidate_class,
             "worlds": args.worlds, "privileged": args.privileged, "max_steps": args.max_steps} for s_idx, name in enumerate(order) for rep in range(args.replications)]
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps({k: v for k, v in vars(args).items()}, indent=2))
    results = []; started = time.perf_counter()
    prior_rows = []; prior_logs = []
    if args.resume and (out / "rows.csv").exists():
        with open(out / "rows.csv") as h:
            prior_rows = list(csv.DictReader(h))
        if (out / "decision_logs.json").exists():
            prior_logs = json.loads((out / "decision_logs.json").read_text())
        done = {(r["scenario"], int(float(r["replication"]))) for r in prior_rows}
        jobs = [j for j in jobs if (j["scenario"], j["rep"]) not in done]
        print(f"resuming: {len(prior_rows)} episodes done, {len(jobs)} remaining", flush=True)
    def all_rows():
        return prior_rows + [x["row"] for x in results]
    if args.workers <= 1:
        for j in jobs:
            r = run_job(j); results.append(r); print(f"{r['scenario']} rep{r['rep']} cost {float(r['row']['total_cost'])/1e6:.1f}M non-anchor {r['non_anchor_fraction']:.2f} {r['seconds']:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(run_job, jobs):
                results.append(r); print(f"{r['scenario']} rep{r['rep']} cost {float(r['row']['total_cost'])/1e6:.1f}M non-anchor {r['non_anchor_fraction']:.2f} {r['seconds']:.0f}s ({time.perf_counter()-started:.0f}s)", flush=True)
                write_rows(all_rows(), out / "rows.csv")
    write_rows(all_rows(), out / "rows.csv")
    audit = {"patients_resampled": sum(x["audit"]["patients"] for x in results), "identical_health_index": sum(x["audit"]["identical_health_index"] for x in results),
             "mean_abs_survival_mismatch": (sum(x["audit"]["abs_survival_mismatch"] for x in results) / max(1, sum(x["audit"]["patients"] for x in results))),
             "clones": sum(x["audit"]["clones"] for x in results), "privileged": args.privileged}
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "decision_logs.json").write_text(json.dumps(prior_logs + [{"scenario": x["scenario"], "rep": x["rep"], "log": x["log"]} for x in results]))
    if prior_rows and (out / "audit.json").exists():
        old = json.loads((out / "audit.json").read_text())
        for k in ("patients_resampled", "identical_health_index", "clones"): audit[k] += int(old.get(k, 0))
    print(json.dumps(audit, indent=1)); print(f"wall {time.perf_counter()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
