"""Per-site, per-patient-profile perfect-information lower bound (Measurement B, tightened).

Extends evaluation/hindsight_lower_bound.py with the two structures the coarse
bound relaxed away and that the simulator provably implies:

1. Patient health. Survival is a deterministic function of age and attributes
   drawn at enrollment (verified: zero violations on ~51,000 simulated
   outcomes), so each patient has a hindsight-known first ineligible age a*.
   Waiting patients are lost at age min(a*, 6). A start at age a completes
   iff a* > a + 3; otherwise it is a doomed start that costs a lost patient
   plus a discarded therapy. Patients are grouped by (origin, arrival epoch,
   profile p = min(a*, 9)); within a group they are divisible (LP relaxation).
2. Geography. Per-site bioreactor and reagent stocks; capacity transfers on
   the capacity edges and reagent transfers on the resource edges with the
   simulator's exact per-edge delays; specimen routing on the specimen edges
   with one route per patient and a one-epoch transit; per-site waiting
   counts; per-site bioreactor- and reagent-shortage penalties.

Still relaxed (permissive): transfer amount caps, per-site idle/reagent caps
that destroy units, reagent expiry, transfer and holding costs, urgency cost,
reagent use and slot use by doomed starts. Aggregation over patients within a
class is a further relaxation.

Validity checks per world: (a) the relaxed cost of the simulator's own MDL-2
trajectory, computed from the simulator's per-site state, must not exceed its
simulator cost; (b) the LP objective must not exceed that relaxed cost.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.prior_oracle_gap_screen import compose_scenarios, make_policy  # noqa: E402
from evaluation.run_full_benchmark import load_benchmark_plan  # noqa: E402
from src.rl.experiment import build_env  # noqa: E402

SCENARIOS = ("routing_nominal_history", "routing_abrupt_regime_shift", "routing_regional_drift", "routing_compound_regional_stress")
SHELF = 6
OCC = 3
PCAP = 9


def first_ineligible_age(model, p, cap: int = PCAP) -> int:
    for a in range(cap + 1):
        if model.survival_at(a, p.health_index, p.deterioration_epoch, risk_multiplier=p.risk_multiplier) < model.config.eligibility_threshold:
            return a
    return cap


def world_data(env_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Exogenous inputs plus the MDL-2 trajectory's per-site aggregates on the same world."""

    env = build_env({"env": copy.deepcopy(env_cfg)}, seed=seed)
    s = env.reset(seed=seed)
    pol = make_policy("mdl2_prior")
    T = int(env.config.episode_horizon); n = env.config.num_facilities
    avail = np.zeros((T, n)); W = np.zeros((T, n)); I = np.zeros((T, n)); R = np.zeros((T, n)); purch = np.zeros((T, n))
    lost600 = 0.0; lost500 = 0.0; total = 0.0
    for t in range(T):
        avail[t] = env.supplier_available
        s, r, d, info = env.step(pol.select_action(s, env=env)); total += -float(r)
        W[t] = env.specimens; I[t] = env.bioreactors[:, 0]; R[t] = env.reagents; purch[t] = info["replenishment"]
        exp_like = np.sum(info["patients_lost_waiting_expired"]) + np.sum(info["patients_lost_transit_expired"]) + np.sum(info["finished_expired"]) + np.sum(info["patients_lost_manufacturing"]) + np.sum(info["transit_loss"])
        lost600 += exp_like; lost500 += np.sum(info["patients_lost"]) - exp_like
        if d: break
    classes: dict[tuple[int, int, int], float] = defaultdict(float)
    for p in env.patient_registry.values():
        classes[(int(p.collection_facility), int(p.enrollment_epoch), min(first_ineligible_age(env.patient_model, p), PCAP))] += 1.0
    costs = env.config.costs
    spec_nb = defaultdict(set)
    for a, b in env.specimen_edges: spec_nb[int(a)].add(int(b)); spec_nb[int(b)].add(int(a))
    cap_edges = [(int(a), int(b), env._transfer_delay_for_edge((int(a), int(b)))) for a, b in env.capacity_edges]
    res_edges = [(int(a), int(b), env._transfer_delay_for_edge((int(a), int(b)))) for a, b in env.resource_edges]
    relaxed_sim = 600_000 * lost600 + 500_000 * lost500 + costs.reagent_purchase * purch.sum() \
        + costs.bioreactor_shortage * np.maximum(W - I, 0).sum() + costs.reagent_shortage * np.maximum(W - R, 0).sum()
    return {"T": T, "n": n, "avail": avail, "classes": dict(classes), "spec_nb": {k: sorted(v) for k, v in spec_nb.items()},
            "cap_edges": cap_edges, "res_edges": res_edges,
            "init_idle": np.asarray(env_cfg["initial_idle_bioreactors"], float), "init_reag": np.asarray(env_cfg["initial_reagents"], float),
            "cap": np.asarray(env_cfg["max_reagent_replenishment"], float),
            "max_idle": np.asarray(env_cfg["max_idle_bioreactors"], float), "max_reagents": np.asarray(env_cfg["max_reagents"], float),
            "max_bio_transfer": float(env_cfg["max_bioreactor_transfer"]), "max_reagent_transfer": float(env_cfg["max_reagent_transfer"]),
            "w_lost": float(env_cfg["weight_patient_lost"]), "w_exp": float(env_cfg["weight_expiry"]),
            "c_R": float(costs.reagent_purchase), "p_B": float(costs.bioreactor_shortage), "p_R": float(costs.reagent_shortage),
            "sim_cost": total, "relaxed_cost_of_sim": float(relaxed_sim), "dominated": bool(relaxed_sim <= total + 1e-6),
            "arrivals": float(sum(classes.values()))}


class LP:
    def __init__(self):
        self.nv = 0; self.cost = []; self.lb = []; self.ub = []
        self.r_ub = []; self.c_ub = []; self.v_ub = []; self.b_ub = []
        self.r_eq = []; self.c_eq = []; self.v_eq = []; self.b_eq = []
    def var(self, cost=0.0, lo=0.0, hi=None):
        self.cost.append(cost); self.lb.append(lo); self.ub.append(hi); self.nv += 1; return self.nv - 1
    def leq(self, terms, rhs):
        k = len(self.b_ub)
        for j, v in terms: self.r_ub.append(k); self.c_ub.append(j); self.v_ub.append(v)
        self.b_ub.append(rhs)
    def eq(self, terms, rhs):
        k = len(self.b_eq)
        for j, v in terms: self.r_eq.append(k); self.c_eq.append(j); self.v_eq.append(v)
        self.b_eq.append(rhs)
    def solve(self, method: str = "highs", time_limit: float | None = None):
        A_ub = coo_matrix((self.v_ub, (self.r_ub, self.c_ub)), shape=(len(self.b_ub), self.nv)).tocsr()
        A_eq = coo_matrix((self.v_eq, (self.r_eq, self.c_eq)), shape=(len(self.b_eq), self.nv)).tocsr()
        options = {} if time_limit is None else {"time_limit": float(time_limit)}
        res = linprog(np.asarray(self.cost), A_ub=A_ub, b_ub=np.asarray(self.b_ub), A_eq=A_eq, b_eq=np.asarray(self.b_eq),
                      bounds=list(zip(self.lb, self.ub)), method=method, options=options)
        if res.status != 0: raise RuntimeError(f"LP status {res.status}: {res.message}")
        return res


def solve_world(d: dict[str, Any], *, method: str = "highs", time_limit: float | None = None, tight: bool = True, objective: str = "cost") -> dict[str, Any]:
    """objective='cost': relaxed total objective (certified cost lower bound).
    objective='loss': count of lost patients only (certified minimum patient loss under the relaxation)."""
    T, n = d["T"], d["n"]; lp = LP()
    if objective == "cost":
        w600 = d["w_lost"] + d["w_exp"]; w500 = d["w_lost"]; c_R = d["c_R"]; p_B = d["p_B"]; p_R = d["p_R"]
    elif objective == "loss":
        w600 = 1.0; w500 = 1.0; c_R = 0.0; p_B = 0.0; p_R = 0.0
    else:
        raise ValueError(objective)
    # ---- per-site stock variables
    B = [[lp.var() for t in range(T)] for _ in range(n)]      # bioreactors on site at decision t
    R = [[lp.var() for t in range(T)] for _ in range(n)]      # reagents on hand at decision t
    U = [[lp.var(cost=c_R, hi=d["cap"][j] * d["avail"][t, j]) for t in range(T)] for j in range(n)]
    S = [[lp.var(cost=p_B) for t in range(T)] for _ in range(n)]
    Q = [[lp.var(cost=p_R) for t in range(T)] for _ in range(n)]
    # capacity/reagent flows: directed, sent end of t, arrive decision t+delay
    Gout = [[[] for t in range(T)] for _ in range(n)]; Gin = [[[] for t in range(T)] for _ in range(n)]; Gsched = [[[] for t in range(T)] for _ in range(n)]
    for a, b, dl in d["cap_edges"]:
        for (src, dst) in ((a, b), (b, a)):
            for t in range(T):
                if t + dl <= T - 1:
                    v = lp.var(); Gout[src][t].append(v); Gin[dst][t + dl].append(v); Gsched[dst][t].append(v)
    Hout = [[[] for t in range(T)] for _ in range(n)]; Hin = [[[] for t in range(T)] for _ in range(n)]; Hsched = [[[] for t in range(T)] for _ in range(n)]
    for a, b, dl in d["res_edges"]:
        for (src, dst) in ((a, b), (b, a)):
            for t in range(T):
                if t + dl <= T - 1:
                    v = lp.var(); Hout[src][t].append(v); Hin[dst][t + dl].append(v); Hsched[dst][t].append(v)
    Dbio = [[lp.var() for t in range(T)] for _ in range(n)]     # disposal of idle units above the site cap (simulator clips)
    Drea = [[lp.var() for t in range(T)] for _ in range(n)]     # disposal of reagents above the site cap
    # ---- per-site event accumulators indexed by epoch (terms added by classes)
    starts = [[[] for t in range(T)] for _ in range(n)]        # completable starts at site, epoch
    Wdelta = [[[] for t in range(T)] for _ in range(n)]        # (var, coef) changes to waiting at end of epoch t
    Wconst = np.zeros((n, T))                                  # constant arrivals into waiting at end of t
    # ---- classes
    lb_loss_terms = []
    for (i, r, p), A in d["classes"].items():
        dests = [i] + d["spec_nb"].get(i, [])
        t_loss = r + min(p, SHELF)
        e_cost = 0.0 if t_loss > T - 1 else (w600 if p >= SHELF else w500)
        X = {}; Z = {}; Y = {}; E = {}
        for m in dests:
            E[m] = lp.var(cost=e_cost)
            for a in range(SHELF):
                ts = r + 1 + a
                if ts > T - 1: continue
                if a <= p - 4: X[(m, a)] = lp.var()
                if p - 3 <= a <= p - 1: Z[(m, a)] = lp.var(cost=w600)
            if m != i:
                for a in range(SHELF - 1):
                    if r + 1 + a <= T - 1 and a <= p - 1: Y[(m, a)] = lp.var()
        # conservation
        lp.eq([(v, 1.0) for v in X.values()] + [(v, 1.0) for v in Z.values()] + [(v, 1.0) for v in E.values()], A)
        # one route per patient
        if Y: lp.leq([(v, 1.0) for v in Y.values()], A)
        # route only while waiting at origin and not yet started there
        for a in range(SHELF):
            terms = [(v, 1.0) for (m, a2), v in Y.items() if a2 <= a] + [(v, 1.0) for (m, a2), v in X.items() if m == i and a2 <= a] + [(v, 1.0) for (m, a2), v in Z.items() if m == i and a2 <= a]
            if terms: lp.leq(terms, A)
        # destination precedence: starts/doomed/unstarted at m require prior arrival
        for m in dests:
            if m == i: continue
            for a in range(SHELF):
                terms = [(v, 1.0) for (mm, a2), v in X.items() if mm == m and a2 <= a] + [(v, 1.0) for (mm, a2), v in Z.items() if mm == m and a2 <= a]
                terms += [(v, -1.0) for (mm, a2), v in Y.items() if mm == m and a2 <= a - 1]
                if terms: lp.leq(terms, 0.0)
            lp.leq([(E[m], 1.0)] + [(v, 1.0) for (mm, a2), v in X.items() if mm == m] + [(v, 1.0) for (mm, a2), v in Z.items() if mm == m]
                   + [(v, -1.0) for (mm, a2), v in Y.items() if mm == m], 0.0)
        # site events
        if r <= T - 1: Wconst[i, r] += A                          # enrolled at end of r at origin
        for (m, a), v in X.items():
            ts = r + 1 + a; starts[m][ts].append(v); Wdelta[m][ts].append((v, -1.0))
        for (m, a), v in Z.items():
            ts = r + 1 + a; Wdelta[m][ts].append((v, -1.0))
        for (m, a), v in Y.items():
            tr_ = r + 1 + a; Wdelta[i][tr_].append((v, -1.0))
            if tr_ + 1 <= T - 1: Wdelta[m][tr_ + 1].append((v, 1.0))
        if t_loss <= T - 1:
            for m in dests: Wdelta[m][t_loss].append((E[m], -1.0))
        lb_loss_terms += [(E[m], e_cost) for m in dests] + [(v, w600) for v in Z.values()]
    # ---- per-site dynamics
    Wv = [[lp.var() for t in range(T)] for _ in range(n)]      # waiting at end of t
    for j in range(n):
        lp.eq([(B[j][0], 1.0)], d["init_idle"][j]); lp.eq([(R[j][0], 1.0)], d["init_reag"][j])
        for t in range(T):
            st = [(v, 1.0) for v in starts[j][t]]
            occ = [(v, 1.0) for tt in (t - 1, t - 2) if tt >= 0 for v in starts[j][tt]]
            # starts <= idle = B - occ
            lp.leq(st + occ + [(B[j][t], -1.0)], 0.0)
            # capacity out <= idle - starts
            lp.leq([(v, 1.0) for v in Gout[j][t]] + st + occ + [(B[j][t], -1.0)], 0.0)
            # starts <= reagents on hand
            lp.leq(st + [(R[j][t], -1.0)], 0.0)
            # reagent out <= R - starts + purchases
            lp.leq([(v, 1.0) for v in Hout[j][t]] + st + [(R[j][t], -1.0), (U[j][t], -1.0)], 0.0)
            if t + 1 <= T - 1:
                lp.eq([(B[j][t + 1], 1.0), (B[j][t], -1.0), (Dbio[j][t], 1.0)] + [(v, 1.0) for v in Gout[j][t]] + [(v, -1.0) for v in Gin[j][t + 1]], 0.0)
                lp.eq([(R[j][t + 1], 1.0), (R[j][t], -1.0), (Drea[j][t], 1.0)] + st + [(U[j][t], -1.0)] + [(v, 1.0) for v in Hout[j][t]] + [(v, -1.0) for v in Hin[j][t + 1]], 0.0)
            if tight:
                # idle after transfers and disposal <= site cap (the simulator clips): I_end - Dbio <= max_idle
                lp.leq([(B[j][t], 1.0), (Dbio[j][t], -1.0)] + [(v, -1.0) for v in Gout[j][t]] + [(v, -1.0) for tt in (t, t - 1) if tt >= 0 for v in starts[j][tt]], d["max_idle"][j])
                # reagents after purchases, transfers and disposal <= site cap
                lp.leq([(R[j][t], 1.0), (U[j][t], 1.0), (Drea[j][t], -1.0)] + [(v, -1.0) for v in starts[j][t]] + [(v, -1.0) for v in Hout[j][t]], d["max_reagents"][j])
                # per-facility net transfer request caps: |scheduled inbound - outbound| <= cap
                lp.leq([(v, 1.0) for v in Gsched[j][t]] + [(v, -1.0) for v in Gout[j][t]], d["max_bio_transfer"])
                lp.leq([(v, -1.0) for v in Gsched[j][t]] + [(v, 1.0) for v in Gout[j][t]], d["max_bio_transfer"])
                lp.leq([(v, 1.0) for v in Hsched[j][t]] + [(v, -1.0) for v in Hout[j][t]], d["max_reagent_transfer"])
                lp.leq([(v, -1.0) for v in Hsched[j][t]] + [(v, 1.0) for v in Hout[j][t]], d["max_reagent_transfer"])
            # waiting recursion: W_t = W_{t-1} + const + deltas
            terms = [(Wv[j][t], 1.0)] + ([(Wv[j][t - 1], -1.0)] if t > 0 else []) + [(v, -c) for v, c in Wdelta[j][t]]
            lp.eq(terms, Wconst[j, t])
            # s >= W_end - I_end, I_end = B - Gout - starts_t - starts_{t-1}
            occ_end = [(v, 1.0) for tt in (t, t - 1) if tt >= 0 for v in starts[j][tt]]
            lp.leq([(Wv[j][t], 1.0), (B[j][t], -1.0), (Dbio[j][t], 1.0)] + [(v, 1.0) for v in Gout[j][t]] + occ_end + [(S[j][t], -1.0)], 0.0)
            # q >= W_end - R_end, R_end = R - starts + U - Hout
            lp.leq([(Wv[j][t], 1.0), (R[j][t], -1.0), (Drea[j][t], 1.0)] + st + [(U[j][t], -1.0)] + [(v, 1.0) for v in Hout[j][t]] + [(Q[j][t], -1.0)], 0.0)
    res = lp.solve(method=method, time_limit=time_limit); x = res.x
    loss_cost = sum(x[v] * c for v, c in lb_loss_terms)
    lost_patients = sum(x[v] for v, c in lb_loss_terms if c > 0) if objective == "loss" else sum(x[v] for v, c in lb_loss_terms if c > 0)
    return {"lower_bound": float(res.fun), "lb_loss_cost": float(loss_cost), "lb_lost_patients": float(lost_patients), "objective": objective,
            "lb_bio_shortage_cost": float(d["p_B"] * sum(x[S[j][t]] for j in range(n) for t in range(T))),
            "lb_reagent_shortage_cost": float(d["p_R"] * sum(x[Q[j][t]] for j in range(n) for t in range(T))),
            "lb_purchase_cost": float(d["c_R"] * sum(x[U[j][t]] for j in range(n) for t in range(T))),
            "lb_starts": float(sum(x[v] for j in range(n) for t in range(T) for v in starts[j][t])),
            "n_vars": lp.nv, "n_rows": len(lp.b_ub) + len(lp.b_eq)}


def run_world(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    d = world_data(job["env"], job["seed"])
    try:
        sol = solve_world(d, method=job.get("method", "highs"), time_limit=job.get("time_limit"), tight=bool(job.get("tight", True)), objective="cost")
        loss_sol = solve_world(d, method=job.get("method", "highs"), time_limit=job.get("time_limit"), tight=bool(job.get("tight", True)), objective="loss")
        sol["min_lost_patients_certified"] = loss_sol["lb_lost_patients"]
    except RuntimeError as exc:
        return {"scenario": job["scenario"], "replication": job["rep"], "world_seed": job["seed"], "arrivals": d["arrivals"],
                "sim_cost_mdl2": d["sim_cost"], "relaxed_cost_of_sim": d["relaxed_cost_of_sim"], "dominated": d["dominated"],
                "lp_leq_relaxed_sim": False, "lower_bound": float("nan"), "error": str(exc)[:120], "seconds": time.perf_counter() - started}
    return {"scenario": job["scenario"], "replication": job["rep"], "world_seed": job["seed"], "arrivals": d["arrivals"],
            "sim_cost_mdl2": d["sim_cost"], "relaxed_cost_of_sim": d["relaxed_cost_of_sim"], "dominated": d["dominated"],
            "lp_leq_relaxed_sim": bool(sol["lower_bound"] <= d["relaxed_cost_of_sim"] + 1e-3),
            **sol, "gap_mdl2_pct": 100 * (d["sim_cost"] - sol["lower_bound"]) / d["sim_cost"], "seconds": time.perf_counter() - started}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="experiments/configs/patient_indexed_specimen_routing_benchmark.json")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--seed-base", type=int, default=99_700_000)
    p.add_argument("--replications", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-root", default="results/hindsight_lower_bound_site")
    p.add_argument("--method", default="highs-ipm")
    p.add_argument("--time-limit", type=float, default=600.0, help="HiGHS time limit per world, seconds")
    p.add_argument("--loose", action="store_true", help="drop the site-cap and transfer-cap constraints (looser bound)")
    p.add_argument("--resume", action="store_true", help="skip worlds already present in <output-root>/rows.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = load_benchmark_plan(Path(args.plan)); scen = compose_scenarios(plan, list(args.scenarios))
    order = [s["name"] for s in plan["scenarios"] if s["name"] in args.scenarios]
    jobs = [{"scenario": name, "env": scen[name]["env"], "rep": rep, "seed": args.seed_base + s_idx * 100_000 + rep,
             "method": args.method, "time_limit": args.time_limit, "tight": not args.loose}
            for s_idx, name in enumerate(order) for rep in range(args.replications)]
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    rows = []; started = time.perf_counter()
    rows_path = out / "rows.csv"
    if args.resume and rows_path.exists():
        with open(rows_path) as h:
            for r in csv.DictReader(h):
                r = {k: (float(v) if k not in ("scenario", "dominated", "lp_leq_relaxed_sim", "error") and v not in ("", "nan") else v) for k, v in r.items()}
                r["dominated"] = r["dominated"] == "True"; r["lp_leq_relaxed_sim"] = r["lp_leq_relaxed_sim"] == "True"
                rows.append(r)
        done = {(r["scenario"], int(r["replication"])) for r in rows}
        jobs = [j for j in jobs if (j["scenario"], j["rep"]) not in done]
        print(f"resuming: {len(rows)} worlds done, {len(jobs)} remaining", flush=True)
    if args.workers <= 1:
        for j in jobs:
            r = run_world(j); rows.append(r); print(f"{r['scenario']} rep{r['replication']}: sim {r['sim_cost_mdl2']/1e6:.1f}M LB {r['lower_bound']/1e6:.1f}M gap {r['gap_mdl2_pct']:.1f}% dom={r['dominated']} lp<=relaxed={r['lp_leq_relaxed_sim']} vars {r['n_vars']} {r['seconds']:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(run_world, jobs):
                rows.append(r)
                print(f"{r['scenario']} rep{r['replication']}: gap {r.get('gap_mdl2_pct', float('nan')):.1f}% dom={r['dominated']} {r.get('error','')} ({time.perf_counter()-started:.0f}s)", flush=True)
                if len(rows) % 5 == 0:
                    keys = sorted({k for rr in rows for k in rr})
                    with open(rows_path, "w", newline="") as h:
                        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows({k: rr.get(k, "") for k in keys} for rr in rows)
    keys = sorted({k for r in rows for k in r})
    with open(rows_path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows({k: r.get(k, "") for k in keys} for r in rows)
    summ = {"seed_base": args.seed_base, "replications": args.replications, "scenarios": order, "wall_seconds": time.perf_counter() - started,
            "all_dominated": all(r["dominated"] for r in rows), "all_lp_leq_relaxed": all(r["lp_leq_relaxed_sim"] for r in rows), "per_scenario": {}}
    for name in order:
        rr = [r for r in rows if r["scenario"] == name]
        rr = [r for r in rr if r.get("error", "") in ("", None) and not (isinstance(r.get("lower_bound"), float) and np.isnan(r["lower_bound"]))]
        if not rr: continue
        summ["per_scenario"][name] = {"n_worlds": len(rr)} | {k: float(np.mean([r[k] for r in rr])) for k in ["sim_cost_mdl2", "lower_bound", "gap_mdl2_pct", "lb_loss_cost", "lb_bio_shortage_cost", "lb_reagent_shortage_cost", "lb_purchase_cost", "lb_starts", "lb_lost_patients", "min_lost_patients_certified", "arrivals"]}
    (out / "summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
