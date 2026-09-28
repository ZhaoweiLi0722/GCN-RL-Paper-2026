"""Certified perfect-information lower bound on the optimal expected cost (Measurement B pilot).

For each world (scenario, replication seed) the exogenous randomness is fixed:
arrivals per clinic per epoch and supplier availability per clinic per epoch
are drawn from the environment RNG independently of the policy (verified
empirically under MDL-2, MDL-3, and a uniform-random policy). A perfect-
information optimizer that knows them can do no worse than any implementable
policy, so the minimum of a RELAXATION of the perfect-information problem is a
valid lower bound on that world's optimal cost, and its average over worlds
bounds the optimal expected cost.

The relaxation keeps only constraints the simulator provably implies and only
cost terms every trajectory provably pays:

  starts        a patient enrolled at the end of epoch r can start at epochs
                r+1..r+6 (specimen age 0..5; age 6 expires), one reagent each;
  occupancy     a start at t holds one bioreactor at decisions t, t+1, t+2;
                the fleet is 210 units (initial idle sum; transfers only move
                or strand units), so x_t + x_{t-1} + x_{t-2} <= 210;
  reagents      cumulative starts <= 2,300 initial units + purchases made at
                earlier epochs; purchases <= max_replenishment * availability;
  losses        an unstarted patient whose window closes inside the horizon
                is lost (500k) and wastes material (100k); a patient may also
                be declared lost early (500k), which is how ineligibility,
                transit losses and manufacturing losses map into the LP;
  transit       a patient may spend one epoch in transit (removed from the
                waiting count), mirroring the simulator's one-route limit;
  penalties     the bioreactor-shortage and reagent-shortage penalties are
                bounded below network-wide by Jensen:
                sum_n max(W_n - I_n, 0) >= max(sum W_n - sum I_n, 0).

Relaxed away (all permissive, so the bound stays valid): geography and edge
feasibility, transfer lead times, per-site capacity and reagent caps, patient
health (every started patient completes), reagent and product expiry, holding,
urgency and transfer costs. The bound is therefore LOOSE; its distance from the
true optimum is unknown. What it certifies is an upper bound on the gap
between any evaluated policy and the optimum.

Aggregation: patients are grouped by (origin clinic, arrival epoch) and
treated as divisible, which is a further relaxation (LP instead of IP).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
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
WINDOW = 6          # start epochs r+1 .. r+6
OCC = 3             # decision epochs a start occupies a bioreactor


def exogenous_trace(env_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Arrivals, supplier availability, and the MDL-2 per-epoch aggregates for the same world."""

    env = build_env({"env": copy.deepcopy(env_cfg)}, seed=seed)
    s = env.reset(seed=seed)
    pol = make_policy("mdl2_prior")
    T = int(env.config.episode_horizon); n = env.config.num_facilities
    A = np.zeros((T, n)); avail = np.zeros((T, n))
    W_end = np.zeros(T); idle_end = np.zeros(T); reag_end = np.zeros(T)
    starts = np.zeros(T); purch = np.zeros(T); lost_exp = np.zeros(T); lost_other = np.zeros(T); lost_mfg = np.zeros(T)
    total_cost = 0.0; comps = {}
    for t in range(T):
        A[t] = env.demand; avail[t] = env.supplier_available
        s, r, d, info = env.step(pol.select_action(s, env=env))
        total_cost += -float(r)
        W_end[t] = env.specimens.sum(); idle_end[t] = env.bioreactors[:, 0].sum(); reag_end[t] = env.reagents.sum()
        starts[t] = np.sum(info["patients_started"]); purch[t] = np.sum(info["replenishment"])
        lost_exp[t] = np.sum(info["patients_lost_waiting_expired"]) + np.sum(info["patients_lost_transit_expired"]) + np.sum(info["finished_expired"])
        lost_mfg[t] = np.sum(info["patients_lost_manufacturing"])
        lost_other[t] = np.sum(info["patients_lost"]) - lost_exp[t] - lost_mfg[t]
        if d:
            break
    costs = env.config.costs
    return {"A": A, "avail": avail, "T": T, "n": n, "fleet": float(np.sum(env_cfg["initial_idle_bioreactors"])),
            "reagents0": float(np.sum(env_cfg["initial_reagents"])),
            "cap": np.asarray(env_cfg["max_reagent_replenishment"], dtype=float),
            "w_lost": float(env_cfg["weight_patient_lost"]), "w_exp": float(env_cfg["weight_expiry"]),
            "c_R": float(getattr(costs, "reagent_purchase", getattr(costs, "reagent_unit_cost", 0.0))),
            "p_B": float(getattr(costs, "bioreactor_shortage", 0.0)), "p_R": float(getattr(costs, "reagent_shortage", 0.0)),
            "mdl2": {"total_cost": total_cost, "W_end": W_end, "idle_end": idle_end, "reag_end": reag_end, "starts": starts,
                      "purch": purch, "lost_exp": lost_exp, "lost_mfg": lost_mfg, "lost_other": lost_other}}


def domination_check(tr: dict[str, Any]) -> dict[str, float]:
    """The relaxed cost of the simulator's own MDL-2 trajectory must not exceed its simulator cost."""

    m = tr["mdl2"]; T = tr["T"]
    relaxed = (tr["w_lost"] + tr["w_exp"]) * m["lost_exp"].sum() + tr["w_lost"] * (m["lost_mfg"].sum() + m["lost_other"].sum()) \
        + tr["c_R"] * m["purch"].sum() \
        + tr["p_B"] * np.maximum(m["W_end"] - m["idle_end"], 0.0).sum() \
        + tr["p_R"] * np.maximum(m["W_end"] - m["reag_end"], 0.0).sum()
    # occupancy and reagent identities on the simulator trajectory (starts include patients later lost in manufacturing,
    # so occupancy is checked against fleet only as a sanity bound, not as the mapped-solution constraint)
    cum_starts = np.cumsum(m["starts"]); cum_purch = np.concatenate([[0.0], np.cumsum(m["purch"])[:-1]])
    reagent_slack = np.min(tr["reagents0"] + cum_purch - cum_starts)
    return {"sim_cost": m["total_cost"], "relaxed_cost_of_sim_trajectory": relaxed, "dominated": bool(relaxed <= m["total_cost"] + 1e-6),
            "min_reagent_slack": float(reagent_slack)}


def solve_world(tr: dict[str, Any]) -> dict[str, Any]:
    T, n = tr["T"], tr["n"]; A = tr["A"]
    classes = [(i, r) for r in range(T) for i in range(n) if A[r, i] > 0]
    C = len(classes)
    # variable blocks: x[c,k] start at r+1+k (k<WINDOW, epoch<=T-1); l[c,k] early loss at same epochs; y[c,k] transit at r+k (k=0..WINDOW-1, epoch r+k);
    # e[c] unstarted at window close; u[t,n] purchases; s[t] bio shortage; q[t] reagent shortage
    idx = {}; nv = 0
    def add(name, count):
        nonlocal nv
        idx[name] = (nv, count); nv += count; return idx[name]
    add("x", C * WINDOW); add("l", C * WINDOW); add("y", C * WINDOW); add("e", C); add("u", T * n); add("s", T); add("q", T)
    def X(c, k): return idx["x"][0] + c * WINDOW + k
    def L(c, k): return idx["l"][0] + c * WINDOW + k
    def Y(c, k): return idx["y"][0] + c * WINDOW + k
    def E(c): return idx["e"][0] + c
    def U(t, j): return idx["u"][0] + t * n + j
    def S(t): return idx["s"][0] + t
    def Q(t): return idx["q"][0] + t
    cost = np.zeros(nv)
    rows, cols, vals, b_ub = [], [], [], []
    eq_rows, eq_cols, eq_vals, b_eq = [], [], [], []
    def ub(coefs, rhs):
        r = len(b_ub)
        for c_, v_ in coefs: rows.append(r); cols.append(c_); vals.append(v_)
        b_ub.append(rhs)
    def eq(coefs, rhs):
        r = len(b_eq)
        for c_, v_ in coefs: eq_rows.append(r); eq_cols.append(c_); eq_vals.append(v_)
        b_eq.append(rhs)
    bounds = [(0, None)] * nv
    # class conservation and costs
    for c, (i, r) in enumerate(classes):
        coefs = [(E(c), 1.0)]
        for k in range(WINDOW):
            t = r + 1 + k
            if t <= T - 1:
                coefs += [(X(c, k), 1.0), (L(c, k), 1.0)]
                cost[L(c, k)] = tr["w_lost"]
            else:
                bounds[X(c, k)] = (0, 0); bounds[L(c, k)] = (0, 0)
            ty = r + k
            if not (ty <= T - 1): bounds[Y(c, k)] = (0, 0)
        eq(coefs, A[r, i])
        cost[E(c)] = (tr["w_lost"] + tr["w_exp"]) if (r + WINDOW <= T - 1) else 0.0
        # one transit per patient, and transit only while still waiting (not yet started or lost by that epoch)
        ub([(Y(c, k), 1.0) for k in range(WINDOW)], A[r, i])
        for k in range(WINDOW):
            ty = r + k
            if ty > T - 1: continue
            coefs = [(Y(c, k), 1.0)]
            for k2 in range(WINDOW):
                if r + 1 + k2 <= ty and r + 1 + k2 <= T - 1:
                    coefs += [(X(c, k2), 1.0), (L(c, k2), 1.0)]
            ub(coefs, A[r, i])
    # purchases
    for t in range(T):
        for j in range(n):
            bounds[U(t, j)] = (0, tr["cap"][j] * tr["avail"][t, j]); cost[U(t, j)] = tr["c_R"]
    # per-epoch: occupancy, reagent cumulative, shortage penalties
    for t in range(T):
        occ = []; cum = []; W = []; occ_end = []
        for c, (i, r) in enumerate(classes):
            for k in range(WINDOW):
                ts = r + 1 + k
                if ts > T - 1: continue
                if ts <= t: cum.append((X(c, k), 1.0))
                if t - OCC + 1 <= ts <= t: occ.append((X(c, k), 1.0))
                if t - 1 <= ts <= t: occ_end.append((X(c, k), 1.0))
        ub(occ, tr["fleet"])
        # cumulative starts <= reagents0 + purchases at epochs <= t-1
        coefs = list(cum) + [(U(tp, j), -1.0) for tp in range(t) for j in range(n)]
        ub(coefs, tr["reagents0"])
        # waiting at end of t: arrivals r<=t minus starts/losses at epochs<=t minus expired (r+WINDOW<=t) minus in transit at t
        for c, (i, r) in enumerate(classes):
            if r > t: continue
            W.append(("const", A[r, i]))
            for k in range(WINDOW):
                ts = r + 1 + k
                if ts <= t and ts <= T - 1: W += [(X(c, k), -1.0), (L(c, k), -1.0)]
                if r + k == t: W.append((Y(c, k), -1.0))
            if r + WINDOW <= t: W.append((E(c), -1.0))
        const = sum(v for kk, v in W if kk == "const"); Wv = [(kk, v) for kk, v in W if kk != "const"]
        # s_t >= W_end(t) - idle_end(t) = W - (fleet - occ_end)  ->  W + occ_end - s_t <= fleet - const
        ub(Wv + occ_end + [(S(t), -1.0)], tr["fleet"] - const); cost[S(t)] = tr["p_B"]
        # q_t >= W_end(t) - R_end(t), R_end = reagents0 + purchases<=t - starts<=t  ->  W + starts<=t - purchases<=t - q_t <= reagents0 - const
        ub(Wv + cum + [(U(tp, j), -1.0) for tp in range(t + 1) for j in range(n)] + [(Q(t), -1.0)], tr["reagents0"] - const); cost[Q(t)] = tr["p_R"]
    A_ub = coo_matrix((vals, (rows, cols)), shape=(len(b_ub), nv)).tocsr()
    A_eq = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(len(b_eq), nv)).tocsr()
    res = linprog(cost, A_ub=A_ub, b_ub=np.asarray(b_ub), A_eq=A_eq, b_eq=np.asarray(b_eq), bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    ex = x[idx["e"][0]: idx["e"][0] + C]; expired = sum(ex[c] for c, (i, r) in enumerate(classes) if r + WINDOW <= T - 1)
    early = x[idx["l"][0]: idx["l"][0] + C * WINDOW].sum(); starts = x[idx["x"][0]: idx["x"][0] + C * WINDOW].sum()
    purch = x[idx["u"][0]: idx["u"][0] + T * n].sum(); sb = x[idx["s"][0]: idx["s"][0] + T].sum(); sq = x[idx["q"][0]: idx["q"][0] + T].sum()
    return {"lower_bound": float(res.fun), "lb_lost_expired": float(expired), "lb_lost_early": float(early), "lb_starts": float(starts),
            "lb_purchases": float(purch), "lb_bio_shortage_units": float(sb), "lb_reagent_shortage_units": float(sq),
            "lb_cost_bio_shortage": float(tr["p_B"] * sb), "lb_cost_patient_loss": float(tr["w_lost"] * (expired + early)),
            "n_vars": nv, "n_rows": len(b_ub) + len(b_eq)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="experiments/configs/patient_indexed_specimen_routing_benchmark.json")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--seed-base", type=int, default=99_700_000)
    p.add_argument("--replications", type=int, default=100)
    p.add_argument("--output-root", default="results/hindsight_lower_bound")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = load_benchmark_plan(Path(args.plan))
    scen = compose_scenarios(plan, list(args.scenarios))
    order = [s["name"] for s in plan["scenarios"] if s["name"] in args.scenarios]  # plan order == screen order
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    rows = []; started = time.perf_counter()
    for s_idx, name in enumerate(order):
        base = args.seed_base + s_idx * 100_000
        for rep in range(args.replications):
            seed = base + rep
            tr = exogenous_trace(scen[name]["env"], seed)
            chk = domination_check(tr)
            sol = solve_world(tr)
            rows.append({"scenario": name, "replication": rep, "world_seed": seed, "arrivals": float(tr["A"].sum()), **chk, **sol,
                         "gap_mdl2_vs_lb_pct": 100.0 * (chk["sim_cost"] - sol["lower_bound"]) / chk["sim_cost"]})
            if rep % 20 == 0:
                print(f"{name} rep{rep}: sim {chk['sim_cost']/1e6:.1f}M  LB {sol['lower_bound']/1e6:.1f}M  gap {rows[-1]['gap_mdl2_vs_lb_pct']:.1f}%  dominated={chk['dominated']}  ({time.perf_counter()-started:.0f}s)", flush=True)
    with open(out / "rows.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = {"seed_base": args.seed_base, "replications": args.replications, "scenarios": order, "wall_seconds": time.perf_counter() - started,
            "all_dominated": all(r["dominated"] for r in rows), "per_scenario": {}}
    for name in order:
        rr = [r for r in rows if r["scenario"] == name]
        summ["per_scenario"][name] = {k: float(np.mean([r[k] for r in rr])) for k in ["sim_cost", "lower_bound", "gap_mdl2_vs_lb_pct", "lb_lost_expired", "lb_lost_early", "lb_starts", "lb_purchases", "lb_cost_bio_shortage", "lb_cost_patient_loss", "arrivals"]}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ["per_scenario"], indent=1)); print("all trajectories dominated:", summ["all_dominated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
