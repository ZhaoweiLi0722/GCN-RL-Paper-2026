"""Prior-oracle gap screen: how much does MDL-2 lose from a wrong demand prior?

Question. Before proposing that a learned policy trained over a broad demand
distribution can beat MDL-2 when MDL-2's demand prior is misspecified, measure
the ceiling of that idea directly: run the *same* MDL-2 decision rule with
(a) the prior it actually executes with and (b) the true current demand rate
supplied by the simulator. The paired difference is the most any policy could
recover purely from better knowledge of the demand distribution, holding the
decision rule fixed. If it is small, distribution knowledge is not the lever.

Arms (all MeanDemandLookahead2Policy, identical sharing rules; only the
demand-rate input to the order-up-to target changes):

``mdl2_prior``         executed anchor: config ``demand_rate_estimates``.
``mdl2_oracle_rate``   true current Poisson mean, including regime drift and
                       transient shock multipliers (``env._effective_demand_rates()``).
``mdl2_oracle_regime`` base rate x regime multiplier only; knows the slow drift
                       but not transient shocks. Separates level knowledge from
                       shock-timing knowledge.
``mdl2_rolling005``    deployable adaptive: existing rMDL-2 (5% weight on the
                       12-epoch rolling mean).
``mdl2_rolling100``    deployable adaptive: 100% weight on the rolling mean.

Scenarios: the four routing-primary scenarios composed exactly as the formal
benchmark composes them for heuristics, plus the two legacy demand-drift
configs re-run under the same routing contract and the current 500k/100k/25k
penalty weights (labelled as such; they are not the 2026-07 studies).

Pairing: every arm in a scenario runs the same replication seeds. Per-episode
arrival fingerprints are recorded so world sharing is verified, not assumed.

No training, no checkpoint, no formal-holdout seed reuse. Seed stream 99.1M+
is unused elsewhere in the repository as of 2026-09-15.
"""

from __future__ import annotations

import argparse
import copy
import re
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

from evaluation.evaluate_formal import evaluate_agent  # noqa: E402
from evaluation.run_full_benchmark import (  # noqa: E402
    load_benchmark_plan,
    make_scenario_env_config,
)
from src.baselines.heuristics import (  # noqa: E402
    MeanDemandLookahead2Policy,
    facility_net_action_from_arrays,
    patient_priority_from_env,
)
from src.rl.experiment import build_env, write_rows  # noqa: E402

ROUTING_SCENARIOS = (
    "routing_nominal_history",
    "routing_abrupt_regime_shift",
    "routing_regional_drift",
    "routing_compound_regional_stress",
)
LEGACY_DRIFT_CONFIGS = {
    "routing_contract_demand_drift": "experiments/configs/20_clinic_patient_condition_geo_demand_drift.json",
    "routing_contract_demand_drift_severe": "experiments/configs/20_clinic_patient_condition_geo_demand_drift_severe.json",
}
CURRENT_WEIGHTS = {"weight_patient_lost": 500000, "weight_expiry": 100000, "weight_urgency": 25000}
ARMS_DEFAULT = ("mdl2_prior", "mdl2_oracle_rate", "mdl2_oracle_regime", "mdl2_rolling005", "mdl2_rolling100")
ARMS = ARMS_DEFAULT
DEFAULT_SEED_BASE = 99_100_000
REPORT_METRICS = (
    "total_cost",
    "patients_lost",
    "completion_service_level",
    "patient_ineligibility_during_manufacturing_rate",
    "patient_loss_cost",
    "expiry_cost",
    "reagent_purchase_cost",
)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


class RateSourceMDL2(MeanDemandLookahead2Policy):
    """MDL-2 whose lookahead demand rate comes from a chosen source."""

    algorithm = "mdl2_variant"

    def __init__(
        self,
        *,
        rate_source: str,
        history_weight: float | None = None,
        order_up_to_multiplier: float = 1.0,
        anticipation_horizon: int = 0,
        lookahead_periods: int | None = None,
    ):
        config: dict[str, Any] = {"local_order_up_to_multiplier": float(order_up_to_multiplier)}
        if lookahead_periods is not None:
            config["lookahead_periods"] = int(lookahead_periods)
        if history_weight is not None:
            config.update({"use_demand_history": True, "demand_history_weight": float(history_weight)})
        super().__init__(config=config)
        self.rate_source = rate_source
        self.anticipation_horizon = int(anticipation_horizon)
        self._fingerprint: list[float] = []

    def reset(self) -> None:
        self._fingerprint = []

    def _rates(self, env) -> np.ndarray:
        if self.rate_source == "prior":
            return np.asarray(getattr(env, "demand_rate_estimates", env.demand_rates), dtype=float)
        if self.rate_source == "oracle_rate":
            return np.asarray(env._effective_demand_rates(), dtype=float)
        if self.rate_source == "oracle_regime":
            return np.asarray(env.demand_rates * env.demand_regime_multiplier, dtype=float)
        if self.rate_source == "anticipate":
            # Regime rate H epochs ahead; transient shocks are unpredictable and excluded.
            return np.asarray(
                env.demand_rates * regime_multiplier_at(env, int(env.t) + self.anticipation_horizon),
                dtype=float,
            )
        if self.rate_source == "regime_max":
            # Robust hedge: knows the set of regimes, not their timing.
            init = np.asarray(env.demand_regime_initial_multipliers, dtype=float)
            fin = np.asarray(env.demand_regime_final_multipliers, dtype=float)
            return np.asarray(env.demand_rates * np.maximum(init, fin), dtype=float)
        raise ValueError(self.rate_source)

    def _facility_net_action(self, env) -> np.ndarray:
        # Arrival fingerprint for the CRN check (demand already drawn for this epoch).
        self._fingerprint.append(float(np.sum(env.demand)))
        return facility_net_action_from_arrays(
            demand=env.demand,
            specimens=env.specimens,
            reagents=env.reagents,
            bioreactors=env.bioreactors,
            supplier_available=env.supplier_available,
            demand_forecast=getattr(env, "demand_forecast", None),
            demand_history_mean=(
                env._demand_history_features()[0]
                if env.config.include_demand_history_state
                else None
            ),
            demand_rates=self._rates(env),
            max_reagent_replenishment=env.max_reagent_replenishment,
            max_specimen_transfer=float(env.config.max_specimen_transfer),
            max_bioreactor_transfer=float(env.config.max_bioreactor_transfer),
            max_reagent_transfer=float(env.config.max_reagent_transfer),
            specimen_edges=env.specimen_edges,
            capacity_edges=env.capacity_edges,
            resource_edges=env.resource_edges,
            settings=self.settings,
            patient_priority=patient_priority_from_env(env, self.settings),
        )

    def fingerprint(self) -> str:
        return hashlib.sha1(np.asarray(self._fingerprint, dtype=np.float64).tobytes()).hexdigest()[:12]


def regime_multiplier_at(env, t: int) -> np.ndarray:
    """Replicates the environment's persistent-regime schedule at epoch ``t``."""

    cfg = env.config
    change = int(getattr(cfg, "demand_regime_change_step", 0))
    duration = int(getattr(cfg, "demand_regime_transition_duration", 0))
    init = np.asarray(env.demand_regime_initial_multipliers, dtype=float)
    fin = np.asarray(env.demand_regime_final_multipliers, dtype=float)
    if t < change:
        return init
    if duration == 0:
        return fin
    progress = float(np.clip((t - change) / duration, 0.0, 1.0))
    return (1.0 - progress) * init + progress * fin


def make_policy(arm: str) -> RateSourceMDL2:
    if arm == "mdl2_prior":
        return RateSourceMDL2(rate_source="prior")
    if arm == "mdl2_oracle_rate":
        return RateSourceMDL2(rate_source="oracle_rate")
    if arm == "mdl2_oracle_regime":
        return RateSourceMDL2(rate_source="oracle_regime")
    match = re.fullmatch(r"mdl2_anticipate(\d{3})", arm)
    if match:
        return RateSourceMDL2(rate_source="anticipate", anticipation_horizon=int(match.group(1)))
    if arm == "mdl2_hedge_regime_max":
        return RateSourceMDL2(rate_source="regime_max")
    match = re.fullmatch(r"mdl2_hedge_out(\d{3})", arm)
    if match:
        # mdl2_hedge_outNNN: executed prior with order-up-to multiplier NNN/100.
        return RateSourceMDL2(rate_source="prior", order_up_to_multiplier=int(match.group(1)) / 100.0)
    match = re.fullmatch(r"mdl_look(\d{3})", arm)
    if match:
        # mdl_lookNNN: MDL family with NNN lookahead periods (MDL-2 is 002).
        return RateSourceMDL2(rate_source="prior", lookahead_periods=int(match.group(1)))
    match = re.fullmatch(r"mdl2_rolling(\d{3})", arm)
    if match:
        # mdl2_rollingNNN: NNN/100 weight on the 12-epoch rolling arrival mean.
        return RateSourceMDL2(rate_source="prior", history_weight=int(match.group(1)) / 100.0)
    raise ValueError(arm)


# --------------------------------------------------------------------------- #
# Scenario composition
# --------------------------------------------------------------------------- #


def compose_scenarios(plan: dict[str, Any], names: list[str]) -> dict[str, dict[str, Any]]:
    by_name = {s["name"]: s for s in plan["scenarios"]}
    nominal_overrides = dict(by_name["routing_nominal_history"]["env_overrides"])
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in by_name:
            env = make_scenario_env_config(plan, "mdl2", by_name[name])
            out[name] = {"env": env, "source": by_name[name]["env_config"], "adapted": False}
        elif name in LEGACY_DRIFT_CONFIGS:
            overrides = dict(nominal_overrides)
            overrides["scenario_name"] = name
            overrides.update(CURRENT_WEIGHTS)
            env = make_scenario_env_config(
                plan, "mdl2", {"env_config": LEGACY_DRIFT_CONFIGS[name], "env_overrides": overrides}
            )
            out[name] = {"env": env, "source": LEGACY_DRIFT_CONFIGS[name], "adapted": True}
        else:
            raise ValueError(f"unknown scenario {name}")
    return out


def mismatch_summary(env_cfg: dict[str, Any]) -> dict[str, Any]:
    rates = np.asarray(env_cfg.get("demand_rates"), dtype=float)
    est = env_cfg.get("demand_rate_estimates")
    est = rates.copy() if est is None else np.asarray(est, dtype=float)
    init = np.asarray(env_cfg.get("demand_regime_initial_multipliers", np.ones_like(rates)), dtype=float)
    fin = np.asarray(env_cfg.get("demand_regime_final_multipliers", np.ones_like(rates)), dtype=float)
    return {
        "prior_total_rate": float(est.sum()),
        "true_base_total_rate": float(rates.sum()),
        "true_initial_regime_total_rate": float((rates * init).sum()),
        "true_final_regime_total_rate": float((rates * fin).sum()),
        "max_abs_clinic_level_misspec_pct_initial": float(100 * np.max(np.abs(rates * init - est) / est)),
        "max_abs_clinic_level_misspec_pct_final": float(100 * np.max(np.abs(rates * fin - est) / est)),
        "demand_shock_probability": env_cfg.get("demand_shock_probability"),
        "demand_shock_multiplier": env_cfg.get("demand_shock_multiplier"),
        "weight_patient_lost": env_cfg.get("weight_patient_lost"),
    }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    env = build_env({"env": copy.deepcopy(job["env"])}, seed=job["seed"])
    policy = make_policy(job["arm"])
    rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    started = time.perf_counter()
    for rep in range(job["replications"]):
        rep_rows = evaluate_agent(
            policy, env, algorithm=job["arm"], seed=job["seed"] + rep, replications=1, max_steps=None
        )
        # evaluate_agent resets with seed+replication where replication==0, so
        # the world seed is job["seed"] + rep; rewrite the bookkeeping columns.
        for row in rep_rows:
            row["seed"] = job["seed"]
            row["evaluation_seed"] = job["seed"]
            row["replication"] = rep
            row["training_seed"] = 0
            row["arm"] = job["arm"]
            row["arrival_fingerprint"] = policy.fingerprint()
        fingerprints.append(policy.fingerprint())
        rows.extend(rep_rows)
    return {
        "scenario": job["scenario"],
        "arm": job["arm"],
        "rows": rows,
        "fingerprints": fingerprints,
        "seconds": time.perf_counter() - started,
    }


def paired_bootstrap(diff: np.ndarray, *, resamples: int, alpha: float, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(results: list[dict[str, Any]], *, resamples: int, alpha: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fp: dict[tuple[str, str], list[str]] = {}
    for r in results:
        by[(r["scenario"], r["arm"])] = sorted(r["rows"], key=lambda x: int(x["replication"]))
        fp[(r["scenario"], r["arm"])] = r["fingerprints"]
    scenarios = sorted({k[0] for k in by})
    arms_run = [a for a in dict.fromkeys(r["arm"] for r in results)]
    out: dict[str, Any] = {"per_scenario": {}, "pooled": {}, "arms": arms_run}
    pooled_diffs: dict[str, dict[str, list[float]]] = {}
    pooled_prior: dict[str, list[float]] = {m: [] for m in REPORT_METRICS}
    for sc in scenarios:
        prior = by[(sc, "mdl2_prior")]
        entry: dict[str, Any] = {"n_worlds": len(prior), "crn_shared": {}, "arms": {}}
        for arm in arms_run:
            if (sc, arm) not in by:
                continue
            entry["crn_shared"][arm] = bool(fp[(sc, arm)] == fp[(sc, "mdl2_prior")])
        prior_vals = {m: np.asarray([float(r[m]) for r in prior]) for m in REPORT_METRICS}
        for m in REPORT_METRICS:
            pooled_prior[m].extend(prior_vals[m].tolist())
        for arm in arms_run:
            if (sc, arm) not in by:
                continue
            rows = by[(sc, arm)]
            arm_entry: dict[str, Any] = {}
            for m in REPORT_METRICS:
                vals = np.asarray([float(r[m]) for r in rows])
                diff = vals - prior_vals[m]  # arm minus executed prior; negative = arm better on cost
                lo, hi = paired_bootstrap(diff, resamples=resamples, alpha=alpha, rng=rng)
                arm_entry[m] = {
                    "mean": float(vals.mean()),
                    "prior_mean": float(prior_vals[m].mean()),
                    "mean_difference": float(diff.mean()),
                    "pct_of_prior": float(100 * diff.mean() / prior_vals[m].mean()) if prior_vals[m].mean() else None,
                    "ci_low": lo,
                    "ci_high": hi,
                    "wins": int(np.sum(diff < 0)),
                    "ties": int(np.sum(diff == 0)),
                }
                pooled_diffs.setdefault(arm, {}).setdefault(m, []).extend(diff.tolist())
            entry["arms"][arm] = arm_entry
        out["per_scenario"][sc] = entry
    for arm, per_metric in pooled_diffs.items():
        out["pooled"][arm] = {}
        for m, diffs in per_metric.items():
            d = np.asarray(diffs)
            lo, hi = paired_bootstrap(d, resamples=resamples, alpha=alpha, rng=rng)
            pm = float(np.mean(pooled_prior[m]))
            out["pooled"][arm][m] = {
                "mean_difference": float(d.mean()),
                "pct_of_prior": float(100 * d.mean() / pm) if pm else None,
                "ci_low": lo,
                "ci_high": hi,
                "n": int(d.size),
            }
    return out


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    ARMS = tuple(summary.get("arms", ARMS_DEFAULT))
    L = ["# Prior-oracle gap screen", ""]
    L.append(f"Seed base `{meta['seed_base']}`, {meta['replications']} paired worlds per scenario, "
             f"{meta['resamples']} bootstrap resamples. Differences are arm minus executed-prior MDL-2; "
             f"negative cost is better for the arm.")
    L += ["", "## Scenario mismatch inventory", "",
          "| Scenario | prior Σλ | true Σλ initial → final | max clinic-level misspec (initial / final) | shocks | adapted |",
          "| --- | ---: | ---: | ---: | --- | :--: |"]
    for sc, m in meta["mismatch"].items():
        L.append(f"| {sc} | {m['prior_total_rate']:.1f} | {m['true_initial_regime_total_rate']:.1f} → {m['true_final_regime_total_rate']:.1f} | "
                 f"{m['max_abs_clinic_level_misspec_pct_initial']:.0f}% / {m['max_abs_clinic_level_misspec_pct_final']:.0f}% | "
                 f"p={m['demand_shock_probability']} x{m['demand_shock_multiplier']} | {'yes' if meta['adapted'][sc] else ''} |")
    L += ["", "## Total cost: arm minus executed prior, per scenario", "",
          "| Scenario | worlds | CRN shared | " + " | ".join(a.replace('mdl2_', '') for a in ARMS[1:]) + " |",
          "| --- | ---: | :--: | " + " | ".join("---:" for _ in ARMS[1:]) + " |"]
    for sc, e in summary["per_scenario"].items():
        cells = []
        for a in ARMS[1:]:
            if a not in e["arms"]:
                cells.append("—"); continue
            t = e["arms"][a]["total_cost"]
            cells.append(f"{t['pct_of_prior']:+.3f}% [{100*t['ci_low']/t['prior_mean']:+.3f}, {100*t['ci_high']/t['prior_mean']:+.3f}]")
        shared = all(e["crn_shared"].values())
        L.append(f"| {sc} | {e['n_worlds']} | {'yes' if shared else 'NO'} | " + " | ".join(cells) + " |")
    L += ["", "## Pooled over scenarios (equal weight)", "",
          "| Arm | Δ cost | % of prior | 95% CI (%) | Δ patients lost | Δ completion (pp) |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    pm_cost = None
    for a in ARMS[1:]:
        if a not in summary["pooled"]:
            continue
        p = summary["pooled"][a]
        c = p["total_cost"]; pl = p["patients_lost"]; cs = p["completion_service_level"]
        base = c["mean_difference"] / (c["pct_of_prior"] / 100) if c["pct_of_prior"] else 1.0
        L.append(f"| {a} | {c['mean_difference']:,.0f} | {c['pct_of_prior']:+.3f}% | "
                 f"[{100*c['ci_low']/base:+.3f}, {100*c['ci_high']/base:+.3f}] | {pl['mean_difference']:+.2f} | {100*cs['mean_difference']:+.3f} |")
    L += ["", "## Patients lost and completion, per scenario (oracle_rate arm)", "",
          "| Scenario | prior lost | oracle lost | Δ lost | prior completion | Δ completion (pp) |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sc, e in summary["per_scenario"].items():
        if "mdl2_oracle_rate" not in e["arms"]:
            continue
        a = e["arms"]["mdl2_oracle_rate"]
        L.append(f"| {sc} | {a['patients_lost']['prior_mean']:.1f} | {a['patients_lost']['mean']:.1f} | {a['patients_lost']['mean_difference']:+.2f} | "
                 f"{100*a['completion_service_level']['prior_mean']:.2f}% | {100*a['completion_service_level']['mean_difference']:+.3f} |")
    return "\n".join(L) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="experiments/configs/patient_indexed_specimen_routing_benchmark.json")
    p.add_argument("--scenarios", nargs="+", default=list(ROUTING_SCENARIOS) + list(LEGACY_DRIFT_CONFIGS))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--replications", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--resamples", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-root", default="results/prior_oracle_gap_screen")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if "mdl2_prior" not in args.arms:
        raise SystemExit("mdl2_prior must be included as the reference arm")
    plan = load_benchmark_plan(args.plan)
    scenarios = compose_scenarios(plan, list(args.scenarios))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx, (name, sc) in enumerate(scenarios.items()):
        seed = int(args.seed_base) + idx * 100_000
        for arm in args.arms:
            jobs.append({"scenario": name, "arm": arm, "env": sc["env"], "seed": seed, "replications": int(args.replications)})
    meta = {
        "plan": args.plan,
        "seed_base": int(args.seed_base),
        "replications": int(args.replications),
        "resamples": int(args.resamples),
        "alpha": args.alpha,
        "arms": list(args.arms),
        "scenario_seeds": {name: int(args.seed_base) + i * 100_000 for i, name in enumerate(scenarios)},
        "mismatch": {name: mismatch_summary(sc["env"]) for name, sc in scenarios.items()},
        "adapted": {name: sc["adapted"] for name, sc in scenarios.items()},
        "sources": {name: sc["source"] for name, sc in scenarios.items()},
    }
    (out_root / "resolved_env_configs.json").write_text(json.dumps({n: s["env"] for n, s in scenarios.items()}, indent=1, sort_keys=True))
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for job in jobs:
            r = run_job(job); results.append(r)
            print(f"done {r['scenario']} {r['arm']} in {r['seconds']:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            for r in ex.map(run_job, jobs):
                results.append(r)
                print(f"done {r['scenario']} {r['arm']} in {r['seconds']:.0f}s", flush=True)
    all_rows = [row for r in results for row in r["rows"]]
    write_rows(all_rows, out_root / "rows.csv")
    summary = summarize(results, resamples=args.resamples, alpha=args.alpha, seed=0)
    summary["meta"] = meta
    summary["wall_seconds"] = time.perf_counter() - started
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    md = render_markdown(summary, meta)
    (out_root / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
