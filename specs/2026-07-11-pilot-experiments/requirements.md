# Phase 7 — Pilot Experiments (requirements)

Feature branch: `phase-7-pilot-experiments`
Roadmap phase: **Phase 7 — Pilot experiments** (pilot gate).
Depends on: Phase 5 (verified baselines + harness), Phase 6 (graph family).
Blocks: Phase 8 (temporal encoder — pilot-gated), Phase 9 (full campaign).

## Goal

Run a **thorough, two-stage multi-seed pilot** across the full algorithm roster
on the patient-condition regime, and from it make three gate decisions, each
recorded in `tech-stack.md`:

1. **Flagship backbone** — which graph agent proceeds as *the* proposed policy.
2. **Temporal encoder go/no-go** — whether Phase 8 is worth building.
3. **Stability + feasibility-projection load** — is training reliable, and how
   much does action-repair contribute (the "projection masks non-learning"
   objection)?

The pilot is a **gate**, not the full campaign (Phase 9): it uses reduced scope
to buy a defensible decision cheaply, then hands a single flagship to the full
matrix.

## Scope

### Two-stage design (decided — lean)

- **Stage A — screen (2-clinic patient dev regime).** Full roster, 5 seeds.
  Cheap; ranks everything and exposes instability early.
- **Stage B — confirm (20-clinic patient regime).** Promote a **minimal
  contender set**: the top graph backbone from Stage A + GNN-DDPG (ablation) +
  flat-DDPG (graph-vs-flat anchor) + MDL-2 (strong-heuristic anchor, train-free),
  5 seeds. This ratifies the flagship at realistic scale.

Only Stage B results feed the flagship decision; Stage A is a screen/ranking.
**Compute reality:** lean two-stage is ~6–8 hrs total; the full-thorough variant
(~50+ hrs) was rejected as unfit for July. The pilot produces a *decision*, not
publication-final numbers — Phase 9 runs the full matrix.

### Roster

Heuristics: MYO, ISO, MDL-1, MDL-2, F-MYO, uMYO. Flat DRL: flat-DDPG. Graph
family: GNN-DDPG (ablation), GNN-TD3, GNN-SAC, GNN-PPO.

### Budget (decided — lean)

- **~30k training steps** per learned agent (config-driven; enough for a ranking
  — agents already beat random at 2k), **5 seeds**.
- Monte Carlo evaluation over multiple replications per trained policy.
- **V4 projection: measurement only.** Build the always-on projection-load metric
  (‖projected − raw‖ per step); **defer the on/off ablation runs to Phase 9**.
- Stability diagnostics (per-seed spread, divergence flags, IQM).

### In scope (what this phase builds/produces)

| Item | Detail |
|------|--------|
| **Patient eval metrics** | Extend the formal evaluator to capture patient-condition outcomes from `info`: `eligibility_rate`, `patients_lost` (+ ineligible/expired split), `material_wasted` (`finished_expired` + expired specimens), `at_risk_unserved`. These do not exist in `EpisodeMetrics` today. |
| **Statistical aggregation** | IQM + 95% bootstrap CIs across seeds, per algorithm × scenario × metric. The Phase 3 protocol mandates IQM (DDPG instability); current `aggregate_results.py` has none. |
| **Pilot configs** | Per-algorithm configs for the patient regime (2-clinic + 20-clinic), including the graph family. Graph agents must receive `config["env"]` (their `build_graph_spec` needs it). |
| **Two-stage runner** | Orchestration: Stage A screen → rank by eligibility IQM → select promotions → Stage B confirm. Smoke-validated at tiny budget in tests, then run at full budget. |
| **V4 projection measurement** | Always-on measurement of per-step projection magnitude (action-repair load) per algorithm. The on/off ablation runs are deferred to Phase 9. |
| **Stability diagnostics** | Per-seed spread, divergence detection, IQM (reuse/extend `check_training_stability.py`). |
| **Decisions recorded** | Flagship, temporal-encoder go/no-go, stability/projection findings written to `tech-stack.md` decision log + a `pilot-findings.md` in this spec dir. |

### Flagship-selection rule (decided)

Rank by **patient eligibility rate** (IQM across 5 seeds) on **Stage B**
(20-clinic), tie-broken by (1) total cost IQM, then (2) cross-seed stability
(narrower CI / no divergent seeds). The winner is the flagship; record the full
ranking, not just the winner.

### Out of scope (this phase)

- The full experiment campaign (Phase 9): full scenario matrix (disruption ×
  forecast-error × patient-stress), 500-replication evaluation, final figures.
  The pilot uses the base patient regime (+ Stage B scale), not the full matrix.
- Building the temporal encoder (Phase 8) — this phase only *decides* whether to.
- MILP/Gurobi heuristics (still deferred).
- Any new third-party dependency.

## Decisions

- **D1 — Eligibility is the headline metric.** It is the paper's problem-speciality
  wedge (serving patients before deterioration), so it decides the flagship; cost
  and stability are tie-breakers. All metrics are still reported.
- **D2 — Two-stage to control compute.** Screen cheaply on 2-clinic, confirm only
  the contenders at 20-clinic. Avoids paying 20-clinic × 8-seed cost for agents a
  cheap screen already eliminates.
- **D3 — IQM everywhere, all seeds shown.** Mandatory for DDPG-family (Phase 5
  instability finding). No single-seed claims; report IQM + 95% CI + per-seed
  values.
- **D4 — Runs are compute-bound and may run in background.** The harness is built
  and smoke-validated in-session (tiny budgets, in tests); the ~6–8 hr lean run
  is launched separately and may span sessions (`--resume`). The decision is
  recorded from the completed run. No fabricated numbers — a partial run is
  reported as partial.
- **D6 — Blended reward, weights unchanged.** Reward = `−(base_cost + patient
  terms)`; kept as-is (not patient-only, not re-weighted — see `tech-stack.md`,
  2026-07-11). The pilot **must report whether condition-aware methods separate
  from condition-blind ones** on eligibility; a non-separation is a finding
  (evidence for the temporal encoder / a Phase 9 weight revisit), not a bug.
- **D5 — Lean over thorough (compute-informed).** After seeing the ~50+ hr cost of
  the full-thorough two-stage run, scope was cut to lean two-stage / 5 seeds /
  ~30k steps / projection-measurement-only. A pilot only needs to *rank* reliably
  and expose instability; convergence-final numbers and the full matrix are
  Phase 9. Stage B scale/seeds remain the dial to turn if compute still bites.

## Context

- **Reuse existing harness:** `train_off_policy_agent` (generic over the agent
  interface — PPO/GNN-PPO buffer internally and update on rollout/done),
  `evaluate_formal.py`, `run_multi_seed.py`, `run_small_pilot.py`,
  `check_training_stability.py`, `plot_results.py`. Extend, don't rebuild.
- **Metrics source:** patient env `step()` `info` already exposes every patient
  metric needed (see `src/env/patient_capacity_planning.py`).
- **Reproducibility (tech-stack.md):** params in configs; large outputs to
  git-ignored `results/`, `runs/`, `checkpoints/`; English only; no fabricated
  numbers; `compileall` + tests before done.
- **July MVP tension:** the thorough choice is heavier than the MVP default. The
  two-stage design and the "screen-then-promote" cut are what keep it tractable;
  if compute forces a trim, Stage B scale/seeds are the first dial to turn — noted
  as an open call, not silently cut.
