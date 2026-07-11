# Requirements — Benchmark Algorithms on the Patient-Condition Env

**Phase:** 5 (roadmap) · **Branch:** `phase-5-benchmark-algorithms` · **Date:** 2026-07-11

Confirm the benchmark baselines run and verify cleanly on the patient-condition
environment (Phase 4), add one patient-aware heuristic so the comparison isn't
trivial, and record each baseline's verification status. This establishes the
"beat these" set before the graph method family (Phase 6).

## Scope

### In scope

- **Confirm existing baselines on the patient env:** MYO, ISO, MDL-1, MDL-2,
  F-MYO heuristics and flat-DDPG. They already build/run (group-7 smoke); this
  phase adds sanity checks and records them as verified.
- **One patient-aware heuristic** (`umyo`, "urgency-aware myopic"): starts from
  the myopic action, then surges replenishment and pulls in bioreactor capacity
  toward clinics with many at-risk / near-expiry patients. Gives a *condition-
  aware* baseline the learned policies must beat — not a straw man.
- **Verification:** RL baselines pass the LQR gate (implementation correctness,
  already run) **and** a patient-env sanity (short training beats random);
  heuristics get patient-env sanity checks (valid actions, beat random,
  deterministic under seed).
- **Provenance table** started: each baseline → source → verification status.

### Out of scope (this phase)

- **Gurobi MILP heuristics — deferred.** MDL-2 (2-period lookahead) serves as the
  strong-lookahead baseline for now; MILP porting waits on the academic license.
- Graph method family (GNN-TD3/SAC/PPO) — Phase 6.
- Temporal encoder, MARL, the full experiment campaign.

### Baseline roster (this phase)

| Baseline | Kind | Condition-aware? | Role | Status target |
|----------|------|------------------|------|---------------|
| MYO | heuristic | no | myopic reference | sanity-verified |
| ISO | heuristic | no | no-sharing lower bound | sanity-verified |
| MDL-1 | heuristic | no | 1-period lookahead | sanity-verified |
| MDL-2 | heuristic | no | **strong lookahead** | sanity-verified |
| F-MYO | heuristic | forecast | forecast-aware myopic | sanity-verified |
| **uMYO** | heuristic (new) | **yes** | condition-aware baseline | built + sanity-verified |
| flat-DDPG | learned | via obs | flat-state RL baseline | LQR gate + patient sanity |

## Decisions (and why)

- **Add a patient-aware heuristic (uMYO).** The learned policies see patient
  condition via the observation; a purely condition-blind heuristic set would let
  RL win too easily. uMYO gives a fair, condition-aware target. Built as a
  subclass of `CapacityHeuristicPolicy` overriding `_facility_net_action`, reading
  the patient env's at-risk / near-expiry counts. (User choice: existing + one
  patient-aware.)
- **Verification standard:** RL through the LQR gate (correctness) + patient-env
  sanity (beats random on our problem); heuristics get patient-env sanity only —
  they don't learn, so the LQR gate doesn't apply. (User choice.)
- **MDL-2 is the strong baseline; MILP deferred.** Protects the July timeline and
  needs no Gurobi. (User choice.)

## Context

- **Heuristics read the env directly** (`env.specimens`, `env.reagents`, …), so
  they already work on the patient env; uMYO additionally reads patient-condition
  accessors on `PatientConditionCapacityEnv` (`patient_queues`, at-risk / near-
  expiry counts) — guard for the base env where those are absent.
- **Verification harness** (`evaluation/verify_algorithms.py`) already exists for
  the LQR gate; the patient-env sanity is a new, small check (train briefly on a
  2-clinic patient config, confirm cost beats a random policy).
- **Constraints:** no new dependencies (no Gurobi this phase); English-only;
  deterministic under seed; register `umyo` in `available_heuristics()` /
  `get_heuristic_class()`; do not modify the base heuristics' behaviour.
- **Open items:** cost-weight sensitivity of uMYO's surge strength — expose as
  config, leave tuning to the Phase 7 pilot.
