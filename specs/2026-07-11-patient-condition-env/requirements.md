# Requirements — Patient-Condition & Product-Expiry Environment Layer

**Phase:** 4 (roadmap) · **Branch:** `phase-4-patient-condition-env` · **Date:** 2026-07-11

The paper's differentiator: a distributed cell-therapy capacity-planning
environment where **individual patients deteriorate while waiting** and
**autologous material expires**. This spec adds that as a new environment layer,
leaving the existing `src/env/capacity_planning.py` (and its passing tests)
untouched.

## Scope

### In scope

- A new environment (new module) that models, per clinic:
  - **Per-patient entities** in a waiting queue, each with a survival state that
    decays over time (a deterioration shock accelerates decay).
  - **Autologous material/product expiry** — specimens and finished product age
    and are wasted if they pass a shelf-life window.
  - **Full reward coupling**: eligibility gate (patient lost if survival drops
    below threshold), expiry-waste cost, and urgency-weighted demand/service.
- A **fixed-width per-clinic observation summary** derived from the per-patient
  queue, so existing flat and graph agents work unchanged in dimensionality.
- New scenario configs (20-clinic patient-condition, plus disruption variants).
- Unit tests, a CPU smoke run, and `compileall`.

### Out of scope (this phase)

- Temporal/recurrent encoders (Phase 8, pilot-gated) and MARL (future work).
- New algorithms (Phases 5–6) — this phase only delivers the environment.
- Real patient data or clinical calibration; parameters are literature-based.
- Multi-health-state (Markov) patient models — not MVP, not planned this phase.
- **Transport-time → product-viability coupling is scoped as a disabled-by-
  default hook** (interface built now, dynamics off): the cold-chain mechanism
  by which inter-clinic shipping time degrades product viability. Built so it can
  be switched on via config without rework, and turned on in the Phase 7 pilot if
  time allows. The *dynamics* are out of scope for the MVP; the *interface* is in.

### Entities & state

| Entity | State it carries |
|--------|------------------|
| Patient (per clinic, per patient) | survival `S ∈ [0,1]`, enrollment epoch, health index `H`, deterioration-shock time, status (waiting / in-production / delivered / lost-ineligible) |
| Specimen / autologous material | age since collection, expiry deadline |
| Finished product | age since production complete, delivery deadline (shelf-life) |
| Clinic (node) | everything the base env tracks (reagents, bioreactor pipeline, …) **plus** the patient queue and aging-material buckets |

### Patient lifecycle (per epoch)

1. **Arrival** — demand draws enroll new patients (survival starts near 1.0,
   per-patient health index sets the decay curve).
2. **Waiting** — each waiting patient's survival decays; after a stochastic
   deterioration time the decay rate increases.
3. **Eligibility gate** — if survival falls below `eligibility_threshold` the
   patient is **lost** (ineligible): removed from the queue, penalty incurred.
4. **Production start** — when specimen + idle bioreactor + reagents coincide;
   deteriorating patients get priority (urgency-weighted).
5. **Production** — occupies the bioreactor pipeline for the production lead time.
6. **Delivery / expiry** — product delivered before its shelf-life → success;
   material/product that ages out before delivery → **expiry waste**, penalty.

### Reward (extends the base cost; reward = −cost)

`total_cost = base_cost + w_lost · patients_lost + w_expiry · material_wasted
             + w_urgency · urgency_weighted_shortfall`

Weights are config-driven. Base cost (reagent/bioreactor/transfer terms) is
unchanged.

### Observation (fixed width — agents unchanged in dimensionality)

Per clinic, summarize the per-patient queue into a fixed vector, e.g.:
waiting count, mean survival, count-at-risk (below a margin), count near expiry,
and a small **survival/age histogram** (fixed buckets). Concatenate with the
base per-facility features. Graph node features get the same summary columns.

## Decisions (and why)

- **Per-patient simulation, summarized observation.** Fidelity where it matters
  (dynamics, eligibility, expiry) without breaking fixed-dim flat baselines.
  Chosen by the user; the summary reconciles it with the existing agents.
- **Full reward coupling.** Eligibility + expiry + urgency all enter the cost, so
  the paper can show condition-blind planners failing on patient outcomes.
- **MVP-lean core + viability hook.** Port the SimPAC mechanism — a survival
  scalar decaying between empirical survival-curve bounds, with a Weibull
  deterioration shock and an eligibility threshold — in a simple, CPU-fast form
  (no multi-state models). **Additionally**, build the material/product aging so
  a transport-time→viability function is a clean, config-gated seam that defaults
  to a no-op. This buys the highest-value fidelity upgrade (network cold-chain
  routing) later without re-architecting, while protecting the July timeline now.
- **New module, base env untouched.** Compose or subclass
  `CapacityPlanningEnv` so its manufacturing dynamics and tests are preserved;
  the patient/expiry layer is additive.

## Context

- **Stack:** pure NumPy env, mirroring `src/env/capacity_planning.py` conventions
  (config-driven, `reset(seed=)`/`step(action)->(next,reward,done,info)`,
  reward = −cost, `info` dict with diagnostics). Python 3.11, no new dependencies.
- **Pattern to reuse:** the base env's **bioreactor production pipeline**
  (age-advancing array) is the template for **age-bucketed, expiry-aware**
  specimen/material inventory.
- **Source of the patient model:** the SimPAC mechanism from the private
  patient-condition paper (`docs/research_howard/...`, git-ignored) — port the
  *mechanism*, do not copy text; re-parameterize for the network.
- **Constraints:** English-only; deterministic under seed; fast enough for
  multi-seed CPU runs; add new diagnostics to `info` (patients_lost,
  material_wasted, eligibility_rate) without removing existing ones.
- **Action space:** keep the base `facility_net` action semantics; this phase
  changes *state, dynamics, and reward*, not the action interface (so agents
  built in Phases 5–6 attach without action-shape changes beyond the new obs).
- **Open questions for implementation:** exact survival-curve parameterization
  and weight magnitudes (`w_lost`, `w_expiry`, `w_urgency`) — start from
  literature/plausible values, expose as config, tune in the Phase 7 pilot.
