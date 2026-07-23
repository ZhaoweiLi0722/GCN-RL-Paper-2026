# Roadmap

Very small, independently reviewable phases, in implementation order. Each phase
becomes its own `feature-spec` (`specs/YYYY-MM-DD-<phase>/`) when started.
Writing phases (intro, methodology, experiment design) can proceed in parallel
with code phases where marked. Two **gates** control risk: a collaborator/scope
gate up front, and a **pilot gate** before the expensive full campaign.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done.

## Timeline note — July 2026 target (read first)

Venue is **EAAI**; the internal aim is to **submit July 2026**. Today is well
into July, so the full plan below will **not** fit that window. Treat the target
as forcing an **MVP-first cut**, most likely:

- **In for the July target:** Phase 0–3, the env layer (Phase 4), baselines +
  the graph family on the **static** encoder (Phases 5–6), a pilot (Phase 7),
  and a **trimmed** experiment matrix (Phase 9) — e.g. fewer scenarios/backbones,
  seeds ≥5.
- **Deferred to a revision / follow-up:** the temporal encoder (Phase 8) and the
  full-scale matrix; MARL stays future work.

Reconcile this with Zhaowei in Phase 0 before committing — if July is firm, we
prioritize a defensible minimal result; if it can slip, we widen scope.

## Now

### Phase 0 — Foundations & alignment  *(gate)*
- [x] Confirm the patient-condition direction and scope with Zhaowei — assumed agreed 2026-07-11; to be ratified at the next meeting via `alignment-brief-2026-07-11.md`
- [x] **Preserve the original manuscript:** tagged `manuscript-v0` before any edits (2026-07-11)
- [ ] Agree the July-2026 MVP cut vs. a slipped timeline (see timeline note) — still open, needs a call
- [ ] Set up the Gurobi academic license (GaTech) for the MILP heuristics; add `gurobipy` when Phase 5 starts
- [x] Reproduce the existing benchmark at smoke budget on this machine; confirm baseline numbers log correctly (no new results claimed) — done 2026-07-10: full pipeline (15 train + 27 eval jobs) runs end-to-end, outputs git-ignored. Note: smoke-scale learned-agent costs are meaningless (untrained); GCN-DDPG's apparent parity is a heuristic warm-start artifact. Confirms plumbing only.
- **Depends on:** nothing. **Blocks:** Phases 2, 4 (design/impl should wait on scope sign-off).

### Phase 1 — Introduction & Literature Review rewrite  ✅ done 2026-07-11  *(merged, PR #1)*
- [x] Problem-first Intro (perishable / identity-bound / patient-condition-driven), five RQs, honest "no algorithmic-novelty" concession
- [x] Literature Review rebuilt as an argument (three-layer stack → design implications); positioning table vs closest prior work
- [x] Citations replaced (all 29 keys resolve); broken `Eq. (14)` + malformed projection equation fixed in the method pass
- [x] Also delivered here: Section 2 constrained-graph-MDP formulation, family-framed Method (§4) + backbone table, regenerated Figure 1, abstract de-DDPG'd, second author added
- **Follow-up (not this phase):** §5–§6 (Experiments/Discussion) still hold the old single-method numbers — updated in Phase 10 from campaign outputs.

### Phase 2 — Finalize methodology plan  *(doc only)*
- [ ] Write the problem formulation extended with patient-condition states and product/material expiry (constrained graph MDP)
- [ ] Specify the graph actor-critic family and the pilot-gated temporal-encoder candidate
- [ ] Define state / action / reward precisely, including expiry deadlines and condition-driven urgency
- **Depends on:** Phase 0. **Blocks:** Phases 4–6.

### Phase 3 — Finalize experiment design  *(doc only)*
- [ ] Define scenarios: disruption levels, forecast-error regimes, and the **new patient-condition / expiry stress regime**
- [ ] Define the ablation matrix (flat vs graph; DDPG/TD3/SAC/PPO; static vs temporal; edge-type ablations)
- [ ] Define metrics (cost, service level, patient eligibility/outcomes, utilization) and the **statistical protocol** (seeds, CIs, IQM, paired tests). **Required:** DDPG-family results report *all* seeds + IQM — the verification harness showed flat DDPG ranging from near-optimal to full divergence across seeds (see `tech-stack.md` 2026-07-11)
- [ ] Define the compute budget and the smoke→pilot→full staging
- **Depends on:** Phase 2. **Blocks:** Phases 7, 9.

## Next

### Phase 4 — Implement the patient-condition + expiry environment layer  ✅ done 2026-07-11
- [x] New env module (base `capacity_planning.py` intact); per-clinic patient queues with survival decay + deterioration shock (`src/env/patient_condition.py`, `patient_capacity_planning.py`)
- [x] Age-bucketed, expiry-aware material inventory (`src/env/aging_inventory.py`, bioreactor-pipeline pattern) + disabled-by-default viability hook; eligibility/expiry/urgency cost terms
- [x] Fixed-width patient observation + graph summary (decision-aligned 4-bucket histogram, config-driven)
- [x] Configs (20-clinic + 3 disruption variants + 2-clinic dev), `build_env` wiring, unit + integration tests, end-to-end smoke; 99 tests green
- See feature spec `specs/2026-07-11-patient-condition-env/`. **Note:** built ahead of formal Phases 2–3 docs; the feature spec served as the design. Methodology/experiment-design write-ups (Phases 2–3) still to be authored.
- **Deferred (as planned):** temporal encoder (Phase 8), inter-clinic patient routing + cold-chain viability dynamics (hook wired but off).

### Phase 5 — Benchmark algorithms  *(verification gate)*
- [x] Build the shared **verification harness** (V2/V3) — done 2026-07-11: `src/verification/` LQR task with an analytic Riccati optimum (dependency-free; chosen over `Pendulum-v1` for a precise reference). Already surfaced DDPG instability. *Still todo:* extend to GNN encoders (Phase 6) and the capacity-planning action projection
- [x] Confirm MYO/ISO/MDL-1/MDL-2/F-MYO + flat-DDPG on the patient env (sanity-verified); add patient-aware **uMYO** (ties MYO — recorded finding) — done 2026-07-11
- [x] Patient-env sanity for flat-DDPG (beats random ~100x); LQR-gate results recorded in `provenance.md`
- [ ] Port MILP heuristics from prior code using the Gurobi academic license — **deferred** (MDL-2 is the strong lookahead for now)
- See feature spec `specs/2026-07-11-benchmark-algorithms/`. TD3/SAC/PPO patient-env sanity carried into Phase 6.
- **Depends on:** Phase 4. **Blocks:** Phase 7 (no algorithm enters the pilot unverified).

### Phase 6 — Graph method family  ✅ done 2026-07-11
- [x] Build GNN-TD3, GNN-SAC, GNN-PPO as *verified GNN encoder ∘ verified backbone*; keep GNN-DDPG as ablation (`src/models/gcn_td3.py`, `gcn_sac.py`, `gcn_ppo.py`; shared plumbing in `graph_features.py` + graph heads in `gcn.py`)
- [x] Verify each: encoder component tests (`tests/test_gcn_heads.py`) + patient-env sanity (`tests/test_gcn_patient_sanity.py`, each beats random ≥2x); provenance table extended
- New agents **learn from scratch** (no residual/imitation) for a clean comparison; the graph plumbing is patient-aware (select-path ≡ replay-path proven). See feature spec `specs/2026-07-11-graph-method-family/`.
- **Deferred to the pilot (Phase 7):** the V4 feasibility-projection ablation and multi-seed IQM curves (V5); no LQR-graph retrofit (backbones already LQR-verified in flat form).
- **Depends on:** Phases 4, 5 (reuses the verification harness).

### Phase 7 — Pilot experiments  ✅ done 2026-07-12  *(pilot gate)*
- [x] Two-stage lean pilot (2-clinic screen → 20-clinic confirm) across baselines + graph family; IQM/CI reporting; projection-load + stability metrics. See `pilot-findings.md`.
- [x] Follow-on scaled + facility_action campaign (undertraining fix): graph ≫ flat (RQ1 clean win); `gcn_ddpg` > `gcn_td3` at scale (flagship reconsidered); heuristics still win at nominal (RQ5 honest negative). See `campaign-scale-plan.md`, `campaign-results.md`.
- [x] **Decisions:** flagship = `gcn_ddpg` (provisional; `gcn_td3` high-variance); **temporal encoder DEFERRED** (condition-awareness not decisive at nominal; the 20-clinic gap was undertraining, not condition-blindness).
- **Depends on:** Phases 5, 6. **Blocks:** Phase 9.

### Phase 8 — Temporal/condition-aware encoder  *(DEFERRED per Phase 7)*
- [ ] Deferred to a revision. Pilot showed eligibility non-discriminating at nominal and the flagship gap was undertraining; priority shifted to the robustness/stress campaign (Phase 9), which must establish *when* learned control wins before a temporal encoder is justified.
- **Depends on:** Phase 9 outcome (revisit if condition-severity regimes reward temporal memory).

## Later

### Phase 9 — Robustness / stress campaign  *("when does DRL win", referee-proof)*
The nominal campaign is done and shows fairly-configured heuristics winning — a strong,
non-crippled baseline. Phase 9 finds the regimes where graph-DRL wins, under three
**design invariants** (see `2026-07-11-pilot-experiments/robust-experiment-design.md`):
fair baselines (include `umyo`/`fmyo`, true demand/forecast — no strawman), **true OOD**
(train on a range, test outside it), and IQM/CI rigor.
- [x] Nominal 20-clinic campaign (disruption 0.3): heuristics win; graph ≫ flat. (`campaign-results.md`)
- [~] **A — Disruption severity:** A1 per-regime sweep (0.05/0.3/0.6) running; A2 single-policy robustness (train on [0,0.4], test OOD 0.5/0.6).
- [ ] **B — Patient-condition stress:** condition-aware DRL vs `umyo`/`fmyo` under hardened deterioration (real claim: beat `umyo`).
- [~] **C — Forecast error, redeemed (flagship):** runner + config built and smoke-validated (`evaluation/forecast_robustness.py`); trains on error U[0,0.4], tests OOD 0.6/0.8, fair `fmyo` baseline. **Launches after A/B.**
- [ ] **D — Non-stationarity:** clustered `demand_shock`; train moderate, test severe (OOD).
- [x] Enabling: per-episode randomization hook (`enable_train_randomization`, resample disruption / forecast error each reset) for the train-on-range → test-OOD splits (2026-07-12).
- [ ] Aggregate; generate figures/tables from logged outputs only.
- See feature spec `specs/2026-07-12-robustness-experiments/`.
- **Depends on:** Phase 7.

### Phase 10 — Write Experiment + Results + Discussion
- [ ] Write from logged outputs; report uncertainty; address each reviewer objection with a result
- **Depends on:** Phase 9.

### Phase 11 — Conclusion
- [ ] Write conclusion and future work (incl. MARL)
- **Depends on:** Phase 10.

### Phase 12 — Update Introduction & Abstract to final results
- [ ] Reconcile Intro + Abstract with measured results; final consistency pass
- **Depends on:** Phases 10, 11.

## Done

| Phase | Shipped |
|-------|---------|
| 1 | Introduction + Literature Review rewrite; Section 2 constrained-graph-MDP formulation; regenerated Figure 1 (merged to main, PR #1, 2026-07-11) |
| 4 | Patient-condition + expiry environment layer (2026-07-11) |
| 5 | Benchmark algorithms + verification harness; uMYO (2026-07-11) |
| 6 | Graph method family: GNN-TD3/SAC/PPO + GNN-DDPG ablation (2026-07-11) |
| 7 | Pilot + scaled/facility_action campaign; flagship + temporal-encoder decisions (2026-07-12) |
