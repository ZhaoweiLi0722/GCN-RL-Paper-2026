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

### Phase 1 — Introduction & Literature Review rewrite  *(can run in parallel with code)*
- [ ] Draft the gap and research questions from `research-context.md` (problem-first framing)
- [ ] Rebuild the literature review as an argument with the three-layer stack, ending in design implications
- [ ] Replace placeholder/decorative citations; fix the broken equation references noted in the draft
- [ ] Position explicitly against the closest prior work (perishability + patient-condition + identity axes)
- **Depends on:** Phase 0 sign-off. **Deliverable:** revised Intro + Lit Review sections (draft).

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

### Phase 7 — Pilot experiments  *(pilot gate)*
- [ ] Small-scale multi-seed pilot across baselines + graph family on the new regime
- [ ] **Decide the flagship backbone** and **whether to build the temporal encoder** (record the decision in `tech-stack.md`)
- [ ] Check training stability and feasibility-projection load
- **Depends on:** Phases 5, 6. **Blocks:** Phases 8, 9.

### Phase 8 — Temporal/condition-aware encoder  *(conditional on Phase 7)*
- [ ] Only if the pilot supports it: implement the temporal encoder + threading through the agent/replay loop; tests
- **Depends on:** Phase 7 decision.

## Later

### Phase 9 — Full experiment campaign
- [ ] Run the full multi-seed × scenario × Monte Carlo matrix per the staged budget; skip/resume supported
- [ ] Aggregate results, generate figures/tables from logged outputs only
- **Depends on:** Phases 7 (+8 if taken).

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
| 4 | Patient-condition + expiry environment layer (2026-07-11) |
| 5 | Benchmark algorithms + verification harness; uMYO (2026-07-11) |
| 6 | Graph method family: GNN-TD3/SAC/PPO + GNN-DDPG ablation (2026-07-11) |
