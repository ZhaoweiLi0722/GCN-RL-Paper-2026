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
- [ ] Confirm the patient-condition direction and scope with Zhaowei (same paper, evolved)
- [ ] **Preserve the original manuscript:** tag `manuscript-v0` (and/or a read-only copy) before any edits
- [ ] Agree the July-2026 MVP cut vs. a slipped timeline (see timeline note)
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
- [ ] Define metrics (cost, service level, patient eligibility/outcomes, utilization) and the **statistical protocol** (seeds, CIs, IQM, paired tests)
- [ ] Define the compute budget and the smoke→pilot→full staging
- **Depends on:** Phase 2. **Blocks:** Phases 7, 9.

## Next

### Phase 4 — Implement the patient-condition + expiry environment layer
- [ ] New env module (leave `src/env/capacity_planning.py` intact); per-clinic patient queues with survival decay + deterioration shock
- [ ] Age-bucketed, expiry-aware specimen/material inventory (reuse the bioreactor-pipeline pattern); eligibility/expiry gate and cost terms
- [ ] New scenario configs; unit tests + a smoke run; `python -m compileall .`
- **Depends on:** Phases 2, 3.

### Phase 5 — Benchmark algorithms  *(verification gate)*
- [ ] Build the shared **verification harness** (V2/V3): reference-task (`Pendulum-v1`) sanity + cross-library curve check (see `tech-stack.md`)
- [ ] Confirm/port MYO, ISO, MDL-1, MDL-2 (and flat DDPG) against the new environment
- [ ] Port MILP heuristics from prior code using the Gurobi academic license (Phase 0)
- [ ] Run each learned baseline through the V1–V5 verification gate; start the provenance table
- **Depends on:** Phase 4. **Blocks:** Phase 7 (no algorithm enters the pilot unverified).

### Phase 6 — Graph method family  *(verification gate)*
- [ ] Build GNN-TD3, GNN-SAC, GNN-PPO as *verified GNN encoder ∘ verified backbone*; keep GNN-DDPG as ablation
- [ ] Run each through the V1–V5 gate (incl. V4 feasibility-projection ablation); extend the provenance table
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
| — | — |
