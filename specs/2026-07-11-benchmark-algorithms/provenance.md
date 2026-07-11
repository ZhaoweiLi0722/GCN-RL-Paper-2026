# Baseline Provenance & Verification

Source and verification status for every benchmark baseline, as of Phase 5
(2026-07-11). Doubles as the reproducibility appendix. "Sanity" = patient-env
checks (valid facility-net actions, deterministic under seed, beats a random
policy). "LQR gate" = the analytic verification harness
(`evaluation/verify_algorithms.py`, V2/V3).

| Baseline | Source | Kind | Verification | Status |
|----------|--------|------|-------------|--------|
| MYO | `src/baselines/heuristics.py` (repo) | heuristic | patient-env sanity | ✅ verified (group 1) |
| ISO | `src/baselines/heuristics.py` | heuristic | patient-env sanity | ✅ verified (group 1) |
| MDL-1 | `src/baselines/heuristics.py` | heuristic | patient-env sanity | ✅ verified (group 1) |
| MDL-2 (strong lookahead) | `src/baselines/heuristics.py` | heuristic | patient-env sanity | ✅ verified (group 1) |
| F-MYO | `src/baselines/heuristics.py` | heuristic | patient-env sanity | ✅ verified (group 1) |
| uMYO (patient-aware) | `src/baselines/heuristics.py` (new, group 2) | heuristic | sanity + `uMYO >= MYO` | ✅ verified; ties MYO (finding) |
| flat-DDPG | `src/baselines/flat_ddpg.py` | learned | LQR gate + patient sanity | ⚠️ patient sanity ✅ (beats random ~100x); **LQR unstable across seeds** |
| TD3 | `src/baselines/td3.py` | learned | LQR gate | ✅ LQR stable (0.83–0.96); patient sanity pending (Phase 6/7) |
| SAC | `src/baselines/sac.py` | learned | LQR gate | ✅ LQR stable (0.97–0.99); patient sanity pending |
| PPO | `src/baselines/ppo.py` | learned | LQR gate | ✅ LQR passes with tuned config (0.89–0.99); patient sanity pending |
| GNN-DDPG (ablation) | `src/models/gcn_ddpg.py` (repo) | learned (graph) | graph tests + patient sanity | ✅ built; **ablation** (expected dominated, per DDPG instability) |
| GNN-TD3 | `src/models/gcn_td3.py` = GCN encoder ∘ verified TD3 (Phase 6) | learned (graph) | encoder tests + patient sanity | ✅ verified; beats random ~0.18x cost |
| GNN-SAC | `src/models/gcn_sac.py` = GCN encoder ∘ verified SAC (Phase 6) | learned (graph) | encoder tests + patient sanity | ✅ verified; beats random ~0.01x cost |
| GNN-PPO | `src/models/gcn_ppo.py` = GCN encoder ∘ verified PPO (Phase 6) | learned (graph) | encoder tests + patient sanity | ✅ verified; beats random ~0.04x cost |
| MILP MYO/MDL (Gurobi) | prior paper (`docs/research_howard/...`, private) | heuristic | — | ⏳ deferred (needs academic license) |

## Notes

- **flat-DDPG:** on the LQR gate it ranged from near-optimal to full divergence
  across seeds (0.996 / 0.401 / −70.4) — the documented DDPG-instability finding.
  It still clearly beats random on the patient env, so it is a valid *baseline*,
  but any DDPG-family result must report **all seeds + IQM** (see
  `specs/tech-stack.md`, 2026-07-11).
- **TD3/SAC/PPO (flat):** verified at the implementation level (LQR gate). Their
  *graph* variants are built and patient-verified in Phase 6 (below); the *flat*
  TD3/SAC/PPO patient-env training runs at the pilot (Phase 7).
- **Graph family (Phase 6):** GNN-TD3/SAC/PPO are each *verified GNN encoder ∘
  verified backbone*. The encoder is certified by component tests
  (`tests/test_gcn_heads.py`: shapes, action bounds, log-prob, adjacency,
  gradient flow); the composed agent is certified by patient-env sanity
  (`tests/test_gcn_patient_sanity.py`: each beats random by ≥2x). The LQR gate is
  **not** retrofitted for graphs — the backbones already passed it in flat form
  and the graph change is confined to the encoder. Deferred to the pilot: the V4
  feasibility-projection ablation and multi-seed IQM curves. The new agents
  **learn from scratch** (no residual anchor / imitation) for a clean flat-vs-
  graph and backbone comparison; GCN-DDPG's residual machinery stays a
  DDPG-only ablation variant.
- **MILP baselines:** MDL-2 serves as the strong lookahead for now; MILP porting
  is deferred until the Gurobi academic license is set up.
