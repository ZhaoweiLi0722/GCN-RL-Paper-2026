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
| MILP MYO/MDL (Gurobi) | prior paper (`docs/research_howard/...`, private) | heuristic | — | ⏳ deferred (needs academic license) |

## Notes

- **flat-DDPG:** on the LQR gate it ranged from near-optimal to full divergence
  across seeds (0.996 / 0.401 / −70.4) — the documented DDPG-instability finding.
  It still clearly beats random on the patient env, so it is a valid *baseline*,
  but any DDPG-family result must report **all seeds + IQM** (see
  `specs/tech-stack.md`, 2026-07-11).
- **TD3/SAC/PPO:** verified at the implementation level (LQR gate). Their patient-
  env training + the graph variants are Phase 6 work; patient sanity will be run
  as they are brought onto the patient env there.
- **MILP baselines:** MDL-2 serves as the strong lookahead for now; MILP porting
  is deferred until the Gurobi academic license is set up.
