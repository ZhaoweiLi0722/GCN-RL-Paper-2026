# Phase 7 — Pilot Experiments (plan)

Groups 1–2 are metric/stat plumbing (unblock everything downstream); 3–5 build
and smoke-validate the pilot harness at tiny budget; 6 runs the thorough pilot
(compute-bound); 7 records decisions and merges. Each group is committed on its
own and smoke-validated before the next.

## Group 1 — Patient-aware evaluation metrics

1. Extend `evaluation/evaluate_formal.py` `EpisodeMetrics` to accumulate, from
   `info`: `eligibility_rate` (episode-final and mean), `patients_lost` (+ split
   `patients_lost_ineligible` / `patients_lost_expired`), `material_wasted`
   (`finished_expired` + expired specimens), `at_risk_unserved`. Guard on key
   presence so base (non-patient) env evaluation is unchanged.
2. Add the new columns to the summary/row schema; keep existing columns intact.
3. Tests (`tests/test_patient_eval_metrics.py`): on the 2-clinic patient env, a
   short rollout produces finite eligibility_rate ∈ [0,1] and non-negative loss
   counts; base-env evaluation still emits the original schema (regression).

## Group 2 — IQM + bootstrap-CI aggregation

1. Add `evaluation/aggregate_stats.py` (or extend `aggregate_results.py`):
   `interquartile_mean(values)` and `bootstrap_ci(values, alpha=0.05,
   resamples=...)`, aggregating per (algorithm, scenario, metric) across seeds.
   Deterministic under a fixed RNG seed (pass seed in; no `Math.random`-style
   nondeterminism).
2. Output a tidy summary CSV: algorithm × metric → mean, IQM, CI-low, CI-high,
   n_seeds, per-seed values (JSON-encoded), divergence flag.
3. Tests (`tests/test_aggregate_stats.py`): IQM of a known vector matches a hand
   computation; IQM discards the top/bottom quartile (a single wild seed does not
   move it much) while the plain mean does — the DDPG-instability motivation;
   bootstrap CI brackets the mean and is reproducible under a fixed seed.

## Group 3 — Pilot configs (patient regime + graph family)

1. Add per-algorithm configs for the patient regime under `configs/` (or
   `experiments/configs/`) for Stage A (2-clinic) and Stage B (20-clinic):
   graph agents (`gcn_ddpg/td3/sac/ppo`), flat-DDPG, and heuristic eval configs.
   Each learned-agent config carries `env` = the patient env config so
   `build_graph_spec` works; hyperparameters inherit each algorithm's verified
   defaults (document any deviation).
2. A pilot manifest (JSON/py) listing roster, seeds (8), budgets, and the two
   scenarios.
3. Test (`tests/test_pilot_configs.py`): every manifest config loads, resolves to
   a buildable env + agent (1 step) at tiny budget, and graph agents get a valid
   graph spec; torch-guarded.

## Group 4 — Two-stage pilot runner

1. `evaluation/run_patient_pilot.py`: Stage A (screen, full roster, N seeds) →
   aggregate (group 2) → rank by eligibility IQM → select promotions (leader +
   GNN-DDPG + MDL-2 + flat-DDPG, configurable) → Stage B (confirm at 20-clinic) →
   aggregate. Writes all rows + summaries to git-ignored `results/`. Supports
   `--smoke` (tiny budgets) and `--resume` (skip completed algorithm×seed).
2. Reuse `train_off_policy_agent` for every learned agent (generic interface;
   GNN-PPO buffers internally) and `evaluate_agent` for Monte Carlo eval.
3. Test (`tests/test_run_patient_pilot.py`): `--smoke` two-stage run completes
   end-to-end on a 2-agent subset, produces the summary CSV with the new metric
   columns and a non-empty ranking; torch-guarded, tiny budgets, outputs to a
   temp dir.

## Group 5 — Projection-load measurement + stability

1. Instrument `project_action` usage to record per-step **repair magnitude**
   (‖projected − raw‖) so the pilot can report projection load per algorithm
   (add an opt-in measurement path; do not change default behaviour). The on/off
   **ablation runs are deferred to Phase 9** — measurement only here.
2. Extend `check_training_stability.py` (or add a helper) to emit per-seed spread,
   divergence flags, and IQM per algorithm from the pilot rows.
3. Tests (`tests/test_projection_measure.py`): repair-magnitude measurement is
   non-negative and zero when the raw action is already feasible; the stability
   helper flags a divergent seed and computes IQM.

## Group 6 — Run the thorough pilot  *(compute-bound)*

1. Smoke-run the full pipeline (`--smoke`) once end-to-end; confirm plumbing.
2. Launch the thorough run (8 seeds, ~100k+ steps, Stage A → Stage B). This is
   compute-bound and may run in the background / across sessions; use `--resume`.
   Log what completed; **report partial results as partial** (D4).
3. Aggregate completed results → IQM/CI summary + stability report + projection
   ablation; generate comparison figures via `plot_results.py`.

## Group 7 — Decisions, changelog, merge

1. Write `specs/2026-07-11-pilot-experiments/pilot-findings.md`: the ranking, the
   chosen **flagship** (eligibility IQM, tie-breakers), the **temporal-encoder
   go/no-go** with its evidence, and the stability/projection findings.
2. Record the three decisions in `tech-stack.md` decision log; update the roadmap
   (tick Phase 7; set Phase 8's conditional based on the go/no-go).
3. Run the changelog; merge `phase-7-pilot-experiments` → `main`; push to `mine`.

## Validation checkpoints (per group)

`python -m compileall src evaluation tests` clean; the group's own tests + the
full suite green before each commit. Heavy runs excluded from the unit suite
(smoke-scale only in tests). See `validation.md`.
