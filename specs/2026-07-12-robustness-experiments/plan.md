# Robustness / stress experiments — Plan

Numbered groups in execution order. Each experiment is a resumable runner under
`caffeinate`; the design invariants (fair baseline / true OOD / IQM+CI) are enforced by
`validation.md`.

## 1. Per-episode randomization hook (enabling)  ✅ done 2026-07-12
1. [x] `CapacityPlanningEnv.enable_train_randomization(disruption_range, forecast_error_range)`
   resamples the stressed parameter uniformly each `reset` (before demand/supplier/forecast
   are drawn). `demand_forecast_error` is now a mutable instance attr; config stays immutable.
2. [x] Off by default; eval envs never enable it, so nominal runs are unchanged. Inherited by
   the patient env via `super().reset`.
3. [x] `tests/test_train_randomization.py` (6 tests): range coverage, fixed-eval isolation,
   reproducibility given episode seed, invalid-range rejection.

## 2. Experiment C — forecast error, redeemed (flagship, do first after hook)  ✅ built, staged
1. [x] Config `20_clinic_patient_condition_forecast.json`: `include_demand_forecast_state`,
   horizon 1, fixed non-stationary `demand_shock` (p=0.10, ×2.0, dur 4, cluster 5) so the
   forecast is genuinely worth using; error swept by the runner (diff-isolated vs nominal).
2. [x] Runner `evaluation/forecast_robustness.py`: trains ONE `gcn_ddpg` (facility_action) +
   `flat_ddpg` contrast per seed with `demand_forecast_error ~ U[0,0.4]`; eval at
   {0,0.2,0.4} in-dist, **{0.6,0.8} OOD**. Smoke-validated end-to-end.
3. [x] Baseline `fmyo` (fair, forecast-aware) + `umyo` + blind (mdl2/iso/mdl1/myo); same forecast.
4. [x] Per-error IQM/CI table, in-dist vs OOD split; resumable per (algo,seed,error) CSV.
5. [ ] **Launch** after Experiments A/B finish (sequential — CPU contention). 5 seeds, 150k.

## 3. Experiment A2 — disruption robustness (single policy)
1. Train ONE `gcn_ddpg` with `supplier_disruption_rate ~ U[0, 0.4]`; test at
   {0.05, 0.3} in-dist and {0.5, 0.6} OOD, vs per-regime heuristics.
2. Runner reuses the sweep harness with the randomization hook.
3. Report alongside A1 (per-regime) so both the tuned and the robust story are visible.

## 4. Experiment B — patient-condition stress (baseline already fixed)
1. `condition_stress.py` already includes `umyo`/`fmyo`; run it (after A1).
2. Headline: `gcn_ddpg` vs `umyo` under hardened deterioration; report eligibility,
   patients-lost, cost with CIs.
3. Optional B2: train on a range of `frail_decay`/`weibull_scale`, test OOD-severe.

## 5. Experiment D — non-stationarity
1. Config: clustered `demand_shock` (probability, multiplier, duration, cluster_size).
2. Train on moderate shock; test on severe/longer/clustered (OOD). Fair `fmyo` baseline.
3. Runner `evaluation/nonstationary.py` (or reuse a generic env-config runner).

## 6. Aggregation & write-up
1. One combined table per experiment (regime × method, IQM + CI), in-dist vs OOD flagged.
2. Figures from logged outputs only.
3. Update `robust-experiment-design.md` status and feed results into Phase 10 (§5–§6).

## Notes
- Consider refactoring the near-identical runners (`campaign_runner`, `disruption_sweep`,
  `condition_stress`) into one generic `env-config × roster` runner to cut duplication.
- Heuristic baseline top-ups (`umyo`/`fmyo`) are no-training and cheap; add to A1 post-hoc.
