# Robustness / stress experiments — Plan

Numbered groups in execution order. Each experiment is a resumable runner under
`caffeinate`; the design invariants (fair baseline / true OOD / IQM+CI) are enforced by
`validation.md`.

## 1. Per-episode randomization hook (enabling)
1. Add a training-time env wrapper / config option that resamples the stressed parameter
   each `reset` from a specified range: `supplier_disruption_rate ~ U[a,b]` and/or
   `demand_forecast_error ~ U[a,b]`. Eval uses a fixed value (the test regime).
2. Keep it off by default; opt-in via config so nominal runs are unchanged.
3. Unit test: over N resets the sampled parameter covers the range; eval path is fixed.

## 2. Experiment C — forecast error, redeemed (flagship, do first after hook)
1. Config: `include_demand_forecast_state = true`, sweep `demand_forecast_error` for eval
   at {0, 0.2, 0.4, 0.6, 0.8} plus a non-stationary `demand_shock` schedule.
2. Train ONE `gcn_ddpg` (facility_action) policy with `demand_forecast_error ~ U[0, 0.4]`
   (train support); test at all eval errors — **0.6, 0.8 are OOD**.
3. Baseline: `fmyo` (fair, forecast-aware) + `umyo` + blind heuristics, same forecast.
4. Runner `evaluation/forecast_robustness.py`; per-error IQM/CI table, in-dist vs OOD split.

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
