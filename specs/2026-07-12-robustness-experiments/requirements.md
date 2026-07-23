# Robustness / stress experiments — Requirements

Phase 9 of the roadmap. Find the regimes where graph-DRL beats fairly-configured
heuristics, in a way a referee cannot dismiss as a rigged comparison.

## Motivation

The nominal campaign (`../2026-07-11-pilot-experiments/campaign-results.md`) shows
strong heuristics beating graph-DRL under matched conditions, and graph ≫ flat. That
strong, non-crippled baseline is an asset: it lets a DRL win *under stress* read as real.
The prior DRL paper (Tseng et al.) established the template — DRL wins under distribution
shift — but with three fixable weaknesses. We adopt those as invariants.

## Scope

In:
- **Experiment A — disruption severity.** A1 per-regime sweep (running); A2 single-policy
  robustness (train on a disruption range, test OOD).
- **Experiment B — patient-condition stress.** Condition-aware DRL vs `umyo`/`fmyo` under
  hardened deterioration.
- **Experiment C — forecast error, redeemed (flagship).** Shared forecast signal; train
  on errors ≤ e, test OOD > e and under non-stationary shocks.
- **Experiment D — non-stationarity.** Clustered `demand_shock`; train moderate, test severe.
- **Enabling code:** a per-episode randomization hook (resample disruption / forecast
  error each `reset`) so training sees a range and testing can go OOD.

Out:
- Temporal encoder (deferred). SAC/PPO `facility_action` (still `global_flat`; separate
  follow-up). New algorithms. MARL.

## Design invariants (must hold for every experiment)

1. **Fair baseline.** Include `umyo` (condition-aware) and `fmyo` (forecast-aware);
   heuristics use the true `demand_rates` and the same `demand_forecast` the DRL sees. The
   headline claim is DRL beating the *aware* heuristic, not only the blind ones.
2. **True OOD.** Any robustness claim trains on a range and tests **outside** it; report
   in-distribution vs OOD columns separately. No "unknown distribution" claim where the
   test set sits inside the training support.
3. **Statistical rigor.** ≥3 seeds (5 for headline), IQM + bootstrap CIs; a win requires
   CIs to separate.

## Env levers (verified)

`supplier_disruption_rate`; `demand_shock_probability/_multiplier/_duration/_cluster_size`;
`include_demand_forecast_state`, `demand_forecast_horizon`, `demand_forecast_error`;
patient block (`frail_decay_rate`, `weibull_scale`, `post_shock_multiplier`,
`eligibility_threshold`). Heuristics: `umyo`, `fmyo`, `mdl1/2`, `iso`, `myo`.

## Decisions

- Budget 150k/seed (kill-prone env; `CAMPAIGN_STEPS` overridable), best graph backbone
  `gcn_ddpg` (facility_action) + `flat_ddpg` contrast, run under `caffeinate`, resumable.
- Experiments run sequentially (CPU contention); design docs and configs prepared ahead.
