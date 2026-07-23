# Referee-proof "when does DRL win" experiment design

The prior DRL paper's Case II showed DRL beating heuristics under forecast error, but a
referee can press three weaknesses. We adopt them as **design invariants** for every
"DRL wins under stress" experiment here, and exploit env levers the prior paper lacked.

## Design invariants

- **I1 — Fair baseline (no strawman).** Always include the strongest *legitimate*
  heuristic, and give heuristics the same information the DRL has. Concretely: include
  `umyo` (urgency/condition-aware: surges capacity + replenishment toward at-risk /
  near-expiry clinics) and `fmyo` (forecast-aware); give heuristics the **true**
  `demand_rates` and the **same** `demand_forecast` signal the DRL sees. Never hand the
  heuristic a fixed wrong point forecast it can't update. The real claim is *DRL beats
  the condition/forecast-aware heuristic*, not just the blind ones.
- **I2 — True OOD, not interpolation.** For any robustness/generalization claim, train
  the DRL policy on a **range** of the stressed parameter and **test outside that range**;
  report in-distribution vs OOD columns separately. The prior paper trained on
  `Uniform(d,2d)` and tested inside it — that is interpolation, not generalization.
- **I3 — Statistical rigor.** Multi-seed IQM + bootstrap CIs (already built). Claim a win
  only when CIs separate, not on point estimates. (Fixes the prior single-policy,
  mean-only reporting the same authors cite Henderson/Agarwal against.)

## Available levers (verified in code)

- Heuristics: `umyo` (condition-aware), `fmyo` (forecast-aware), `mdl1/mdl2/iso/myo`
  (condition-blind); all use the **true** `demand_rates` (fair) and `env.demand_forecast`.
- Env config: `supplier_disruption_rate`; `demand_shock_probability/_multiplier/_duration/
  _cluster_size` (non-stationary surges); `include_demand_forecast_state`,
  `demand_forecast_horizon`, `demand_forecast_error` (a shared forecast signal with
  controllable error — the fair way to do the prior paper's Case II); patient block
  (`frail_decay_rate`, `weibull_scale`, `post_shock_multiplier`, `eligibility_threshold`).

## Experiments

### A — Disruption severity
- **A1 (running, per-regime).** Train DRL at each disruption rate; compare vs heuristics
  at that rate (0.05 / 0.3-from-campaign / 0.6). Fair: heuristics have true demand;
  disruption is a *structural* challenge both face at eval. Answers "does DRL's advantage
  grow with disruption?" **Fix pending:** add `umyo`/`fmyo` to its baseline (cheap,
  no-training runs; fill after A1 finishes).
- **A2 (robustness, the stronger claim).** Train **one** DRL policy with disruption
  randomized per episode over [0, 0.4]; test at held-out rates incl. **OOD** 0.5, 0.6.
  One robust policy vs per-regime heuristics → genuine generalization (I2). Needs a small
  per-episode disruption-randomization hook.

### B — Patient-condition value (running after A)
- Stress config `20_clinic_patient_condition_stress.json` (disruption held at 0.3; only
  patient dynamics hardened) isolates the condition effect. Baseline now includes `umyo`
  + `fmyo` (I1). **Real claim: DRL beats `umyo`** — learning the condition response beats
  a hand-crafted urgency rule. Beating only blind heuristics is the weak version.
- **B2 (optional, OOD over severity).** Train on a range of condition severities, test at
  held-out severe settings (I2).

### C — Forecast error, redeemed (flagship robustness experiment)
Direct fix of the prior paper's Case II. Turn on `include_demand_forecast_state` with
`demand_forecast_error` so **both** heuristics and DRL consume the **same** forecast
signal (I1 — no fixed-wrong-point handicap). Train DRL on errors in [0, e_train]; test at
errors **> e_train** (I2, OOD) and under `demand_shock` non-stationary schedules. Fair
baseline: `fmyo`. Claim: DRL degrades more gracefully than the forecast-aware heuristic,
especially OOD. IQM/CI throughout (I3).

### D — Non-stationarity
`demand_shock_*` with clustering; train on moderate shocks, test on severe/clustered
(OOD). Extends the prior paper's non-stationary case with a fair forecast-aware baseline.

## The narrative these buy (honest, referee-proof)

Because the campaign already shows heuristics beating DRL **fairly** at nominal (strong,
non-crippled baseline), a DRL win in A2/B/C/D under distribution shift is credible in a way
the prior paper's fixed-wrong-forecast comparison was not: *"Tuned heuristics win under
nominal, well-specified conditions; graph-DRL's advantage emerges specifically as supply
disruption rises, patient deterioration sharpens, and demand forecasts drift — including
regimes the policy never trained on."*
