# Demand Drift Robustness Plan

Date: 2026-07-20

## Motivation

The current patient-condition + geography benchmark is too favorable to tuned
heuristics when their demand-distribution assumptions match the simulator. This
is not the most realistic PRM setting. The previous DRL-CaP paper framed its
advantage around exactly this issue: conventional capacity-planning methods
depend on demand and disruption distributions that are hard to know in PRM
supply chains, and DRL-CaP became strongest when ground-truth demand forecasts
differed from prior estimates.

This project should therefore evaluate graph-aware residual RL under explicit
forecast/prior misspecification:

- Heuristics plan using prior demand-rate estimates, `demand_rate_estimates`.
- The simulator generates patient arrivals from true rates, `demand_rates`.
- Graph residual RL receives state feedback, patient-risk summaries, geography,
  transfer pipelines, and a heuristic anchor action, and learns corrections
  across randomized demand regimes.

## Literature Read

Tseng et al. (2025), the earlier decentralized PRM DRL paper, is the closest
template. In Case Study II, the heuristic methods generated plans from a
forecasted Poisson demand distribution while the simulator tested shifted
ground-truth Poisson rates, including constant forecast errors and a
non-stationary forecast-error table. The reported advantage of DRL-CaP was
largest when the true demand differed materially from the priori estimate.

Dehaybe, Catanzaro, and Chevalier (2024) study DRL for inventory optimization
with non-stationary uncertain demand. Their key methodological idea is also
useful here: train policies over randomized environments and provide forecast
information in the state so the learned policy can interpolate on previously
unseen rolling-horizon forecasts instead of re-solving a deterministic model at
each period.

Boute et al. (2022) emphasize that DRL for inventory control should be designed
around the structural insights of inventory theory rather than treated as a
black-box replacement. This supports our residual design: the learned actor
does not relearn a full ordering/transshipment policy from scratch; it learns
bounded graph-aware corrections around a strong MDL/patient-aware heuristic.

Kotecha and del Rio Chanona (2025) are especially aligned with our graph story.
Their supply-chain inventory framework combines graph neural networks with
reinforcement learning and parameterizes heuristic inventory policies instead
of directly searching over a huge action space. This is close in spirit to our
GCN residual actor-critic, where a GCN policy uses network state and geography
to modify heuristic decisions.

Schett et al. (2026) integrate probabilistic demand forecasting with DRL reorder
optimization and explicitly discuss that forecast accuracy alone is not enough;
inventory performance depends on how forecast uncertainty is translated into
actions. This supports our choice to evaluate downstream patient/cost outcomes
rather than only demand-forecast error.

## Implemented Design

Code now supports a split between true and prior demand rates:

- `CapacityPlanningConfig.demand_rates`: true Poisson arrival rates used by the
  simulator.
- `CapacityPlanningConfig.demand_rate_estimates`: prior estimates used by
  mean-demand heuristics and residual heuristic anchors.

Training randomization now supports:

- `disruption_range`
- `forecast_error_range`
- `demand_rate_multiplier_range`

The training loop reads these from `config["train_randomization"]`, applies
them only during training, and records the ranges in the training CSV. Formal
evaluation remains fixed to the scenario config.

## New Scenario

`patient_condition_geo_demand_drift` extends the patient-condition geography
scenario with explicit prior/true demand mismatch:

- prior estimates: the original four clinic-type demand rates
  `[7.5, 3.2, 6.0, 3.5]` repeated across the 20 clinics by group.
- true rates: regional/clinic-type drift, with high-demand western and eastern
  clusters above their prior estimates and one lower-demand cluster below prior.
- all patient-condition, transfer lead-time, geography, supplier disruption,
  and demand-shock dynamics remain active.

The follow-up sensitivity scenario `patient_condition_geo_demand_drift_severe`
uses the same operational model but increases the prior/true demand mismatch.
It is intended as a robustness stress test, not as the nominal manuscript case.

## Paper-Safe Claim To Test

Under correctly specified demand distributions, tuned heuristics may remain very
strong. Under realistic demand drift, a graph-aware residual actor-critic should
be able to use observed congestion, patient-risk queues, geography, and transfer
pipelines to correct the heuristic anchor and degrade more gracefully than the
fixed-prior heuristic.

The headline method remains RL:

`GCN residual DDPG = heuristic anchor + bounded graph-aware deterministic policy-gradient residual`

GCN residual TD3 is the configuration-matched stability companion, not a
replacement name for the proposed DDPG method.

The shield selector remains a supporting distillation/safety baseline, not the
main proposed method.

## Immediate Experimental Sequence

1. Smoke test the new `patient_condition_geo_demand_drift` scenario.
2. Run a targeted 100-episode pilot for:
   - `gcn_residual_mdl2_shield_td3`
   - `gcn_residual_mdl2_td3`
   - `gcn_mdl2_shield_selector`
   - `mdl2`
   - `mdl2_shield`
   - `pmyo`
   - `pmyo_shield`
3. Compare:
   - best learned nonzero residual validation gap versus MDL-2
   - service-level and eligibility gaps
   - whether fallback still selects anchor
   - inference/runtime gap versus online shield
4. If nonzero residual improves under demand drift, scale to 300 episodes and
   then multi-seed formal evaluation.

## Revised Sequence After Targeted-100 Evidence

1. Keep `gcn_residual_mdl2_replenish_ddpg` as the canonical proposed method and
   `gcn_residual_mdl2_replenish_td3` as its matched stability companion.
2. Build advantage-filtered shield demonstrations: keep a nonzero residual
   target only when the shield correction beats MDL-2 on paired short rollouts.
3. Use zero residual targets elsewhere and retain the MDL-2 anchor action as an
   actor input.
4. Fine-tune DDPG and TD3 under the same randomized demand/forecast regimes,
   anchor-relative actor objective, and deployment trust region.
5. Re-run mild and severe drift for both backbones with at least five training
   seeds before any manuscript claim of outperformance.
6. Treat the wider transfer/capacity residual variants as ablations unless
   advantage-filtered targets make them stable.

## Sources

- Tseng et al. (2025), "Deep reinforcement learning approach for dynamic
  capacity planning in decentralised regenerative medicine supply chains",
  International Journal of Production Research,
  https://doi.org/10.1080/00207543.2023.2262043
- Dehaybe et al. (2024), "Deep Reinforcement Learning for inventory optimization
  with non-stationary uncertain demand",
  https://doi.org/10.1016/j.ejor.2023.10.007
- Boute et al. (2022), "Deep reinforcement learning for inventory control: A
  roadmap", https://doi.org/10.1016/j.ejor.2021.07.016
- Kotecha and del Rio Chanona (2025), "Leveraging graph neural networks and
  multi-agent reinforcement learning for inventory control in supply chains",
  https://doi.org/10.1016/j.compchemeng.2025.109111
- Schett et al. (2026), "Integrated demand forecasting and reinforcement
  learning for order point optimization in inventory planning",
  https://doi.org/10.1016/j.ijpe.2026.110111
