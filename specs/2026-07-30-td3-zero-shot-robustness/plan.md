# TD3 zero-shot robustness plan

## Decision context

The 100-episode conservative screen established that the final graph TD3
checkpoint outperforms both MDL-2 and the matched flat TD3 policy on the
regional-drift holdout. The frozen graph checkpoint also outperforms both
comparators, while the pooled final-versus-frozen confidence interval crosses
zero. The next experiment must therefore separate graph-aware AFR
initialization from the contribution of online TD3 updates.

## Fixed policies

- GCN residual TD3, seeds 0-2, episode-100 checkpoints.
- Matched flat residual TD3, seeds 0-2, episode-100 checkpoints.
- The corresponding frozen-pretrain checkpoints.
- MDL-2 anchors generated within each paired evaluation.

No policy is retrained in this phase. The deployment scale, projection,
clinical margins, horizon, and action masks remain fixed.

## Zero-shot scenarios

1. `patient_condition_geo_nominal_history`
2. `patient_condition_geo_demand_drift_severe_history`
3. `patient_condition_geo_regional_drift`
4. `patient_condition_geo_abrupt_regime_shift`
5. `patient_condition_geo_compound_regional_stress`

The holdout uses 100 replications per scenario and training seed with a new
common-random-number stream (`40900000`). Final and frozen-pretrain evaluations
must have identical scenario, training-seed, evaluation-seed, and replication
keys.

## Required comparisons

- Final GCN TD3 versus MDL-2, pooled and by scenario.
- Final flat TD3 versus MDL-2, pooled and by scenario.
- Final GCN TD3 versus matched flat TD3, pooled and by scenario.
- Final versus frozen pretrain for both representations.
- Difference-in-differences:
  `(GCN final - GCN frozen) - (flat final - flat frozen)`.
- Completion service, patients lost, and manufacturing-period ineligibility
  under the pre-registered clinical noninferiority margins.
- Residual usage and inference cost by method, seed, and scenario.

## Advance gates

The graph-aware controller advances only if:

1. Final GCN TD3 has a pooled cost confidence interval below zero versus MDL-2.
2. Final GCN TD3 is clinically noninferior in every scenario.
3. Final GCN TD3 has a pooled cost confidence interval below zero versus the
   matched flat TD3 policy.
4. At least four of five scenario-level GCN-versus-flat mean cost differences
   favor GCN; no adverse scenario may violate clinical noninferiority.
5. Online TD3 attribution is classified explicitly:
   - **Strong:** pooled final-versus-frozen GCN CI is below zero.
   - **Scenario-specific:** at least three scenario CIs are below zero and none
     shows clinically inferior final performance.
   - **Not established:** neither condition holds.

## Follow-on decision

- If online TD3 attribution is strong or scenario-specific, reproduce the
  distillation and TD3 pipeline for seeds 3 and 4, then run a five-seed
  confirmation with untouched CRN streams.
- If graph and anchor gates pass but online attribution is not established,
  run a GCN-only 300-episode development screen before paying for a matched flat
  campaign. Use an independent development stream and retain the frozen
  checkpoint control.
- If graph or clinical gates fail, train a multiscenario policy spanning
  regional drift, abrupt shift, and compound stress before increasing the
  episode budget.

Do not update the manuscript until this zero-shot phase has been audited.
