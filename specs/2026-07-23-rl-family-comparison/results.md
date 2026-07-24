# RL-Family Comparison Results

Date: 2026-07-23

## Objective

Compare the canonical advantage-filtered residual GCN-DDPG policy with:

- a configuration-matched advantage-filtered residual GCN-TD3 policy;
- pure GCN-DDPG, GCN-TD3, GCN-SAC, and GCN-PPO policies trained from scratch;
- the MDL-2 heuristic anchor.

The matched residual DDPG and TD3 arms use the same graph encoder, MDL-2
anchor, replenishment-only residual action space, advantage-filtered teacher
data, persistent distillation regularization, demand randomization,
checkpointing, and deployment fallback. Their RL backbone is the intended
experimental difference.

## Experiment

Scenario:

- `patient_condition_geo_demand_drift`
- 20 clinics with patient deterioration, real geographic locations,
  distance-dependent transfer delays, regional disruption, clustered demand
  shocks, and mismatch between the heuristic demand prior and true demand.

Budget:

- `targeted_100`
- training seeds 0 and 1
- 100 training episodes
- 50 paired Monte Carlo evaluation replications per seed

This is a progression gate, not final manuscript evidence. The final comparison
requires at least five training seeds and paired confidence intervals.

## Aggregate Results

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved | Inference ms |
|---|---:|---:|---:|---:|---:|---:|
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.266296 | 0.43162 | 0.53072 | 3656.87 | 5190.67 | 1.18 |
| `mdl2` | 1.266334 | 0.43146 | 0.53064 | 3657.94 | 5192.11 | 0.19 |
| `gcn_residual_mdl2_replenish_td3_afd` | 1.266379 | 0.43160 | 0.53070 | 3657.00 | 5190.82 | 1.08 |
| `gcn_td3` | 1.644431 | 0.37696 | 0.48707 | 4038.05 | 5627.98 | 0.21 |
| `gcn_pure_ddpg` | 1.981265 | 0.34638 | 0.47203 | 4258.51 | 5892.80 | 0.20 |
| `gcn_ppo` | 3.058631 | 0.29842 | 0.40210 | 4600.18 | 6324.51 | 0.40 |
| `gcn_sac` | 3.303912 | 0.27355 | 0.37564 | 4778.33 | 6574.73 | 0.20 |

## Paired Comparisons

The comparisons use 10,000 two-level bootstrap resamples over training seeds
and paired evaluation replications.

Matched residual GCN-TD3 versus MDL-2:

- mean cost difference: +44,984.58;
- relative gap: +0.003552%;
- 95% confidence interval: [-111,898.66, 242,327.77];
- seed-level differences: seed 0 = +89,969.17, seed 1 = 0;
- 26 wins and 50 ties across 100 paired replications.

Residual GCN-DDPG AFD versus matched residual GCN-TD3 AFD:

- mean DDPG-minus-TD3 cost difference: -83,810.19;
- relative gap: -0.006618%;
- 95% confidence interval: [-225,323.73, 0.00];
- seed-level differences: seed 0 = -167,620.38, seed 1 = 0;
- 33 DDPG wins and 50 ties across 100 paired replications.

## Deployment Diagnostics

For matched residual GCN-TD3:

- seed 0 selected the episode-100 checkpoint and deployed residual scale 3.0;
  its validation cost was 1,259,220,852.96 versus 1,259,232,806.92 for MDL-2;
- seed 1 selected the episode-50 checkpoint and fell back to MDL-2;
  its best nonzero candidate, scale 2.0, had validation cost
  1,236,542,400.51 versus 1,236,123,246.10 for MDL-2.

## Interpretation

1. The anchored residual formulation is the main source of performance at this
   budget. Every pure graph RL policy remains substantially worse than MDL-2.
2. Among pure graph policies, GCN-TD3 is clearly stronger than pure GCN-DDPG,
   GCN-PPO, and GCN-SAC, so TD3 is the most credible from-scratch secondary
   baseline.
3. The matched residual GCN-TD3 arm is stable enough to advance, but it does not
   beat MDL-2 at this gate. Its point estimate is slightly worse and its
   confidence interval crosses zero.
4. The canonical residual GCN-DDPG arm has the best point estimate in this
   two-seed gate. The difference from TD3 is driven by one seed and is not yet
   sufficient for a final backbone claim.

## Progression Decision

- Advance matched residual GCN-TD3 to `300 episodes x 5 seeds`.
- Keep pure GCN-TD3 as an anchor-ablation baseline.
- Do not spend the full budget on GCN-SAC or GCN-PPO until their reward scale,
  action saturation, entropy, and update diagnostics are corrected.
- Preserve residual GCN-DDPG as the canonical proposed method because its
  existing `300 episodes x 5 seeds` result already beats MDL-2 with a paired
  confidence interval below zero.

## Targeted-300 Five-Seed Follow-up

The matched residual GCN-TD3 arm was advanced to:

- 300 training episodes;
- five training seeds;
- six checkpoint candidates per seed;
- 50 fallback-calibration replications;
- 100 independent paired evaluation replications per seed.

All five seeds passed the validation gate and deployed learned residuals. The
selected checkpoints were episodes 50, 250, 100, 50, and 50. The selected
residual scales were 0.75, 1.0, 2.0, 3.0, and 3.0.

| Algorithm | Cost (B) | Service | Eligibility | Waiting time | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|---:|
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.268049 | 0.430321 | 0.530524 | 3.052012 | 3663.27 | 5203.86 |
| `gcn_residual_mdl2_replenish_td3_afd` | 1.268241 | 0.430317 | 0.530523 | 3.052019 | 3663.30 | 5203.88 |
| `mdl2` | 1.268257 | 0.429982 | 0.530313 | 3.052625 | 3665.59 | 5206.41 |

Matched residual GCN-TD3 versus MDL-2:

- mean cost difference: -15,823.34;
- relative gap: -0.001248%;
- 95% two-level bootstrap interval: [-142,607.46, 99,437.30];
- 268 wins across 500 paired replications;
- per-seed differences: +91,498.90, +96,771.41, -14,394.43,
  -67,363.39, and -185,629.19.

TD3 therefore preserves the small patient-facing improvements of the residual
policy, but its independent cost evidence is mixed across seeds and its
confidence interval includes zero. It does not establish outperformance over
MDL-2.

Residual GCN-DDPG AFD versus matched residual GCN-TD3 AFD:

- mean DDPG-minus-TD3 cost difference: -191,662.15;
- relative gap: -0.015112%;
- 95% two-level bootstrap interval: [-234,659.45, -148,400.79];
- 377 DDPG wins across 500 paired replications;
- every training-seed mean favors DDPG.

This matched comparison supports residual GCN-DDPG as the canonical proposed
backbone. TD3 should remain a stability-oriented family ablation: it deploys
nonzero corrections reliably and improves patient metrics over MDL-2, but its
twin-critic conservative target suppresses part of the sparse positive
correction signal in this problem.

## Final Family Decision

1. Keep advantage-filtered residual GCN-DDPG as the proposed method.
2. Report matched residual GCN-TD3 as the principal RL-backbone ablation.
3. Report pure GCN-DDPG, GCN-TD3, GCN-SAC, and GCN-PPO as from-scratch
   secondary baselines at a common screened budget; do not imply that SAC or
   PPO use the matched residual formulation.
4. Spend the next full experiment budget on graph-specific ablations and
   robustness regimes, because the current graph-versus-flat confidence
   interval still crosses zero and is the more important unresolved claim.
