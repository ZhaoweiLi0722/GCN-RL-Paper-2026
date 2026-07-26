# Regional-Regime Network AFR Results

Date: 2026-07-25

## Primary Question

Can a graph-aware anchored residual policy learn geographically structured
resource-transfer corrections that improve on fixed-prior MDL-2 under an
integrated patient-condition, geographic, transfer-delay, and abrupt regional
demand-shift scenario?

## Teacher Headroom

The 80-trajectory end-to-go teacher cache contains 4,160 states and nine
clinically screened options per state:

- full teacher cost gap versus MDL-2: -2.0689%;
- completion-service difference: +0.015314;
- patients-lost difference: -86.15;
- correction rate: 28.22%.

Transfer-focused options retain most of this opportunity. Restricting the
teacher to reagent transfer, combined transfer, and combined network actions:

- retains 599 correction states (14.40%);
- retains 72.63% of positive dense advantage above the $500,000 gate;
- yields mean selected advantage of $7.04 million when correcting.

The restricted cache is:

`results/network_residual_teacher_abrupt_shift_80traj_network_transfer/teacher_cache.npz`

## Locked Graph Policy

The candidate is Network AFR-GCN-DDPG with:

- MDL-2 as the deterministic anchor;
- GCN node encoder and conservative reagent/capacity edge-flow heads;
- specimen transfer fixed to zero;
- transfer-focused advantage-filtered teacher targets;
- residual deployment scale 0.5;
- grouped gate thresholds `(reagent=0.7, capacity=0.8, replenishment=1.0)`.

The deployment configuration was selected on 30 paired validation episodes and
then locked. Validation cost improved by 1.2646%, completion service increased
by 0.001864, and patients lost decreased by 3.37.

## Three-Seed Holdout

Three independently initialized GCN policies were evaluated on the same 100
previously unseen paired Monte Carlo replications per seed
(`evaluation_seed=6,600,000`).

Per-seed cost gaps versus MDL-2 were:

- seed 0: -1.1516%;
- seed 1: -1.0555%;
- seed 2: -1.1938%.

The two-level bootstrap over training seeds and paired replications gives:

- mean cost gap: -1.1337%;
- mean cost difference: -$25.39 million per 52-week episode;
- 95% CI: [-$29.76 million, -$20.95 million];
- 237 wins in 300 paired episodes;
- completion-service difference: +0.001357;
- completion-service 95% CI: [0.000240, 0.002458];
- patients-lost difference: -0.70;
- patients-lost 95% CI: [-6.42, 5.07].

The cost and completion results pass the pre-registered 1% progression gate.
Patient loss is noninferior in aggregate but is not significantly improved.
Seed 1 increases patients lost by 1.84, so the strict per-seed clinical gate
passes for two of three seeds rather than all three.

Traceable summaries:

- `results/network_afr_regime_transfer_only_holdout/gcn_vs_mdl2_total_cost.json`
- `results/network_afr_regime_transfer_only_holdout/gcn_vs_mdl2_completion.json`
- `results/network_afr_regime_transfer_only_holdout/gcn_vs_mdl2_patients_lost.json`

## Matched Flat Attribution

The matched flat residual DDPG uses the identical states, teacher cache,
training seeds, action targets, deployment scale, grouped thresholds, and
evaluation CRNs. Its actor, critic, and gate contain 528,404 trainable
parameters versus 524,775 for the graph model.

Flat cost gaps versus MDL-2 were -0.0568%, +0.0687%, and -0.0637%. Every
confidence interval crossed zero, completion declined for every seed, and
patients lost increased by 8.05, 11.72, and 7.70.

Direct GCN versus flat two-level comparisons give:

- cost difference: -$25.01 million;
- relative cost gap: -1.1166%;
- cost 95% CI: [-$29.35 million, -$20.61 million];
- patients-lost difference: -9.86;
- patients-lost 95% CI: [-15.24, -4.31];
- completion-service difference: +0.002836;
- completion-service 95% CI: [0.001766, 0.003898].

All three seed-level means favor the graph model. This is the strongest current
evidence that geographic graph representation, rather than model size or
teacher supervision alone, creates deployable transfer value.

Traceable summaries:

- `results/network_afr_regime_transfer_only_holdout/gcn_vs_flat_total_cost.json`
- `results/network_afr_regime_transfer_only_holdout/gcn_vs_flat_completion.json`
- `results/network_afr_regime_transfer_only_holdout/gcn_vs_flat_patients_lost.json`

## Negative Algorithm Ablations

Dense high-value teacher weighting did not improve the original continuous
policy. Its validation cost gap was -0.3225%, completion declined by 0.001132,
and patients lost increased by 8.57.

Five hundred offline replay updates exposed value overestimation:

- GCN-DDPG weakened to a -0.1158% cost gap with a CI crossing zero;
- GCN-TD3 assigned positive anchor advantage to 99.6% of sampled actions;
- ungated GCN-TD3 at deployment scale 0.025 increased cost by 8.11% and
  patients lost by 303.

These runs are rejected. Increasing actor-critic updates without a conservative
offline-RL objective is not an acceptable performance strategy.

## External Robustness Screen

The abrupt-shift checkpoint did not generalize zero-shot:

- gradual regional drift: +0.3798% cost, completion -0.007194, patients lost
  +51.8;
- compound regional stress: -0.1228% cost with CI crossing zero, completion
  -0.002627, patients lost +25.83.

The manuscript must not claim cross-scenario robustness from the current
checkpoint. The positive claim is limited to the integrated abrupt regional
shift for which the policy was trained.

## Decision

The transfer-focused Network AFR-GCN-DDPG passes the primary within-scenario
cost and graph-attribution gates. It significantly outperforms MDL-2 and a
parameter-matched flat residual policy while improving completion service.

The current checkpoint is primarily advantage-filtered teacher distillation;
the tested offline DDPG/TD3 fine-tuning mechanisms were rejected. Before a
strong reinforcement-learning claim or final 5-seed x 500-replication campaign:

1. train a scenario-conditioned or multi-scenario transfer policy using
   regional-drift, abrupt-shift, and compound-stress teacher coverage;
2. add a conservative actor-critic objective such as TD3+BC-style behavior
   regularization or an explicit lower-confidence advantage constraint;
3. require held-out patient-loss noninferiority for every selected seed;
4. rerun pure GCN-DDPG, matched residual TD3, robust MDL-2, forecast-aware
   MDL-2, ISO, MYO, pMYO, SAC, and PPO under the locked final protocol;
5. advance to five training seeds and 500 paired replications only after the
   multi-scenario policy passes the same clinical gates.
