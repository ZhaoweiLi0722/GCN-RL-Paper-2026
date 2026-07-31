# Detailed Research Weekly Update

- Reporting period: July 20-30, 2026
- Audience: GCN-RL project team
- Primary question: Can graph-aware residual reinforcement learning improve
  adaptive capacity and resource-transfer decisions in a patient-condition-aware
  20-clinic regenerative medicine manufacturing network?

## Executive Summary

This week we separated three claims that had previously been discussed
together:

1. whether learning bounded corrections around a strong heuristic is useful;
2. whether the graph representation contributes value beyond a matched flat
   residual policy;
3. whether online actor-critic updates add value beyond graph-policy
   pretraining.

The evidence now supports the first two claims under clearly defined scenarios.
AFR-GCN-DDPG produced a small but statistically stable improvement over MDL-2
under demand-prior drift. The network extension produced a materially larger
and statistically significant advantage over both MDL-2 and a
parameter-matched flat policy under geographically structured regional demand
shifts. A subsequent online TD3 screen also produced a final graph policy that
outperformed MDL-2 and matched flat TD3. However, the incremental effect of
online TD3 updates relative to the frozen pretrained graph policy remains
uncertain at the pooled level.

The current five-scenario zero-shot campaign is designed to resolve that last
attribution question without retraining or changing the completed
checkpoints.

## 1. Environment And Experimental Design Updates

The current environment extends the patient-condition framework introduced in
Howard's pull request with the following network features:

- 20 geographically located clinics;
- distance-based transfer lead times of one to three periods;
- delayed transfers of fungible reagents and idle bioreactor capacity;
- identity-bound patient specimens with specimen pooling and transfer
  prohibited;
- patient deterioration while waiting and during manufacturing;
- therapy discard and bioreactor cleaning following manufacturing-period
  ineligibility;
- regional demand drift, abrupt regime shifts, and compound regional stress;
- paired common-random-number evaluation for operating and clinical outcomes.

These changes align the action space with the graph structure. Earlier
replenishment-only residual policies had limited ability to convert geographic
information into network decisions. The network residual policy can modify
node-level replenishment and edge-level reagent and idle-capacity transfers.

## 2. Proposed Method And Terminology

The canonical proposed method remains AFR-GCN-DDPG.

AFR means advantage-filtered residual and describes the complete controller,
not a new actor-critic algorithm. It combines:

1. an MDL-2 operational anchor;
2. a GCN state encoder for clinic and geographic-edge information;
3. a DDPG actor that produces bounded corrections to the anchor action;
4. advantage-filtered distillation, which retains a nonzero teacher correction
   only when paired look-ahead predicts an improvement over the anchor;
5. feasibility projection and deployment safeguards.

Advantage-filtered distillation is a training-data construction and
regularization step inside AFR. It is not a second deployed policy. TD3 is
retained as a matched stability and backbone comparison. SAC and PPO remain
secondary family baselines rather than the primary research direction.

## 3. Verified Experimental Results

### 3.1 AFR-GCN-DDPG Under Demand-Prior Drift

Protocol:

- 300 training episodes;
- five independent training seeds;
- 100 paired evaluation replications per seed;
- 500 paired outcomes in total;
- integrated patient-condition, geography, and demand-prior drift setting.

AFR-GCN-DDPG versus MDL-2:

- mean cost difference: -$207,485;
- relative cost difference: -0.016360%;
- paired 95% confidence interval: [-$332,641, -$91,367];
- paired wins: 323 of 500;
- completion service difference: +0.000340;
- eligibility difference: +0.000211;
- average waiting-time difference: -0.000612;
- patients-lost difference: -2.32;
- at-risk-unserved difference: -2.55.

The result establishes a statistically stable anchor improvement, but the
absolute economic effect is small.

The matched flat residual DDPG policy also improved on MDL-2:

- relative cost difference: -0.013216%;
- paired 95% confidence interval: [-$286,899, -$46,230].

The GCN policy improved the point estimate by a further 0.003145% relative to
the matched flat policy, but the graph-versus-flat confidence interval
[-$89,702, $14,025] included zero. This replenishment-only experiment supports
residual learning, but does not independently establish a graph advantage.

Traceable result:
[demand-drift robustness results](../../specs/2026-07-20-demand-drift-robustness/results.md).

### 3.2 Matched DDPG And TD3 Comparison

The matched TD3 study used the same 300-episode, five-seed protocol and the
same anchor, graph representation, residual action, teacher targets, and
evaluation streams.

AFR-GCN-TD3 versus MDL-2:

- relative cost difference: -0.001248%;
- paired 95% confidence interval: [-$142,607, $99,437].

AFR-GCN-DDPG versus AFR-GCN-TD3:

- relative cost difference: -0.015112%;
- paired 95% confidence interval: [-$234,659, -$148,401];
- every training-seed mean favored DDPG.

This comparison supports DDPG as the canonical proposed backbone. TD3 remains
useful as a stability ablation, but its conservative twin-critic target appears
to suppress part of the sparse positive correction signal in this setting.

Traceable result:
[RL family comparison](../../specs/2026-07-23-rl-family-comparison/results.md).

### 3.3 Network AFR-GCN-DDPG Under Regional Abrupt Shift

This experiment combined patient-condition dynamics, clinic geography,
distance-based transfer delays, regional demand heterogeneity, and an abrupt
regional regime shift. The action space included graph-edge corrections for
reagent and idle-bioreactor transfers.

Protocol:

- three independently initialized graph policies;
- 100 paired holdout replications per training seed;
- 300 paired outcomes in total;
- parameter-matched flat residual DDPG attribution control.

Network AFR-GCN-DDPG versus MDL-2:

- relative cost difference: -1.1337%;
- mean cost difference: -$25.39 million per 52-week episode;
- paired 95% confidence interval:
  [-$29.76 million, -$20.95 million];
- paired wins: 237 of 300;
- completion-service difference: +0.001357;
- completion-service 95% confidence interval: [0.000240, 0.002458];
- patients-lost difference: -0.70;
- aggregate patient-loss noninferiority passed.

Network AFR-GCN-DDPG versus matched flat residual DDPG:

- relative cost difference: -1.1166%;
- mean cost difference: -$25.01 million;
- paired 95% confidence interval:
  [-$29.35 million, -$20.61 million];
- completion-service difference: +0.002836;
- patients-lost difference: -9.86;
- patients-lost 95% confidence interval: [-15.24, -4.31].

This is the strongest current evidence that graph representation creates
deployable value when the policy directly controls geographically structured
edge actions.

The limitation is important: this checkpoint derived most of its value from
advantage-filtered teacher distillation. Earlier offline DDPG and TD3
fine-tuning attempts were unstable. The result therefore supports a
graph-policy advantage but does not, by itself, prove that online
actor-critic learning generated the improvement.

Traceable result:
[regional network AFR results](../../specs/2026-07-25-regional-regime-network-afr/results.md).

### 3.4 Conservative Online TD3 Screen

The RTX 4090 screen trained matched GCN and flat TD3 residual policies for 100
episodes across seeds 0, 1, and 2.

Each run completed:

- 5,200 update calls;
- 5,073 actual critic updates;
- 2,286 actor updates;
- finite optimization diagnostics;
- nonzero actor drift;
- no Traceback, NaN, OOM, or CPU fallback.

Final GCN-TD3 versus MDL-2:

- seed 0 relative cost difference: -0.051936%;
- seed 1 relative cost difference: -0.139249%;
- seed 2 relative cost difference: -0.119193%;
- pooled mean cost difference: -$2.544 million;
- pooled 95% confidence interval:
  [-$3.872 million, -$1.131 million];
- clinical noninferiority passed for all three seeds.

Matched flat TD3 versus MDL-2:

- seed 0 relative cost difference: +0.049693%;
- seed 1 relative cost difference: +0.040470%;
- seed 2 relative cost difference: +0.097624%;
- clinical noninferiority failed for all three seeds, primarily because of
  increased patient loss.

Final GCN-TD3 versus matched flat TD3:

- relative cost difference: -0.165951%;
- mean cost difference: -$4.083 million;
- pooled 95% confidence interval:
  [-$5.633 million, -$2.436 million];
- completion-service difference: +0.001203;
- patients-lost difference: -7.05.

The final graph controller therefore significantly outperformed MDL-2 and
matched flat TD3 in the regional-drift screen.

The final-versus-frozen comparison was less conclusive:

- pooled GCN final-minus-frozen cost difference: -$0.350 million;
- 95% confidence interval: [-$1.032 million, $0.155 million];
- GCN-minus-flat difference-in-differences: -$0.485 million;
- 95% confidence interval: [-$1.489 million, $0.173 million].

Both point estimates favor online updates, but both intervals include zero.
The graph-aware final controller advantage is established within this screen;
the independent incremental contribution of online TD3 updates is not yet
established.

## 4. Experiment In Progress

The RTX 4090 is running an evaluation-only five-scenario zero-shot campaign.
It reuses the six completed final checkpoints and six frozen-pretrain
checkpoints. It does not retrain policies or perform online updates.

Scenarios:

1. nominal demand history;
2. severe global demand drift;
3. regional demand drift;
4. abrupt regional regime shift;
5. compound regional stress.

Protocol:

- GCN and parameter-matched flat policies;
- training seeds 0, 1, and 2;
- 100 paired holdout replications for every scenario and seed;
- identical common-random-number keys for final and frozen checkpoints;
- 52-week horizon;
- fixed deployment scale and clinical noninferiority criteria.

The campaign will classify online TD3 attribution as:

- Strong;
- Scenario-specific;
- Not established.

## 5. Supported And Unsupported Claims

Supported by current evidence:

- Anchored residual learning can improve on a misspecified MDL-2 policy under
  demand-prior drift.
- DDPG is the stronger canonical backbone in the matched five-seed
  replenishment-residual comparison.
- A graph representation provides significant value when geographically
  structured edge-level transfers are part of the action space.
- The final graph TD3 controller can outperform MDL-2 and matched flat TD3
  while satisfying clinical noninferiority in regional drift.

Not yet supported:

- Reinforcement learning outperforms all heuristics in every scenario.
- All graph-policy improvement is attributable to online actor-critic updates.
- The current checkpoint is robust across unseen demand and disruption
  regimes.
- The three-seed TD3 screen is sufficient for a final five-seed,
  500-replication manuscript claim.

## 6. Proposed Next Steps

If online TD3 attribution is Strong:

1. retain AFR-GCN-DDPG as the proposed controller;
2. retain AFR-GCN-TD3 as the principal backbone ablation;
3. add training seeds 3 and 4;
4. run the final five-seed, 500-paired-replication confirmation.

If attribution is Scenario-specific:

1. identify the scenarios in which online updates add value;
2. train a scenario-conditioned or multi-scenario policy;
3. retain nominal demand as a no-harm case;
4. use regional and compound stress to test adaptive network value.

If attribution is Not established:

1. do not increase the same training budget without changing the learning
   problem;
2. return to the AFR-GCN-DDPG main line;
3. add multi-scenario replay, persistent advantage-filtered regularization,
   and a conservative advantage constraint;
4. frame the current contribution around graph-aware residual control and the
   patient-condition/geography testbed, while reporting the limited online-RL
   increment honestly.

## 7. Decisions Requested From The Team

1. Confirm AFR-GCN-DDPG as the proposed method and TD3 as the principal matched
   stability ablation.
2. Present graph representation, anchored residual control, and online-RL
   attribution as separate empirical claims.
3. Confirm which robustness scenarios belong in the primary manuscript table.
4. Decide whether a non-significant online increment should trigger
   multi-scenario GCN-DDPG fine-tuning or a narrower paper claim.
