# Demand Drift Robustness Results

Date: 2026-07-20

## Targeted-100 Pilot

Scenario:

- `patient_condition_geo_demand_drift`
- 20 clinics with patient-condition deterioration, geography, transfer lead
  time, regional supplier disruption, clustered demand shocks, and explicit
  mismatch between true demand rates and prior demand-rate estimates.

Budget:

- `targeted_100`
- seeds 0 and 1
- 100 training episodes
- 50 Monte Carlo evaluation replications per seed

Algorithms evaluated:

- `gcn_residual_mdl2_shield_td3`
- `gcn_residual_mdl2_td3`
- `gcn_residual_mdl2_replenish_td3`
- `mdl2`
- `mdl2_shield`
- `pmyo`
- `pmyo_shield`

## Aggregate Result

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved | Inference ms |
|---|---:|---:|---:|---:|---:|---:|
| `mdl2_shield` | 1.266013 | 0.43185 | 0.53090 | 3655.2 | 5189.4 | 196.97 |
| `gcn_residual_mdl2_shield_td3` | 1.266334 | 0.43146 | 0.53064 | 3657.9 | 5192.1 | 0.19 |
| `gcn_residual_mdl2_td3` | 1.266334 | 0.43146 | 0.53064 | 3657.9 | 5192.1 | 0.19 |
| `mdl2` | 1.266334 | 0.43146 | 0.53064 | 3657.9 | 5192.1 | 0.19 |
| `gcn_residual_mdl2_replenish_td3` | 1.266373 | 0.43160 | 0.53070 | 3657.0 | 5190.8 | 1.03 |
| `pmyo_shield` | 1.362150 | 0.41497 | 0.52707 | 3744.8 | 5397.1 | 202.71 |
| `pmyo` | 1.364144 | 0.40962 | 0.52122 | 3781.9 | 5442.6 | 0.22 |

## Interpretation

The original MDL-2 residual TD3 variants still fell back to the MDL-2 anchor.
Their nonzero residuals moved too much transfer/capacity action and degraded
service, which caused the validation gate to reject them.

The narrower `gcn_residual_mdl2_replenish_td3` variant is more promising:

- It is still a GCN-TD3 residual RL policy, not a selector.
- It keeps MDL-2 as the anchor but restricts the learned correction to positive
  replenishment only.
- It trains under demand-rate multiplier randomization and forecast-error
  randomization.
- It is the first variant where a nonzero learned residual passed the
  validation fallback gate.

Per-seed fallback diagnostics:

- Seed 0 selected the learned residual at residual scale 1.0.
  - validation cost gap versus MDL-2: -0.00107%
  - validation service gap: +0.00028
  - all nonzero residual scales passed the gate.
- Seed 1 still fell back to MDL-2.
  - best nonzero residual service gap: +0.00018
  - best nonzero residual cost gap: +0.03492%

So the direction is correct but not yet strong enough: replenishment-only
residuals improve patient-facing metrics and can pass validation on one seed,
but aggregate cost is still slightly worse than MDL-2 and still behind
`mdl2_shield`.

## Paper Implication

This supports the revised paper narrative:

1. Tuned heuristics remain strong under nominal conditions.
2. Under demand-prior drift, graph-aware residual RL begins to find deployable
   corrections when the action space is constrained to clinically/plausibly safe
   corrections.
3. Online shield policies are slightly stronger but roughly 1000x slower than
   heuristic / residual policy inference.
4. The next proposed-method direction should be GCN residual TD3 with a
   structure-preserving residual action space and service/eligibility-aware
   actor objective.

## Next Technical Step

Do not broaden the residual action space yet. The next improvement should add a
patient-facing actor objective or advantage term so the critic/actor explicitly
rewards service and eligibility gains, not only total cost. The immediate target
is to make `gcn_residual_mdl2_replenish_td3` pass fallback on both seeds while
keeping the aggregate cost no worse than MDL-2.

## Service-Proxy Actor Loss Follow-up

After the first targeted-100 pilot, we added a config-driven
`patient_service_proxy_actor_loss` to `GCNTD3Agent` and enabled it only for
`gcn_residual_mdl2_replenish_td3`. The loss rewards replenishment residuals that
align with high resource/patient pressure and penalizes low-pressure
replenishment. This keeps the method in the GCN-TD3 residual RL family while
adding a structure-preserving patient-service prior.

Follow-up targeted-100 result:

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved | Inference ms |
|---|---:|---:|---:|---:|---:|---:|
| `mdl2_shield` | 1.266013 | 0.43185 | 0.53090 | 3655.2 | 5189.4 | 196.97 |
| `mdl2` | 1.266334 | 0.43146 | 0.53064 | 3657.9 | 5192.1 | 0.19 |
| `gcn_residual_mdl2_replenish_td3` | 1.266373 | 0.43160 | 0.53070 | 3657.0 | 5190.8 | 1.28 |

Fallback diagnostics:

- Seed 0 still selected the learned residual at residual scale 1.0.
  - best nonzero cost gap versus MDL-2: -0.00092%
  - best nonzero service gap: +0.00028
- Seed 1 still fell back to MDL-2.
  - best nonzero cost gap improved from +0.03492% to +0.03426%
  - best nonzero service gap remained positive at +0.00018

Interpretation: the service-proxy loss did not yet make the residual pass both
seeds, but it preserved the useful direction: service and eligibility improve,
patient loss drops slightly, and one seed deploys the learned residual. The
remaining bottleneck is a small cost penalty in seed 1. Next, the actor objective
should become cost/service balanced rather than purely service-aligned; otherwise
the learned residual will keep improving patient-facing metrics while missing
the cost gate by a small margin.

## Cost-Aware Objective and Severe-Drift Follow-up

We next added two configurable safeguards to `GCNTD3Agent`:

- a negative anchor-advantage penalty, which discourages residual actions that
  the twin critics rank below the MDL-2 anchor;
- a replenishment-cost proxy, which counterbalances the patient-service pressure
  term and discourages broad positive replenishment.

The mild-drift targeted-100 result was essentially unchanged. Seed 0 continued
to deploy a nonzero replenishment residual, while seed 1 continued to fall back
to MDL-2. This showed that adding another actor-side proxy was not sufficient to
solve the cross-seed instability.

We therefore added the traceable sensitivity scenario
`patient_condition_geo_demand_drift_severe`. It retains the same 20-clinic
patient-condition, geography, transfer-delay, regional disruption, and clustered
demand-shock model, but increases the mismatch between prior and true Poisson
demand rates. Prior rates remain `[7.5, 3.2, 6.0, 3.5]` by clinic group, while
true rates become `[13.5, 4.8, 10.2, 5.25]`.

Targeted-100 severe-drift aggregate:

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved | Inference ms |
|---|---:|---:|---:|---:|---:|---:|
| `gcn_residual_mdl2_replenish_td3` | 1.915119 | 0.34695 | 0.44904 | 5756.3 | 8129.4 | 1.29 |
| `gcn_residual_mdl2_shield_td3` | 1.915084 | 0.34691 | 0.44901 | 5756.6 | 8129.8 | 0.20 |
| `gcn_residual_mdl2_td3` | 1.915084 | 0.34691 | 0.44901 | 5756.6 | 8129.8 | 0.22 |
| `mdl2` | 1.915084 | 0.34691 | 0.44901 | 5756.6 | 8129.8 | 0.20 |

The deployment trust-region sweep was extended from a maximum residual scale of
1.0 to 3.0. For `gcn_residual_mdl2_replenish_td3`:

- seed 0 selected the learned residual at scale 3.0;
  - validation cost gap: -0.00660%;
  - validation service gap: +0.000139;
- seed 1 still selected the MDL-2 anchor;
  - best nonzero scale: 0.05;
  - validation cost gap: +0.00386%;
  - validation service gap: +0.000139.

The wider multi-resource variants were not competitive at 100 episodes:

- `gcn_residual_mdl2_td3` fell back on both seeds; its best nonzero residuals
  raised validation cost by 0.33%-0.37% and reduced service by about 0.007.
- `gcn_residual_mdl2_shield_td3` also fell back on both seeds; direct imitation
  of the wider shield action was difficult and its nonzero residuals reduced
  service even more.

## Canonical DDPG Comparison

To preserve continuity with the earlier PRM paper and remove an inconsistency in
the manuscript, we implemented a configuration-matched DDPG counterpart:
`gcn_residual_mdl2_replenish_ddpg`. It uses the same graph encoder, MDL-2
anchor, replenishment-only residual space, imitation teacher, demand
randomization, patient-service proxy, and fallback calibration as the TD3 arm.
The only substantive backbone difference is standard single-critic DDPG versus
TD3's twin critics, delayed actor updates, and target smoothing.

Mild-drift targeted-100 aggregate:

| Algorithm | Cost (B) | Cost gap vs MDL-2 | Service | Eligibility | Patients lost | At-risk unserved | Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mdl2` | 1.266334 | 0.0000% | 0.43146 | 0.53064 | 3657.94 | 5192.11 | 0.19 |
| `gcn_residual_mdl2_replenish_ddpg` | 1.266374 | +0.0031% | 0.43160 | 0.53070 | 3657.00 | 5190.82 | 1.15 |
| `gcn_residual_mdl2_replenish_td3` | 1.266375 | +0.0032% | 0.43160 | 0.53070 | 3657.00 | 5190.82 | 1.25 |

DDPG selected a nonzero residual on seed 0 at scale 3.0 and fell back to MDL-2
on seed 1. TD3 displayed the same one-seed-deploys, one-seed-falls-back pattern.
The two backbones therefore have indistinguishable patient-facing performance at
this budget; DDPG is marginally cheaper, but neither is yet cheaper than MDL-2
on the independent aggregate evaluation.

This evidence supports naming residual GCN-DDPG as the canonical proposed
instantiation and treating GCN-TD3 as its stability-enhanced variant. It does not
support a claim of statistically stable heuristic outperformance yet.

## Current Method Direction

The evidence now favors a narrower proposed method:

`MDL-2 anchor + graph-aware replenishment residual DDPG + demand randomization + trust-region fallback`

GCN-TD3 remains the primary stability-oriented variant and should be reported
beside GCN-DDPG under matched settings.

The current learned policy has a real patient-facing signal: in both mild and
severe drift it improves service/eligibility and reduces patient loss slightly,
and one seed deploys a nonzero residual. It does not yet beat MDL-2 on aggregate
cost across both seeds, so the paper must not claim stable outperformance yet.

The next algorithmic step is advantage-filtered teacher distillation. Training
data should retain only states where a shield correction is measurably better
than MDL-2; all other states should use a zero residual target. TD3 fine-tuning
should then operate inside the replenishment-only action subspace. This keeps
the contribution centered on graph-aware RL while avoiding the unstable broad
transfer corrections observed in the severe-drift pilot.

## Advantage-Filtered Residual GCN-DDPG

We implemented the next step as
`gcn_residual_mdl2_replenish_ddpg_afd`. The method remains a GCN-DDPG policy:
MDL-2 supplies the anchor action, while the graph actor learns a continuous,
replenishment-only residual. The additional advantage-filtered distillation
(AFD) stage uses common-random-number lookahead to compare candidate residuals
with the MDL-2 anchor at the same state. It retains a nonzero correction target
only when the candidate improves the configured cost/patient score. All other
states receive a zero-residual target.

To prevent the useful but sparse correction targets from being overwhelmed by
anchor labels, positive-correction and zero-residual classes receive equal total
weight. The resulting weighted demonstrations are retained as an actor
regularizer throughout DDPG fine-tuning rather than being used only for a
one-time warm start. The deployment calibration still selects a residual scale
or falls back to MDL-2.

### Mild demand drift

At `targeted_100`, the AFD teachers found 67 and 61 improving states out of 416
states for training seeds 0 and 1, respectively. Both datasets assigned 50% of
their total weight to improving corrections. Seed 0 deployed the learned policy
at residual scale 3.0; seed 1 fell back to MDL-2.

| Algorithm | Cost (B) | Cost gap vs MDL-2 | Service | Eligibility | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|---:|
| `mdl2` | 1.266334 | 0.0000% | 0.431465 | 0.530638 | 3657.94 | 5192.11 |
| `gcn_residual_mdl2_replenish_ddpg` | 1.266374 | +0.003114% | 0.431597 | 0.530702 | 3657.00 | 5190.82 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.266296 | -0.003066% | 0.431617 | 0.530715 | 3656.87 | 5190.67 |

This is the first aggregate result in which the proposed GCN-DDPG arm has lower
mean cost than MDL-2 while also improving all reported patient-facing metrics.
However, the paired mean cost difference was -38,825.61 over 100 replications,
with a 95% paired bootstrap interval of [-192,365.47, 105,859.40]. The interval
crosses zero, and one training seed still used fallback. This is promising pilot
evidence, not yet statistically stable heuristic outperformance.

### Severe demand drift

The stronger mismatch produced denser teacher signal: seeds 0 and 1 yielded 90
and 65 improving states out of 416, with mean lookahead improvements of 51,305
and 56,306. Both training seeds selected episode 100 and deployed the learned
GCN-DDPG residual at scale 3.0. Their calibration cost gaps were -0.01387% and
-0.000323%, respectively; this removes the previous one-seed-fallback pattern
on the severe scenario.

Independent evaluation was more conservative:

| Algorithm | Cost (B) | Cost gap vs MDL-2 | Service | Eligibility | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|---:|
| `mdl2` | 1.915083961 | 0.000000% | 0.346914 | 0.449007 | 5756.62 | 8129.82 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.915084409 | +0.000023% | 0.346955 | 0.449050 | 5756.20 | 8129.42 |

The pooled paired mean cost difference was +448.27 across 100 replications,
with a 95% paired bootstrap interval of [-115,602.49, 136,333.61]. The learned
policy won 58 of 100 paired replications and improved mean service by 0.0000416,
but its mean cost was effectively tied with MDL-2. The validation gains therefore
did not fully generalize to the independent evaluation seeds.

## Updated Decision

AFD materially improves the proposed method: it changes the severe-drift result
from one learned deployment plus one fallback to learned deployment for both
training seeds, and it produces the first lower-cost aggregate signal under mild
drift. It still does not justify the manuscript claim that GCN-DDPG reliably
outperforms MDL-2. The next experiment should test five training seeds at 100
episodes before increasing the training horizon. If the deployment rate and
paired effect remain favorable, the surviving configurations should advance to
300 episodes and formal multi-seed evaluation. If instability remains, the next
model change should improve out-of-sample calibration and teacher diversity,
not expand the residual action space.

## Five-Seed Targeted-100 Stability Check

We expanded the AFD comparison to five training seeds while retaining 50 paired
Monte Carlo replications per seed. Confidence intervals below use a paired
two-level bootstrap: training seeds are resampled first, followed by
replications within each selected seed. This avoids treating repeated evaluation
trajectories from the same trained policy as independent training outcomes.

### Mild demand drift

Three of five seeds deployed the learned residual at scale 3.0; the other two
used the MDL-2 fallback. The learned deployments were seeds 0, 3, and 4.

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|
| `mdl2` | 1.261660 | 0.430691 | 0.530362 | 3651.47 | 5181.93 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.261559 | 0.430837 | 0.530451 | 3650.49 | 5180.88 |

The mean paired cost difference was -100,624.13, or -0.007976%. The 95%
two-level bootstrap interval was [-260,658.74, 18,842.24]. The candidate won 88
of 250 paired replications, tied 100 because of fallback, and lost 62. The mean
effect and every patient-facing metric favor AFD, but the interval still crosses
zero slightly.

### Severe demand drift

Four of five seeds deployed the learned residual at scale 3.0. Seed 3 fell back
because its nonzero candidate reduced service on validation, even though its
validation cost was marginally lower than the anchor.

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|
| `mdl2` | 1.914292 | 0.346939 | 0.449184 | 5753.47 | 8121.90 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.914232 | 0.346984 | 0.449226 | 5753.05 | 8121.36 |

The mean paired cost difference was -60,544.91, or -0.003163%. The 95%
two-level bootstrap interval was [-165,288.45, 32,762.33]. The candidate won 123
of 250 paired replications, tied 50, and lost 77. Severe drift increased learned
deployment from 3/5 to 4/5, consistent with greater correction opportunity when
the MDL-2 demand prior is more misspecified.

The reproducible paired summaries are stored in
`mild_five_seed_paired.json` and `severe_five_seed_paired.json` beside this note.

## Three-Seed Targeted-300 Gate

Both five-seed means favor AFD and most seeds deploy the learned policy, so the
method passes the progression gate for a longer diagnostic run. It does not yet
pass the statistical-claim gate because both two-level intervals include zero.
The next run uses mild drift and training seeds 0, 1, and 2 for 300 episodes.
This set includes one previously learned deployment and two fallbacks, directly
testing whether additional training converts the weak seeds. Checkpoints are
saved every 50 episodes, checkpoint validation is increased to 20 replications,
fallback validation to 50 replications, and residual scales remain available up
to 3.0. A 100-replication independent evaluation follows selection.

## Three-Seed Targeted-300 Result

The targeted-300 mild-drift run completed for seeds 0, 1, and 2. All three
deployed the learned residual at scale 3.0. Seed 0 selected episode 300, whereas
seeds 1 and 2 selected episode 50. Longer interaction therefore produced a new
best checkpoint for seed 0, but seeds 1 and 2 benefited primarily from the more
reliable checkpoint/fallback validation rather than from late training.

| Algorithm | Cost (B) | Service | Eligibility | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|
| `mdl2` | 1.266548 | 0.430731 | 0.530479 | 3658.97 | 5194.97 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.266418 | 0.431029 | 0.530653 | 3656.91 | 5192.95 |

All three independent per-seed mean cost differences favored AFD:

| Seed | Selected checkpoint | Mean paired cost difference | Cost gap |
|---:|---:|---:|---:|
| 0 | 300 | -86,944.51 | -0.006926% |
| 1 | 50 | -146,821.10 | -0.011517% |
| 2 | 50 | -154,954.76 | -0.012207% |

Across 300 paired replications, the mean difference was -129,573.46, or
-0.010230%. The paired two-level 95% bootstrap interval was
[-238,744.59, -24,526.53], fully below zero. The proposed policy won 183 of 300
paired replications and had no fallback-induced ties. This is the first result
that supports a statistically stable lower-cost signal while preserving the
patient-facing improvements. Because the inference includes only three training
seeds, the next gate is to extend the same frozen configuration to seeds 3 and
4 before making the final manuscript claim.

## Five-Seed Targeted-300 Confirmation

The frozen targeted-300 configuration was extended to seeds 3 and 4. All five
training seeds deployed the learned residual at scale 3.0. Selected checkpoints
were episodes 300, 50, 50, 50, and 150 for seeds 0--4, respectively. Additional
training created a later best checkpoint for seeds 0 and 4; the other three
seeds retained early checkpoints, confirming that training quality is not
monotone in episode count and that checkpoint selection remains necessary.

| Algorithm | Cost (B) | Service | Eligibility | Waiting time | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|---:|
| `mdl2` | 1.268257 | 0.429982 | 0.530313 | 3.052625 | 3665.59 | 5206.41 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.268049 | 0.430321 | 0.530524 | 3.052012 | 3663.27 | 5203.86 |

Every independent per-seed mean cost difference favored AFD:

| Seed | Selected checkpoint | Mean paired cost difference | Cost gap | Paired wins |
|---:|---:|---:|---:|---:|
| 0 | 300 | -86,944.51 | -0.006926% | 61/100 |
| 1 | 50 | -146,821.10 | -0.011517% | 59/100 |
| 2 | 50 | -154,954.76 | -0.012207% | 63/100 |
| 3 | 50 | -267,271.04 | -0.020973% | 74/100 |
| 4 | 150 | -381,436.04 | -0.030098% | 66/100 |

Across 500 paired replications, the proposed method reduced mean total cost by
207,485.49, or 0.016360% relative to MDL-2. The paired two-level 95% bootstrap
interval was [-332,641.11, -91,367.07], fully below zero. It won 323 of 500
paired replications and produced no fallback ties. Mean service increased by
0.000340, eligibility by 0.000211, average waiting time fell by 0.000612,
patients lost fell by 2.32, and at-risk unserved patients fell by 2.55.

This is the first five-training-seed result that supports the claim that the
proposed advantage-filtered residual GCN--DDPG policy outperforms its MDL-2
anchor under demand-prior drift while improving patient-facing metrics. The
absolute cost effect is small, so the manuscript should report both the
percentage effect and the paired interval rather than describing the advantage
as operationally large. The reproducible statistical output is stored in
`mild_five_seed_targeted_300_paired.json` beside this note.

## Matched Flat-State Ablation

To isolate the graph representation from the residual-learning recipe, a
matched flat-state residual DDPG ablation was trained for the same 300 episodes
and five seeds. It used the same MDL-2 anchor, anchor-action features, pressure
projection, advantage-filtered labels, persistent weighted regularization,
domain randomization, auxiliary actor losses, checkpoint candidates,
validation replications, residual-scale candidates, and 100 formal evaluation
replications per seed. The encoder was the intended difference: the proposed
method used the full graph, whereas the ablation used a flat MLP state encoder.

All five flat policies passed validation and deployed learned corrections.
Their selected checkpoints were episodes 300, 50, 50, 50, and 50, with
residual scales 3.0, 3.0, 3.0, 2.0, and 3.0, respectively.

| Algorithm | Cost (B) | Service | Eligibility | Waiting time | Patients lost | At-risk unserved |
|---|---:|---:|---:|---:|---:|---:|
| `mdl2` | 1.268257 | 0.429982 | 0.530313 | 3.052625 | 3665.59 | 5206.41 |
| `flat_residual_mdl2_replenish_ddpg_afd` | 1.268089 | 0.430320 | 0.530523 | 3.052012 | 3663.28 | 5203.86 |
| `gcn_residual_mdl2_replenish_ddpg_afd` | 1.268049 | 0.430321 | 0.530524 | 3.052012 | 3663.27 | 5203.86 |

The matched flat policy also improved on MDL-2. Its mean paired cost difference
was -167,609.55, or -0.013216%, with a paired two-level 95% bootstrap interval
of [-286,899.37, -46,230.03]. It won 319 of 500 paired replications. Four of
five per-seed mean differences favored flat residual DDPG; seed 0 was
2,115.46 higher than MDL-2.

The GCN point estimate remained best, reducing mean cost by a further 39,875.94
relative to matched flat residual DDPG, or 0.003145%. It won 285 of 500 direct
paired comparisons. However, the paired two-level 95% bootstrap interval was
[-89,701.54, 14,024.91], which includes zero. Per-seed GCN-minus-flat mean cost
differences were -89,059.97, 48,635.33, 1,830.68, -103,815.48, and -56,970.25.

The defensible conclusion is therefore narrower than the anchor comparison.
The advantage-filtered residual RL design has a statistically stable advantage
over the misspecified MDL-2 anchor in this scenario. The graph encoder has the
best mean result and a positive directional signal, but the current five-seed
campaign does not establish a statistically significant graph-specific gain.
The exact comparisons are stored in
`mild_five_seed_targeted_300_flat_vs_mdl2_paired.json` and
`mild_five_seed_targeted_300_gcn_vs_flat_paired.json`.
