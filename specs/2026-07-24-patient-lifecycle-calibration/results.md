# Patient-lifecycle clinical calibration

Date: 2026-07-24

## Decision

The exploratory patient-condition parameters produced a 77%-83% probability
that a patient who started manufacturing became ineligible before completion.
This is not suitable for the nominal experiment. It is retained only as an
extreme-stress diagnostic.

The metric is now reported as
`patient_ineligibility_during_manufacturing_rate`. The legacy
`manufacturing_loss_rate` field remains as a compatibility alias. This patient
outcome is distinct from a technical process or product-release manufacturing
failure.

## Calibration protocol

The reproducible calibration plan is:

`experiments/configs/patient_lifecycle_calibration.json`

The runner is:

`python -m evaluation.calibrate_patient_lifecycle`

The screen used:

- the integrated `patient_condition_geo_demand_drift` environment;
- ISO and MDL-2, with common evaluation seeds;
- 100 Monte Carlo replications per scale-policy pair;
- decay scales `0.10`, `0.12`, `0.13`, `0.14`, and `0.15`;
- a pre-specified 10%-15% manufacturing-period patient-ineligibility target.

The reference rates were `0.01` healthy decay, `0.12` frail decay, and `0.01`
waiting-time decay. Only these three rates were scaled; the risk mixture,
Weibull shock distribution, post-shock acceleration, and eligibility threshold
were held fixed.

## Calibration screen

| Decay scale | ISO ineligibility | MDL-2 ineligibility | Both in target |
|---:|---:|---:|:---:|
| 0.10 | 0.0406 | 0.0272 | No |
| 0.12 | 0.1073 | 0.0837 | No |
| 0.13 | 0.1466 | 0.1223 | Yes |
| 0.14 | 0.1849 | 0.1648 | No |
| 0.15 | 0.2203 | 0.2082 | No |

The selected scale is `0.13`, giving nominal rates:

- `healthy_decay_rate = 0.0013`;
- `frail_decay_rate = 0.0156`; and
- `waiting_time_decay_rate = 0.0013`.

The complete screen is stored at:

`results/patient_lifecycle_calibration/summary.csv`

## Objective calibration

The exploratory objective used a one-time patient-loss penalty of 50,000,
whereas one patient remaining short of both reagent and idle-bioreactor
capacity could incur 136,778.1 in operating penalties in one epoch. This
directionally rewarded early patient exit in some comparisons.

The nominal integrated scenarios now use:

| Component | Legacy | Calibrated |
|---|---:|---:|
| Patient loss | 50,000 | 500,000 |
| Material/therapy waste | 40,000 | 100,000 |
| At-risk unserved patient per epoch | 5,000 | 25,000 |

These are policy-objective weights, not claimed monetary valuations of a human
life. Their role is to preserve the declared ordering in which patient loss
dominates avoidable operating and material cost.

## Paired heuristic pilot

After writing the selected risk parameters and cost weights into the nominal
scenario, ISO and MDL-2 were rerun for 100 paired replications with seed 95000.

| Metric | ISO | MDL-2 |
|---|---:|---:|
| Total cost | 3,990,455,613 | 3,895,694,268 |
| Completion service | 0.3376 | 0.3719 |
| Eligibility | 0.3775 | 0.4210 |
| Patients lost | 4,021.92 | 3,698.87 |
| Manufacturing-period ineligibility | 0.1464 | 0.1214 |
| In-process therapies discarded | 435.65 | 386.25 |
| Turnaround time | 4.257 | 5.522 |

MDL-2 minus ISO mean cost was `-94,761,344`, with a paired 95% interval of
`[-108,810,889, -80,711,800]`. MDL-2 had lower cost in 92 of 100 replications,
lost 323.05 fewer patients on average, discarded 49.40 fewer therapies, and
increased completion service by 0.03434.

Raw and summary outputs are stored under:

`results/patient_lifecycle_calibration/pilot/`

## Consequence for learned policies

All learned-policy checkpoints produced under the exploratory 77%-83%
high-attrition environment are invalid for nominal manuscript comparison. The
next campaign must retrain, in order:

1. AFR-GCN-DDPG;
2. matched AFR-flat-DDPG;
3. AFR-GCN-TD3; and
4. the fixed heuristic set.

The first screen should use 100 episodes x 3 training seeds. A variant should
advance to 300 episodes x 5 seeds only if its held-out correction improves on
MDL-2 without degrading completion service or patient loss.

## Calibrated learned-policy screen

The pre-specified screen was completed with 100 training episodes for each of
three training seeds. The matched policies were:

- AFR-GCN-DDPG;
- AFR-GCN-TD3; and
- AFR-Flat-DDPG.

All three used the same MDL-2 anchor, advantage-filtered demonstrations, bounded
replenishment correction, train-time demand/forecast randomization, checkpoint
schedule, and Monte Carlo streams. Checkpoints were selected from episodes 25,
50, 75, and 100 using independent validation replications. A separate
20-replication deployment calibration selected the residual scale or returned
the MDL-2 anchor. The deployment guard required the correction not to reduce
completion service or increase patient loss.

Results are stored under:

`results/residual_policy_benchmark/lifecycle_pilot/`

### Aggregate results

Each row below contains 300 formal evaluation replications: 100 paired
replications for each of three training seeds.

| Method | Total cost | Completion service | Eligibility | Patients lost | Mfg. ineligibility |
|---|---:|---:|---:|---:|---:|
| AFR-Flat-DDPG | 3,927,584,168 | 0.3703593 | 0.5070304 | 3,730.357 | 0.1222230 |
| AFR-GCN-DDPG | 3,927,593,628 | 0.3703593 | 0.5070305 | 3,730.353 | 0.1222226 |
| AFR-GCN-TD3 | 3,927,628,451 | 0.3703578 | 0.5070299 | 3,730.363 | 0.1222229 |
| MDL-2 | 3,927,620,146 | 0.3703588 | 0.5070299 | 3,730.360 | 0.1222231 |
| ISO | 4,011,547,363 | 0.3375213 | 0.4600442 | 4,043.103 | 0.1464684 |
| pMYO | 4,180,485,055 | 0.3433113 | 0.4967760 | 3,894.087 | 0.1333322 |
| MYO | 4,194,548,795 | 0.3428887 | 0.4965860 | 3,897.010 | 0.1335821 |

### Paired inference

Differences are candidate minus comparator. Cost intervals use the
training-seed/replication two-level bootstrap with 20,000 resamples.

| Comparison | Mean cost difference | Relative gap | 95% interval |
|---|---:|---:|---:|
| AFR-GCN-DDPG vs MDL-2 | -26,519 | -0.000675% | [-89,588, 40,330] |
| AFR-GCN-TD3 vs MDL-2 | +8,305 | +0.000211% | [-22,710, 51,126] |
| AFR-Flat-DDPG vs MDL-2 | -35,979 | -0.000916% | [-101,471, 33,347] |
| AFR-GCN-DDPG vs AFR-Flat-DDPG | +9,460 | +0.000241% | [-7,951, 24,621] |

AFR-GCN-DDPG deployed a nonzero residual in all three seeds. AFR-GCN-TD3 and
AFR-Flat-DDPG each fell back to MDL-2 in one seed. Nevertheless, every cost
interval includes zero, and patient-facing differences are effectively zero.
The current evidence therefore supports safe parity with MDL-2, not material
improvement, and does not identify a graph advantage over the matched flat
policy.

### Mechanistic diagnosis

The accepted DDPG correction trades approximately 67,774 additional reagent
purchase cost and 5,611 additional holding cost for 94,726 lower reagent
shortage cost per replication. It changes completed patients by only 0.0033 on
average. This explains both the small cost improvement and the absence of a
clinically meaningful effect.

The present residual is restricted to replenishment. That decision is mostly
node-local and gives the graph encoder little opportunity to exploit geographic
edges, transfer pipelines, or regional disruption propagation. Increasing the
same replenishment-only training budget is therefore not justified by this
screen. The next matched screen should add small, distance-aware reagent and
idle-bioreactor transfer corrections, retain the clinical deployment
guardrails, and compare GCN against the same flat residual action space.
