# Patient-lifecycle retraining: 300 episodes x 3 seeds

Date: 2026-07-24

> Superseded for manuscript inference: this campaign used the exploratory
> high-attrition patient parameters and legacy patient-facing cost weights.
> It remains an extreme-stress diagnostic, but its policy rankings must not be
> used as evidence for the nominal calibrated environment. See
> `specs/2026-07-24-patient-lifecycle-calibration/results.md`.

## Scope

This campaign retrained the residual-policy family after adding the complete
patient lifecycle to the environment:

- patient deterioration while waiting and during manufacturing;
- therapy discard after manufacturing-period deterioration;
- bioreactor cleaning before capacity becomes available again;
- completion/infusion-based patient service;
- geographic transfer dynamics; and
- demand drift and forecast error.

The evaluated scenario was `patient_condition_geo_demand_drift`. Each learned
method used 300 training episodes for each of three training seeds. Checkpoints
were saved every 50 episodes, selected with 20 held-out validation
replications, calibrated against the anchor with 50 additional validation
replications, and evaluated with 100 paired Monte Carlo replications per
training seed.

Results are stored under:

`results/residual_policy_benchmark/lifecycle_targeted_300/`

## Runtime

The three algorithm families were launched in parallel on CPU. Per training
seed, the measured training times were approximately:

| Method | Minutes per seed |
|---|---:|
| AFR-GCN-DDPG | 28.8 |
| AFR-GCN-TD3 | 21.8 |
| AFR-Flat-DDPG | 36.2 |

The full campaign contains 2,700 training episodes. Parallel execution reduced
the training wall-clock time to approximately 1.8 hours before checkpoint
selection, fallback calibration, and final evaluation.

## Checkpoint and fallback decisions

| Method | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| AFR-GCN-DDPG | episode 150, anchor | episode 50, anchor | episode 100, learned scale 2.0 |
| AFR-GCN-TD3 | episode 250, anchor | episode 150, anchor | episode 300, anchor |
| AFR-Flat-DDPG | episode 50, anchor | episode 50, learned scale 2.0 | episode 100, anchor |

The safeguard therefore rejected the learned correction in seven of the nine
trained policies. AFR-GCN-TD3 reduced exactly to MDL-2 for all three seeds.

## Aggregate results

The table aggregates 300 final replications per method. Lower cost, fewer
patients lost, fewer discarded therapies, shorter turnaround time, and higher
eligibility/completion service are preferable.

| Method | Total cost | Start service | Eligibility | Completion service | Patients lost | Therapies discarded | Turnaround |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISO | 1,505,208,233 | 0.5971 | 0.1384 | 0.1285 | 5,782.49 | 3,304.55 | 3.236 |
| AFR-Flat-DDPG | 1,565,826,942 | 0.6477 | 0.1074 | 0.0995 | 5,974.73 | 3,888.47 | 3.773 |
| AFR-GCN-DDPG | 1,565,885,376 | 0.6479 | 0.1074 | 0.0995 | 5,974.78 | 3,889.37 | 3.773 |
| AFR-GCN-TD3 | 1,565,902,041 | 0.6476 | 0.1074 | 0.0995 | 5,974.92 | 3,887.75 | 3.773 |
| MDL-2 | 1,565,902,041 | 0.6476 | 0.1074 | 0.0995 | 5,974.92 | 3,887.75 | 3.773 |
| pMYO | 1,730,593,076 | 0.6413 | 0.0574 | 0.0530 | 6,289.63 | 4,199.20 | 4.815 |
| MYO | 1,779,858,391 | 0.6398 | 0.0569 | 0.0526 | 6,292.58 | 4,191.62 | 4.821 |

The start-service metric alone is misleading after introducing manufacturing
deterioration. ISO starts fewer patients but completes more eligible therapies,
loses fewer patients, discards fewer therapies, and incurs lower cost.

## Paired comparisons

Differences below are learned method minus comparator, so a negative cost
difference favors the learned method. The intervals are conditional 95%
normal intervals over the 300 paired final-evaluation replications; three
training seeds are not enough for strong across-training-seed inference.

| Comparison | Mean paired cost difference | Conditional 95% interval |
|---|---:|---:|
| AFR-GCN-DDPG vs MDL-2 | -16,665 | [-125,966, 92,637] |
| AFR-Flat-DDPG vs MDL-2 | -75,099 | [-157,126, 6,929] |
| AFR-GCN-TD3 vs MDL-2 | 0 | [0, 0] |
| AFR-GCN-DDPG vs ISO | +60,677,143 | [58,076,865, 63,277,420] |
| AFR-GCN-DDPG vs AFR-Flat-DDPG | +58,434 | [-78,342, 195,210] |

## Interpretation

The 300-episode campaign is sufficient as a decisive pilot. It shows that the
current MDL-2-anchored residual design does not provide a stable or material
improvement under the complete patient lifecycle.

- AFR-GCN-DDPG marginally improves on MDL-2 only because one training seed
  passed fallback calibration; the paired interval includes zero.
- AFR-GCN-TD3 never passed fallback calibration.
- AFR-Flat-DDPG has a slightly better mean than AFR-GCN-DDPG, but the
  graph-versus-flat paired interval includes zero.
- ISO is the strongest tested heuristic by a large margin and also produces
  better patient-facing outcomes.

These results do not support claims that the current graph residual policy
outperforms the strongest heuristic or that its gain is caused by the graph
representation. Increasing the same training run from 300 episodes without
changing the method is unlikely to resolve the structural gap.

## Next experiment

The next targeted experiment should treat ISO, rather than MDL-2, as the
operational anchor:

1. diagnose which ISO decisions reduce manufacturing-period losses;
2. implement matched GCN and flat ISO-anchored residual policies;
3. let the residual modify only decisions for which the observation contains
   predictive graph information, initially replenishment and transfer;
4. use completion service, patient loss, manufacturing loss, and total cost in
   checkpoint selection;
5. run a 100-episode x 3-seed screen before escalating promising variants to
   at least 300 episodes x 5 training seeds and 500 paired replications.
