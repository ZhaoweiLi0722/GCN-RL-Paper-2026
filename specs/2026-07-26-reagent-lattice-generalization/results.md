# Reagent-lattice residual generalization results

Date: 2026-07-26

## Question

Can a causal, graph-aware reagent-transfer residual reliably improve the
MDL-2 anchor across the integrated geography, patient-condition, demand-shift,
and disruption scenarios?

This stage was diagnostic. No result below is manuscript-ready evidence of
heuristic outperformance.

## Changes to the experimental protocol

- Added complete-trajectory train/validation splits. Rows from one simulated
  trajectory can no longer appear in both supervised fitting and checkpoint
  selection.
- Added validation-loss early stopping with restoration of the best actor,
  correction gate, and optimizer state.
- Added a configurable false-positive penalty for correction gates.
- Collected 40 new causal DAgger trajectories (1,570 labelled states) across
  the four integrated scenarios, with 10% teacher-behavior mixing.
- Replaced the original 100k teacher-label margin with a pre-specified 500k
  margin for actor training.
- Added actor-specific gate calibration from saved CRN outcomes, including
  classification and continuous advantage-regression modes.

## Teacher and supervised-fit diagnostics

The broad 500k-margin cache retained 410 corrections from 1,570 states
(26.11%). Mean positive look-ahead advantage was 882,355, with scenario
correction rates from 12.73% to 32.71%.

The trajectory-held-out GCN fit used 28 training trajectories and 12 validation
trajectories. It stopped after 180 epochs and restored epoch 140. Held-out gate
precision/recall were 60.67%/40.30%.

For the actor's exact scale-0.5 correction, only 7 of 1,570 states (0.45%) had
at least 500k clinically feasible advantage. At a 100k margin, 14.65% were
positive. The continuous advantage gate achieved a saved target correlation of
0.649 and an advantage MAE of 78,135.

Traceable artifacts:

- `results/multiscenario_reagent_lattice_broad_collection/manifest.json`
- `results/multiscenario_reagent_lattice_broad/cache_manifest.json`
- `results/multiscenario_reagent_lattice_broad/multiscenario_reagent_lattice_broad_margin_train_seed1/training_manifest.json`
- `results/multiscenario_reagent_lattice_broad_advantage_gate_seed1/training_manifest.json`

## Paired rollout results

The initial 10-rep deployment screen selected the continuous-advantage gate at
threshold 0.15 and residual scale 0.5:

- validation cost gap: -0.003017%
- validation mean paired difference: -83,924
- validation 95% CI: [-245,642, 77,794]
- all four scenario-level clinical guardrails passed
- one-rep holdout cost gap: -0.002871%

Because the interval crossed zero and the effect was very small, the candidate
was frozen and evaluated with new seed blocks using 100 paired replications per
scenario plus a 50-replication holdout allocation.

The 100-rep confirmation rejected the candidate:

- aggregate cost gap: +0.005591%
- mean paired difference: +155,203
- 95% CI: [-294,466, 604,872]
- completion-service delta: -0.00003815
- patients-lost delta: +0.2125
- clinical noninferiority: failed
- compound-stress cost gap: +0.012408%
- regional-drift cost gap: +0.025150%

Validation therefore selected the scale-0 fallback, so the 50-rep holdout
evaluated MDL-2 unchanged.

Traceable artifacts:

- `results/multiscenario_reagent_lattice_broad_advantage_gate_seed1_eval/summary.json`
- `results/multiscenario_reagent_lattice_broad_advantage_gate_seed1_confirm/summary.json`

## Decision

Do not claim that this reagent-only AFR-GCN policy outperforms MDL-2. Do not
expand this rejected candidate to three training seeds or a matched flat
ablation.

The experiment does establish that:

1. Local look-ahead reagent-transfer headroom exists.
2. Continuous advantage prediction ranks actor actions better than binary
   correction classification.
3. Repeated reagent-only corrections do not convert that local headroom into a
   robust full-horizon advantage.

The next algorithm stage should change the decision class rather than add more
epochs: learn explicit network transfer corrections, represent recent
correction/pipeline history, and impose a temporal cooldown or cumulative
transfer budget. That stage should first pass a one-seed, fresh-seed paired
gate before any multi-seed expansion.
