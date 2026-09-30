# R6: training fit improved; held-out critic ranking did not generalize

## Verified scope and decision

The single approved attempt completed at source
`97c79a6b316740b835d82bc3f2be2683624908f9`, exit0, in1895.705s (31.6min).
It used18 frozen parent trajectories,72 states,3456 logical continuations,
73232 actual simulator steps and exactly1000 updates for each of three fresh
critics. Actors/gates/reward/dynamics stayed unchanged; zero actor or DDPG
updates. There was no extra attempt, expansion, scenario, holdout or remote action.
All three predeclared triage components failed. **Do not advance to actor
training from this result.** The engineering execution succeeded; the scientific
generalization screen did not. This does not prove that DDPG can never help.

The independent verifier reconciles every raw trace, cost component, outcome,
lineage, label, split, training-only constant choice, sealed selection and
reported contrast. All848 shared tails pass exact physical/RNG/cost/event
equivalence. Their savings explain the smaller actual-step count. Reconstructed
fresh initial weights match; all final Adam states record1000 steps; saved
MPS predictions reproduce exactly (maximum difference0 for every model).
Actor source hashes and R4's24 inputs/83 source locks match; R3 remains intact.
Stderr is zero bytes; no related collector/verifier remains at final scan.

Prelaunch62 related tests and full compileall passed. A post-run read-only
accounting helper adds one synthetic test:63 tests/compileall pass. Its repeated
derived output matches exactly. It takes no simulator step, performs no fitting,
changes no choice, and is descriptive analysis, not a new predeclared gate.

## Fit versus independent trajectories

Each model has16 training states from4 trajectories and8 test states from2
different trajectories. The critic has238433 parameters; sample coverage is
small. A training/test gap is observed, but this study does not causally separate
coverage, label noise, representation aliasing, architecture or regularization.

Accuracy is the fraction of correctly ordered certified action pairs whose
sampled cost means differ. Label ties are excluded; predicted ties are not
correct. Pairs/states are dependent, so these denominators are not independent
trials and no binomial significance claim is made.

| Frozen actor | Training accuracy | Test accuracy | Final training normalized MSE | Test normalized MSE | Zero-advantage test MSE |
| --- | --- | --- | --- | --- | --- |
| 60 | 118/150,78.7% | 24/75,32.0% | 0.13955 | 1.64468 | 0.39006 |
| 61 | 130/165,78.8% | 28/60,46.7% | 0.10911 | 2.13262 | 0.25426 |
| 62 | 61/90,67.8% | 37/80,46.3% | 0.51566 | 5.55800 | 2.84963 |

MSE uses each model's saved training-only RMS scale, equal state weights and
equal certified-choice weights within a state. Final errors use final saved
predictions, not the last pre-update loss log. Zero-advantage means predicting
every action equivalent to frozen; it is a value-prediction reference, not a
random controller. All three critics have worse test squared error than this
reference. The predeclared ranking criterion (>0.5 perpolicy) also fails.

As a post-run descriptive noise check, mean squared paired-label SE in these
same normalized units is0.46761,0.07911,0.88760 on the three test sets. Eight
draws give uncertain conditional estimates; these values do not identify a
ground-truth error decomposition. Noise is a live issue but cannot simply be
asserted to explain the whole observed generalization failure.

## All six test trajectories

Numbers below average the four prespecified state contrasts and their eight
paired futures. They mix overlapping remaining horizons: they are **not** the
cost or clinical outcome of one deployed full-episode critic-controlled policy.
Cost units are millions of original simulator units; negative is lower cost.
Lost-patient delta is also a mean remaining-window contrast, not six patient
cohorts. The constant family was selected using training labels only, with
public-support fallback: +0.10 for60/61; MDL2-first for62.

| Actor/test trajectory | Cost vs frozen,M | Cost vs training-constant,M | Lost patients vs frozen | Completed vs frozen | Adverse clinical mean vs frozen |
| --- | --- | --- | --- | --- | --- |
| 60/0 | +0.07270 | +2.00602 | +5.90625 | -4.37500 | yes |
| 60/1 | -0.88012 | -0.58561 | -0.18750 | +1.03125 | no |
| 61/0 | 0 | +0.66803 | 0 | 0 | no; all choices frozen |
| 61/1 | +0.04634 | +1.25534 | +2.81250 | -1.46875 | yes |
| 62/0 | +0.04396 | -0.47395 | -1.06250 | +0.84375 | no |
| 62/1 | -1.30929 | -0.71984 | -1.06250 | +1.00000 | no |

Only2/6 have lower cost than frozen,3/6 have higher cost,1/6 is tied. Three of
six beat the training-chosen constant; two show an adverse clinical mean versus
frozen. All exact service-level/manufacturing-loss deltas and MDL2-first
comparisons remain in `seed*/test_summary.json` and independent verification.
"No adverse mean" is not safety/noninferiority; averaging also hides individual
state harms, which are retained below. No threshold or clinical veto was
retroactively applied to the selected actions.

## All 24 test states

Action0=frozen;4=specimen-0.10;5=specimen+0.10. Choices1/2/3 were available
subject to the unchanged public-map certificate but never selected here.
Cost and lost-patient deltas are paired eight-draw means versus frozen-followup.

| State | Choice | Cost delta,M | Lost-patient delta |
| --- | --- | --- | --- |
| 60/test0/t0 | 0 | 0 | 0 |
| 60/test0/t13 | 4 | +0.214991 | +16.625 |
| 60/test0/t26 | 4 | +0.075827 | +7.000 |
| 60/test0/t39 | 0 | 0 | 0 |
| 60/test1/t0 | 0 | 0 | 0 |
| 60/test1/t13 | 0 | 0 | 0 |
| 60/test1/t26 | 5 | -1.456103 | -1.000 |
| 60/test1/t39 | 4 | -2.064360 | +0.250 |
| 61/test0/t0 | 0 | 0 | 0 |
| 61/test0/t13 | 0 | 0 | 0 |
| 61/test0/t26 | 0 | 0 | 0 |
| 61/test0/t39 | 0 | 0 | 0 |
| 61/test1/t0 | 0 | 0 | 0 |
| 61/test1/t13 | 0 | 0 | 0 |
| 61/test1/t26 | 4 | +5.394406 | +9.750 |
| 61/test1/t39 | 4 | -5.209031 | +1.500 |
| 62/test0/t0 | 0 | 0 | 0 |
| 62/test0/t13 | 5 | -1.572068 | -7.500 |
| 62/test0/t26 | 4 | +3.328597 | +4.750 |
| 62/test0/t39 | 5 | -1.580706 | -1.500 |
| 62/test1/t0 | 0 | 0 | 0 |
| 62/test1/t13 | 5 | +0.519728 | -0.750 |
| 62/test1/t26 | 4 | -0.557240 | +4.750 |
| 62/test1/t39 | 5 | -5.199666 | -8.250 |

Twelve choices remain frozen. Seven of24 states have more mean patient losses;
three of those also have lower mean total cost. Those conflicts are disclosed,
not removed. These are conditional descriptive outcomes, not clinical findings.

## What this changes, and what it does not

R3 established limited sampled local headroom; R6 does not convert it into a
reliable learned action selector. The optimizer did run and training errors
fell, but held-out ranking and cost consistency were insufficient. Calling the
pretrained policy globally optimal would still be unsupported. Calling this
online-DDPG improvement would be equally unsupported: this was MC-supervised
critic fitting with a frozen continuation, not Bellman/actor learning.

Reward redefinition is **not** established as the remedy. Raw cost/return
accounting passed, and the critic failed to generalize even for the existing
fixed objective. At the same time, cheaper-versus-more-loss cases expose a
separate objective/clinical-priority question. Any clinical constraint or cost
weight revision needs a defensible prespecified rationale, not weights chosen
to make RL look better. Do not alter the historical task or manuscript claim.

Next decision: retain this negative screen and first use saved data to examine
coverage/label uncertainty and observable state-action representation. An
alternative regularized estimator, different labels, more data, a revised
clinical objective or an actor experiment would each require a separate bounded
prospective proposal. Do not expand this run or automatically start any of them.
Even successful ranking would not by itself validate a DDPG action gradient
through integer execution or the hard deployment gate.

## Evidence and reproduction

- Raw root: `results/clean_critic_generalization_20260930/`
- Independent report: `reports/2026-09-30-clean-critic-generalization/verification.json`
- Descriptive fit/error report: `reports/2026-09-30-clean-critic-generalization/fit_generalization.json`
- Source protocol/config: this directory and `experiments/configs/clean_critic_generalization_20260930.json`
- Preservation receipt/status: `reports/2026-09-30-clean-critic-generalization/README.md`

Verification SHA256:
`57aa08ed2ef38a48bc698fffbd47bc16374b8bfd43d16f897d65811f2e72573b`.
Raw inventory SHA256:
`29ddb322f5da8b003f68008cf8fba95c2cc8489e30f71f0b9bfd27b2cf38bb35`.
Use new audit output filenames (writers refuse overwrite):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m evaluation.verify_clean_critic_generalization \
  results/clean_critic_generalization_20260930 --replay-predictions --output NEW_AUDIT.json
python -m evaluation.summarize_clean_critic_generalization \
  results/clean_critic_generalization_20260930 --output NEW_ACCOUNTING.json
```

This finite packet is complete scientifically; it does not authorize a new
campaign, a remote Git operation, reopening Stage E, or a stronger paper claim.
