# Candidate calibration acceptance did not pass

2026-10-01. The single approved engineering packet completed without execution
error, but passed0/9 full numerical acceptance cases. Do not launch a patient
pilot with this candidate. Preserve this negative result rather than tuning or
repeating it. No patient trajectory, research model fit or reward change occurred.

## Execution and independent checks

- Implementation/config/protocol frozen at
  `4c90d201da2b8e8cfa67137d071ed3ceb9037106`.
- 87 zero-optimizer contract/regression/archive tests, including14 new tests,
  passed before execution; full compileall and diff check passed.
- Nine artificial fits completed128 Adam calls each, exactly1,152 total, in
  6.250 seconds of recorded numerical execution (cap30 minutes). No restart,
  additional trial, parameter sweep or unused-budget extension.
- All nine final models were sealed before their artificial test coordinates
  were evaluated. Every initial/final checkpoint, Adam moment, old-distribution
  tensor, optimizer charge and update receipt is retained.
- Original P2 source/document locks and full payload/launcher inventories are
  unchanged. The new type remains rejected by real patient collectors/kernels.
- Both numerical runner and independent scalar verifier exited0. No related
  research Python remains live. An execution success is not a passed gate.

Root:`results/candidate_calibration_engineering_20261001`.
Readback:`reports/2026-10-01-candidate-calibration-engineering/verification.json`.
The verifier reconciled all2,343 run files,1,152 charges/updates,18 model envelopes,
Adam counters, frozen source hashes, oracle labels/returns, softmax probabilities,
ranking margins, value errors and gate outcomes without another neural forward
or optimizer call. It also verified that raw acceptance artifacts were unchanged.

## Results

Every fitted model still chose the reference on all8 test coordinates. The
invented task requires the reference in4 cases and the known alternative in4,
so every actor remains at50% accuracy, identical to its own frozen control.
Only flat-102 passes both value gates; no case passes the ranking gates.

| Representation | Seed | Action accuracy | Worst winner margin | Value explained variance | Value RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Graph | 101 | 50% | -3.858 | 0.822 | 0.371 |
| Graph | 102 | 50% | -3.436 | 0.867 | 0.320 |
| Graph | 103 | 50% | -2.995 | 0.845 | 0.345 |
| Self-only | 101 | 50% | -3.814 | 0.805 | 0.387 |
| Self-only | 102 | 50% | -3.508 | 0.873 | 0.312 |
| Self-only | 103 | 50% | -2.933 | 0.848 | 0.342 |
| Flat | 101 | 50% | -3.276 | 0.851 | 0.339 |
| Flat | 102 | 50% | -3.608 | 0.959 | 0.177 |
| Flat | 103 | 50% | -2.951 | 0.905 | 0.270 |

Prespecified gates:100% action accuracy, winner margin>=0.1, value explained
variance>=0.90 and RMSE<=0.25, for every fixture. None was relaxed.

## What changed and what remains unresolved

The calibrated output span upper bounds are60.13-63.61, above the3.8067 prior
gap. The old L1 impossibility certificate therefore no longer excludes a flip.
This upper bound alone does not prove that attainable hidden features or the
optimizer can realize the needed state-dependent scores. Observed choices still
did not flip. Simply enlarging output range was insufficient in this packet.

The value head now predicts substantial variation on this invented task,
whereas the corresponding frozen zero-valued head explains none. This is not a
controlled comparison against P2 patient data: the input dimension, labels and
sampling procedure are different. It is a partial engineering result, not a
new clinical or scientific performance gain.

The actor actually increased reference probability from90% to92.46%-97.22%
across the artificial test coordinates, including those needing the alternative.
An analytic check makes this direction understandable without claiming a full
causal explanation of the neural fit. At initialization, consider raising the
reference logit equally in both public-cue contexts:

- Reference-good context:expected return derivative `0.9*0.1*0.25 = +0.0225`.
- Reference-bad context:the sole good alternative has probability0.02, giving
  derivative `-0.9*0.02*0.25 = -0.0045`.
- Balanced-context average is+0.009. The entropy term contributes-0.003426,
  leaving+0.005574 along this shared reference-preference direction.

Thus correct expected-surrogate arithmetic can initially reinforce a common
reference preference even when half the contexts need another action. Learning
the context-action interaction is still necessary. This local derivative is not
proof that no later optimizer trajectory can learn it, nor proof that a different
policy head will work. It does rule out treating known favorable action labels
as sufficient by themselves under this setup.

The new artificial gradient instrumentation records initial value-gradient
norms80-106 times the actor-gradient norm,664/1,143 defined shared-encoder cosine
values negative, and global clipping on278/1,152 calls. Probability-ratio clipping
still never occurs. These observations concern only the toy packet. They do not
retroactively measure P2's missing gradients or prove that shared optimization
caused failure; no separate-optimizer control was run. Adam and separate output
parameters further limit causal conclusions from global norms.

## Scope and scientific interpretation

The fixture supplies all artificial action values exactly. It bypasses noisy
rollout labels, exploration coverage, temporal credit assignment, GAE and patient
dynamics. Its public cue makes the answer deliberately elementary. Failure even
here means this candidate has not met a necessary engineering check. It is not
evidence that the patient simulator has no improvement headroom, that all PPO
or DDPG methods fail, or that the reward weights need changing.

Graph/self-only parameter counts are3,938 each; flat is4,002 and unmatched.
The toy public cue is shared by all representations. No graph superiority,
deployment adaptation, online patient benefit or publication-readiness claim
can be drawn. Original P2 null findings and its inherited R4 gains stand.

## Next decision

Do not spend another patient run on this calibrated shared-head/prior setup.
Propose a separate, simpler actor-only positive control, not yet authorized:

1. Replace the permanent90% preference with only an explicit infinitesimal
   reference tie-break at initialization. Keep the same initial greedy reference,
   but disclose that initial sampling is no longer90% reference and that this is
   a new policy, not historical-policy continuation.
2. Use a minimal state-conditioned categorical scoring head with an isolated
   actor optimizer. Retain the same invented oracle task so the action-learning
   question is not obscured by critic fitting. No reward or patient-scenario change.
3. Freeze one implementation and gates before its single numerical packet. Use
   no more than the existing9 fixtures/128 optimizer calls each/1,152 total/30
   numerical minutes. Preserve failures; no extra variants or expansion.

This would test a combined positive-control design, not separately identify the
causal effects of the prior, head and optimizer. Passing would justify designing
a fair bounded follow-up, not immediately launching one. Failure would argue for
further formulation review rather than continued large patient pilots. User
approval is required for those new artificial fits; they have not started.

## Preservation

The run and new implementation/report packet are preserved separately from P2.
See `reports/2026-10-01-candidate-calibration-engineering/preservation.json` for
archive member hashes and Dropbox-local byte-copy verification when present.
Cloud synchronization and Howard access are not verified, and no sharing or
remote Git operation is included. This finite engineering chain is closed.
