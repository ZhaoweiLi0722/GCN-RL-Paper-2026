# P2 learning mechanism diagnosis

2026-10-01. Completed posthoc analysis of saved training evidence only.
P2 still has zero observed incremental greedy-policy gain. The new finding is
that its final learned scorer cannot mathematically overcome its fixed reference
prior on any admissible finite input, and its recorded value baseline barely
explains return variation. This is a limitation of the attained parameterization
and fitting, not evidence that the operating environment is near optimal or
that RL can never help. The conservative P2 design needs revision before another
scientific test would be informative.

## Evidence and scope

- Frozen reader commit: `3b90bef2e3d984f2cc0b38f161774729f63cd9d7`.
- Nine PPO fits, three blocks, 288 training episodes, 14,976 decisions,
  72 saved four-episode rollouts and 1,152 recorded minibatch updates.
- All 675 consumed input files hashed. Full original payload and launcher
  inventories unchanged before/after; original source/document locks pass.
- Independently recomputed backward sums of raw negative costs, advantages,
  normalization, first-minibatch losses and update index coverage.
- Second scalar `math.fsum` reader rehashed all 675 inputs and independently
  checked final score/value bounds, target coverage, costs and clipping counts.
- No new patient simulation, model forward/backward, optimizer update or test
  evaluation. No scientific retry. Both readers exited 0; process scan empty.
- 28 artificial arithmetic, closure and archive tests passed, with full
  repository compilation and diff checks. No numerical neural training tests.
- This is posthoc development evidence. The protocol discloses preliminary
  weight inspection; neither reader creates an independent scientific sample.

Machine-readable evidence: `reports/2026-10-01-p2-training-diagnostic/result.json`
and `crosscheck.json`. The original P2 outcome remains in
`specs/2026-10-01-reference-prior-residual/terminal_readout.md`.

## A fixed weight barrier to changing the action

The shared scorer ends in `Linear -> Tanh -> Linear(32,1)`. For fixed final
output weights w, any two candidate residual scores differ by at most
`2 * sum(abs(w))`. The common output bias cancels, and each hidden coordinate
is between -1 and 1. This bound holds irrespective of the preceding encoder.

The fixed 90% reference prior requires an alternative to overcome a logit gap
`log(9*(K-1))`: 2.197225 even for K=2, and 3.583519/3.806662 for the actually
observed K=5/6. All training states had K=5 or 6; none was a singleton.

| Block | Graph bound | Self-only bound | Flat bound |
| --- | ---: | ---: | ---: |
| 60 | 0.374272 | 0.212757 | 0.277901 |
| 61 | 0.393381 | 0.358977 | 0.487638 |
| 62 | 0.258558 | 0.095106 | 0.809345 |

Every bound is below even the K=2 prior gap, with at least 1.387880 margin.
Therefore all nine final policies must retain the reference greedy choice on
every finite input for this implementation and prior, not merely on the saved
test states. That explains why changing weights did not change greedy actions.
It does not establish which training component caused the small final weights,
nor what a future fit, alternative parameterization or sampled deployment would
do. The trainable model class is not globally incapable of larger scores.

Sources: `src/models/candidate_policy.py` and
`src/models/reference_prior_candidate.py`; both verified against the run locks.

## The critic did not learn useful return variation

The final value head has the analogous fixed-weight interval
`[bias - L1(w), bias + L1(w)]`. Its lower limits are only -1.256 to -1.274,
whereas observed once-scaled training returns reach -3.319. Across the saved
training sets, 9,001/14,976 targets (60.10%) lie outside their respective final
head's entire possible range. The associated unavoidable MSE lower bounds are
0.497 to 0.597. These compare final weight ranges with old training targets;
they are not new final-model forward evaluations.

Recorded behavior-time value predictions explain almost none of the variation
in their matched return targets: explained variance ranges from -0.000654 to
0.002063 across the 72 rollouts. MSE falls from roughly 2.56-3.30 at the first
rollout to 0.73-0.97 at the eighth, but a declining average loss alone does not
establish a useful state-dependent baseline.

Within each four-episode rollout, 90.73%-99.52% of advantage sum-of-squares is
between episode-time positions. This is a descriptive decomposition, not a
causal attribution: remaining-horizon structure dominates these targets, and
the recorded baseline has not removed it. We have not measured hidden-unit
saturation or separate actor/critic parameter gradients.

## What clipping and sampled advantages do not establish

Probability-ratio clipping occurred in 0/1,152 minibatches. Conversely, global
gradient clipping occurred in 1,145/1,152 (99.39%). The typical recorded common
gradient multiplier is around 0.10. These are different mechanisms. Global
norms and scalar value-loss size cannot prove that critic gradients dominated
actor gradients; Adam's state also prevents treating the multiplier as the
actual parameter-update shrinkage.

Nonreference sampled actions have a positive mean normalized advantage in
29/72 rollouts, or 33/72 after removing the within-rollout mean at each time
position. The combined reference-logit return/entropy score-function signal
is positive in 39/72. Directions are mixed. These are different visited states
and sampled actions, not paired same-state counterfactuals, independent trials,
significance tests or a reliable action-ranking oracle. No stable alternative
action gain has been demonstrated by this diagnostic.

## Reward arithmetic is consistent

Every checked reward is negative raw total cost, scaled once by 1e-9. The maximum
independent return/advantage discrepancy is 1.2662e-6, consistent with float32
accumulation; the largest first-minibatch objective discrepancy is 3.4178e-7.
No episode boundary leakage or index-coverage error was found.

Aggregate raw training cost shares are 44.19% patient loss, 43.02% bioreactor
shortage and 8.82% expiry. Their large shares neither validate clinical cost
weights nor prove they are wrong. Small transport-fee shares do not imply that
routing has a small effect on downstream patient outcomes. Changing those
weights to obtain a positive result would change the scientific objective;
the current evidence first supports numerical and learning-mechanism work.

## Next decision

Recommend a separate engineering-only acceptance packet before any new pilot.
The proposed checks are:

1. Preserve an explicit same-start frozen reference comparator, but demonstrate
   that the chosen policy parameterization can actually reverse candidate
   ranking within its declared update cap on invented, known-answer examples.
   Test both advantageous and disadvantageous alternatives and state-dependent
   choices. A general L1 bound exceeding the prior gap is necessary for this
   certificate to stop binding, but is not sufficient proof of learnability.
2. Calibrate public-input and value/target numerical scales, including time to
   go. Verify useful value prediction on invented horizon-dependent returns,
   not just smaller average loss. Keep raw cost, reward semantics and inverse
   transforms auditable; normalization is not a new cost-weight objective.
3. Record separate policy/value gradient norms and their interaction in those
   artificial tests before deciding whether separate optimization is needed.
   Do not infer the need solely from the existing total norm.
4. Freeze one proposed repair, its tests and prospective finite pilot protocol.
   Keep any subsequently approved real comparison same-start frozen/PPO/BC,
   use fresh streams, and disclose every changed mechanism. A combined repair
   would identify the package effect, not isolate each component's contribution.

Proposed engineering cap, NOT approved or executed: nine predetermined invented
fixtures (graph/self-only/flat, three initialization seeds), at most 128
optimizer.step calls per fixture, 1,152 total, 30 minutes of numerical execution,
one candidate implementation, no expansion or search if a gate fails. No real
checkpoints or patient traces enter optimization, no new patient trajectories,
no scientific model fit, no reward/cost change and no remote action. Implementation
and artificial acceptance criteria must be frozen before those numerical tests.
If separate optimizers are proposed, each optimizer.step consumes the same cap.
A later scientific pilot requires its own explicit bounded authorization.

No paper claim of new RL benefit, repaired-DDPG superiority, deployment online
adaptation or publication readiness follows from this work. The productive next
step is to make the learning test mechanically informative, then measure benefit
honestly. Original negative results and clinical trade-offs stay in the record.
Flat parameters remain unmatched, all arms share the graph-based R4 reference,
and this packet does not provide a new isolated graph-representation gain.

## Preservation

The new diagnostic packet is separate from immutable P2 and is covered by
`reports/2026-10-01-p2-training-diagnostic/preservation.json` when present.
Its receipt distinguishes archive-member verification and Dropbox-local byte
verification from cloud synchronization and Howard access; the latter two
remain unverified. No source result was overwritten and no sharing was changed.
