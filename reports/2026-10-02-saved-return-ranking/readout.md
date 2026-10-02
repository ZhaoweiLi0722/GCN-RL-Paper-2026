# Saved return-ranking diagnosis

## Decision-changing answer

The completed continuation result remains unchanged: no incremental greedy PPO
benefit. The new diagnosis identifies a weak value baseline, not missing updates
or missing sampled actions. It does not establish that the reward is wrong or
that the restricted decision space has no headroom.

Read-only reconstruction covers all 96 PPO training episodes, 4,992 decisions,
24 rollouts and 384 recorded minibatches. All 99 consumed phase/event files
match the completed payload inventory; event files also match phase indexes.
No model load, model forward, environment call or optimizer update was performed.
Results are post-hoc development diagnostics, not independent confirmation.

## What the records show

| Block | Return/step correlation | Advantage/step correlation | Last collected rollout value explained variance | Last value SD | Last return SD |
|---|---:|---:|---:|---:|---:|
| 60 | 0.9685 | 0.8853 | 0.002817 | 0.0014 | 0.8222 |
| 61 | 0.9754 | 0.8864 | 0.001061 | 0.0005 | 0.7798 |
| 62 | 0.9587 | 0.9185 | 0.002136 | 0.0013 | 0.9837 |

Rewards and values here are scaled objective units (raw negative cost times
1e-9). Explained variance is `1 - Var(return - value) / Var(return)` within
each 208-row rollout. These are **collection-time** values before the final
update, not a new score of the final checkpoint. The final checkpoint's value
quality on fresh trajectories remains unmeasured by this diagnosis.

The zero-initialized value output moves toward negative values, but on the last
collected rollout its mean is about -1.11 in each block while its statewise SD
is only 0.0005-0.0014. Returns vary by roughly 0.78-0.98 SD. Thus the recorded
baseline mainly moves its level, without explaining the large within-rollout
differences. Full-horizon absolute returns naturally depend on remaining time;
high time correlation alone is not a reward defect. It does show that the
baseline has left much of this predictable variation in the learning target.
The independent source check confirms that normalized time actually reaches
the critic; this is not a missing-time-input wiring bug. Raw feature scales,
tanh activations and limited value updates are possible mechanisms, not proven
causes. Actor and critic have separate gradients, clipping and optimizers.

The recorded critic gradient exceeds the 0.5 clipping threshold in
128/128, 127/128 and 127/128 minibatches. Actor clipping occurs in only 1/128,
0/128 and 0/128. Clipping is an observed implementation behavior, not proof of
the causal bottleneck or permission to relax the threshold.

## Action signal and entropy limits

After centering returns within the same update and time step (four trajectories
per cell), the observed nonreference-minus-reference mean return is +0.002280,
-0.009807 and -0.023737 across blocks. First-half versus second-half signs are
opposite in blocks 60 and 61; block 62 is negative in both halves. These are
descriptive sampled-action associations: action probabilities, patient states
and downstream sampled actions differ. They are **not** estimates of the value
of changing one action in the same state. No confidence interval treats the
4,992 correlated steps as independent samples.

Recorded actor norms combine the surrogate and entropy terms. Separate
parameter gradients, their alignment and the actual Adam contribution of each
term were not saved. Therefore we cannot assert that entropy caused the null.
An analytic old-policy **logit-space** check finds mean entropy-gradient norm
only 0.018%-0.208% of the surrogate-gradient norm across rollouts. This is not
a parameter-space ratio: the network Jacobian and cancellation across states
can change it. Loss magnitudes also cannot substitute for gradient attribution.

## Method and verification

`diagnose.py` uses only standard-library JSON and scalar arithmetic. With
gamma=lambda=1 and terminal bootstrap zero, it reconstructs return-to-go and
`return - saved_value`, then population-normalizes advantages once across the
four-episode rollout. It checks all four recorded epoch partitions and compares
the first minibatch's policy/value loss against saved receipts at each update.
Maximum absolute discrepancy is 3.503e-7, below the declared 2e-5 float64 versus
float32 reconciliation tolerance. Later minibatch losses are read, not
reconstructed without their changed model predictions.

Seven artificial scalar tests cover return accumulation, normalization, grouped
centering, degenerate correlation, analytic gradient signs and finite differences,
and nonfinite rejection. This is analysis validation, not a performance gate.
The source-contract companion identifies the exact frozen implementation.
Recorded minibatch summaries in the JSON are unweighted minibatch summaries,
not row-weighted full-rollout losses at one fixed policy.

Reproduction (existing JSON only; output creation is exclusive):

```bash
python reports/2026-10-02-saved-return-ranking/diagnose.py \
  --root results/dynamic_candidate_continuation_recovery_20261001 \
  --output NEW_NONEXISTING_OUTPUT.json
```

`diagnostic.json` records the script SHA256, input inventory SHA256 and all
99 consumed file hashes. No old experiment or report is changed by this script.
The original completed comparison and archives are reused, not re-audited.

## Next decision

Prepare one bounded value-baseline intervention, keeping the actor's saved
starting behavior, candidate support, absolute-cost reward, entropy coefficient,
scenario and deterministic evaluation rule fixed. Make time/feature scaling
explicit and distinguish value-learning improvement from a patient-performance
gain. The exact value representation and initialization must be chosen and
recorded prospectively, not searched until favorable.

The minimum informative comparison must retain current PPO as an active
comparator as well as the same-start frozen control; BC-CONTINUE remains the
imitation attribution control. Reusing old test worlds can be diagnostic only,
not a fresh confirmation. Specify fresh training/test streams, complete
simulation/update/time budgets and one-attempt stopping rules before seeking
execution approval. Do not request a separate fit-only gate before the comparison.

This direction is a proposal, not an execution permit. If improved value
learning still yields no greedy gain, report it and reconsider scientifically
justified decision headroom rather than tuning reward to obtain a win. Routing
plus flexible capacity remains a separate possible scope, not evidence from this
restricted experiment. No reward change is justified by these records alone.

Preparation remains open: the exact single intervention, fresh stream mapping
and full comparison budget have not yet been frozen. The active same-thread
automation owns that finite next deliverable and must pause at its consolidated
execution-approval question. No experiment is currently running.
