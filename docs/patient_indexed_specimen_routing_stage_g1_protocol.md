# Stage G1 Legal-Action Critic Feasibility Protocol

## Locked question

Can a GCN-DDPG critic initialized from frozen pretraining learn a
remaining-episode ranking over the five actions that actually execute, and
generalize that ranking to an entirely unseen training seed and an independent
counterfactual replication stream?

This is the critic-first gate required by Stage G0. It is an offline
development audit. It does not optimize an actor, interactively collect replay,
select a checkpoint, reuse a formal stream, or authorize a new online campaign.

## Fixed dataset

- Reuse the immutable F1 control frozen-pretrain GCN checkpoints for seeds
  60-62, assigned to persistent hotspot clusters 1-3.
- Generate one frozen-pretrain trajectory per seed and inspect all 52 decision
  steps: 156 states total.
- At every state, evaluate MDL-2 and the four specimen corrections
  `+/-0.05`, `+/-0.10`. All five actions must remain distinct after execution
  quantization.
- Estimate both four-step and remaining-episode cost with MDL-2 continuation.
- Use three discovery replications and five independent validation
  replications. Discovery labels may train the ranker; validation labels are
  evaluation-only.
- Fresh stream bases are 101000000 for live trajectories, 103000000 for
  discovery rollouts, and 107000000 for validation rollouts. Every stream at or
  below 100500000 is forbidden, including the formal holdout and all prior
  development work.

## Single frozen ranker design

For each held-out seed, initialize a separate critic from that seed's immutable
frozen-pretrain checkpoint. Train only that critic on discovery labels from the
other two seeds. The deterministic actor, correction gates, target networks,
and original checkpoints remain untouched.

The primary target is remaining-episode cost advantage relative to MDL-2.
Within each state, divide advantages by the larger of `$1,000,000` and the
median absolute non-anchor advantage. Optimize the fixed sum of:

1. unit-weight smooth-L1 regression on action-versus-anchor Q differences; and
2. unit-weight pairwise soft-margin ranking for legal action pairs separated by
   at least `$250,000`, with margin `0.10`.

Use the existing critic learning rate `3e-4`, full-state batches, and exactly
200 epochs. There is no early stopping, validation-based epoch choice,
hyperparameter sweep, or saved fitted checkpoint. Four-step values are a
prespecified horizon-alignment diagnostic and do not train the primary ranker.

## Leave-one-seed-out evaluation

Run three folds. In each fold, the ranker never sees discovery or validation
labels from the held-out seed. Evaluate both the frozen critic and fitted
ranker on all 52 held-out states against:

- held-out discovery labels, for replication sensitivity; and
- held-out validation labels, for the primary feasibility decision.

Report exact top-1 legal-action accuracy, pairwise ranking accuracy for pairs
above the fixed cost gap, selected-action material-improvement rate, per-seed
gain over the frozen critic, discovery-validation best-action agreement, loss
components, and critic parameter drift.

## Prospective gate

The data are stable only if discovery and validation best actions agree in at
least `70%` of states. Conditional on that requirement, the fitted ranker must:

- reach at least `40%` aggregate validation top-1 accuracy;
- reach at least `30%` validation top-1 accuracy in every held-out seed;
- reach at least `65%` aggregate validation pairwise accuracy;
- improve validation top-1 by at least `5` percentage points in at least two
  seeds; and
- lose no more than `10` percentage points from held-out discovery to held-out
  validation top-1 accuracy.

Passing authorizes only a separately locked actor-transfer smoke. It does not
authorize online training. Failure closes the legal-action-aligned DDPG
extension and preserves the Stage E claim that online DDPG attribution is not
established.

