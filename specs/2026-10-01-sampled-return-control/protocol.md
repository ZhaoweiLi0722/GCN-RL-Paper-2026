# Sampled return artificial control

## Status and authority

2026-10-01. Engineering protocol; numerical execution remains pending the
explicit 9-fit/2,304-call approval. The user's latest "continue" progresses
already authorized engineering; it does not resolve the separate pending
approval question by inference. No patient environment calls, patient-data
fitting, reward revision, new scenario, remote operations or Dropbox export.
The non-authorizing draft must remain unchanged. The separate committed
`execution_authorization.json` must record actual user approval, exact scope,
implementation commit, all tracked source/test hashes, protocol/config hashes
and the runtime before fitting. No such execution authorization is supplied
by engineering preparation. A mocked test authorization is not user approval.

## Fixed task and comparison

Reuse the actor-only positive-control task, bank, neural units, actor and
initialization seeds 101/102/103 for graph/self_only/flat. Preserve the tiny
trainable reference tie-break and approximately uniform initial sampling law.
This is not continuation of the earlier 90%-reference patient policy. Compare
each final actor with its own tensor-identical frozen start.

Use an independent linear state-value model over normalized public raw inputs,
including time/cue, nodes, physical links, reference and anchor. Parameters
start at zero with output gain32; no actor parameter storage is shared. The
critic is identical in form across representations, not a graph-critic study.
The invented value trend is linear, so this is deliberately a simple necessary
learnability check. Actor changes, sampled supervision and independent critic
are not a single-component causal ablation versus historical patient models.

## Observations and updates

Each fit has32 rollouts. Each rollout visits the12 public training contexts
four times in fixed order, sampling48 actions from the current actor using a
private CPU generator. Collect the whole rollout before any update. Only the
chosen action's scalar reward is returned to the learner. The generator may
know the task rule, but a full Q table or winner label must not enter learner
targets, policy inputs or advantages. Rewards are deterministic conditional
on context/action: this diagnoses sampling, not exogenous reward noise.

These are one-step invented decisions with a time-dependent scalar return,
not a52-step patient trajectory. There is no GAE or clinical long-horizon
claim. Advantage = observed return minus pre-update independent value, then
normalize once across48 samples. Shuffle with a separate private generator;
split into four disjoint minibatches of12, used once each. Preserve behavior
log probabilities throughout those four updates. The old-policy distribution
is refreshed by collecting the next rollout, not during minibatch optimization.

Each minibatch takes one actor Adam step and one independent critic Adam step.
Actor uses existing clipped PPO loss (clip0.2, entropy0.01), value MSE coefficient
zero in the actor objective. Critic uses MSE to observed scalar returns. Both
optimizers use lr0.0003, gradient norm cap0.5 and default Adam betas/epsilon,
zero weight decay, CPU float32, one Torch thread. Critic gradients cannot enter
the actor; normalized advantages and behavior values remain detached.

Sampling generator seed =610000+initialization seed; shuffle generator seed
=620000+initialization seed. Use the same streams across representations, but
do not claim identical sampled actions after policies diverge. The cue bypasses
graph aggregation; no graph advantage can be inferred. Flat parameter mismatch
and the independent critic count must be disclosed.

## Bounds and prospective gates

Exactly1536 training action observations and128 actor+128 critic calls per
completed fit; maximum9 fits,13,824 observations,2,304 total calls,1800 seconds
from numerical setup through final evaluation. Charge observation batches and
each optimizer call before execution; no refund after a failed call. One
configuration and one attempt, no tuning, retry, extra seed or early selection.

All9 final models must be sealed before8 held-out interpolation contexts are
constructed per fit. Final evaluation contains72 contexts,144 actor decisions
(frozen/final) and72 final critic values. These reuse the same8 invented
contexts and are not72 independent statistical replications. No extra evaluation
sampling or optimizer updates are permitted. The exact oracle table is allowed
only here for scalar scoring. Report frozen/final expected and greedy return,
accuracy, margins and every gate even if they fail.

Retain actor-control gates: all8 choices correct, minimum correct-action logit
margin0.1, in every fit. Require at least one training selection of every
context/action pair. Compare final critic MSE to a constant baseline equal to
the mean of observed training returns, using final-policy exact expected
values as evaluator-only targets; require MSE <=0.5*baseline MSE. This is a
prospective engineering tolerance, not a scientific significance threshold.
If baseline MSE is zero, require critic MSE zero; do not divide by zero or
silently drop the case. Report all raw errors and counts regardless of pass.

## Evidence and remaining engineering

Preserve initial/final actors, frozen-copy hashes, critic, both optimizers,
private/global RNG, whole pending batch, minibatch permutation/cursor, sample
receipts, charged budgets, raw losses/gradients and runtime. Failed paired
updates roll back model/optimizer payloads, but their attempted calls stay
charged. Continuation/resume is not authorized; recovery payloads are evidence.
The runner must mark failure terminally, retain partial state and never retry.

Before a run: zero-optimizer tests, full compileall, committed protocol/config,
explicit scope approval, source/runtime/input locks and a unique process claim.
The separate verifier must recompute sampled rewards, probabilities, normalized
advantages, budgets, model seals, test metrics and gates from raw receipts with
no new policy forwards or optimizer calls. Verify immutable P2, calibration and
actor-only inventories before/after. Locally archive every member with hashes.

The engineering implementation includes models, sampler, disjoint loss/update
core, nonrefundable accounting, snapshots, serial campaign orchestration and
an independent saved-packet verifier. The runner requires an unchanged clean
committed source tree, exact branch, explicit frozen approval packet and the
single unused result root. Authorization is checked before model construction.
The original config remains `artificial_execution_authorized=false`; only the
separate approval-bound packet can enable a run. Its readiness flag does not
grant research authority.

Run entry point after approval/freeze only:
`python -m experiments.scripts.run_candidate_sampled_return_control`.
Readback entry point:
`python -m experiments.scripts.verify_candidate_sampled_return_control`.
The verifier checks each rollout/update checkpoint and chronological charge
receipt, final seals, Adam counters and model/optimizer payloads. Private RNG
replay uses recorded logits and permutations, not new model forwards or task
observations. This establishes recorded sampling consistency, not a second
independent neural computation of each stored logit.

Real Adam/SGD and patient calls are forbidden in preparatory tests. The full
serial mock test substitutes sample receipts, optimizer moments and evaluation
values; its gate outputs are not experimental results. Passing this packet
would not authorize a patient pilot or dynamic-bank architecture fit. Do not
manually loop the core to bypass the execution entry point.
