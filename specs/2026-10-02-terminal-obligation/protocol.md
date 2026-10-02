# One-Shot Cohort-Objective Comparison

Status: prospective complete numerical scope, not execution approval. Engineering
is partial; no scientific call may occur until full integration, relevant mock
tests, compileall and committed source/runtime/input/authorization locks exist.
The old time-baseline and continuation attempts stay closed. This is development
in a simulator, not a clinical study or deployment-time online adaptation.

## Single Scientific Change

Question: does including the standardized consequences after the 52-step cutoff
in training improve the learned first-window policy relative to its own frozen
initializer, continued imitation, and PPO trained on the original window?

All arms keep the same 52-step, 20-facility `routing_nominal_history` prefix,
three qualified graph initializers for blocks60/61/62, specimen-route message
graph, candidate bank, original reward components, Adam3e-4, entropy0.01,
clip0.2, gradient cap0.5, gamma1/lambda1 and scale1e-9. No initializer refit,
requalification fit, new architecture, action channel, learning-rate search or
time-baseline variant. Both PPO arms use collected-value advantages. Plain R4
and full MDL-2 remain contextual comparators, not interchangeable controls.

After the prefix is recorded unchanged, enrollment closes. The same full MDL-2
rule receives zero current and prospective demand/rate priors and controls every
arm while patients remain active. Historical observations are not rewritten.
When active count first reaches zero, all new orders/transfers stop, but physical
resources and supplier evolution continue until the common economic endpoint.
The exact idle action is `[0]*60 + [-1]*20`, not an all-zero vector.

The source-derived patient bound is8tail transitions. The maximum generic
transfer delay is3, giving11tail transitions and economic endpoint63 for every
arm. A patient unresolved after tail8 or pending flow after tail11 fails the
contract and consumes the attempt; it never triggers a larger tail or a zero
liability. See [bound and settlement](bound-and-settlement.md).

No salvage, liquidation, free disposal, new penalty or external monetary
calibration is introduced. Original purchase cost is charged at order placement
only; on-hand holding continues to the fixed endpoint. Final resources are
retained and reported without a terminal credit. Source clipping of resource
overflow is preserved and must be disclosed, not asserted to be conservation.
This is a fixed63step economic observation, not lifetime financial settlement.

## Matched Acquisition and Training

Three training arms, each32prefix-plus-tail episodes/block: `window_ppo`,
`cohort_ppo`, `bc_continue`. Each rollout contains4complete acquired episodes;
8rollouts/model,4epochs/rollout, batches64/64/64/16 over the208prefix actions.
No learned decisions or optimizer calls occur in the11step tail.

- Window PPO logs the full tail but trains on original52step rewards only.
- Cohort PPO appends negative total tail primitive cost once to the last prefix
  reward, then applies the original1e-9scale and return calculation once. Both
  actor advantages and critic targets use that same declared objective.
- BC uses only the original prefix imitation examples. It receives the same
  acquisition budget; the tail is logged, never relabeled as training examples.
- Own-frozen uses exactly the same saved initial weights and no updates.

Keep raw environment rewards/costs separate from the new training-target receipt.
Do not modify `OneStepRecord.raw_reward` to pretend the environment charged the
tail during step51. Versioned objective provenance must survive optimizer,
pending-rollout, RNG and session recovery. No update may begin until all four
rollout tails close successfully. No new fit-only acceptance gate.

## Numerical Limits

| Phase | Main episodes | Prefix + tail calls | Extra calls | Optimizer calls | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input/runtime/saved-model binding | 0 | 0 | 0 | 0 | 300 |
| Same-start and recovery preflight | 18 | 1,134 | 300 | 0 | 600 |
| Window PPO, all3blocks | 96 | 6,048 | 0 | 768 | 2,250 |
| Cohort PPO, all3blocks | 96 | 6,048 | 0 | 768 | 2,250 |
| BC continuation, all3blocks | 96 | 6,048 | 0 | 384 | 1,200 |
| All-model sealing | 0 | 0 | 0 | 0 | 120 |
| Final six-controller evaluation | 216 | 13,608 | 0 | 0 | 1,620 |
| Independent raw verification | 0 | 0 | 0 | 0 | 600 |
| Payload archival | 0 | 0 | 0 | 0 | 900 |
| Supervisor closure/archive | 0 | 0 | 0 | 0 | 600 |

Totals: **522main episodes,32,886main calls +300verification calls =33,186
environment calls,1,920optimizer calls,10,800seconds globally,one attempt**.
Phase caps sum to10,440seconds; remaining global slack is not transferable to
any phase/owner. No numerical work follows a failed preflight or terminal failure.

The300extra calls include18prefix clones x4,18tail clones x4, and3full52step
original-engine prefix parity traces using each block's own-frozen preflight
requests. These three traces are explicitly budgeted additional episodes without
tails, not hidden free smoke tests. No extra policy inference is needed for them.
Allow528environment builds (3layout+522main+3parity),36restored clone instances,
6historical checkpoint loads (3references+3initializers). Initialization and
qualification optimizer budgets are0. Only existing qualified models are reused.

Per PPO block:32episodes,2,016main calls,128actor+128critic calls,750seconds.
Per BC block:32episodes,2,016main calls,128actor calls,400seconds.
Per evaluation controller/block:12episodes,756main calls,90seconds.
Both source-level and runtime budgets include unsuccessful calls before invoking
the operation. Neither rollback nor restoration refunds spent work.

## Evaluation and Interpretation

Seal all12learned artifacts before opening36new test-world starts, each evaluated
under all six controllers (216main evaluation episodes). Use the new namespace
in proposal.json and a bounded historical local seed check; no prior observed
test data are independent confirmation. Starts are paired; state-dependent RNG
consumption after divergence does not imply identical exogenous event paths.

Primary contrasts: cohort PPO minus own frozen, window PPO, and BC-CONTINUE.
Primary outcome: raw63step cost, equal weighting across the3training blocks.
For each contrast report all3block means, pooled absolute/relative differences,
and descriptive95%paired block-then-world bootstrap intervals using the same
predeclared analysis procedure and10,000draws as the base scope. The3blocks,
not216controller episodes, are the independent training replicates.

Development promotion screen: all three primary contrasts have a negative pooled
mean and negative cost difference in each of3blocks; cohort losses do not
increase in any block against any primary comparator. There is no borrowed1%
practical-effect threshold or claim that the zero loss margin is clinically
validated noninferiority. Failing any part closes this particular one-shot route.
Positive directional consistency nominates an independent confirmation study;
it does not itself establish significance, practical value or clinical benefit.

Always report original52step cost and patient outcomes, tail and post-resolution
cost components, completions/losses by cause, waiting/turnaround, resolution time,
greedy prefix changes, resource stocks/overflow and R4/MDL-2 context. Final active
must be0in every completed episode, so it is a contract check rather than a
useful ranking metric. At fixed enrollment, completions and losses are complements;
do not count their opposite signs as two independent confirmations. Waiting or
window-service worsening remains an explicit trade-off even if cost/loss screens
pass. No post-hoc endpoint, penalty, sample-size or comparator selection.

## Execution Admission and Stop

First complete the additive public-producer bridge, PPO target/recovery binding,
two-part recorder, serial campaign, independent bundle verifier and watchdog.
Fake-only tests must reject premature update/test access, modified raw rewards,
changed input support, missing tails and budget refund. Freeze implementation,
effective configs, source/runtime/input locks and fresh-stream inventory locally.
Record one exact user authorization against this scope before a unique result
root is claimed. All routine phases then proceed without repeated approvals.

Scientific failure consumes the attempt: preserve partial/failed evidence and
archives where possible, diagnose saved data, do not repair-and-retry. A null or
only diagnostic gain also closes this objective amendment; no extra epochs,
weights, algorithms or further seeds are permitted. No automatic formal follow-on,
holdout, StageE reopening, remote action, Dropbox export or Howard sign-off claim.
