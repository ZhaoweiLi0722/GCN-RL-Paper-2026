# One-step rollout-based policy improvement: prospective package

Status: local preparation only. No scope-specific execution approval, source
freeze, scientific forward, environment call or optimizer update has occurred.
Config: `experiments/configs/paired_cohort_improvement_20261002.json`.
This is not a continuation or retry of any closed attempt.

## Decision and hypothesis

The completed 216-world cohort comparison found identical greedy requests and
outcomes for cohort PPO, frozen, window PPO, BC and R4. The earlier saved-data
continuation diagnostic already showed broad non-reference sampling and changed
probabilities without a changed greedy winner. These results do not establish
absent headroom, an entropy failure, bad reward weights, or global optimality.

Stop further PPO baseline/horizon adjustments in this restricted recipe. Test
one different mechanism: learn directly from the complete remaining cost of
each available request at the same state, under a fixed reference continuation.
This removes learned critic ranking and sampled-action credit assignment from
the training target. It does NOT ensure low-variance labels, representative
contexts or better greedy closed-loop performance. Those remain empirical risks.

This is model-assisted one-step policy improvement, conceptually related to
[rollout-based approximate policy iteration](https://aaai.org/papers/icml03-057-reinforcement-learning-as-classification-leveraging-modern-classifiers/),
not a claim to reproduce that paper's algorithm or guarantees. It is neither
model-free PPO/DDPG nor deployed online parameter adaptation. The comparison
tests a replacement learning package, not the isolated effect of pairing.

## Fixed scientific surface

- Reuse the three qualified graph initializers, blocks 60/61/62, from the
  initializer records in the closed cohort-objective frozen packet. No refit,
  qualification rerun, architecture modification or graph ablation in this pilot.
- Preserve 20-facility public inputs, specimen-route message graph, exact
  candidate-generation contract, maximum six original requests and canonical
  deduplication. Submit the original float64 representative request. A request
  class is not proof of realized-action equivalence or physical feasibility;
  keep existing projection/blocking and record realized effects.
- Preserve original costs, patient dynamics, 52-step enrollment and the existing
  11-step fixed accounting tail. No new weight, penalty, salvage, patient scenario,
  optimizer search or entropy search. All evaluation controllers get the same tail.
- Policies see only existing public inputs. Simulator state is available to the
  TRAINING label generator, not the actor or test-time action selector. This extra
  simulator access and its cost must be disclosed.

## Training data and targets

Collect four fresh complete R4 cohorts per block. Save full environment and
public-observation snapshots immediately after prefix steps 4, 20 and 36,
before the next decision. Thus 12 states/block and 36 overall, from only twelve
context worlds; they are not 36 independent training replications.

At every snapshot enumerate all unique request classes, without outcome-based
filtering. For each of two prespecified future-RNG replications, clone the same
full current state for every candidate. Replace ONLY the environment generator's
future RNG state from a fresh recorded seed; retain current patient attributes,
in-flight jobs, hidden conditions and accumulated counters. This estimates a
conditional continuation value at the saved latent state; it does not integrate
over all hidden states compatible with the public observation. Same future seed
within a replication is a coupling, not a guarantee of event-aligned CRN after
action-dependent consumption diverges. Preserve that limitation in all results.

Apply the candidate once, then fixed deterministic R4 to step52, then the exact
existing closed-cohort MDL-2 tail to step63. Charge/read raw primitive costs only
from the branch start onward, including the first action and each tail charge
once. Never append already sunk cost. Save every branch's original requests,
realized actions, primitive components, patients, closure, RNG and provenance.

For state s and class k, c(s,k) is the two-replication mean remaining cost;
b(s) is the reference-class cost. Minimize

`L = mean_s sum_k softmax(logits(s))[k] * (c(s,k) - b(s)) / 1e9`.

Subtract and accumulate raw costs in float64. The CPU float32 actor logits are
differentiably promoted to float64 for this loss; never round labels to float32
before subtracting the shared large cost. Labels are detached, each state has
equal weight, and there is no entropy bonus or critic loss. Subtracting b(s)
does not change the exact gradient; enumerating action-conditioned costs is the
substantive change, not another state-baseline proposal.

For each block, fork paired-cost and BC actors from the same initializer with
fresh empty Adam state. Exactly 128 full-batch actor updates per arm, all12states
every update, Adam lr3e-4/betas0.9,0.999/eps1e-8/weight_decay0/gradientcap0.5.
BC uses the same public states/support but negative log probability of the R4
class. Neither uses held-out worlds or trains a critic. Keep reference bias
trainable, do not manually move it. No action-margin threshold or adaptive scale.
No data refresh, extra epoch or cherry-picked best checkpoint. Equal costs imply
no expected-cost gradient, not permission for more labels or another attempt.

## Complete comparison

Seal all12 learned artifacts before opening fresh test worlds: paired-cost,
continued-BC, unchanged own-frozen, and the saved cohort-PPO model per block.
Also evaluate R4 and full MDL-2. Use twelve fresh paired worlds/block and the
unchanged deterministic greedy class/tie rule: 216 full63step evaluations.
Old test worlds are not reused as independent confirmation.

Primary comparisons are paired-cost minus own-frozen and minus BC. Saved PPO,
R4 and MDL-2 are secondary context. Saved PPO had different training data and
compute, so this is NOT an equal-budget PPO-superiority experiment. Likewise this
graph-only pilot cannot establish an isolated GCN effect, general joint-resource
optimization, calibrated clinical utility, or deployment online adaptation.

Independently aggregate primitive raw costs and final patient outcomes. Report
each block and equal-block means, all loss reasons, completions, waiting,
turnaround, 52-step versus tail costs, resource stocks, requested/executed action
differences and simulator/optimizer/time costs. Preserve adverse trade-offs.
Use the existing paired block-then-world 10,000-draw descriptive95% intervals
with a fresh analysis seed; three training blocks are weak inferential support.

The development nomination screen is unchanged from the closed cohort recipe:
strictly lower mean cost in all3blocks for BOTH primary comparisons, and no
observed increase in cohort patient losses in any block for either comparison.
Report waiting/expiry deterioration even if this screen passes. This is not a
clinical noninferiority test or publication claim. Passing only nominates a
separately approved confirmation; nothing follows automatically. No favorable
secondary contrast can replace a primary null. If greedy behavior remains
unchanged, close this one-shot mechanism without another baseline/epoch variant.

## Single-attempt budget

All counts are hard caps, charged before work; aliases do not release budget for
new contexts. Every clone step, preflight and failure counts. Sequential execution.

| Phase | Environment calls | Actor updates | Seconds |
| --- | ---: | ---: | ---: |
| Input/runtime binding | 0 | 0 | 300 |
| Three full R4 preflight originals and exact same-start clones | 378 | 0 | 600 |
| Twelve reference context cohorts | 756 | 0 | 600 |
| Up to432 candidate/future branches | 18,576 | 0 | 3,600 |
| Three paired-cost actors | 0 | 384 | 1,200 |
| Three BC actors | 0 | 384 | 600 |
| Seal all models | 0 | 0 | 120 |
| 216 final evaluations | 13,608 | 0 | 4,320 |
| Independent raw comparison | 0 | 0 | 600 |
| Local verified archive | 0 | 0 | 1,200 |
| Closure | 0 | 0 | 600 |

Total:33,318 environment calls,768actor/0critic updates,435clone instances,
234fresh environment builds (including three layout builds),nine historical
model/reference loads. Branch formula is
`3*4*2*6*((63-4)+(63-20)+(63-36)) = 18576`.
Phase caps total13,740s; global14,400s. Per-block phase caps and240s per
controller/block evaluation cap are in config, with no cross-phase transfers.
Reuse existing correctly charged archive/watchdog/budget mechanisms. Wall-clock
limits are upper bounds, not completion-time promises.

Fresh stream namespace is in config. Bind3preflight,12context,72conditionalfuture,
36test,6neural and1analysis seeds before any scientific work, audit their local
historical overlap once, commit exact values and hashes. Layouts and immutable
initializers use the existing pinned inputs. No unavailable external seed-audit
claim. No model fitting, test scoring or environment preflight during preparation.

## Execution and failure boundary

Before any science: obtain one explicit approval for this complete numerical
package; finish versioned branch collector/update/serial/raw-verifier integration;
test actual persisted metadata through the real entry with fake backends and
zero optimizer steps; run focused tests/compileall; commit source/config/input/
runtime/authorization locks and create an exclusive new result root. Reuse prior
valid tests; no new toy-fitting readiness campaign. Do not run a partial package.

Preflight verifies exact restore/R4 continuation using one already budgeted
original/clone pair per block. This is implementation parity, not a performance
gate. Failure, nonfinite values, missing provenance, limit overrun or restore
mismatch terminates this attempt. Preserve all spent calls/partial outputs; no
automatic repair-and-retry. Correct mock failures BEFORE execution only. Persist
optimizer/global/local RNG, cursor, plan hashes and append-only budget receipts;
restoration cannot refund costs or reopen a terminal attempt.

Archive versioned local artifacts with per-file hashes/readback, preserving old
results and sealed models. No Dropbox/export/remote actions/holdout/StageE change
or coauthor approval assertion. Completion of this package is not permission
for reward search, new scenarios, graph ablations or a second round.
