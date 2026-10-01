# Readiness decision before new optimization

2026-10-01. The artificial sampled-return diagnostic has an implementation,
bounded serial runner and independent saved-artifact verifier. The reward and
dynamic-request source audits are complete. No new optimization or patient
experiment has run. There is no new performance estimate.

## Current gates

| Gate | State | Evidence or remaining requirement |
| --- | --- | --- |
| Actor can rank an exact-value invented task | Previously passed | Closed actor-positive-control packet; not sampled RL or patient evidence |
| Sampled-return learning implementation | Implemented and tested without fitting | New core, campaign, scalar/packet verifiers and mocked serial test |
| Sampled-return learning result | Not run | Explicit bounded approval, then approval/source/runtime freeze and one run |
| Dynamic request interface | Source and mock audit complete | `candidate-contract-readout.md`; new integrated scorer still absent |
| Reward arithmetic | Existing saved audit passes; inspected source unchanged | `reward-decision.md` and `review-inputs.json` |
| Full-lifecycle objective | Not established | Current objective stops at 52 steps; unfinished obligations separately reported |
| New patient pilot | Not ready or approved | Must first resolve the learning gate and actual integrated-policy qualification |

## Exact next decision

The pending numerical question remains the same: one artificial-only packet,
9 fits, at most 128 actor and 128 independent critic calls per fit, 2,304 total,
13,824 invented training action observations, CPU float32, one configuration,
30 numerical minutes, one attempt. No patient model, patient episode, reward
change, tuning, retry or scope expansion. Prior approvals for other packets do
not authorize this one. A previous question already presents this scope; do
not repeatedly ask it while its answer is pending.

If approved, bind the completed implementation commit and all input/runtime
hashes in a separate committed authorization, recheck processes and immutable
evidence, then launch exactly once. Preserve all nine results, including a null
or failed gate. Independently verify the saved packet and locally archive it.
The artificial numerical work cap is 30 minutes; verification and preservation
are additional work and no end-to-end delivery time is guaranteed.

If any required artificial gate fails, close that attempt and diagnose saved
evidence only. Do not adjust the reward or create extra fits automatically.
If it passes, the next decision is a bounded dynamic-policy engineering and
qualification package, not immediate patient training. Any patient pilot then
requires all initialization/preflight/clone/training/evaluation steps to have
explicit caps; the draft 19,344 training/evaluation steps alone is not a complete
launch budget.

## Implications for the paper

Keep the existing graph/distillation evidence and report P2's zero incremental
greedy-policy effect honestly. The present diagnosis supports a specific
implementation limitation, not global optimality or impossibility of RL.
Neither mock acceptance nor an eventual artificial pass supplies the missing
patient RL contribution. Keep patient losses, completions and unfinished work
visible alongside raw total cost in the next prospective comparison.

Implementation/protocol/config freeze:
`2bf2403d346c708338a7ee20a3a6ee8bf190589e`. The 121-test no-fitting suite and
full repository compileall pass. The app confirmed deletion of this finite
preparation schedule on 2026-10-01 at 17:52 UTC; see `closure-receipt.json`.
Only approval-gated work remains. No new experiments or repeated waiting
notifications should be created. No remote action, Dropbox export, formal
holdout use, Howard sign-off or Stage E reopening is implied.
