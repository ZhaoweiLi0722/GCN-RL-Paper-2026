# Paired-cohort terminal readout and remaining-work decision

Date: 2026-10-02. Status: failed, consumed attempt; saved-data handoff only.
Run: `results/paired_cohort_improvement_20261002`.
The approved original package is recorded by intent commit `e617f1f`, frozen
packet commit `dfa9b9b`, and authorization commit `3da1a09`. The original
proposal/protocol preparation wording remains a historical snapshot. This
readout neither changes those locks nor authorizes another run.

## Runtime cause and responsibility

`launcher/supervisor.json` records `reason=wall_clock_deadline`, child exit
`-15` (termination signal), no forced kill, and elapsed global time
1,437.574030 seconds against 14,400 seconds. The active owner was
`paired_branches/block60`, whose separate hard cap was 1,200 seconds.
Its ledger begin clock was 951837.238675; the enforced deadline was
953037.238675. The global budget was not exhausted and could not be borrowed
to extend that owner. `launcher/branch-failure-block60.json` records the
wall-clock `TimeoutError`, 117 completed branches, and no refund.
`launcher/terminal.json` correctly records failed and scientific completion
not verified. Both stdout and stderr are zero bytes; silence is not success.
The parent separately confirmed launcher PID 92429 and child PID 92443 absent.

The per-block 20-minute estimate was too tight. That is my execution-budget
planning error, not evidence that reinforcement learning, the reward, or the
paired-cost mechanism failed. The observed complete-branch pace implies about
100 additional seconds for the unfinished block-60 matrix. No optimizer update
or final six-controller comparison had started when the owner deadline stopped
the run. The existing terminal attempt stays closed; unused caps are not launch
authority or transferable compute credit.

## Saved work and exact schedule

The parent independently verified 117 complete branches and 819 file references
without replay. Its report is
`reports/2026-10-02-paired-cohort-closure/saved-branch-verification.json`, SHA256
`5074c0dc7445206e6db309841bf0d22fc4d35795fb42614ebba41557aa252fc6`.
This readout reuses that validation; it does not add another raw-validation gate.

All twelve reference cohorts and three same-start original/clone preflights
completed. The 36 saved context envelopes under
`payload/contexts/block{b}/cohort{c}/after{t}.pt` were read as data with
`weights_only=True`, their existing envelope/context digests checked, and their
canonical request classes reconstructed by the existing pure context validator.
No neural-weight checkpoint, model constructor, forward pass, environment
construction/restore/step, or optimizer was used in this handoff.

Every context has six original requests, but deduplication yields five or six
canonical classes. The exact class counts are:

| Block | Cohort | After 4 | After 20 | After 36 | Branches, both futures | Calls |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | 0 | 5 | 6 | 6 | 34 | 1,430 |
| 60 | 1 | 5 | 5 | 5 | 30 | 1,290 |
| 60 | 2 | 5 | 5 | 6 | 32 | 1,344 |
| 60 | 3 | 5 | 5 | 6 | 32 | 1,344 |
| 61 | 0 | 6 | 6 | 6 | 36 | 1,548 |
| 61 | 1 | 6 | 6 | 6 | 36 | 1,548 |
| 61 | 2 | 6 | 5 | 6 | 34 | 1,462 |
| 61 | 3 | 6 | 5 | 6 | 34 | 1,462 |
| 62 | 0 | 5 | 6 | 6 | 34 | 1,430 |
| 62 | 1 | 6 | 6 | 5 | 34 | 1,494 |
| 62 | 2 | 5 | 6 | 6 | 34 | 1,430 |
| 62 | 3 | 5 | 6 | 5 | 32 | 1,376 |

Canonical execution order is block 60,61,62; cohort 0,1,2,3; snapshot
4,20,36; replication 0,1; then canonical class index 0 through K-1.
Each branch uses its frozen conditional-future seed and takes `63-t` calls:
59,43,27 respectively, including the unchanged eleven-step tail. Thus the
actual complete matrix is 402 branches / 17,158 calls, not the maximum
432 / 18,576. Deduplicated aliases do not permit replacement samples.

| Block | Planned branches / calls | Complete branches / calls | Remaining branches / new calls |
| --- | --- | --- | --- |
| 60 | 128 / 5,408 | 117 / 5,111 | 11 / 297 |
| 61 | 140 / 6,020 | 0 / 0 | 140 / 6,020 |
| 62 | 134 / 5,730 | 0 / 0 | 134 / 5,730 |
| Total | 402 / 17,158 | 117 / 5,111 | 285 / 12,047 |

The 117 receipts are exactly the first 117 canonical schedule entries, ending
at `block60/cohort3/after36/rep0/class0`. For block 60 the remaining entries are
replication 0 classes 1-5, then replication 1 classes 0-5, all at that context.
All scheduled entries in blocks 61 and 62 remain.

## Debit reconciliation and interrupted state

The saved append-only ledger's 6,246 environment debits reconcile exactly:

| Work | Calls |
| --- | --- |
| Three preflight originals and clones | 378 |
| Twelve complete reference cohorts | 756 |
| 117 complete branches | 5,111 |
| Interrupted branch, already charged | 1 |
| Total | 6,246 |

The ledger has zero optimizer debits. Its branch total is 5,112, with 118
conditional branch clones admitted: 117 complete and one interrupted.

Interrupted branch:
`payload/branches/block60/cohort3/after36/rep0/class1`.
Its future seed is `231525053205570908202738518863346282227`, from the unchanged
replication-0 stream. Its only files are `header.json`, `initial.pt`, and
`events.jsonl`. The raw log has one complete row: branch step 0, absolute step
36. That row's before-state hash matches the saved initial-state token.

The 951,811-byte `initial.pt` is a readable checksum-valid
`paired-cohort-branch-state-v1` envelope with zero rows, no failure marker,
and a manifest matching the header. It is a valid initial data envelope, NOT
a post-step resume checkpoint. There is no `final.pt`, `states.json`,
`receipt.json`, `verified.json`, `launcher/branch-failure-block60.pt`, or
leftover temporary `candidate-state-*.pt`. Therefore no saved full state after
the charged step is available for a 26-call continuation. The one raw row is
failure evidence, not a complete label or sufficient full-state checkpoint.

The proposed new run must recompute this one interrupted branch from its
original context, same request and same future seed, for all 27 calls.
It must preserve the old partial row and debit without refund and exclude that
partial row from the assembled complete-label matrix. Of the 285 remaining
branches, 284 were never started and only this one is a recomputation.

## One bounded remaining-work package

Decision: propose one separately approved, new-result-root, reuse-only completion
package; do not restart the full campaign and do not reopen the failed root.
Reuse the completed preflights, twelve reference cohorts, all 36 context states,
and the 117 independently validated branches. No new reference cohort, training
context, preflight episode, replacement sample, or completed-branch replay is
proposed. Keep the same rewards, models/initializers, candidate rules, reference
continuation, seeds, samples, losses, greedy rule, and primary decision screen.

The remaining numerical work is exactly:

- 285 branch instances / 12,047 environment calls, including the one full
  interrupted-branch recomputation. Assemble all 402 complete labels' branches
  for the original 36 contexts; do not fit on the partial 117-branch subset.
- Six original actor fits, 128 updates each: 768 actor and zero critic updates.
  No fit has already consumed an optimizer update. Preserve the original
  paired-cost and BC objectives, optimizers, and final-checkpoint rule.
- Twelve model seals and the unchanged 216 paired 63-step evaluations:
  13,608 environment calls. No extra test world or controller is introduced.
- The unchanged independent raw comparison and descriptive 10,000-draw
  hierarchical bootstrap with its already frozen analysis seed, then ordinary
  local archive/closure. No follow-on confirmation is included.

New environment calls total **25,655 = 12,047 + 13,608**. Including the consumed
failed attempt gives **31,901 = 6,246 + 25,655** cumulative charged calls.
The deduplicated original unique-work schedule is 31,900; the one-call
difference is the preserved failed debit repeated within the 27-call restart.
Neither `33,318 - 6,246` nor a 26-call partial-branch continuation is the correct
new-work count. No new calls were made to obtain these figures.

### Proposed finite time allowance

Observed consecutive clone-admission timestamps cover 1,197.221418 seconds for
the 117 complete branches, including their intervening recording/verification.
Mean time per branch by snapshot 4/20/36 was 11.071 / 10.423 / 9.047 seconds.
Applying those rates to the exact remaining schedule estimates 99.513 seconds
for block 60, 1,424.241 for block 61, and 1,363.318 for block 62: 2,887.072
seconds, about 48.1 minutes of remaining branch collection. These are
extrapolations from block 60, not measurements of blocks 61/62 or guarantees.

Propose these hard caps for the ONE new package, with no automatic extension:

| Remaining phase | Proposed seconds | Owner allocation |
| --- | --- | --- |
| Binding and saved-work import | 300 | No scientific environment step |
| Remaining branches | 5,400 | Block 60: 600; block 61: 2,400; block 62: 2,400 |
| Paired-cost fits | 1,200 | 400 per block, unchanged |
| BC fits | 600 | 200 per block, unchanged |
| Twelve seals | 120 | Unchanged |
| 216 evaluations | 6,480 | 360 per controller/block; 18 owners |
| Raw comparison | 600 | Unchanged |
| Local archive | 1,200 | Unchanged |
| Closure | 600 | Unchanged |
| Phase total | 16,500 | 4 hours 35 minutes |
| Global hard cap | 18,000 | 5 hours, including boundary/supervisor overhead |

The 90-minute branch allowance is about 1.87 times the observed-rate estimate;
each unstarted full block receives 40 minutes rather than 20. Reference
collection took 126.980 seconds for 756 calls, while the preflight phase took
95.819 seconds for 378 calls. Their throughputs would imply roughly 38-58
minutes for 13,608 calls, but they do not benchmark all six evaluation roles.
The proposed 108-minute evaluation allowance and six minutes per role/block
provide explicit margin rather than retaining the previous four-minute owners.
Actor-fit and import timings remain unmeasured in this failed run; their caps
are limits, not completion promises. These time changes expand no numerical
sample or scientific hypothesis and confer no execution authority.

## Known readiness and inference boundary

Known: all contexts are readable, digest-bound data envelopes with canonical
support; the interrupted initial envelope is readable and bound; the parent
has verified all 117 complete branches; preflights previously passed; the exact
remaining schedule and nonrefundable charges are determined.

Unknown: this handoff has not instantiated environments or models to demonstrate
live restoration of these particular persisted context states through a new
authorized runner, nor demonstrated end-to-end saved-work import and completion.
Envelope validity is not a live restore guarantee or an assertion that the
existing single-attempt entry can be relaunched as a resume command. No recovery
implementation, repair, test campaign, new audit chain, freeze, launch, or retry
is performed or authorized here. The parent owns the existing failed-run
archive, workflow/Live update, and automation pause; this memo does not claim
their completion. One new explicit approval must cover the complete numerical
and time package before any new scientific work.

Scientifically, this stop supplies runtime and saved-work evidence, not a
negative RL result. The 117 branches are incomplete training-label coverage
conditional on saved latent states and fixed reference continuation. They are
not independent test patients, a learned-policy comparison, or evidence of
clinical benefit/harm. There were zero actor updates and zero final evaluations,
so the primary development screen is unassessed, not failed. Do not select
favorable branches or infer absent headroom, bad reward weights, optimality,
clinical noninferiority, an isolated GCN effect, deployment adaptation, or
publication readiness. Even a completed recovery would retain the original
three-block descriptive-inference limits and twelve-context-world dependence.
