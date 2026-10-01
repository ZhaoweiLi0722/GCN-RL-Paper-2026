# S1 terminal readout: raw-verifier runtime failure

Executed2026-10-01 from af1fe201779477b71200d069ce7b446f39ee4b84,using source
db8f17e38586083aa6531bae0964880dbf773b4a and the prospectively approved packet.
**Status:failed,exit1,attempt consumed.** No repair,relaunch or retry occurred.

## Supported result

Input binding,three real prototype preflights,24 R4 demonstration episodes and
three256-update initializers completed. All12 qualification trajectories were
persisted. The independent raw reader then raised `ValueError: non-integer raw
vector` before qualification decisions or same-start forks were produced.
The failure is in execution-data verification,not a rejected performance gate.
PPO/BC continuation and all180 final evaluation episodes were never started.
No conclusion about PPO increment,clinical trade-offs or deployment adaptation
is available. The old A1 failed1/9 result remains unchanged.

The trace is saved in
`results/dynamic_candidate_pilot_20261001/launcher/failure.json`:
`dynamic_candidate_campaign._qualify` -> `read_dynamic_raw_episodes` -> `_enrich`
at dynamic_candidate_verification.py:164 -> integer `vector` at
candidate_pilot_verification.py:51. The failed qualification report was not
written; absence must not be read as either passing or failing competence.

## Independently located cause

Fermat's finite read-only diagnosis identified the first reached offending
value in
`payload/episodes/qualification/block60/r4/world00/events.jsonl`,line1:
`event.info.capacity_transfers[0] = 3.0000019669532776`. The coordinator read
that same persisted vector. This is not merely a nearly-integer tolerance
issue: the earliest prototype row also hascapacity_transfers[5] =
-8.600000560283661. Fractional reagent transfer and replenishment values are
present too. Rounding only capacity would both distort evidence and leave
the same contract defect in other channels.

The simulator's contract is channel-specific: specimen/patient lots are
integer counts; capacity and reagent transfers are signed continuous facility
net flows; replenishment is nonnegative continuous quantity. See
`src/env/patient_capacity_planning.py:300` and
`src/env/capacity_planning.py:1999`. The verifier incorrectly applies the
same integer-only vector helper to all four executed-resource channels.
The sampled row's recorded capacity equals its continuous requested action
times the configured18-unit scale,with a zero net transfer sum. Relevant
source files matched their frozen execution copies in the independent review.
The observed fractional values are valid under that simulator contract;
this check does not establish full raw-patient accounting or qualification.

This is an implementation integration defect in our reader. The155 passing
engineering tests did not catch the actual simulator/reader channel mismatch.
The remedy must preserve integer checks on patient/specimen counts and use
finite,dimension,sign and flow-accounting checks appropriate to continuous
resources. A regression should use the saved real fractional rows without
executing a new environment. No remedy was applied to the frozen attempt.

## Actual consumption

| Item | Completed/persisted evidence |
| --- | --- |
| Prototype | 3 full52-step episodes and12 cloned calls |
| Demonstrations | 24 full52-step episodes |
| Initializers | 3 saved final states,256 actor steps each |
| Qualification | 12 full52-step trajectories; decision not reached |
| Raw episode rows | 2,028 across39 completed episode receipts |
| Durable environment debits | 2,040 =2,028 trajectory+12 clone |
| Durable optimizer debits | 768 actor,0 critic |
| Continued PPO/BC training | 0 |
| Final evaluation | 0/180 |
| Closed scopes | 6/33; qualification active at failure |
| Supervisor elapsed | 288.88377199997194seconds |

Ledger sequence count2,822,last SHA256
`73cba36d83d41b537e9826362040a067f869361dd0d163f8c2bb218b6b9d7ad9`.
Independent ledger reading reconciled owner counts and39 raw episode files,
each with52 rows. Debits are pre-execution accounting; completed episode/phase
receipts supply additional return evidence. The qualification failure does not
refund its calls or grant the remaining budget for a new attempt.

All768 persisted initialization cross-entropies and gradient norms are finite.
First/last minibatch losses were1.728842/1.324039,1.743792/1.331886 and
1.701023/1.190234 for blocks60/61/62. These are learning diagnostics,not paired
patient-performance claims. Raw-verifier failure prevents acceptance.

## Closure and preservation

Child42704 and supervisor42349 exited; an approved read-only host command scan
found no remaining matching process. Scientific exec session25711 ended exit1.
The supervisor reports incomplete scopes,not timeout; forced_kill=false.
stderr contains0bytes but failure.json records the real exception and trace.
A35,213,901-byte failure-state payload was saved without a preservation error.

The complete result root was archived locally without overwriting any evidence:
606 files,141,125,221bytes,SHA256
`b1e1ffd33ca659d9258b88a892ff61db3dfdd01ae96ac7897901f69e41a6a904`.
Every archive member and the unchanged original inventory were verified.
Receipt:`terminal-preservation.json`. This post-failure preservation made no
environment/model/optimizer calls. No Dropbox copy or collaborator access is
claimed. Only gcn-rl-s1 was deleted; receipt:`s1-monitor-closure.json`.

## Next decision boundary

The exact saved-field cause is now located. Propose a separate reader-repair
and saved-artifact qualification-only package,with no new trajectories,fitting,
PPO continuation or final evaluation. Do not round raw values,relax integer
patient counts or mutate historical results to make verification pass.

The minimum candidate recovery boundary is after qualification collection,
before `_qualify()`. Three saved initialization files and
`launcher/recovery/000027-collection-block-complete.pt` exist. Their contents
and full restorable ownership have not been loaded or validated in this
diagnosis. A blind `advance()` is not a valid recovery: the collection cursor
has already exhausted its descriptors,and failure-state restoration is
explicitly prohibited. Saved-input qualification additionally needs frozen
model forwards,not only arithmetic. That bounded work must be described and
authorized prospectively; qualification remains unknown. Any later training
continuation requires a separate source-bound decision,not an implicit resume.

The remaining priority is reaching the actual PPO/frozen/BC comparison,not
adding another unrelated artificial learning campaign. No new execution scope
is approved by this readout.
