# Resource Repair: Executed Engineering Regression, No New Trial

2026-10-03 UTC. Entry d0c94ee. Zhaowei requested `火速推进` immediately
after the explanation proposing a targeted predictor repair and remaining-work
preparation. This authorizes that engineering chain, not an unasked numerical
scientific retry. No Howard approval is implied.

## Exact Reproduced Cause

The saved controller/public state at real epoch47 reproduces the old failure:
candidate0, quantile0.5, forecast horizon5 (forecast epoch52). At epoch51,
production's `floor(min(ready, reagents, idle) + 1e-12)` allowed an integer
patient start without a full unit of reagent. Site3 reagent stock became
`-1.1102230246251565e-16`. All transfer requests were zero, but the transfer
loop used this negative stock as its minimum flow and wrote that value into
`reagent_transfers[0,0]`. The next public-view validator correctly rejected it.

This is now reproduced, not a roundoff hypothesis. The real saved inputs
remain valid. Native patient production already uses the strict floor in
src/env/patient_capacity_planning.py; the predictor's epsilon was inconsistent.
The prior recovery1 service-prefix fix remains intact.

## Delivered Correction

New, unregistered src/baselines/capacity_completion_control_recovery2.py:
- floor actual available stock, with no epsilon, clipping or fractional patient;
- reject negative/nonfinite stocks and invalid pipelines before transfer;
- preserve original common operation, cost, support, filter and learning rules;
- report exact field/index/epoch for any invalid internal forecast resource;
- explicit public-state migration from recovery1, separate versioned snapshots.

Legacy modules/results/configs were not edited. The original failure state was
reduced to a compressed public-only fixture, excluding environment/learner data:
tests/fixtures/capacity_resource_failure_20261003.json.gz
SHA256 09656bf842c307c2d843fc0561dcb72acdba429c7133e042061d67c2c89871ac.
Source failure-state SHA256:
2c224da4ba33c44b990b19521d0a1eba5f13c54b2e631d66ec94f6d7c24c80b0.

Eleven focused tests pass: exact saved failure, full corrected384-query decision,
sub-integer inventory/no borrowing, transfer rejection before mutation, explicit
field diagnostics, controller restoration, and the existing five horizon/work/
identity/resource contracts rebound to this new predictor. All neural fitting,
native environments and real new episodes remain zero. No performance signal
was calculated. Full compileall passed.

The first test invocation caught a fixture list/tuple restoration mismatch;
the fixture loader was corrected. That was an artificial test, not a scientific
attempt. No frozen runtime data was altered.

## Reuse And Interpretation

Hypatia delivered remaining-work.md from saved metadata/source. Poincare's
finite efficiency review advised ending diagnosis, using that one ledger, and
avoiding another review cycle. Both agents are completed and closed.

Reuse block0's sealed model,35complete trajectories and47saved partial steps.
Do not claim the old teacher decisions would all match the corrected predictor:
the rounding correction can change forecasts near integer boundaries. Preserve
historical teacher provenance and report the mixed-initializer limitation.
Within each block, the online and both frozen arms start from the exact same
sealed model. This still targets conditional online-update attribution; it is
not homogeneous re-training under one corrected initializer or confirmation.

## Single Next Decision

proposal.json defines a one-shot remainder:16,145new native steps,
6,208optimizer calls,216final evaluations;90minutes and2GiB with explicit
stage/owner limits. It includes751saved-receipt reconstruction updates and one
counted partial-environment restore. Old failed reservations stay charged.

The working saved-boundary/replay continuation entry and its runtime/input locks
are NOT yet delivered. Do not launch the old runner: it starts block0 again.
One approval would cover finishing that exact integration then freezing and
running once, without routine-stage questions. No recollection fallback,
reward changes or automatic further retry. Same automation remains PAUSED;
no experiment is currently running and no completion ETA is claimed.

