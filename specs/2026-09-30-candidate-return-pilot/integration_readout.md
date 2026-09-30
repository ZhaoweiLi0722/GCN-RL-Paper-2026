# P1 integration checkpoint

Updated 2026-09-30T16:02Z. Base `c6cde24d8b332935f250c3b66c24e3f4cdf61c92`,
persistent local integration branch. Engineering/readiness only, not a scientific
result. No real patient constructor/reset/step, scientific initialization/fit,
qualification or evaluation has run. P1 output root is absent. Original proposal
and protocol remain byte-identical.

## Implemented and tested

- `src/rl/candidate_imitation.py`: opt-in reference-class cross-entropy
  initialization and continued imitation. Fixed replacement batches for
  initialization; whole shuffled epochs for continuation; private RNGs, split
  rejection, consumed-example lineage, capped Adam/gradient clipping, unchanged
  value-head weights, atomic publication and optimizer/RNG restoration. Shared
  encoder updates may change value predictions despite no value-head fitting.
- `src/rl/candidate_patient_session.py`: public-state-only reference/options,
  original float64 requests, raw absolute-cost/route/identity receipts, explicit
  finite horizon and whole collector/environment/kernel checkpoints. Restoration
  checks candidate reconstruction, old probabilities, decisions, cost accounting,
  boundary/lineage and exact private sampler advancement. Only complete sampled
  training episodes can become PPO segments.
- Full MDL-2 baseline receipts use MDL-2 as reference, so a same-class R4 alias
  cannot replace its original action. This baseline-only context is forbidden
  in training/demo/qualification collection; learned-policy support/canonical
  representatives remain unchanged. These are request classes, not certified
  equal-outcome or physically feasible classes.
- `src/rl/candidate_pilot_resources.py`: prospective numeric streams and an
  exclusive append-only/fsynced/hash-chained ledger. Debit before each simulator
  or optimizer call. Global, phase and model caps are independent. Rollback cannot
  refund calls; failed writes poison the owner. No resume/refund/reset API.
  Independent ledger reading checks chain, clocks, scope order, counts and caps.
- PPO adds a pre-optimizer-step callback for external durable accounting. Its
  objective/defaults and whole-update rollback remain unchanged.

This is not yet the complete campaign orchestrator. Phase scheduling,
four-episode update-boundary bundles with external ledger/cursor, final-model
seals, an outer wall-clock watchdog, raw outcome reporting and an independent
outcome verifier still need implementation and integrated tests. Budget checks
at operations/boundaries do not interrupt a hung native call; do not claim the
six-hour watchdog exists yet.

## Validation and engineering failures

31 new invented-fixture tests plus the previous199: **230 tests pass in6.425s**
on the final rerun (the earlier complete run passed in6.028s).
Full repository `compileall -q .` exits0. Python3.9.6, PyTorch2.8.0. Session tests
reject patient construction/reset and replace step/state methods with invented
accounting. Optimizer tests use tiny invented tensors and strict update bounds.

Checks include independent first-Adam-step CE arithmetic, value-head preservation,
test/qualification leakage rejection, epoch coverage, exact next fit/sample,
mid-episode restoration through the next PPO update, corrupted checkpoint rejection,
model/RNG/cursor drift, baseline action preservation and irreversible external
budget debits. Global RNG states remain unchanged.

Two failures were resolved before any scientific run:
1. First10-test session suite:1 error from PyTorch rejecting an extensionless
   dot-prefixed temporary checkpoint name. Persistent-directory staging now uses
   a `.pt` suffix. No-overwrite, failed-publication cleanup and roundtrip pass.
2. First read-only numeric audit stopped at preserved truncated `diagnosis.json`
   and its archive copy. Both are183-byte aggregate-score prefixes, SHA256
   `5713c90bbf396d3448447cfab2de9a5ca7626f0bb43795d1df4f0170905473ff`.
   These inspected non-seed files are explicitly hash-locked exclusions; complete
   `diagnosis.v2.json` is scanned. Unknown malformed files still fail. Old files
   were not repaired, and this was not a P1 scientific attempt/retry.

## Input and stream audit

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' \
  -m experiments.scripts.audit_candidate_pilot_readiness \
  --output reports/2026-09-30-candidate-pilot-integration/readiness.json
```

The saved output is no-overwrite; later audits need distinct output names.
Seven R4 policy/config/archive-manifest locks pass; all three environment dicts
match.174 environment starts,37 neural streams and3 initialization seeds have no
cross-purpose collisions. Numeric values, including decimal strings, were checked
against **888** historical tracked JSON configs and local result/report JSON/JSONL
files: **zero collisions**. Every scanned path/hash and both exclusions are saved.
Unavailable external evidence is not claimed scanned. Eight earlier preservation
fingerprints in the research-direction memo remain unchanged.

## Remaining finite work

Approval is recorded in `authorization.md`; no additional routine-step approval
is needed for this exact scope. Complete orchestrator/verifier and fake integration
tests, commit source/effective execution config, then perform only the single
budgeted preflight/attempt. Scientific terminal failure closes the attempt without
repair/relaunch. No reward/scenario/threshold/count/time-cap changes are permitted.

New in-thread heartbeat `gcn-rl-p1` is confirmed ACTIVE with saved cadence every15min,
finite remaining P1 chain only. Prior `gcn-rl` remains deleted. A heartbeat is not
an experiment process. Approved host scan returned only its own shell/filter;
test/compile/audit sessions ended. No Dropbox copy, cloud sync, collaborator access,
remote Git, Howard sign-off, formal holdout or Stage E reopening is claimed.
