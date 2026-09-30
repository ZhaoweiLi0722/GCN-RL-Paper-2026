# P1 terminal preflight failure: no performance result

2026-09-30T17:16Z. The single authorized P1 attempt is **closed, failed**.
No automatic repair, retry, resume, reward change or additional experiment ran.

## Frozen execution and actual outcome

- Implementation: `e1de58ab1b358ff41cb4a37c26babfac88ecff4b`.
- Execution HEAD: `3de710bbb84cf200101d8ba8d22e864145bce680`.
- Effective config SHA256:
  `e1090015d1f963f071f2bfbb2a8e76b7399e0545d13db894262438cad9a0b8ba`.
- Supervisor78259, child78540, child PPID78259. Child exited1 in4.9169s;
  no forced termination. Post-exit host scan found only its shell/filter.
- The real factory constructed the first preflight environment and loaded the
  strict frozen reference, then observation-producer construction rejected it.
  This is not zero environment construction: an initial reset occurred. However,
  **environment step calls0, optimizer steps0, completed preflight episodes0,
  demonstration/qualification/training/test episodes0**.
- Budget ledger has exactly2 records (claim and preflight begin), no debits.
  Last seal `a05519006bd3073d0a99ff8fbc865e4c34e0a055bb29050cedfaefd19201afbe`.
- No cost/clinical comparison or positive/negative RL performance estimate exists.
  It is invalid to label this attempt an ineffective PPO or reward-function result.

Raw evidence root: `results/candidate_return_pilot_20260930`.
Authoritative records: `launcher/failure.json`, `failure-state.pt`,
`budget.jsonl`, `claim.json`, `child.json`, `supervisor.json`, `terminal.json`.
Stderr is empty, but stdout and failure.json contain the caught traceback;
empty stderr must not be described as a clean successful experiment.

## Binding compatibility failure

`PatientObservationProducer.__init__` in `src/rl/patient_replay_collector.py`
rejects any `include_central_capacity_hub=True` configuration. All three exact
R4 effective configs set that field totrue. The failed check's combined message
is `overtime, hub and procurement layouts are not supported`; the observed
binding field is the central capacity hub, not evidence of overtime being on.

This conflict was inspectable from the already locked JSON files and should
have been caught in pre-execution static compatibility checking. The miniature
fixtures intentionally had no hub, so their passing271 tests did not establish
compatibility with this full R4 configuration. The preparation missed this
known-layout coverage, although the real preflight correctly failed closed.

Read-only source diagnosis: the existing raw facility observation and the
environment's graph_observation are different interfaces. The latter appends
a derived hub node/indicator and hub capacity edges. Simply switching off the
hub in the inherited config would change the declared reference environment;
simply removing the guard without checking topology, anchor equivalence and
information parity would be unvalidated. Neither was done.

## Separate timing defect discovered during read-only diagnosis

A two-second artificial parent/child clock probe, with no environment or fitting,
returned parent monotonic0.014125125 before waiting and2.058593708 afterward,
while the newly spawned child returned0.012836708. Wall timestamps agreed within
about2ms (parent1790788499.344533, child1790788499.342473).
Thus this Python3.9.6 runtime's observed monotonic origins are not directly
comparable between processes. The supervisor currently compares child ledger
deadlines with its own monotonic values. Its cross-process timing must be fixed
and tested with differently aged processes before a future run. This did NOT
cause this attempt's failure: the child exited on the hub guard after4.9s, with
zero steps, and the supervisor reported child_failed, not timeout.
No runtime/source fix or further scientific launch occurred after closure.

## Preservation and handoff

The complete15-file failure tree, including the source Git bundle and original
locks, was archived without changing the failed root. Every archive member was
read and compared to its source hash:

`results/candidate_return_pilot_20260930_failure_archive/failed-attempt-3de710b.tar.gz`

Archive SHA256 `3d29ccdff06ff52cb7b2fe1280f0ec5015d7fef252101fca3522da06ee3c877c`,
80,491,902 bytes. Per-file manifest and `local-backup-receipt.json` are alongside.
Archive and manifest were copied byte-verified into the approved Dropbox-local
P1 folder. **Cloud synchronization and Howard access are unverified.** No shared
permissions, remote Git, messages, formal holdout or Stage E status changed.

The finite `gcn-rl-p1` automation was deleted; the app confirmed
`deleteStatus=deleted`. It must not keep retrying or waiting indefinitely.

One new decision is required: authorize P1-R1 compatibility/timing repair,
invented-fixture acceptance and at most one new bounded attempt, only if the
original environment, reward, information, architecture/parameter boundaries,
95% gate, training/evaluation counts and all budgets can remain unchanged.
Preserve P1, use a new output root/committed recovery packet and explicitly
account for already consumed preflight initialization versus untouched streams.
If preserving that design is impossible, return with the specific changed
scientific scope instead of silently editing the protocol. This is a proposal,
not approval or a claim that a recovery experiment has started.
