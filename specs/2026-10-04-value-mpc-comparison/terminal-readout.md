# Value-MPC Strong-Baseline Comparison: Interrupted Before Evaluation

2026-10-04T12:14Z handoff. The single approved attempt terminated with child
and supervisor exit code 1 at approximately11:47:52Z, after2544.009seconds.
The12:07 heartbeat detected the terminal failure; the previous11:40 snapshot
was genuinely running. PIDs67458/67481 and the matching experiment commands
are now absent. Samegcn-rl isPAUSED, confirmed by the app tool; no retry,
repair, model forward, optimizer update or environment call was made by this
monitor. The prior5-6hour remaining estimate no longer applies.

## Failure And Scientific Meaning

`launcher/child-failure.json` records:

`ValueError('midpoint forecast incompatible with its public work intervals')`

The traceback ends at
`src/baselines/capacity_completion_control_recovery2.py:103-105`, called from
the public MPC candidate forecast, not an optimizer. The forecast constructs
a midpoint-based completed-patient prefix, then `_condition_prefix` returns
zero compatibility with the stored work intervals. That helper can reject
infeasible bounds or zero interval weight. The saved traceback identifies the
failed consistency contract, but not which numerical branch caused it. No
forecast replay was authorized or performed, so floating-point rounding versus
an interval-update defect is still unresolved. Do not loosen this check or
change reward weights on the basis of this failure.

Active world: continuation,block0,condition2 (fast fluctuation),replicate7,
seed63010207,flat_value_mpc. There are37 saved native rows (epochs0-36);
failure occurred while planning the next native action at epoch37. The last
periodic progress event was epoch31. All3040 persisted loss/gradient receipts
are finite. There is no OOM/native-crash/timeout marker. Empty stderr does NOT
mean success: the exception was captured in child-failure.json.

The comparison is incomplete, not a null performance finding. None of the240
frozen evaluations opened. All six performance contrasts, three conditions,
five-block intervals and both development screens are unavailable, not zero
or failed statistical screens. Existing20261003 results and their limitations
remain unchanged. This run establishes neither strong-MPC superiority nor a
GCN-over-flat gain; it also does not establish that RL cannot improve performance.

## Preserved Progress And Budget

| Item | Completed or charged | Planned maximum |
|---|---:|---:|
| Shared initialization worlds | 24 | 120 |
| Graph continuation worlds | 24 | 120 |
| Flat continuation worlds | 23, plus one partial | 120 |
| Complete native worlds | 71 | 600 |
| Frozen evaluation worlds | 0 | 240 |
| Graph value updates | 1536 | 7680 |
| Flat value updates | 1504 | 7680 |
| Total value updates | 3040 | 15360 |
| Actor updates | 0 | 0 |
| Initial / final seals | 2 / 0 | 10 / 10 |
| Native steps | 4581 | 38400 |
| Native operations | 4725 | 39600 |
| Neural forward calls | 5428 | 33120 |
| Charged planner epochs | 1323264 | 9953280 |
| Filter transitions | 458100 | 3840000 |

The ledger charged72 trajectory constructions, including the partial world.
Its final planner chunk reserved384 epochs and dispatched13 before failure;
the reservation is not refundable. Unused nominal limits are not a new permit.
Graph completed its first block's1536updates, but no final model seal exists:
the runner had not reached the shared block-final sealing boundary.

Two initial models,95 complete after-fit states,71 complete episode states,
and `payload/failure-state.pkl.gz` are retained in the original run directory.
The graph's latest after-fit state is
`payload/states/continuation-b0-c2-j7-graph_value_fit-after-fit.pkl.gz`.
Failure-state gzip decompresses successfully; model deserialization, exact
restore and continuation have NOT been tested. Existing public-filter/planner
state and the interrupted charged chunk must be reconciled before any future
remaining-work continuation can be claimed feasible.

## Independent Saved-Data Check

Read-only standard-library parsing checked all71 saved complete summaries and
4544 raw rows: epochs0-63, finite costs, component sums, reward=-cost, and
terminal unique patient counts/lost/delivered/enrolled match their summaries.
No discrepancies were found. The37 partial rows bring stored native rows to
4581, matching the ledger; they are not a settled cohort or performance sample.
No partial-training advantage or bootstrap interval was computed.

All551 current scientific source hashes and the protocol, proposal and approval
intent match the frozen packet. The canonical packet digest remains
`3fb325a35dee08a17e3d7955635b2233f34e840fc77c94aa344f72234d79e8e9`.
Execution42f1ed5f12d44d147d2b25984d1217ef2aba8462 and frozen implementation
875af2644a46130aaece6efa9f82ebb4c7d60139 remain the recorded execution basis.
Both initial model hashes match their metadata. Historical source/results were
not changed. The existing24zero-update tests andcompileall are reused, not
misreported as a check of a new repair.

Selected immutable evidence (paths below are relative to the run root):

| File | Bytes | SHA256 |
|---|---:|---|
| launcher/terminal.json | 258 | a63dbbab6335127598be140fe93aceb6d05721b669ebfb86432b3d91223e42eb |
| launcher/child-failure.json | 16568 | 6911545070719a03582da708307fb8df5b5304c2a6795cdfe5dae497ca85c210 |
| payload/failure-state.pkl.gz | 131415 | b9a624487e5deda8b80a940d26e28e167bbbcd7a1d38fec603ef24b34d427b2e |
| payload/progress.jsonl | 663886 | 785e943a0e82381ee555472e7926503c20c41ecdf49272d1e39cdb58ab5eb87b |
| launcher/budget.jsonl | 13080568 | fc7668d0a8563b993070d92008c7101bcc45e936ea1c3296e440b1a0af6731b0 |
| payload/states/continuation-b0-c2-j7-graph_value_fit-after-fit.pkl.gz | 131386 | 3b41cc05abd0024d8698b40c2d0983ceb5f5aa2a1fe875ce18d8878d13dd9d71 |

Stderr,stdout and detached report log remain0bytes. Failure-state preservation
has no error marker. No completion/comparison, artifact inventory, archive or
archive receipt was produced: the exception path saves state but does not run
the success-only archive branch. Local failure bytes/hashes are verified;
verified archive/cloud backup/Howard access must NOT be claimed. No duplicate
archive or external copy was created by this read-only monitor.

## Next Decision

Recommend a targeted predictor-consistency repair and saved-boundary recovery
assessment, not a reward search or restart from scratch. This is proposed new
work, not authorized by the consumed attempt. The next approval should cover
that repair/preparation; any real forecast replay, resumed scientific execution
or additional call budget needs a separately specified recovery package.
Preserve all71 complete worlds and3040updates. Reuse is the objective, not yet
a verified restore guarantee. If a corrected predictor changes subsequent
collection, disclose the mixed historical/corrected training data rather than
calling the result a pristine from-scratch comparison. No seed/threshold/sample
or reward change, automatic retry, holdout, StageE reopening or remote action.

The manuscript now records this interruption separately from the completed
three-block study. Static document checks only; no newly compiled PDF is claimed.
