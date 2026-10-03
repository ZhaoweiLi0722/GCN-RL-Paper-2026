# DRAFT: Recovery2 Remaining Work And Saved Boundaries

Date: 2026-10-03. Scope: metadata/source-only sidecar for
`dynamic_capacity_adaptation_20261003_recovery1`.

**Not an execution authorization, executable resume package, or verified repair.**
Recovery1 is a consumed, closed attempt. The current instruction permits narrow
engineering preparation, not another scientific invocation. All numerical
continuation budgets below are conditional on compatible saved-state reuse and
an unchanged scientific method. The parent owns diagnosis, code repair, and any
later consolidated continuation proposal. Old artifacts remain immutable.

## Evidence Boundary

Paths below are relative to this worktree. Let `R` denote
`results/dynamic_capacity_adaptation_20261003_recovery1` and `P` denote
`R/payload`.

- Read `AGENTS.md`, `recovery1/terminal-readout.md`,
  `experiments/scripts/run_dynamic_capacity_recovery1.py`,
  `src/rl/capacity_pilot_runner.py`, and
  `src/rl/capacity_pilot_recovery1_execution.py`; the remaining evidence is
  existing JSON/JSONL metadata and the existing archive manifest.
- The app-terminal tool reported no attached session. The current persisted
  terminal readout and `R/launcher/terminal.json` are the terminal evidence:
  failed, exit 1, elapsed 793.920531 seconds, scientific completion false.
- Frozen implementation: `56a85d27b87399fec5bf02d800712df3aee7f7a6`;
  execution head recorded by the readout: `18ee090`; packet:
  `1e6a59ff5f83d2c11e61a5a047aff3fff0d376b3b9e72402739841a2a379186c`.
- Reuse the existing findings in
  `reports/dynamic_capacity_recovery1_20261003_partial_analysis.json` and
  `reports/dynamic_capacity_recovery1_20261003_terminal_evidence.json`.
  The former records 35 complete trajectories passing primary checks and one
  incomplete trajectory; the latter records finite update receipts and the
  previously verified seal hash. These checks were not rerun here.
- The existing failed-run archive manifest records 154 files, 26,836,217 bytes,
  and verification by reading every member. No archive was opened or rehashed.
  No pickle/checkpoint was deserialized; no model, filter, planner, environment,
  optimizer, analysis bootstrap, or scientific retry was executed.

## Exact Reusable Boundaries

Trajectory IDs follow `<phase>-b<block>-c<condition>-j<replicate>-<role>`.
Each complete trajectory has 48 control epochs plus 16 settlement epochs.
Serial order is replicate outermost, condition innermost.

| Saved boundary | Exact membership | Reuse and limit |
|---|---|---|
| Block0 teacher | `teacher_bc_critic_warmup-b0-c{0,1,2}-j{0,1,2,3}-teacher`: all 12 | 768 native rows; 576 teacher transitions represented in the seal; do not recollect |
| Block0 offline | `offline_ddpg-b0-c{0,1,2}-j{0,1,2,3}-learner`: all 12 | 768 native rows; 576 offline transitions and DDPG pairs; do not retrain |
| Block0 model | `P/models/block0-seal.pt` and its JSON seal | Reuse the fixed offline endpoint for all block0 learned evaluation forks; no block0 fitting remains |
| Block1 complete teacher | `c=0,1` with `j=0..3`, plus `c=2` with `j=0..2`: 11 | 704 native rows and 528 teacher transitions to reconstruct, not recollect |
| Block1 partial teacher | `teacher_bc_critic_warmup-b1-c2-j3-teacher`, seed `62601023` | Saved epochs 0..46: 47 native rows; next real action is epoch 47; not settled or cost-complete |
| Block1/2 models and offline work | No completed initializer or seal for these blocks; no offline trajectories | Both blocks still require their original fitting budgets |
| Final evaluation | 0 of 216 trajectories | All 36 paired worlds and six arms remain; no evaluation savings or RL benefit established |

Block0 model seed is `520260301`. Its seal contains the endpoint after 256 BC
actor steps, 256 critic-only warmup steps, and 576 offline DDPG pairs:
832 actor + 832 critic optimizer calls, 4,480 neural forwards, and 106,496
optimizer example presentations. The intended offline replay size is 1,152
rows, enforced immediately before sealing by the runner. Metadata identifies
7,543,716 bytes and SHA256:
`68e5ee091c6083e2d67a917d50abed8bd3053ff8d7b248f4763ed8e95b0d6d7d`.
The 12 teacher and 12 offline trajectories remain provenance/training evidence,
not evaluation samples. Snapshot field restoration was not tested in this task.

The partial boundary is `P/failure-state.pkl.gz`, recorded SHA256
`2c224da4ba33c44b990b19521d0a1eba5f13c54b2e631d66ec94f6d7c24c80b0`.
Its already-saved exogenous tape is
`P/tapes/teacher_bc_critic_warmup-b1-c2-j3-exogenous.json`, SHA256
`2232da193fc0b66568e29fb876dd5ae2261da4f4bff2459eab1cd39f246b2e89`.
The immutable 47-row raw prefix has SHA256
`d95857cbfb9b172bd62082c46d2e460f9f6ae9bf7b404d596b022e2f30e1749d`.
These are existing manifest identities, not new byte-verification claims.

## Partial Teacher Continuation Semantics

The recovery exception handler saves `active`, `epoch`, environment
`state_dict()`, controller `state_dict()`, learner state, and budget snapshot,
with `resume_authorized=false` (recovery execution lines 200-207). At this
boundary the runner has set `learner=None` for block1 teacher collection.
Failure occurred inside the forecast for epoch 47, before the real environment
step. The real 47-step prefix must not be replayed as new native work.

A prospective continuation would restore the real epoch-47 environment and
the corresponding controller/filter/history state, retain the exact tape,
and compute the still-missing action. It then needs **one control step at 47
and 16 tail steps at 48..63: 17 native steps**, 17 new receipt/filter updates,
and 1,700 hypothesis transitions. Receipt 47 was already observed after the
previous real step; observing it again on restore would double-update the
filter. No teacher neural action or optimizer call is needed for these 17
steps. The last control transition becomes terminal only after the complete
tail cost and final state are attached.

Important persistence limitations visible directly in the runner:

- Both the per-episode `teacher_rows` and the block-level accumulated
  `teacher_rows` are local variables, absent from `_save_state()` and the
  exception snapshot. There is no directly saved block1 typed replay buffer.
  Reuse therefore means reconstructing **575 existing teacher transitions**
  (11 x 48 + 47), then adding the final settled transition to reach 576.
  Preserve world order, epoch order, executed actions, feature timing,
  reward divisor 100,000, and terminal tail-cost attachment. Existing raw
  public inputs/filter summaries and end-state snapshots are candidate inputs;
  metadata-only inspection does not prove their typed-feature reconstruction
  is complete or identical. Do not silently recollect the 11 complete worlds.
- Running cost totals, `tail_total`, `public`, `states`, and the in-flight
  forecast's local candidate state are not top-level snapshot fields either.
  Prefix cost must come from saved rows; tail cost is initially zero here.
  The pending forecast cannot be declared resumable at its 15th call from a
  budget counter alone. Plan one complete replacement decision, separately
  accounted below, not reuse of its 14 dispatched calls as completed output.
- `noise_rng` and the precomputed noise array are episode locals. The runner
  generates a keyed 48 x 4 Gaussian array before each episode; the teacher
  does not use it. Retain the frozen integer substreams and exact saved tape,
  not newly chosen seeds. The wrapper calls environment/controller
  `state_dict()` but exposes no JSON inventory of their internal RNG fields;
  complete RNG/filter restoration and absence of mutation by the failed
  forecast are **not established by this sidecar**. Do not claim bitwise
  recovery; frozen runtime metadata also records `deterministic=false`.
- Remaining model seeds are block1 `520260302` and block2 `520260303`. The
  runner initializes each learner's replay RNG from that block's final
  teacher world (`c2,j3`), not its first world. Read those exact integer seeds
  directly from the frozen stream manifest, avoiding float conversion.
- The current runner has no resume entry: it creates fresh directories with
  `exist_ok=False`, starts block0, and demands exact original final counters.
  Saving failure state does not implement continuation. A parent-owned new
  runner must distinguish imported completions, the one resumed logical
  trajectory, new starts, restoration overhead, and historical failed work.

## Remaining Scientific Work

This table assumes semantic compatibility of the repair, reconstruction of the
saved block1 teacher data, and restoration at epoch 47 without native replay.

| Remaining phase | Logical trajectories to finish | Control | Tail | Native steps | Actor calls | Critic calls | Neural forwards |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher + BC/warmup | 1 partial + 12 new | 577 | 208 | 785 | 512 | 512 | 2,048 |
| Offline DDPG, blocks1/2 | 24 new | 1,152 | 384 | 1,536 | 1,152 | 1,152 | 6,912 |
| Six-arm evaluation | 216 new | 10,368 | 3,456 | 13,824 | 1,440 | 1,440 | 12,384 |
| Total | 253 completions, 252 new starts | 12,097 | 4,048 | 16,145 | 3,104 | 3,104 | 21,344 |

Remaining training alone is 2,321 native steps, 512 BC actor steps, 512
critic warmups, and 1,152 offline DDPG pairs. That is 3,328 optimizer calls
and two new seals; per block the unchanged fitting budget is 256 BC + 256
warmup + 576 offline pairs. Offline trajectories use seeds
`62604000 + 1000*b + 10*c + j` for `b=1,2`, `c=0..2`, `j=0..3`.
Block2's 12 teacher seeds are `62602000 + 10*c + j`.

All three seals precede evaluation in the original runner. Evaluation remains
`3 blocks x 3 conditions x 4 worlds x 6 arms = 216`, with seeds
`62610000 + 1000*b + 100*c + j`. Arms, in fixed order:
`frozen_history`, `frozen_matched_exploration`, `online_matched_fork`,
`adaptive_rule`, `id_mpc`, `fixed_allocation_reference`.
The 108 learned-arm restorations are independent seal resets. Online fitting
is 36 worlds x 40 pairs = 1,440 pairs / 2,880 optimizer calls; receipt 48's
update remains deferred until settlement. Keep matched exploratory noise and
the frozen/noisy-frozen contrasts; do not substitute training results.

## Numerical Limits Needed In A Later Proposal

These are new-work counts, **not transferred authorization**. Old charged work
stays in the historical ledger. Minimal scientific counts are exact under the
conditions above; restoration/reconstruction mechanics are not yet implemented.

| Counter | New-work amount or required distinction |
|---|---:|
| New trajectory admissions | 252, plus 1 separately identified partial restoration |
| Fresh actor/critic pairs | 2 |
| Imported sealed model artifacts | 1: block0 only; no new fit |
| Learned evaluation-arm restorations | 108 |
| Native constructions / constructor-triggered resets | 252 / 252 for new worlds; see restoration allowance below |
| Native steps / total native operations | 16,145 / 16,649 before restoration overhead |
| Explicit extra resets / native clones | 0 in the minimal schedule |
| BC / critic-only warmup / DDPG pairs | 512 / 512 / 2,592 |
| Actor / critic / combined optimizer calls | 3,104 / 3,104 / 6,208 |
| Optimizer example presentations | 397,312 = 6,208 x 64 |
| Neural module forwards / maximum batch | 21,344 / 64; no selection forwards |
| Receipt updates / hypothesis transitions | 16,145 / 1,614,500 before optional reconstruction overhead |
| Teacher / evaluation planner decisions | 577 / 1,728 |
| Total planner decisions / candidate-quantile rollouts | 2,305 / 110,640 |
| Planner forecast epochs | 885,120 = 2,305 x 16 candidates x 3 quantiles x 8 epochs |
| Analysis bootstrap | Original 2,000 resamples, only after complete paired evaluation |

**The planner subtraction trap:** Recovery1 charged 1,152 teacher decisions
and 442,368 reserved forecast epochs, but only 1,151 decisions / 441,984
forecast epochs are verified complete. Its last reserved chunk is 384, with
14 dispatched calls; a dispatched call is not necessarily completed. Simple
limit-minus-debit yields 576 remaining teacher decisions and 884,736 forecast
epochs, short by the failed decision. The prospective schedule needs 577 and
885,120 respectively. Preserving old debits gives cumulative charged totals
of 1,729 teacher decisions, 3,457 total decisions, 165,936 candidate rollouts,
and 1,327,488 reserved forecast epochs. The original scientific schedule still
targets 3,456 successful decisions in total; only 1,151 have completed so far.
Do not erase the old failed
reservation or treat its undispatched 370 slots as a resumable forecast.

**Explicit implementation allowances, not hidden science:**

- If restoring the partial environment requires one construction and its
  constructor reset, budget 253 constructions, 253 constructor resets and
  **16,651 total native operations** instead of 16,649. If restoration is
  allocation/state-only, retain 252/252/16,649. Any separate reset must also
  be counted; neither path may execute the 47 saved native steps again.
- Importing the block0 seal bytes is distinct from a neural model
  instantiation. Bind one existing seal import explicitly; retain 108 actual
  evaluation restorations. The old `historical_model_loads=0` cannot silently
  forbid reuse or hide an additional load. If the implementation separately
  instantiates a model during import, identify and budget that one extra
  restoration; no diagnostic forward is implied or budgeted.
- If block1 feature reconstruction requires replaying saved observations
  through the unchanged filter, a full pass over its completed histories
  and prefix is **751 saved receipts = 11 x 64 + 47**, or **75,100 additional
  hypothesis transitions**. Separate these from new native receipts:
  combined transition work would be **1,689,600**. Direct exact extraction
  needs no such filter pass. This is a conditional accounting option, not a
  verified implementation or permission to rerun science. Other reconstruction
  work has no measured cost and cannot be assigned zero wall time.

For a later proposal, the existing conservative envelope can be retained as
a **fresh, explicitly approved ceiling**, not as a demonstrated ETA:
CPU, 1 worker, 4 compute threads; 4,294,967,296-byte RSS;
1,073,741,824-byte raw and archive caps each; 2,147,483,648-byte combined disk;
2,000 files; flush every 8 epochs. Charge new imports/copies, deserialization,
recording, reconstruction and archive I/O to their actual owners. The closed
old run/archive remain preserved; define whether new disk accounting references
or duplicates them rather than silently counting their storage as free.

| Wall-clock owner | Candidate fresh hard ceiling, seconds |
|---|---:|
| Admission / binding | 300 |
| Remaining initialization, training, restoration and seals | 2,100 |
| Evaluation and settlement | 2,100 |
| Analysis and new-output archive verification | 600 |
| Failure flush / shutdown reserve | 300 |
| Global | 5,400 |

These are the existing numeric caps reused for planning, not permission to
consume another 90 minutes. Historical arithmetic would leave 1,310.344129
training-owner seconds and 4,606.079469 global seconds, but unused budget is
expressly nonreusable. Recovery1's 793.920531-second failure gives no measured
evaluation, restored-state, or repaired-predictor throughput. There is no
defensible completion ETA or finite preparation-time estimate in the allowed
metadata, and no running job is claimed here.

## Does A Changed Teacher Invalidate Fair Reuse?

**A behavior-changing teacher algorithm invalidates unconditional same-method
reuse.** Completed teacher actions remain valid historical observations, but
block0's seal and offline trajectories depend on the old teacher. Reusing them
as if all three blocks had the newly changed initializer would confound block
and algorithm; recomputing only block1's failed decision does not remove that.
Block1's saved teacher prefix would likewise be a mixed-algorithm trajectory.

A narrow resource-accounting repair permits this remaining-only plan only to
the extent it preserves the intended teacher semantics and prior valid
decisions. Metadata establishes neither that equivalence nor the specific
invalid field. A change to forecast resource dynamics, candidate rankings,
tie-breaking, horizon, filter, features, reward, or real environment is not
automatically a harmless precision repair. Do not weaken validation or the
ID-MPC comparator to make reuse convenient.

If the repair changes teacher behavior, the old seal could still define an
explicit historical initializer with matched forks, but that is a different
scientific interpretation, not the original homogeneous three-block method.
Whether to retain that amended estimand or replace affected training belongs
in the parent's single prospective scope decision; this draft authorizes
neither and does not invent a replacement training budget. The 216 evaluation
trajectories remain unexecuted in either case. Engineering failure is not an
RL null result, and successful training receipts are not patient benefit.

## Efficiency Recommendations

1. Reuse the existing terminal analysis, finite-receipt checks, seal identity,
   stream manifest and archive receipt. Do not repeat the full-history/archive
   audit or recollect block0's 24 complete trajectories.
2. Implement only the named predictor repair and the missing saved-boundary /
   teacher-row continuation path. Reuse unchanged tests; the parent owns only
   repair-affected validation and its required compileall, not another toy
   campaign or duplicate reviewer chain.
3. Put the exact remaining counts, conditional restore/reconstruction costs,
   and teacher-compatibility interpretation in one continuation package. Keep
   all original seeds, six arms and settled-cost/patient analyses; do not add
   selection episodes, extra fitting, or approval stages inside routine work.

## Delivery State

Only this DRAFT was created. No scientific code/config was changed, no commit
was made, and no execution was launched. A documentation-only sidecar does not
run compileall or tests; validation/integration remain parent-owned. The source
and metadata inspection supports the counts above, not a claim that a working
resume loader, repaired teacher, or completed comparison now exists.
