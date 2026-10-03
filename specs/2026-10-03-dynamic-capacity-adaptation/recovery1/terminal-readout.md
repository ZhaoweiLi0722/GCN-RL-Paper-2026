# Capacity Recovery1: Training Milestone, Incomplete Comparison

2026-10-03T04:17:42Z read-only handoff. Execution18ee090, implementation56a85d2,
packet1e6a59ff5f83d2c11e61a5a047aff3fff0d376b3b9e72402739841a2a379186c.
This single correction attempt is consumed; no automatic retry or resume.

## Terminal State

Supervisor29491/PPID1 and child29504/PPID29491 exited with code1 after
793.920531seconds, about13.23minutes. Read-only process inventory confirms no
related launch/child remains. This was not a time, storage or update-budget
exhaustion. The exception is recorded in launcher/child-failure.json:
`ValueError: invalid public resource value`.

Failure boundary: teacher_bc_critic_warmup,block1,condition2,replicate3,
seed62601023,teacher,epoch47. The internal ID-MPC forecast failed while building
PublicSupportOperations from a resource matrix. Raw stdout/stderr and detached
log are empty; the captured traceback is authoritative, not a clean result.
No launch-failure, supervisor-overrun or failure-preservation-error exists.

The saved real public matrices (bioreactors, reagent transfers, capacity
transfers and orders) are all finite and nonnegative. The invalid matrix value
was produced within the forecast, not present in that saved real input. The
specific field/value and arithmetic origin are not preserved in the traceback.
They have NOT been reproduced under this read-only handoff. Do not assume a
roundoff cause, relax validation, change reward, or claim the implementation is
fixed. No additional forecast, environment or neural execution was performed.

## Actual Progress

| Item | Completed / Limit |
|---|---:|
| Full teacher trajectories | 23 / 36 |
| Additional partial teacher trajectory | 47 / 64 steps |
| Full offline learner trajectories | 12 / 36 |
| Final evaluation trajectories | 0 / 216 |
| Offline sealed models | 1 / 3 |
| Actor optimizer steps | 832 / 3,936 |
| Critic optimizer steps | 832 / 3,936 |
| Combined optimizer calls | 1,664 / 7,872 |
| Native environment steps | 2,287 / 18,432 |
| Native operations including construct/reset | 2,359 / 19,008 |
| Neural forwards | 4,480 / 25,824 |
| Filter hypothesis transitions | 228,700 / 1,843,200 |
| Planner epochs reserved | 442,368 / 1,327,104 |

There are441,984verified completed forecast epochs plus14dispatched calls in
the final reserved384-call chunk. Reserved work is not completed work.

Block0 has256BC actor updates,256critic warmup updates and576DDPG actor/critic
pairs. All1,088update receipts have finite persisted numeric fields. Thus real
neural fitting occurred, not just preparation. Deployment-online update count
is0. The model seal exists and its7,543,716bytes match the recorded SHA256:
68e5ee091c6083e2d67a917d50abed8bd3053ff8d7b248f4763ed8e95b0d6d7d.
No checkpoint load or model forward was needed to verify these bytes.

## Evidence And Interpretation

- reports/dynamic_capacity_recovery1_20261003_partial_analysis.json independently
  reconstructs35complete trajectories and the47-row partial trajectory from
  saved costs and patient records. All35complete trajectories pass its primary
  checks. The partial trajectory lacks its last17rows/summary and has live
  patient liabilities; do not treat it as a complete cost/outcome.
- reports/dynamic_capacity_recovery1_20261003_terminal_evidence.json records
  optimizer-mode counts, finite receipts, seal hash and saved public matrices.
- reports/dynamic_capacity_recovery1_20261003_archive_receipt.json verifies the
  new154-file local archive,26,836,217bytes, with every member read and original
  files unchanged. Archive SHA256:
  a3ba7f2473f5d0184d910858b561e60191e24efee777730e42df39379688b291.

Original proposal/protocol, correction intent/protocol and previous-failure
input hashes still match. No scientific source differs from implementation
56a85d2. Prior checks and prior archive were reused, not repeated. No Dropbox,
cloud-sync or Howard-access claim; no remote operation.

Online-versus-frozen/adaptive/MPC contrasts are unavailable, not zero. The
signal classification is engineering_failure_not_rl_null; training loss and
the sealed model do not establish initialization competence or RL improvement.
There is no completion ETA because no related process remains running.

## Next Decision

Authorize a narrowly scoped correction of the predictor's resource accounting
and preparation of a remaining-work continuation package, or retain this
terminal evidence without further execution. Prefer reuse of the sealed block0
model and35complete trajectories; do not train them again by default. Determine
the exact failed field from the saved public boundary before designing a fix.
Any later scientific continuation requires its own complete numeric scope,
source/runtime/input locks and new directory. None is authorized here, and old
unused budget is not a permit. The original scientific objective and fair
comparators must not be weakened to avoid the error.

Same gcn-rl automation is tool-confirmedPAUSED with a visible handoff. No new
experiment, automatic repair-and-retry, reward search or repeated waiting loop.
