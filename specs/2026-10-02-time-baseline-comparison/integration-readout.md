# Time-baseline integration checkpoint

2026-10-02T05:05Z. Entry commit:42997b35e2d49ff0f933a7f40621914d2a216753.
Engineering only. No new scientific model access, patient episodes, parameter
updates or result directory. The numeric execution question remains pending;
the latest instruction to start follows the automation request and is applied
to the already authorized integration work, not treated as an extra permit.

## Usable implementation

- `src/rl/time_baseline_ppo.py`: versioned original-V and leave-one-episode-out
  target selection. Both retain canonical raw segments, original critic targets,
  old probabilities, separate actor/critic owners and pre-call budget charging.
  Saved target receipts bind ordered training identities, independent seeds,
  behavior digests, raw rewards, collected values and segment hashes. Restoration
  checks the unchanged float32 return arithmetic and consumed episode inventory.
- `src/rl/time_baseline_collection.py`: public-input collection and continuation
  adapters for the new exact kernel type. The episode's declared seed, split,
  source and identity are bound to its training manifest. Existing float64 raw
  requests, transition receipts, terminal failures and nonrefund accounting are
  reused. Old source/type guards are not edited.
- `src/rl/time_baseline_plan.py`: pure JSON budget, serial-section and random-stream
  adapters. The unchanged proposal produces 33 sections, nine training owners,
  eighteen evaluation owners and paired arm streams with independent preflight
  copies. Decimal seed strings preserve large integers. Internal uniqueness is
  checked; historical freshness is explicitly not certified here.
- `src/rl/time_baseline_sequence.py`: ordered six-controller schedule, twelve
  learned-model byte seals, no test before the complete seal, and atomic cursor
  restoration. Modified artifacts or failed cursors cannot resume as successes.

## Verification performed

53 focused/regression tests passed in2.792s across the scalar transform, new
kernel/collection integration, budget/stream plan and existing dynamic updates
and continuation. Four additional sequence tests passed in0.078s. Full repository
`compileall -q .` passed after the source changes.

All numerical fixtures use invented two-step data. Adam and SGD real steps are
blocked; the fake optimizer only records counters and zero moments, never
changes parameters. No artificial fit was performed. Tests establish unchanged
original-method losses/counters, complete/interrupted restoration, malformed
provenance/target rejection, independent-owner failure rollback with retained
debits, ordered test access and model-byte sealing. They are not evidence of
patient performance or successful scientific execution.

Helmholtz completed the disjoint plan/stream adapter and19tests, then was closed.
The coordinator integrated the remaining modules and tests. Carson's completed
efficiency advice is reused: one decisive full comparison, no extra toy fit or
review prerequisite. There is no always-running pair of agents.

## Remaining critical path

1. Wire a new campaign/factory for current PPO, time-baseline PPO, BC, frozen,
   R4 and full MDL-2. Keep saved initializer reuse and fresh continuation owners;
   do not rename raw records or modify old campaign globals.
2. Add the versioned independent raw-cost/patient verifier and six-controller
   contrasts, retaining continuous-resource checks and integer patient checks.
3. Bind the existing exclusive claim, startup admission, watchdog, full recovery
   and local archive path to this campaign. Complete a mock-only end-to-end test,
   scoped historical seed conflict check and committed source/runtime/input locks.
4. Execute only after the complete numeric package has explicit approval and
   the implementation/authorization locks are committed. Approval alone cannot
   substitute for the missing entrypoint. No extra scientific acceptance gate.

The proposal's522episodes,27216environment calls,1920optimizer calls and10800s
remain unchanged and unconsumed. A null or diagnostic-only result closes this
baseline route and triggers the existing reward-design memo, not automatic
reward fitting, expanded samples or a repeated baseline attempt.

The current `gcn-rl` heartbeat was read back ACTIVE at30-minute cadence. It
continues finite engineering from Live/workflow; this does not mean a research
process is running. No remote action, export, holdout or Stage E change.
