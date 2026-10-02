# Time-baseline integration checkpoint

## Completed preparation at2026-10-02T05:35Z

The six-controller comparison is implemented and locally frozen, not executed.
Implementation commit:1baa0a958a5656e38ab823304e4ff5cba2c5386d.
Frozen packet: `frozen.json`; content SHA256
5279726f8fc069905ec1097fc961c13136c25241136a99383edae78bcd3c6300.
This supersedes the remaining-engineering list in the previous checkpoint below.
The original proposal/protocol/reward-pivot and old scientific artifacts are intact.

### Named blockers removed

- `time_baseline_factory.py`, `time_baseline_backend.py` and
  `time_baseline_campaign.py` connect original PPO, LOEO PPO, BC, own frozen,
  R4 and full MDL-2 through the complete33-section schedule. Qualified saved
  initializers are imported once, with old initialization metadata kept separate
  from fresh comparison streams. Preflight sampling owners do not consume the
  training RNG state. Collection/update/phase/optimizer/RNG restoration retains
  the live nonrefundable budget and cannot rewind recorded raw rows.
- `time_baseline_verification.py` independently reads raw costs and patient
  outcomes, exact decimal world seeds, continuous resource requests and integer
  patient/specimen fields. It enforces the216-world/controller matrix, greedy
  test selection for all four learned controllers and six paired contrasts.
  It reports patient adverse directions and whether the greedy requests changed;
  lower training loss alone cannot satisfy the prespecified performance screen.
- `time_baseline_target_verification.py` independently recomputes unchanged
  float32 critic returns and original-V/whole-trajectory-excluded LOEO targets
  from saved raw training events and phase receipts. Its baseline/advantage
  variance diagnostics are not measurements of policy-gradient variance.
  Segment hash syntax is checked, but full segment-payload reconstruction is
  explicitly not claimed by this scalar reader.
- `time_baseline_execution.py`, `time_baseline_watchdog.py` and the thin
  `experiments/scripts/run_time_baseline_comparison.py` supply source/input/runtime
  binding, exact human-authorization matching, exclusive one-attempt ownership,
  separate load/build caps, and an external watchdog for setup/owner/phase/global
  and post-child closure limits. All scientific execution fails closed without
  the new authorization. Old execution modules and artifacts were not edited.

### Verification and freeze

124focused/regression tests passed in26.972s across the scalar baseline,
integration/plan/sequence, campaign, raw and target verifiers, admission/watchdog,
and existing dynamic update/continuation modules. Full repository compileall
and staged diff-check passed. Only invented data and metadata-only fake optimizers
were used; actual Adam/SGD calls are blocked. The complete two-step FakeEnv fixture
uses54episodes,126environment debits,30fake optimizer debits and12model seals;
both local archive boundaries are checked. This is engineering evidence only.

Preparation bound371source files,12input files and the runtime. The local numeric
collision scan parsed520named historical declaration files, including preserved
original-attempt declarations, against138world and19neural/analysis streams.
No collisions were found. Scope is local tracked JSON configs and named
seed/stream/manifest/config/frozen JSON in results/reports/specs, excluding this
prospective directory and new result root. Unavailable external files are not
covered. No scientific model was decoded or scored by the freeze/readback.

The hash-only freeze and binding readback passed. The prospective result root
does not exist and no authorization.json is present. Host process inspection
found no matching scientific job. A first artificial campaign test failed because
its test fixture selected a draft without the effective candidate-support fields;
the fixture was corrected to the committed prior effective config before this
passing suite. No scientific attempt failed or was consumed. A TOML read utility
also required the available vendored parser because this Python lacks tomllib;
that administrative failure made no research call or artifact change.

Aquinas delivered the disjoint raw verifier and watchdog and was closed. The
coordinator integrated the factory/backend/campaign, independent target reader,
admission, tests and freeze. Prior Carson efficiency advice was reused, not
turned into another release gate.

### Next decision, not another preparation cycle

The already-presented numeric package remains unapproved:3blocks,32episodes per
training arm/block,522episodes total,27216environment calls,1920optimizer calls,
10800seconds,one attempt with fixed phase/owner limits. No reward, architecture,
support, initialization, entropy or evaluation change is authorized. No old
unused allocation transfers into this package.

After an explicit reply to this package, record the verbatim authorization and
change control, commit them against frozen.json, verify the clean binding, and
run the complete package once using the entrypoint (without --freeze). Do not
launch the command or its --child mode before that approval. No additional
qualification rescore, toy fit, critic gate or routine per-stage question is
needed. The independent performance readout governs the prewritten reward pivot;
a terminal scientific failure preserves evidence and does not license a retry.

Automation gcn-rl is PAUSED and retained in the list, read back from the same
thread's TOML. Approval is the sole remaining execution blocker. No new RL
performance result can be claimed from this preparation.

## Previous checkpoint: target and collection integration

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
