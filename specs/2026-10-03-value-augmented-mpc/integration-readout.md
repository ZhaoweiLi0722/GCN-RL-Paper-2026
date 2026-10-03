# GCN Value-MPC Integration Readout

2026-10-03T20:49Z. Preparation complete; numerical execution approval pending.
No new scientific model loads, forwards, native worlds or optimizer steps ran.
The previous fixed-budget DDPG study remains closed with its negative result.

## Delivered

- One public-belief schema for observed and forecast inputs; a fresh GCN
  residual value network; complete-cohort eight-step TD targets retaining the
  16-step settlement tail. Zero residual reproduces the original MPC score.
- The corrected public predictor, action candidates, information and query
  budget are shared by plain, initial-value and updated-value MPC. No actor
  is trained. Learned values replace the heuristic through a residual
  parameterization; the heuristic is not counted twice.
- A serial, admitted entry through initialization, same-state continuation,
  all-model seal barrier, frozen evaluation, independent raw cost/patient
  reconstruction and verified local archive. Existing IO/watchdog/ledger
  helpers are reused. There is no scientific resume or retry mode.
- Complete value model, Adam, minibatch RNG, pending targets, public/native
  state and durable budget receipts. A failed update rolls back mutable
  learner state without refunding dispatched-call charges.
- The single prospective package is in protocol.md and the original draft
  config remains scientific_execution_authorized=false. Exact authority,
  committed source/runtime/input/stream locks and a new result root are
  required before its one execution.

## Necessary Validation

20 zero-update tests passed with unittest in 4.874s. The actual build_runner
entry, scheduler and independent reader completed an artificial 288-world
fixture, all 144 test worlds behind all three final seals, with every declared
phase/global counter matched. These are mock accounting events, not scientific
training. Real optimizer.step, native construction/step and real predictor
construction/step were forbidden in those tests.

Tests also covered residual target units, terminal masking, batched 48-candidate
scoring, zero-head plain-MPC parity, snapshot ownership/RNG/pending restore,
failure charging and authorization rejection. The independent worker exposed
a partial-mutation bug when pending snapshot metadata was absent; validation
now precedes the atomic swap and the regression passes. Model seeds were
spaced before freezing to avoid overlap with their minibatch-RNG seed+1 streams.
No experimental outcome informed either correction.

Whole-repository compileall and git diff --check passed. Pytest is not installed;
the project's unittest entry was used. No dependency installation was needed.
No related scientific Python process was present at the final read-only check.
The run directory and scientific frozen/authorization files do not yet exist.

## Manuscript And Next Action

The manuscript now states the proposed value-MPC mechanism, TD target, numerical
design and fair primary comparison, while preserving historical AFR-GCN-DDPG
and fixed-budget negative results. No value-MPC performance is claimed.
The manuscript worker and one efficiency adviser completed; the bounded test
worker also completed and closed. No extra review gate was added. PDF rendering
remains unavailable locally because no TeX engine is installed.

The only remaining decision is the already-asked complete preparation-plus-
execution package: 3 blocks; 24 initial plus 24 continued training cohorts/block;
144 frozen evaluations; 288 total worlds; 18432 native steps; 4608 value updates;
4644864 planner epochs; 4h and 4GiB including IO; one attempt, no automatic retry.
After approval record the exact reply, commit change control and locks, then
launch directly and complete the comparison without another launch question.
Expected runtime is roughly 2-4h based on the prior planner workload, not a
measured guarantee for this new value method. No new toy fit or screen is needed.
Until that reply the same automation stays PAUSED, not a training process.
