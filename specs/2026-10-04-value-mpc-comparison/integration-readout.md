# Value-MPC Comparison Engineering Handoff

2026-10-04T10:48Z. Preparation only. Zhaowei's latest `继续推进` follows the
four-controller recommendation, not a previously proposed numerical package.
The complete600-world/15360-update/9h/8GiB question was asked once this turn.
No exact reply has arrived. Scientific execution is NOT approved; no freeze,
launch, scientific model loading, patient environment, actual optimizer step or
test-world access occurred. Existing gcn-rl automation stays PAUSED.

## Delivered

- Additive `capacity_value_comparison_*` learner/resources/runner/execution/
  analysis modules and thin `run_capacity_value_comparison.py` entry. Historical
  sources and results are unchanged. The runner reuses the native trajectory IO,
  TD update loop, full-state saving, causal public predictor and nonrefundable
  accounting rather than adding another diagnostic experiment.
- Flat global residual reads the identical4x31 public features. Parameter counts
  are3169(graph) and3155(flat):14/3169=0.0044177974 gap. No padding. Graph and
  flat are parameter-matched, not depth-matched; this compares inductive biases,
  not graph edges in isolation. Neither uses hidden state or extra information.
- Shared initial collection:120 worlds,240 fits. Both learners' full states are
  saved at interleaved fit boundaries; each fit references a raw-data digest.
  Continuation:240 worlds/fits from paired tapes, but different endogenous
  trajectories may result. Twenty seals (ten initial,ten final); tests stay
  closed until all final seals. Frozen evaluation:240 worlds/four controllers.
- Exact accounting:600 worlds,38400 steps,39600 native operations,15360 value
  updates,0 actor,33120 forwards,9953280 planner epochs and3840000 filter
  transitions. Five-block independent raw readout; the old three-block bootstrap
  is intentionally NOT reused. No post-result sample or checkpoint selection.
- Raw reader checks all expected trajectories, full64-step cost and patient
  outcomes, paired tapes/cohorts, model hashes/config, all480 fit boundaries,
  shared-data digests, sequential update receipts and test-sealing order. It
  reports all six contrasts, conditions and blocks, not only the favorable one.

## Validation

`PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache
 /Users/lizhaowei/GCN-RL\ Paper\ 2026/.venv/bin/python -m unittest
 tests.test_capacity_value_comparison tests.test_capacity_matched_value -v`

24 tests pass in14.833s:17 model/learner tests and7 new schedule/binding/raw-reader
tests. The actual new entry completed the entire600-world schedule with fake
backends and forbidden real optimizer/environment constructors. It is NOT600
scientific trajectories. Artificial test tensors/fake optimizer receipts are not
training evidence. Tests cover all-site connectivity, zero head, gradients,
complete snapshot/RNG/optimizer/pending recovery, atomic failed fake updates,
architecture routing, shared data, seal barrier, tampered-input rejection, five-
block intervals, budget nonrefund and unapproved admission rejection.

First run:23 pass,1 artificial budget-fixture error because the test ledger was
placed at the temporary root, making the inherited storage guard scan its parent.
Corrected only the fixture to its real launcher layout; second run24/24. This is
not a scientific failed attempt. No consumed experiment was repaired or retried.

Whole-repository `python -m compileall -q .` with the same cache/runtime exits0.
No dependency changes. Reuse historical passed tests; no new toy fitting gate.

## Delegation And Remaining Action

Franklin01a1067c-5ac9-7d33-b756-ac5feb719568 delivered the matched model/learner
and17 tests, then closed. Read-only efficiency advisor
McClintock01a1067c-5b38-71f1-8220-19f6f5e59726 delivered three reuse choices,
then closed. The coordinator implemented the schedule, metadata binding, raw
reader, numerical protocol and integrated tests concurrently with these finite
tasks. No extra reviewer gate or repeated historical audit was added.

Only remaining decision: approve or decline the already asked complete package.
After approval, preserve the exact reply, append change control, commit current
source/runtime/input and scoped seed-conflict locks, then execute the single
package through training/evaluation/readout without another launch question.
No extra smoke episode. Local commits only. Prior throughput suggests5-7h;
the hard cap is9h includingIO, not a guarantee of speed or benefit.

The prior13.75% continuation signal and2.46% uncertain plain-MPC contrast are
unchanged. This deliverable makes the stronger comparison executable after
approval; it establishes no new performance improvement. No reward change,
holdout, StageE reopening, Howard approval, remote action or Dropbox export.
