# Reference-prior design engineering readout

2026-10-01. Base111ec7bc87225487bd6499e82a323fbe5c796f91. Zhaowei approved
design/artificial testing, not another scientific attempt. No new patient
simulation, optimizer steps, existing-result fitting or P2 claim occurred.

## Completed

Added opt-in `src/models/reference_prior_candidate.py`, using the existing
CandidatePolicy scorer and unchanged information/parameter widths. It adds a
fixed reference log prior to learned class residuals, with explicitly zero
scorer/value output layers at initialization. The default legacy policy is
unchanged. Its new manifest binds epsilon and prior semantics; current receipts,
PPO and imitation constructors explicitly accept only the two known concrete
types, not arbitrary subclasses. No campaign factory or launch flag was added.

The proposed epsilon=.10 is passed through the design config, not hard-coded
in the model. At initialization the reference is always the unique greedy
class and gets90% sampling mass where alternatives exist. The remaining mass
is uniform over unique other requested-net classes. A singleton gets1.0.
Alias counts, option order and reference/anchor overlap cannot duplicate mass.
This does not guarantee useful exploration, improved outcomes or safety.

18 new tests plus116 relevant existing tests pass:134 total in0.715s. Full
repository compileall and git diff check exit0. Tests include:
- All1..6 class supports, every reference position, all three representations,
  float32/float64 and exact original-reference request preservation.
- Full20x28+1 invented layout,1041 packed values and expected59,602/59,602/238,658
  parameter counts; same graph/self-only tensors and numerical information.
- Actual prior likelihood in behavior receipts, unit unchanged PPO ratios,
  entropy, finite derivatives and rejection of changed-prior/legacy receipts.
- Nonzero first output-layer gradients and zero first encoder gradients;
  no optimizer step or claim of useful learned representations.
- Positive initial alternative mass and10,000 invented categorical RNG draws
  only, not environment actions/episodes. A fabricated residual can override
  the prior, so it is not a permanent reference-action mask.
- Same-start frozen/PPO/BC probabilities and sampler sequence, pending receipt
  and private-RNG restoration with EMPTY Adam state, atomic changed-prior and
  malformed zero-update state rejection. Nonempty Adam recovery is not tested
  for the new type in this authorized design-only step.
- Explicit optimizer-step/imitation-fit guards in new tests, unchanged global
  RNG/input tensors, legacy manifests and failure on nonfinite/schema errors.

An initial test run passed17 tests but emitted a scalar-conversion warning.
The test assertions were changed to detach tensors when inspecting numerical
values; a design-config authority/budget test was added. The134-test run is
clean. This was artificial test maintenance, not a scientific retry.

Command:
```
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests.test_reference_prior_candidate tests.test_candidate_policy_rollout \
  tests.test_candidate_ppo_objective tests.test_routing_candidate_contract \
  tests.test_matched_inputs tests.test_formal_replay_contract \
  tests.test_validated_returns -q
```

## Preservation and remaining gates

The two old failed trees,15 and279 files, their archive/manifest SHA256 values,
the original P1 proposal/protocol and P1-R1 effective packet are rechecked
unchanged. Evidence receipt:
`reports/2026-10-01-reference-prior-design/acceptance.json`.
No patient datasets, Dropbox copies, permissions or Howard work were changed.
New source naturally differs from the old execution source lock; this is a
prospective implementation, never a claim that old runs used this model.

The new protocol/config remain design-only and cannot launch the existing
runner. P2 readiness is FALSE. Before an approved new attempt, implement and
test the distinct phase order, analytical prior qualification, complete
optimizer/rollout/phase/budget restoration, frozen-model/test barrier and
independent raw verification. Also audit fresh streams against history and
freeze actual code/input/runtime locks. Update-boundary acceptance must include
bounded invented optimizer updates under the next explicit scope; current
zero-update recovery is not a substitute.

The proposed single P2 removes demo/BC-init fitting: maximum51,480 environment
calls,2,304 PPO/BC-CONTINUE optimizer steps and6h, all nontransferable subcaps in
protocol.md. Proposed396 final evaluations retain old loop shapes and check
closed-loop frozen/R4 equivalence; duplicate baseline lineage is not independent
evidence. No new reward/scenario, DDPG fit, holdout or Stage E reopening.

The startup answer is conditional: once the remaining integration/acceptance,
explicit bounded-execution approval and committed freeze are complete, the
single budgeted preflight can start in that same work session. No extra BC fit
or external data wait is required by this design. The6h cap is not an ETA;
neither an exact start time nor an RL improvement is yet established.

Next single decision: approve completing the P2 runner/artificial-update
acceptance and, only after every gate passes, one new bounded P2 attempt under
the proposed51,480/2,304/6h caps. Design consent alone is not that approval.
