# Completion-count rule: recorded-evidence result

## Decision

A simple fixed event-count rule reaches the restricted completion-information
optimum in the two persistent-change, nonbinding-downstream cells. Thus even
without exact remaining work, feedback value in these examples does not require
online neural updates. It does not reach the coupled-queue bound. That gap is
not evidence that RL would beat a competent planner or frozen history policy.

All values below are deterministic expected **synthetic cost units** with the
unchanged terminal closure. Smaller is better. No confidence intervals or
operational/clinical meaning are inferred from these finite worlds.

| Releases | Downstream | Response | Count rule | Fixed balanced | Completion-view known-law bound | Rule minus bound |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Batch | Nonbinding | Nominal | 81.25 | 81.25 | 81.00 | 0.25 |
| Batch | Nonbinding | Persistent change | 103.25 | 109.00 | 103.25 | 0.00 |
| Batch | Shared bottleneck | Nominal | 109.25 | 109.25 | 102.25 | 7.00 |
| Batch | Shared bottleneck | Persistent change | 121.25 | 125.00 | 115.00 | 6.25 |
| Booked flow | Nonbinding | Nominal | 71.25 | 71.25 | 71.00 | 0.25 |
| Booked flow | Nonbinding | Persistent change | 93.25 | 99.00 | 93.25 | 0.00 |
| Booked flow | Shared bottleneck | Nominal | 99.25 | 99.25 | 92.25 | 7.00 |
| Booked flow | Shared bottleneck | Persistent change | 111.25 | 115.00 | 105.00 | 6.25 |

The rule balances for the first three decisions, then allocates to the side
with more support-stage jobs after a persistent change. It performs no
parameter estimation, model simulation, optimization or neural update. Nominal
paths remain balanced throughout. Both individual changed-world paths are
recorded, not only their weighted means. Batch/flow remain offset copies, not
independent replications of a mechanism.

The richer completion-view bound can use stages, downstream state, commitments,
public costs and the known finite-world law. Our rule voluntarily uses fewer
fields and ignores downstream congestion. Therefore the bottleneck gap can
include ordinary scheduling/information-processing value. It is **not** a
clean test of online learning, remaining-work observability, or system
identification. No new rule was fitted to close it after seeing the table.

## Provenance and verification

- Fixed-rule source/protocol commit: `5650ac1eccfdbf39160d63b0c7c41b7dd5f46d65`.
- Inputs: the original 16,368 tree edges and the previously saved
  completion-information diagnostic. No environment or planner was invoked by
  the replay calculation. Original action grid, delays, costs, releases and
  closure were unchanged.
- Outputs: [summary](../../reports/2026-09-29-completion-count-comparator/run/summary.json),
  [12 paths](../../reports/2026-09-29-completion-count-comparator/run/episodes.json),
  [60 decisions](../../reports/2026-09-29-completion-count-comparator/run/decisions.json),
  and [source/input/output hashes](../../reports/2026-09-29-completion-count-comparator/run/provenance.json).
- Independent upstream row accounting and all 59 inventory hashes passed
  before and after replay. The three output files reproduced byte for byte in
  a second in-memory computation. A separate count/action/cost walk also
  verified all 12 paths without calling the comparator function.
- Fifty-seven focused tests and full Python compilation passed. Tests reject
  full-state input, poison unavailable progress fields, and verify no new
  transition/planner call, closure inclusion, invalid inputs, and no overwrite.

## Reproduction

Run from the integration worktree; use a fresh destination, never the archived
`run` directory. This reads the frozen inputs and writes additive analysis only.

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m evaluation.audit_service_queue_completion_rule \
  --output /private/tmp/completion-count-reproduction-new
"$PY" -m unittest tests.test_service_queue_completion_rule
```

The source-lock check intentionally refuses uncommitted source differences.
The 12 paths and 60 decisions are extracted from existing exhaustive evidence;
they are not new independent episodes or a new scientific validation sample.

## Remaining boundary

C1's bounded software question is answered. A practical censored-data
identification-MPC and a frozen history-conditioned neural policy have not been
compared here. Implementing either with a chosen statistical model is a new
methodological decision, not a parameter-free continuation of this result.
The E1 operational contract remains uncalibrated. Proceed to documentation and
reproducibility consolidation, not a DDPG campaign on these solved small cells.
