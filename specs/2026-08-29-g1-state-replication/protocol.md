# Protocol

## Scientific role

Adjudicate one question: **on Stage G1's own states, is best-action label
agreement budget-limited?** No policy is trained, no checkpoint is selected,
no formal holdout stream is touched, and no existing evidence file is
modified.

## Inputs (all immutable, hash-verified before and after)

- Stage F1 training trees (control and candidate), from which G1 drew its
  frozen-pretrain trajectory states.
- The G1 configuration and summary
  (`experiments/evidence/patient_indexed_specimen_routing_ddpg_legal_action_ranker_g1/`,
  summary SHA256 `9e96deec850d6033fb5a2fb9e2f5090f8a040468df9faf8634015a14875d1c41`).
- Its three actor hashes, asserted unchanged on completion.

Any hash mismatch aborts the run before it produces a row.

## States and actions

- **The same 156 frozen-pretrain states G1 used**, reconstructed by the same
  procedure, with the scenario reconstructed from the benchmark plan — never
  from a snapshot's stored `env`, which is the defect that superseded the
  original F0/G0/G1 scenario labels.
- **The same five executed legal specimen actions** G1 enumerated. The state
  set and action family are copied, not redesigned: the whole point is to
  change *only* the replication budget.

## Worlds

- 64 worlds per state, from a seed family disjoint from G1's
  (`101000000`, `103000000`, `107000000`), from the formal holdout
  (`91100000`), from the dev streams, and from every reserved family.
- Complete pool: every action on every world. 156 × 5 × 64 = **49,920
  rollouts**. At the laptop's measured 0.155 s/rollout this is roughly 2.2
  hours single-threaded; the 4090 host should be faster and the run is
  embarrassingly parallel across states.

## Precondition gate (evaluated FIRST, before any other analysis)

Draw many random 3-versus-5 world splits from the pool and compute
best-action agreement, exactly G1's budget shape.

| Result | Consequence |
| --- | --- |
| Agreement within **±0.08 of 0.545** | Baseline reproduced. Proceed to the budget analysis. |
| Anything else | **`replication_failed_uninformative`.** Report the measured value against 0.545 and stop. Draw no conclusion about budget, and do not adjust the state set, action family, or seeds to chase the target — that would be fitting the replication to the answer. |

This gate exists because the laptop study lacked it. A prospective threshold
protects against choosing a number after seeing data; it does not protect
against an experiment failing to reproduce the condition it was meant to
probe.

## Budget analysis (only if the gate passes)

Agreement-versus-budget curve for k ∈ {1, 2, 4, 8, 16, 32} worlds per group,
≥200 random disjoint splits each, plus convergence against the full-pool
label as a reference (not as ground truth).

| Outcome | Classification | Consequence |
| --- | --- | --- |
| Agreement at k = 32 ≥ 0.70 | `g1_labels_budget_limited` | G1's finding is reclassified as a budget statement. The extension it closed is **reopened as a question** — reopening requires its own specification, and no manuscript claim changes on this study alone. |
| Agreement at k = 32 < 0.60 and last-doubling gain < 0.02 | `g1_labels_fundamentally_unstable` | G1's conclusion stands on 8× the evidence. The manuscript's online-attribution negative is strengthened. |
| Otherwise | `inconclusive_at_this_budget` | Recorded as such. No quiet extension. |

## Pre-registered prediction

To be written into this file and committed **before** the run, per house
practice. My current expectation, recorded now: the baseline reproduces
(the procedure is copied), and the curve climbs but lands in the inconclusive
band — G1's states carry genuinely harder labels than MDL-2-anchored ones, and
the 4-step-versus-remaining-horizon mismatch G0 documented (51.9% agreement
between horizons) is a structural cause that budget cannot remove.

## Prohibitions

1. No training, fine-tuning, or checkpoint selection.
2. No modification of G1, G0, F1, or Stage E artifacts; new output root only.
3. No adjustment of states, actions, or seeds after seeing any outcome.
4. One run, one analysis pass, one classification.
