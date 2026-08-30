# Protocol

## Scientific role

This is a **measurement of an estimator**, not a policy experiment. The
quantity under study is how reliably the best legal specimen action can be
identified as a function of simulation budget and allocation scheme. No policy
is trained, no checkpoint is selected, no formal stream is touched, and no
manuscript claim changes on the basis of this study alone.

## Relationship to Stage G1 (what this can and cannot say)

Stage G1's manifold was 156 frozen-pretrain trajectory states drawn from the
Stage F1 training runs, whose checkpoint artifacts live on the RTX 4090 host
and are not available on this machine. This study therefore uses **fresh
anchor-policy decision states** rather than G1's exact states.

Consequence, stated up front: a budget-sensitivity finding here transfers to
G1 as an *argument about the estimator class* (same simulator, same action
family, same CRN mechanics, same replication regime), not as a literal
re-analysis of G1's rows. If the finding warrants it, an exact-state
replication on the 4090 host is the follow-up — it is not part of this study.

## States

- Scenarios: `routing_nominal_history`, `routing_abrupt_regime_shift`,
  `routing_compound_regional_stress` — the exploratory set. The reserved
  confirmation scenario (`routing_regional_drift`) and the reserved seed
  families (`97100000`–`97300009`) are untouched, and the existing
  code-level reservation guard applies.
- Environment: the routing-primary patient environment with
  `enable_specimen_routing` on and **overtime off** — this is a routing study.
- State generation: roll the MDL-2 anchor from seeds `98100000`–`98100002`,
  snapshotting at epochs `30, 55, 80` → 27 candidate states.
- **Distinct-execution filter (from G1):** a state enters the study only if
  all five actions produce distinct executed integer routings there. States
  failing the filter are recorded and excluded; the count is reported. If
  fewer than 18 states survive, add generation seeds `98100003`–`98100004`
  before any outcome is examined.

## Actions

Five arms per state: the MDL-2 anchor, plus the anchor with the specimen
net-flow block shifted by `−0.10, −0.05, +0.05, +0.10` along a **centred
state-derived pressure pattern**, clipped to [−1, 1].

The pattern is `p = (waiting − min(idle capacity, reagents))`, mean-centred and
scaled to unit maximum absolute value, computed once per state and frozen, so
every world sees the identical action vector for a given (state, arm).

**Amendment, 2026-08-29, before any pool collection and before any outcome was
observed.** The first draft specified a *uniform* shift across clinics. A
pre-run probe showed that produces **zero executed routes for every arm**:
specimen net flow is conserved, so shifting all clinics in the same direction
leaves no counterparties and nothing can move. The five arms would have been
behaviourally identical and the study vacuous — the requested integer net
differed, but no patient was routed. Centring the shift makes some clinics
offer and others request; the same probe then yields five distinct executed
flows at epochs 30, 55 and 80, moving 5–13 patients. Centred pressure patterns
are also the form the project's existing structured-exploration machinery uses
(`project_tensor_to_pattern_basis`: "transfer corrections use one centred
pressure pattern").

The distinct-execution filter below operates on **executed** flows
(`specimen_transfers`), never on requested ones — the probe showed requests can
differ while executions are identical, so filtering on requests would admit
degenerate states.

## Worlds and pairing

- World seeds: `98200000`–`98200063` — a pool of **64 worlds per state**.
- Every arm is evaluated on every world (pool collection is uniform and
  complete): restore the state snapshot, reseed the environment RNG with the
  world seed, apply the arm's action once, roll the anchor to episode end,
  record remaining-episode cost and the clinical counters.
- CRN pairing is exact by construction (verified property of this simulator);
  a per-row provenance assertion checks scenario identity, as in every screen
  since the E2 defect.

Budget: ≤27 states × 5 arms × 64 worlds = **≤8,640 rollouts**, ~30 minutes at
the measured ~0.19 s/rollout.

## Offline analyses (no further simulation)

All allocation questions are answered by replaying policies against the pool.

**A. Agreement-versus-budget curve (primary).** For each
`k ∈ {1, 2, 4, 8, 16, 32}`: draw many (≥200) random splits of the 64 worlds
into two disjoint groups of size `k`; in each group, label each state with the
arm of lowest mean cost; record the fraction of states where the two groups
agree. Report the mean curve with a percentile band over splits.

**B. G1-replication point.** The curve evaluated at G1's actual budget shape
(groups of 3 and 5) is the direct comparison to G1's 54.5%.

**C. Allocation comparison.** At matched total simulator budget
`B ∈ {5·2, 5·4, 5·8, 5·16}` calls per state per group, replay uniform
allocation and sequential halving (`evaluation/sequential_halving_labeling.py`)
against the pool and report each scheme's two-group agreement curve. Sequential
halving consumes pool worlds in a fixed order, so the replay is exact and
unbiased.

**D. Convergence diagnostic.** Agreement of each k-group labelling against the
full-pool (k = 64) labelling, which estimates how far each budget sits from
the pool's own limit. The full-pool label is *not* treated as ground truth —
it is itself an estimate — and is reported only as a reference point.

## Prospective reading rule (fixed before any data)

Let `A(k)` be the mean two-group agreement at k worlds per group, and
`A_G1` the value at the G1 budget shape (analysis B).

| Outcome | Classification |
| --- | --- |
| `A(32) ≥ 0.70` | `g1_negative_underpowered` — the 70% gate is reachable with budget; the G1 conclusion is reclassified as a budget statement and the question it closed is **reopened** (further work needs its own spec) |
| `A(32) < 0.60` and the curve's last doubling gains < 0.02 | `g1_negative_confirmed_fundamental` — labels do not stabilize at 8× the G1 budget; the closed conclusion stands with stronger evidence |
| otherwise | `inconclusive_at_this_budget` — reported as such; any extension to larger pools requires a new change-control entry, not a quiet re-run |

Secondary, not gating: whether sequential halving reaches a given agreement at
materially (≥2×) lower budget than uniform — this calibrates the labeller for
any future screen regardless of the primary outcome.

## Pre-registered prediction

Recorded before execution, per house practice. Based on the overtime channel's
slow gain (~+0.02 agreement per budget doubling from a 0.83 base) and G1's
much lower 0.545 base, I predict the curve **climbs but does not reach 0.70 by
k = 32** — i.e. the likeliest outcome is `inconclusive_at_this_budget`, with
`g1_negative_confirmed_fundamental` next. A clean `g1_negative_underpowered`
result would surprise me and would be the most consequential outcome, which is
exactly why the study is worth running.

## What this study must not do

1. Touch the reserved overtime confirmation scenario or seeds, any formal
   holdout stream (`91100000`), the dev streams (`94000000`/`94100000`), or
   any G1 seed family (`101000000`, `103000000`, `107000000`).
2. Train, fine-tune, or select any policy or critic.
3. Modify any existing result root, evidence file, or the G1 record.
4. Rerun or extend itself after seeing outcomes. One pool, one analysis pass,
   one classification. Extensions require a new change-control entry.
