# Held-Out Confirmation Set — Reservation

Reserved: 2026-08-29, **before** any exploratory work beyond Stage E4.

## Why this document exists

Stages E2 through E4 developed the screening protocol while looking at
results. Two gates were amended after a screen failed. Each amendment was
individually defensible and ran under change control with fresh configs and
output roots, but a sequence of choices made after seeing data is a garden of
forking paths: it converges on a passing result whether or not one exists.

The remedy is not to stop exploring. It is to **separate exploration from
confirmation**, and to fix the boundary in advance so the confirmation is
genuinely uncontaminated. Everything done so far is hereby designated
exploratory. This file reserves the data on which the eventual confirmatory
run will be executed.

## Reserved and untouchable

**Scenario:** `routing_regional_drift`

Never used in E2, E2b, E3, or E4, all of which used only
`routing_nominal_history`, `routing_abrupt_regime_shift`, and
`routing_compound_regional_stress`.

**Generation seeds:** `97100000`–`97100009`
**Discovery replication seeds:** `97200000`–`97200004`
**Validation replication seeds:** `97300000`–`97300009`

None of these seed families has been used by any screen. They are disjoint
from every exploration family (`96100000`, `96200000`, `96300000`,
`96400000`, `96500000`, `96600000`, `96700000`, `96800000`), from the routing
campaign's development streams (`94000000`, `94100000`), and from the formal
holdout (`91100000`).

## Rules

1. **No exploratory run may touch the reserved scenario or any reserved
   seed.** Not for debugging, not for a smoke test, not for a "quick look".
2. **The confirmatory run happens once.** If it fails, that is the result. It
   is reported, not repeated with adjustments.
3. **The protocol must be frozen and committed before the confirmatory run
   starts**, including every gate, threshold, feature set, model class, and
   hyperparameter, with the commit hash recorded in the results.
4. **Both phases are always reported together.** The exploratory findings are
   real evidence about what the environment does; they are simply not
   confirmatory evidence, and must never be presented as such.
5. If exploration ends without a protocol worth confirming, the reserved set
   stays unused and the honest report is the exploratory negative.

## What confirmation can and cannot establish

It can establish that the frozen protocol's verdict replicates on a scenario
and seed families that took no part in developing it. That answers the
forking-paths objection with data instead of argument.

It cannot fully undo the fact that the protocol was *designed* while looking
at the exploratory scenarios. A stronger design would draw the confirmation
regime from a different environment family altogether. This limitation should
be stated plainly in any write-up rather than glossed.

## Status

- Reserved: 2026-08-29
- Exploration phase: **open**
- Confirmatory run: **not started**, protocol not yet frozen
