# Protocol

## Scientific role

Test whether making reagent procurement lead times **stochastic with order
crossing** creates a regime in which a tuned look-ahead heuristic is
structurally misspecified, opening headroom that a state-dependent policy can
capture. The study's first job is to measure the heuristic's own optimality
gap; everything else is contingent on that gap being materially larger than
the ~1% we see today.

## Environment extension (flag-gated, default off)

New configuration surface on `CapacityPlanningConfig`:

```
reagent_purchase_lead_time: int = 0          # deterministic component
reagent_lead_time_distribution: str = "deterministic" | "geometric" | "discrete"
reagent_lead_time_probabilities: tuple[float, ...] = ()   # order-crossing allowed
enable_stochastic_procurement: bool = False
```

Mechanics, mirroring the existing transfer pipeline:

- An order placed at epoch *t* enters a procurement pipeline and arrives at
  *t + L*, where *L* is drawn per order from the configured distribution.
- **Order crossing is permitted**: a later order may arrive before an earlier
  one. This is the property that breaks conventional base-stock reasoning and
  is the reason the literature reports large heuristic gaps under random lead
  times.
- With the flag off, behaviour is bit-identical to the current environment —
  the same non-negotiable regression test used for overtime.
- State exposure: outstanding on-order quantity by remaining age, appended to
  per-facility features (flat and graph alike), so the flat baseline is
  representation-matched.

`reagent_purchase_lead_time` already has a consumer in the heuristics and no
producer in the environment; this study supplies the producer.

## Comparators (built and tuned BEFORE any learned policy)

1. **MDL-2** unchanged — the lead-time-blind incumbent, which will look bad
   and must not be the headline comparison.
2. **MDL-2-LT** — the same planner with `reagent_purchase_lead_time` set, so
   its order-up-to horizon covers the commit lead. This is the honest
   incumbent and the baseline every later claim is measured against.
3. **MDL-2-LT tuned** — a development sweep over its safety-stock multiplier
   under the stochastic regime.

## Deliverable 1: the heuristic's optimality gap (a gate, not a result)

Against a hindsight benchmark (a policy with knowledge of realised lead times,
or a strong rollout policy where an exact benchmark is intractable), measure
the tuned MDL-2-LT optimality gap under the stochastic regime.

| Measured gap | Consequence |
| --- | --- |
| ≥ 3% | Regime change succeeded. Proceed to the gating protocol. |
| 1–3% | Marginal. Report and decide explicitly whether to continue; do not drift onward by default. |
| < 1% | **Regime change failed.** The heuristic is robust to lead-time uncertainty too. Report that as the finding and stop — it materially strengthens the "heuristically saturated" conclusion. |

A sub-1% result here is a genuinely valuable negative: it would show that the
literature's 7.1% figure does not transfer to this problem class, which is a
sharper claim than anything the current manuscript makes.

## Deliverable 2: the gating protocol (only if the gap gate passes)

Unchanged from the overtime study, applied to the procurement decision:

1. **Headroom** over the tuned MDL-2-LT anchor: ≥30% of states with a
   validated, clinically noninferior material saving.
2. **Value of state-dependence** (primary): prospective, discovery-selected
   and validation-scored, against the best tuned *constant* order-up-to
   adjustment — ≥0.5% of anchor cost, plus an interior-optimum fraction.
3. **Label stability** on CRN streams disjoint from those used to select.
4. **Learnability** from decision-time state, leave-one-seed-out, with
   per-fold floors and a state-blind comparator.

Only then does any training specification become discussable, and it needs its
own sign-off.

## Reservation, from the outset

A held-out scenario and seed families are reserved **before exploration
begins** and enforced in `validate_config`, exactly as for the overtime study.
Exploration is labelled exploratory; one confirmatory run on reserved data,
with predictions committed beforehand.

## Pre-registered expectation

Recorded now, before any implementation. The literature's 7.1% comes from
single-echelon inventory control where lead-time uncertainty directly governs
the ordering decision. Here reagents are one of three binding resources
(specimens, capacity, reagents), so lead-time noise on one of them should
produce a **smaller** gap than 7.1% — I expect **2–4%**, landing in or just
above the marginal band. That is still 3–6× the current ceiling and would be
the largest headroom this project has measured.

## Prohibitions

1. No training until the gap gate and all four screens pass.
2. No modification of any existing result root, evidence file, or closed
   campaign artifact; new output root only.
3. Flag-off behaviour must remain bit-identical, enforced by regression test.
4. The lead-time-blind MDL-2 may never be the headline comparator.
