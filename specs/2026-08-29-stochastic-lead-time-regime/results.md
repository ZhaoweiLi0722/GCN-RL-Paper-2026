# Results

## Optimality-gap gate — 2026-08-29

### Decision

**GATE FAILED. The regime change did not work. The study stops here, as the
protocol requires.**

Pooled cost of lead-time variability for the tuned lead-time-aware planner:
**+0.176%**, against a 1% floor and the literature's 7.1%.

No gating protocol was run, no learned policy was built, and no held-out data
was spent.

### What was measured

The gate asks whether making procurement lead times stochastic leaves the
tuned heuristic materially misspecified. Operationalised as the **cost of
lead-time variability**: the same tuned planner under stochastic lead times
versus under a lead fixed at the identical mean.

The comparison is exactly paired. `procurement_lead_override` still *draws*
the lead before discarding it, so both runs consume an identical random
stream and differ only in whether the lead varies. Lead distribution
`(0.1, 0.2, 0.4, 0.2, 0.1)` over 0–4 epochs: mean exactly 2.0, variance 1.2,
order crossing possible.

24 evaluation seeds per scenario, tuned safety multiplier 1.0 selected on 10
separate development seeds.

| Scenario | Variability gap | 95% interval | Seeds worse |
| --- | ---: | ---: | ---: |
| routing_abrupt_regime_shift | +0.113% | ±0.064 | 20/24 |
| routing_compound_regional_stress | +0.160% | ±0.172 | 21/24 |
| routing_nominal_history | +0.256% | ±0.210 | 17/24 |
| **Pooled** | **+0.176%** | | |

The effect is real — consistently positive, and worse on most seeds in every
scenario — and it is roughly **40× too small** to matter.

### Why the literature's 7.1% did not transfer

Two reasons, and the second is the interesting one.

**Reagents are one of three binding resources.** Production is
`min(specimens, idle capacity, reagents)`. The published 7.1% comes from
single-echelon inventory control where the ordering decision *is* the problem.
Here, lead-time noise on one input is absorbed by slack in the others: when a
reagent order is late, capacity or specimens are frequently the binding
constraint anyway, so the delay costs nothing.

**A correctly specified planner absorbs almost all of it.** Under stochastic
lead times the lead-blind MDL-2 loses 0.81% against the no-lead baseline, and
the lead-aware planner recovers it: −0.91% against lead-blind, better on 12/12
seeds, landing within 0.11% of the no-lead cost. Two standard corrections were
enough — order against inventory *position* (on hand plus on order) rather
than on-hand, and cover the lead plus the lookahead window. Lead-time
uncertainty is not a source of structural misspecification here; it is a
source of misspecification only for a planner that ignores it.

### Pre-registered prediction versus outcome

| Prediction | Outcome | |
| --- | --- | --- |
| Gap of 2–4%, below the literature's 7.1% but well above the 1% floor | **+0.176%** | **wrong by an order of magnitude** |

I predicted the direction correctly — smaller than 7.1% because reagents are
one of three resources — and badly underestimated how much smaller. This is
the third optimistic prediction this session to be refuted by measurement
(after the overtime re-calibration and the routing label budget study), and
the pattern is consistent: my priors about how much headroom this simulator
holds run high, and the measurements keep returning near zero.

### What this establishes

This is a substantive negative, not a failed experiment.

The deep-research finding was that an achievable learned-policy margin is
bounded by the incumbent heuristic's own optimality gap, and that random lead
times are a documented regime where that gap reaches 7.1%. We implemented that
regime faithfully — order crossing included — and measured **0.176%**.

The conclusion is therefore stronger than "we could not find headroom here":
**a regime the literature identifies as favourable to learned control does not
produce exploitable headroom in this problem class.** The reason is
structural — multi-resource coupling means no single input's uncertainty
dominates — and it applies to any lever that perturbs one resource at a time.

Combined with the closed routing and overtime channels, the picture is
consistent: this simulator's operational decisions are heuristically
saturated, and the sub-1% ceiling is a property of the problem rather than of
the method.

### What was built and kept

The environment extension is committed and tested, flag-gated and off by
default. It remains available if a future study wants lead-time uncertainty
for a different purpose.

- Stochastic procurement with order crossing, drawn once per facility per
  epoch so the random stream stays action-independent and exact CRN pairing
  survives.
- `MDL-2-LT`, a lead-time-aware planner ordering against inventory position.
- 20 tests, including patient-environment wiring, pipeline conservation, CRN
  preservation under changed order quantities, and the inventory-position rule.

### Implementation note

The first wiring pass touched only the base environment. Every routing and
overtime study uses the **patient** environment, which overrides `step()` and
computes its own reagent balance, so the extension was inert exactly where it
mattered — while the base-environment equivalence test passed. The symptom was
two configurations returning identical costs to four significant figures.
Patient-environment wiring tests were added so this cannot recur silently.
