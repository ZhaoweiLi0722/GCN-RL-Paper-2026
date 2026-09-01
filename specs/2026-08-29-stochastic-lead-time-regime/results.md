# Results

## Exploratory variability diagnostic — 2026-08-29

### Decision

**No formal gate decision is supported.** The unsigned draft protocol required
an optimality gap against a hindsight benchmark or strong rollout policy. The
executed analysis instead measured the cost of stochastic lead-time
variability relative to a fixed-mean lead. It is retained as an exploratory
diagnostic and did not authorize any learned-policy work.

The reported pooled cost of lead-time variability for the tuned
lead-time-aware planner was **+0.176%**. The draft protocol's 1% floor and the
literature's 7.1% concern heuristic optimality gaps, so they are context rather
than valid thresholds for this different estimand.

No downstream gating protocol was run, no learned policy was built, and no
held-out data was spent.

### What was measured

The diagnostic asks how much stochasticity costs the same tuned planner. It
compares stochastic lead times against a lead fixed at the identical mean.
This is useful, but narrower than asking how far the planner lies from a
hindsight or rollout benchmark.

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

The point estimates are consistently positive and stochastic leads are worse
on most seeds in every scenario. PR #9 did not include the original rows or
the executable configuration that produced this table, so the exact values
and the reported safety multiplier cannot be reconstructed from the original
artifacts.

### Publication reproduction — 2026-09-01

A post-hoc executable reproduction was added after the original `+0.176%`
result was known. It uses fresh development seeds to select the safety
multiplier, fresh evaluation seeds, 24 exactly paired replications per
scenario, and the same five-point lead distribution. It is a reproducibility
audit of the fixed-mean variability estimand, not an independent confirmation
and not the optimality-gap gate described by the unsigned draft.

| Scenario | Reproduced gap | Normal 95% half-width | Seeds worse | Exact RNG end state |
| --- | ---: | ---: | ---: | ---: |
| routing_abrupt_regime_shift | +0.0824% | +/-0.0789% | 17/24 | 24/24 |
| routing_compound_regional_stress | +0.1559% | +/-0.1153% | 21/24 | 24/24 |
| routing_nominal_history | +0.1124% | +/-0.0542% | 21/24 | 24/24 |
| **Pooled** | **+0.1169%** | **+/-0.0479%** | **59/72** | **72/72** |

The pooled interval clusters the three scenario outcomes by their shared
evaluation seed (24 independent seed clusters); scenario intervals use their
24 seed-level outcomes directly.

The reproduction selected safety multiplier `1.25`, rather than the reported
`1.0`. It therefore does not reproduce the original tuning choice or exact
point estimates. It does reproduce the scientifically relevant narrow result:
with a lead-aware planner, stochastic lead times impose a small positive cost
relative to fixing the same lead at its mean. The executable config, all 72
paired rows, summary, and artifact inventory are stored under
`results/stochastic_procurement_variability_reproduction/`.

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

This is a useful exploratory negative, not a formal gate result.

The motivating finding was that an achievable learned-policy margin is bounded
by the incumbent heuristic's own optimality gap, and that random lead times are
a documented regime where that gap reaches 7.1%. The implementation includes
order crossing, but the executed diagnostic measured a different estimand:
**0.176%** variability cost relative to a fixed-mean lead.

The supported conclusion is narrower: **this diagnostic found little cost from
lead-time variability after giving the heuristic lead-time information.** It
suggests that multi-resource coupling absorbs much of the perturbation. It
does not establish the tuned heuristic's optimality gap and therefore cannot
rule out exploitable headroom without the benchmark required by the draft
protocol.

Combined with the routing and overtime evidence, the observation is consistent
with a heuristically saturated simulator, but it is corroborating rather than
decisive evidence for that claim.

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
