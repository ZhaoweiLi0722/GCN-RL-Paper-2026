# Change-control proposal: replace the MDL-2 anchor with a coverage-correct MDL-3

Date: 2026-09-15. Status: **proposal for the user's approval; not appended to
the locked plan's amendment log and not in effect.** No formal artifact,
config, or evidence file is changed by this document.

## What is proposed

For every study launched after approval, the heuristic anchor and primary
comparator becomes MDL-3: the existing `mdl2` heuristic class with
`lookahead_periods = 3` (`base_policy_config: {"lookahead_periods": 3}`),
all other settings unchanged. The closed routing-primary campaign and its
published evidence are not reopened or re-labelled; the MDL-2 anchor remains
the comparator of record for those results.

## Why

1. **The executed anchor under-covers by one lead time.** MDL-2 targets two
   epochs of demand; production takes three epochs and capacity transfers up
   to three. Extending the lookahead to three lowers total cost by 3.65%
   pooled over six scenarios and 4.14% pooled over the four routing-primary
   scenarios, on seeds disjoint from the ones used to select it. The effect
   is 5.6% in the nominal scenario, where the demand prior is exactly right
   and no regime change exists, so it is a calibration defect, not a
   robustness story. Source: `specs/2026-09-15-prior-oracle-gap-screen/results.md`, Part 2.
2. **It is the largest available gain and it sits on a lever the formal
   learned policy could not touch.** The formal residual had authority only
   over specimen transfer (group scale 0.1; reagent, capacity, and
   replenishment 0.0). The learned policy's −0.66% and the anchor's −3.65%
   are therefore different levers, but a reviewer will ask why the anchor was
   not calibrated before a learned correction was layered on it.
3. **No configuration in the repository ever set a lookahead other than 2 or
   an order-up-to multiplier other than 1.0 for this anchor.** The historical
   "tuned heuristics" language referred to the choice among MYO, ISO, MDL-1,
   and MDL-2, not to calibrating MDL-2's coverage.

## Why MDL-3 rather than an order-up-to multiplier

Both repair the same defect. Head to head on fresh seeds, MDL-3 minus
order-up-to ×1.4:

| Pool | Δ cost | wins | Δ completion (pp) | Δ mfg-ineligibility (pp) | Δ lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| four routing scenarios | −0.63% [−0.74, −0.52] | 285/400 | −0.01 | +0.003 | +1.0 |
| all six scenarios | −0.39% [−0.46, −0.31] | 387/600 | −0.06 | +0.102 | +5.3 |

MDL-3 is cheaper in five of six scenarios with an identical clinical
profile on the routing set. It also needs no tuned constant: the lookahead
equals the production lead time, which is the standard coverage rule. The
multiplier has an interior optimum between 1.3 and 1.5 that would have to be
justified. MDL-4 is already over-covering (regional drift −0.6%, patients
lost +43 on the routing pool). Recommendation: MDL-3.

## The clinical guardrail problem, stated plainly

Under the formal protocol's own rule (paired 95% CI, margins: completion may
not fall more than 0.1 pp, manufacturing ineligibility may not rise more than
0.1 pp, patients lost may not rise more than 1.0; pooled, not per scenario),
MDL-3 on the four routing scenarios:

| Metric | Δ vs MDL-2 | 95% CI | passes |
| --- | ---: | ---: | :--: |
| completion service | +1.03 pp | [+0.87, +1.18] | yes |
| patients lost per episode | −22.2 | [−30.9, −13.4] | yes |
| manufacturing ineligibility rate | +0.203 pp | [+0.132, +0.274] | **no** |

The breach is real, not noise, and every retuned variant shares it. The
mechanism is mechanical: MDL-3 starts more patients, so more patients are in
manufacturing when they deteriorate past eligibility. The same rule, applied
to all six scenarios, passes (−0.55 pp), because the two drift scenarios
dominate.

This means the formal gate, as written, would veto a change that raises
completion by a full percentage point and saves 22 patients per episode in
order to avoid a 0.2 pp rise in one intermediate rate. Whether that is the
intended clinical weighting is a decision, not a measurement. Options:

- **A. Adopt MDL-3 and re-specify the manufacturing-ineligibility margin**
  (for example 0.5 pp, or replace the three separate margins with a single
  patient-outcome index). This must be fixed before any new study is run and
  recorded as its own amendment.
- **B. Keep the gate as written and keep MDL-2.** Then the paper must say
  that a one-parameter change to the heuristic lowers cost by 3.7% and
  patient loss by 22 per episode but is excluded by the ineligibility
  margin. That is defensible only if the margin has a clinical rationale the
  manuscript can state.
- **C. Keep the gate and search for an anchor variant that passes it.** Not
  recommended: it selects the anchor on the guardrail, which is the same
  post-hoc selection the protocol was written to prevent.

Recommendation: A, with the margin decision made now and frozen.

## What approval would and would not authorize

Authorizes: writing MDL-3 into the anchor slot of new development
configurations; running the specimen-routing headroom and label-stability
screen on the MDL-3 anchor (no training); updating
`specs/2026-09-15-network-input-dependence/plan.md` to name MDL-3 as its
anchor.

Does not authorize: retraining any learned policy, reusing the formal
holdout stream, editing the manuscript, or changing any published number.

## Open item this proposal does not settle

Whether the specimen-routing gain survives on a coverage-correct anchor is
unknown. The headroom screen (`specs/2026-09-16-routing-headroom-anchor-screen/results.md`)
has since been run: on MDL-3 the routing channel's prospective state-dependent
value is 0.02% of cost and labels replicate at 0.62, both below the
pre-training gates, and both weaker than on MDL-2. Room is not established;
only retraining could show a multi-step policy captures more than the probe.

Evidence files: `reports/anchor_decision_2026-09-15/anchor_decision_packet.{md,json}`;
row-level outputs under gitignored `results/prior_oracle_gap_screen_*`.
