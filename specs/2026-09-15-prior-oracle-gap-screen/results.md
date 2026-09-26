# Prior-oracle gap screen: results

Date: 2026-09-15. Branch `e2b-run`. Diagnostic study; no training, no
checkpoint, no formal-holdout seed reuse, no change to any existing artifact.

## Question

Before designing a study where "MDL-2 misjudges the demand distribution while
RL is trained over a broader distribution", measure the ceiling of that idea:
how much does MDL-2 lose from its wrong demand prior, holding its decision rule
fixed? The paired difference between MDL-2 with the executed prior and MDL-2
with the simulator's true current rate is the most any policy could recover
purely from better distributional knowledge under the same decision rule.

## Design

- **Arms.** One decision rule, `MeanDemandLookahead2Policy`, five demand-rate
  inputs: executed prior (`mdl2_prior`, verified bit-identical to the stock
  anchor on 10 worlds); true current Poisson mean including regime drift and
  transient shocks (`mdl2_oracle_rate`); base rate times regime multiplier
  only (`mdl2_oracle_regime`); and deployable adaptive variants that blend the
  prior with the 12-epoch rolling arrival mean at weights 0.05, 0.10, 0.25,
  0.50, 1.00 (`mdl2_rollingNNN`).
- **Scenarios.** The four routing-primary scenarios composed exactly as the
  formal benchmark composes them for heuristics, plus the two 2026-07 demand-
  drift configs re-run under the routing contract and the current
  500k/100k/25k weights. The latter are labelled `routing_contract_*` and are
  not the July studies.
- **Worlds.** 100 paired replications per scenario on seed stream 99,100,000
  plus 100,000 per scenario index, unused elsewhere in the repository. Arrival
  fingerprints were recorded per episode and matched across all arms in every
  scenario, so world sharing is verified rather than assumed.
- **Statistics.** Paired world-level percentile bootstrap, 20,000 resamples,
  equal scenario weights when pooling.
- **Code.** `evaluation/prior_oracle_gap_screen.py`;
  `evaluation/trace_prior_oracle_abrupt_shift.py` for the mechanism trace.
  Row files (276 MB and 215 MB) stay under gitignored `results/`; their
  SHA-256 are `18ccc95e…9ae36` and `bf87e7f9…de4e2`. Summaries, metadata, and
  resolved environment configs are versioned under `artifacts/`.

## Mismatch inventory

| Scenario | prior Σλ | true Σλ initial → final | max clinic-level misspecification | shocks |
| --- | ---: | ---: | ---: | --- |
| routing_nominal_history | 101.0 | 101.0 → 101.0 | 0% | p=0.12, ×2.4 |
| routing_abrupt_regime_shift | 101.0 | 109.0 → 97.3 | 25% → 40% | none |
| routing_regional_drift | 101.0 | 101.0 → 117.5 | 0% → 40% | none |
| routing_compound_regional_stress | 101.0 | 101.0 → 121.1 | 0% → 50% | p=0.16, ×2.6 |
| routing_contract_demand_drift | 101.0 | 123.4 (constant) | 40% | p=0.12, ×2.4 |
| routing_contract_demand_drift_severe | 101.0 | 168.8 (constant) | 80% | p=0.12, ×2.4 |

## Result 1: in the routing-primary scenarios the distribution-knowledge ceiling is at most 0.08%

Total cost, arm minus executed prior, percent of prior, 95% paired CI:

| Scenario | oracle_rate | oracle_regime | rolling005 |
| --- | ---: | ---: | ---: |
| routing_nominal_history | −0.076% [−0.140, −0.020] | 0 (no regime) | −0.213% [−0.406, +0.002] |
| routing_regional_drift | +0.020% [−0.057, +0.097] | +0.020% [−0.058, +0.097] | −0.085% [−0.310, +0.174] |
| routing_compound_regional_stress | −0.008% [−0.079, +0.064] | −0.052% [−0.106, +0.003] | −0.174% [−0.298, −0.044] |
| routing_abrupt_regime_shift | **+2.449%** [+2.027, +2.868] | +2.449% [+2.035, +2.871] | +0.095% [−0.170, +0.426] |

A clairvoyant demand rate buys MDL-2 nothing under slow drift (regional drift,
compound stress) and 0.08% under transient shocks (nominal, where the oracle
knows shock timing). Under the abrupt shift it is 2.4% *worse* than the stale
prior. The formal routing-primary evidence therefore contains no headroom that
a "better estimate of the demand distribution" could unlock; the learned
policies' 0.44–0.76% gains over MDL-2 come from something else.

## Result 2: why the oracle loses under the abrupt shift

Thirty-world per-tier trace (`trace_prior_oracle_abrupt_shift.py`). Before
epoch 26 the oracle knows tiers 5–9 and 15–19 run at 0.80×/0.85× of prior, so
it lowers their order-up-to targets and lets capacity flow to the 1.25×/1.15×
tiers. Idle bioreactors in tier 5–9 fall from 20.9 to 17.3 per epoch. At epoch
26 those tiers jump to 1.40×/1.30×. Bioreactors are the binding resource,
capacity transfers take up to three epochs, and MDL-2 has no anticipation of
the flip, so the deficit persists: post-shift bioreactor shortage rises by 319
(tier 5–9) and 212 (tier 15–19) unit-epochs and patient loss by 51.5 and 27.8
per world. The gains in tier 0–4 (−36 lost) do not compensate. Net: +43.7
patients lost, +26.8M bioreactor-shortage cost, +21.9M patient-loss cost.

The "wrong" balanced prior acts as an accidental hedge against regime change.
Knowing the current mean is not the same as knowing a change is coming; a
point-estimate oracle removes the hedge without adding anticipation.

## Result 3: persistent level misspecification does create headroom, roughly proportional to its size

| Scenario | misspec | oracle_rate | oracle_regime | Δ patients lost (oracle) | Δ completion (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| routing_contract_demand_drift | 40% | −0.783% [−1.046, −0.519] | −0.879% [−1.141, −0.615] | −19.7 | +0.42 |
| routing_contract_demand_drift_severe | 80% | −2.854% [−3.008, −2.701] | −2.916% [−3.074, −2.764] | −177.1 | +1.90 |

Components in the 40% case: the oracle buys 12.3M more reagent, cuts reagent
shortage by 8.7M and bioreactor shortage by 22.0M, and reduces patient loss by
9.9M. This is the regime the 2026-07 demand-drift study targeted. Under that
study's own protocol (50k weights, replenishment residual, no routing) the
learned residual captured −0.016%; the ceiling measured here under the current
contract is about −0.8%. The severe case has 8.25% completion service under
the prior, so the network is overwhelmed and the oracle mostly reallocates
loss rather than preventing it.

## Result 4: the simple deployable estimator does not capture the headroom, and noise-following is costly

Rolling-mean weight sweep, total cost versus executed prior:

| Scenario | w=0.05 | w=0.10 | w=0.25 | w=0.50 | w=1.00 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routing_nominal_history | −0.213% | +0.365% | +0.519% | +0.921% | +4.390% |
| routing_regional_drift | −0.085% | +0.500% | +0.978% | +1.430% | +4.815% |
| routing_compound_regional_stress | −0.174% | +0.129% | +0.695% | +1.012% | +3.387% |
| routing_abrupt_regime_shift | +0.095% | +1.260% | +1.968% | +4.232% | +10.689% |
| routing_contract_demand_drift | +0.017% | +0.233% | +0.276% | +0.616% | +3.117% |
| routing_contract_demand_drift_severe | −0.286% | −0.423% | −0.911% | −1.281% | −1.276% |

Even in the nominal scenario, where the prior is exactly right and the rolling
mean is an unbiased estimator of it, following the rolling mean costs 0.4% at
weight 0.10 and 4.4% at weight 1.00. The order-up-to rule amplifies Poisson
noise and transient shocks into inventory swings. Only the overwhelmed severe
case rewards aggressive adaptation. A history-following heuristic is not a
free comparator; it needs regularization, and the pooled effect of every
weight above 0.05 is negative.

## Interpretation

1. **"MDL-2 misjudges the distribution" is not a lever in the existing
   routing-primary scenarios.** The ceiling is ≤0.08% where it is positive and
   negative under the abrupt shift. Any proposed RL advantage there cannot be
   attributed to broader training distributions.
2. **Where headroom exists, it is level misspecification that persists for
   the whole horizon**, of order 0.8% at 40% misspecification. That is a
   plausible, bounded target, but the July study already attempted it and
   captured 2% of the ceiling. A proper adaptive estimator with explicit
   regularization (for example, exponential smoothing with a change-point
   guard) is the mandatory comparator before any learned policy is credited.
3. **The abrupt-shift result reframes what "knowing the distribution" should
   mean.** The value is in anticipating change and hedging capacity across
   tiers, not in tracking the current mean. That is a structural, network-
   level decision: where to hold idle bioreactors given that demand will move.
   It is also exactly the kind of decision a static point-estimate heuristic
   cannot express and a graph policy with history could in principle learn.
   This is the strongest mechanistic argument so far for the
   [network-input-dependence](../2026-09-15-network-input-dependence/plan.md)
   direction, and it argues for a hedging-aware heuristic baseline in that
   study's comparator list.
4. **Nothing here measures what a learned policy would capture.** The screen
   bounds the value of information under a fixed decision rule. A policy with
   a different decision rule (hedging, anticipatory transfers) can in
   principle exceed the oracle-MDL-2 bound, as the abrupt-shift case shows,
   because the bound is on knowledge, not on rules.

## Proposed next screen (not launched)

Add to the network-input-dependence Stage 1 a fourth oracle arm: MDL-2 given
the *future* rate at horizon equal to transfer plus production lead time
(anticipation oracle). Its gap over the executed prior under the abrupt shift
measures the value of anticipation separately from the value of the current
mean. If it is large and a regularized adaptive heuristic cannot capture it,
that is the environment signature the follow-up study should build on.

---

# Part 2: anticipation versus hedging versus anchor calibration

Same worlds (seed base 99,100,000, 100 per scenario) unless marked *fresh*
(seed base 99,700,000, disjoint). Same MDL-2 rule throughout; only its inputs
or its two scalar settings change. Arrival fingerprints matched across arms in
every run.

## Result 5: anticipation is worth nothing; the executed anchor is under-covering by roughly one lead time

Total cost, arm minus executed MDL-2, percent of prior:

| Scenario | anticipate 4 ep | anticipate 8 ep | regime-max hedge | order-up-to ×1.3 | order-up-to ×1.5 | MDL-3 | MDL-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal history | 0.000 | 0.000 | 0.000 | −4.80 | −5.11 | **−5.52** | −4.75 |
| regional drift | −0.07 | −0.32 | −0.44 | −2.35 | −2.00 | −2.77 | −0.18 |
| compound stress | −0.09 | −0.14 | −0.63 | −2.19 | −1.87 | −2.87 | −2.16 |
| abrupt shift | +2.47 | +2.47 | −4.11 | −3.98 | −5.17 | −4.93 | −4.64 |
| demand drift (40%) | −0.88 | −0.88 | −0.88 | −2.61 | −2.58 | −3.51 | −3.26 |
| demand drift severe (80%) | −2.92 | −2.92 | −2.92 | −2.55 | −3.49 | −2.91 | −4.51 |
| **pooled** | −0.83 | −0.87 | −1.64 | −2.90 | −3.26 | **−3.53** | −3.44 |

- **Anticipation** (knowing the regime rate four or eight epochs ahead,
  shocks excluded) equals the current-rate oracle in every scenario. Under
  the abrupt shift it is still +2.47% worse than the stale prior. Timing
  knowledge does not help MDL-2 because its rule reacts to the level it is
  given; a higher future rate simply moves the same mistake earlier.
- **Hedging without any future knowledge** captures the whole abrupt-shift
  penalty and more: the regime-max hedge (knows the set of regimes, not the
  timing) is −4.11%; a uniform ×1.3 order-up-to multiplier, which knows
  nothing about regimes at all, is −3.98%.
- **The same multiplier improves the nominal scenario by 4.8 to 5.1%**, where
  the prior is exactly right and no regime exists. The effect is therefore
  not about uncertainty. The executed anchor is under-covering: MDL-2 targets
  two epochs of demand while production takes three and capacity transfers
  take up to three. Extending the lookahead to three epochs (MDL-3) is the
  most natural repair and is the best single arm pooled (−3.53%). MDL-4 is
  already too long; regional drift collapses to −0.18%.
- The order-up-to sweep (1.05 → 3.0) has an interior optimum between 1.3 and
  1.5 in every scenario, then degrades to +5 to +13% at 3.0. This is a
  standard newsvendor-shaped response, not a monotone "more is better".

*Fresh-seed confirmation* (99.7M, disjoint from selection): ×1.3 −3.00%,
×1.4 −3.28%, ×1.5 −3.38%, MDL-3 −3.65%, MDL-4 −3.62%, all with 95% CIs
excluding zero. Selection on the development stream did not inflate the
effect.

## Result 6: the mechanism is capacity pooling, and it trips the manufacturing-ineligibility guardrail

Components, nominal scenario, ×1.2 versus prior: bioreactor-shortage penalty
−87.9M, patient-loss cost −19.8M, reagent purchase +11.1M, transshipments
+164 per episode. A higher target workload raises the shortage signal that
drives MDL-2's sharing rules, so more idle capacity is moved toward clinics
under pressure. Reagent spend rises modestly; the saving is almost entirely
the bioreactor-shortage term, which is the largest single component of the
objective (0.96–1.11 billion of 2.2–2.6 billion).

Applying the formal study's own scenario-level clinical guardrails
(completion may not fall by more than 0.1 pp, manufacturing ineligibility may
not rise by more than 0.1 pp, patients lost may not rise), point differences
on the fresh stream:

| Scenario | MDL-3 Δcost | Δcompletion (pp) | Δmfg-inelig (pp) | Δlost | passes |
| --- | ---: | ---: | ---: | ---: | :--: |
| nominal history | −5.60% | +1.58 | +0.21 | −39.2 | no (ineligibility) |
| abrupt shift | −5.38% | +1.36 | +0.30 | −24.8 | no (ineligibility) |
| regional drift | −3.05% | +0.27 | +0.48 | +11.9 | no |
| compound stress | −3.09% | +0.90 | −0.18 | −36.6 | yes |
| demand drift | −3.67% | +1.35 | −0.35 | −46.5 | yes |
| demand drift severe | −2.82% | +1.85 | −3.76 | −169.1 | yes |

Every retuned arm raises completion service by 1 to 2 percentage points and,
in five of six scenarios, lowers patient loss by 25 to 170 per episode. But
more patients started means more patients become ineligible *during*
manufacturing, and that rate rises by 0.2 to 0.5 pp in the nominal, abrupt,
and regional-drift scenarios, breaching the 0.1 pp margin. Regional drift
also loses 12 more patients. Under the strict per-scenario gate the retuned
heuristics would fail exactly where the formal learned policy's seeds 10 and
14 also failed (abrupt shift). Whether a 0.1 pp rise in manufacturing
ineligibility should veto a 1.5 pp rise in completion and 40 fewer lost
patients is a clinical weighting question the guardrail design has not
answered; the cost objective already says yes.

## What this means for the existing evidence

1. **The formal anchor is miscalibrated on a lever the learned policy could
   not touch.** The formal residual has group scales
   `specimen_transfer 0.1, reagent 0.0, capacity 0.0, replenishment 0.0`.
   The −0.66% learned gain is specimen routing only. The −3.5% here is
   reagent and capacity coverage, outside the learned action authority. The
   two numbers are not competing for the same headroom, but a reviewer will
   ask why the anchor was not tuned before a learned correction was layered
   on it, and whether the routing gain survives on top of a tuned anchor.
   That last question needs retraining and is not answered here.
2. **"Tuned heuristics" in the historical notes were not tuned on this
   setting.** No configuration in the repository sets
   `local_order_up_to_multiplier` or a lookahead other than 2 for the
   routing-primary anchor. The 2026-07 robustness conclusions that "tuned
   heuristics beat learned control" were correct in direction and understated
   in magnitude.
3. **The RL-beats-heuristic framing loses another 3.5 points of ground; the
   graph-versus-flat and structural findings do not.** Graph-minus-flat is a
   matched comparison on the same anchor and is unaffected. The reporting
   audit, the crossed-bootstrap result, and the online-RL null stand.
4. **For the follow-up environment study**, the treatment must be defined
   relative to a coverage-correct anchor (MDL-3 or ×1.4, chosen on a
   development stream and frozen), or any learned gain will be confounded
   with the anchor's under-coverage. The plan's comparator list should name
   this explicitly.

## Not done here

No learned policy was retrained on a retuned anchor. No guardrail margins
were changed. No formal artifacts were touched. Choosing MDL-3 versus ×1.4 as
the new anchor, and whether the clinical gate should be re-weighted, are
protocol decisions for the change-control process, not results of this
screen.
