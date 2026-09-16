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
