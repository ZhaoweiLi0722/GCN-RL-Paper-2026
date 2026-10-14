# Follow-Up Study Line: Numbers and Evidence Appendix

Reference sheet for the three decision channels screened after the online-DDPG
attribution post-mortem. Status date 2026-09-01. Routing and overtime are
closed. The stochastic lead-time study was not formally authorized and did
not execute its drafted optimality-gap gate; its exploratory diagnostic did
not justify advancement. **No policy was trained at any point** in this line
of work.

Companion documents: `docs/online_rl_attribution_postmortem_and_followup_brief.md`
(why these channels were chosen), and the per-study `results.md` files cited
below.

---

## 1. Summary of outcomes

| Channel | Decision question | Verdict |
| --- | --- | --- |
| Patient-indexed specimen routing | Do counterfactual best-action labels replicate? | **No** — 54.5% agreement vs a 70% gate (Stage G1, pre-existing) |
| Overtime capacity | Is the best setting predictable from decision-time state? | **No** — 0.267 top-1 vs a 0.50 gate, worse than a state-blind constant |
| Stochastic procurement lead times | Did the exploratory diagnostic justify a formal study? | **No advancement** — reported +0.176%, post-hoc reproduction +0.1169%; formal optimality-gap gate not executed |

The evidence identifies two closed-channel mechanisms and one exploratory
observation: the routing target does not replicate; the overtime target is
stable but does not generalize from decision-time state; and a lead-aware
heuristic appears insensitive to lead-time variability in the reported
diagnostic.

---

## 2. Reference results the study line rests on

From the frozen Stage E formal holdout (5 seeds x 4 scenarios x 100 paired
replications, holdout seed 91100000). Unchanged by this work.

| Comparison | Delta cost | Relative | 95% interval |
| --- | ---: | ---: | --- |
| AFR-GCN-DDPG minus MDL-2 | -17.690M | -0.658% | [-19.361, -15.725]M |
| AFR-Flat-DDPG minus MDL-2 | -9.086M | -0.338% | [-10.587, -7.510]M |
| AFR-GCN-DDPG minus AFR-Flat-DDPG | -8.604M | -0.321% | [-11.065, -6.428]M |

Attribution noise floor (Stage F1 final-versus-frozen): +/-0.001% to 0.004%.
Every screen threshold below is set relative to this floor.

---

## 3. Overtime capacity channel

### 3.1 Screens, in order

| Screen | Criterion | Threshold | Measured | Outcome |
| --- | --- | ---: | ---: | --- |
| Headroom (first pass) | states with validated material saving | >=0.30 | 1.00 / 0.33 | passed |
| State-dependence (first pass) | prospective value, out of sample | >=0.005 | **-0.00083** | **failed** |
| Headroom (re-calibrated) | states with validated material saving | >=0.30 | 1.00 / 1.00 | passed |
| State-dependence (re-calibrated) | prospective value, out of sample | >=0.005 | **+0.006446** | passed |
| Interior optimum | best rung strictly interior | >=0.30 | 0.926 | passed |
| Label stability, argmax | fresh-stream best-action agreement | >=0.70 | 0.926 | passed |
| Label stability, pairwise | material-pair sign agreement | >=0.80 | 0.990 | passed |
| Learnability, pooled | held-out-seed top-1 | >=0.50 | **0.252** | **failed** |
| Learnability, worst fold | held-out-seed top-1 | >=0.50 | **0.148** | **failed** |
| **Confirmatory, state-dependence** | prospective value on reserved data | >=0.005 | **+0.004500** | **failed** |
| Confirmatory, learnability | held-out-seed top-1 | >=0.50 | **0.267** | **failed** |

### 3.2 The calibration change between the two passes

| Parameter | First pass | Re-calibrated |
| --- | ---: | ---: |
| `max_overtime_fraction` | 0.3 | 0.6 |
| `weight_overtime_quadratic` | 5,000 | 20,000 |
| Marginal cost crosses the 50,274 shortage benefit at | s\*=1.76 (outside range) | s\*=0.88 (inside) |

Made under change control with a new config name and output root, after the
first pass produced a corner solution (best rung was the maximum in 26 of 27
states).

### 3.3 The exploratory-versus-confirmatory gap

| Measurement of state-dependence value | Result |
| --- | ---: |
| Re-calibrated exploratory screen | +0.6446% |
| Independent re-measure on the stability screen's rows | +0.6182% |
| **Confirmatory run, reserved scenario** | **+0.4500%** |

Three internally consistent exploratory replications, including one with
1.000 cross-family best-arm agreement, did not predict the held-out result.
The reserved scenario and seed families were fixed and code-enforced before
exploration continued, and predictions were committed before the run.

### 3.4 Representation and model class do not change the verdict

Best result per feature family over a ridge penalty grid from 0.1 to 10,000
(gate: top-1 >= 0.50):

| Features | Dimension | Best top-1 | Best realized cost vs constant | Headroom captured |
| --- | ---: | ---: | ---: | ---: |
| Network aggregates | 15 | 0.319 | -0.2655% | 41.2% |
| + distributional shape | 27 | **0.356** | -0.2599% | 40.3% |
| + graph message passing | 34 | 0.341 | -0.2529% | 39.2% |
| Graph features alone | 7 | 0.215 | — | — |

Gradient-boosted trees, same states and folds:

| Features | Model | Top-1 | Worst fold | vs constant | Headroom |
| --- | --- | ---: | ---: | ---: | ---: |
| Aggregates | depth 1, 60 rounds | 0.393 | 0.074 | -0.2721% | 42.2% |
| Aggregates | depth 3, 60 rounds | 0.244 | 0.148 | -0.0701% | 10.9% |
| Aggregates | depth 3, 150 rounds | 0.311 | 0.148 | -0.0886% | 13.7% |
| All families | depth 1, 60 rounds | 0.333 | 0.185 | -0.2386% | 37.0% |
| All families | depth 3, 60 rounds | 0.178 | 0.111 | -0.0514% | 8.0% |
| All families | depth 3, 150 rounds | 0.230 | 0.148 | -0.0439% | 6.8% |

Depth 3 is strictly worse than depth 1 in every pairing: capacity for
interactions overfits rather than helps, so interactions are not the missing
ingredient. Realized cost plateaus near 42% of the available headroom
regardless of model class.

---

## 4. Routing label budget study

Tested whether Stage G1's 54.5% was a property of the channel or of its
eight-world replication budget. 27 states, 5 legal specimen actions, 64
worlds, 8,640 paired rollouts.

| Worlds per group | Best-action agreement |
| ---: | ---: |
| 1 | 0.785 |
| 2 | 0.796 |
| 4 | 0.822 |
| 8 | 0.855 |
| 16 | 0.912 |
| 32 | 0.952 |

**At Stage G1's exact budget shape (3 vs 5 worlds): 0.821 here, against G1's
reported 0.545.** Budget cannot explain a 0.28 gap measured at the same
budget, and the curve never starts below the 0.70 gate. The frozen reading
rule returned `g1_negative_underpowered`; that classification's precondition
is violated and was **not** adopted. Honest classification:
`precondition_violated_cannot_adjudicate_g1`. **Stage G1's negative stands
unchanged.**

Secondary, allocation comparison at matched budget:

| Simulator calls per arm | Uniform | Sequential halving |
| ---: | ---: | ---: |
| 2 | 0.793 | 0.784 |
| 4 | 0.826 | 0.804 |
| 8 | 0.869 | 0.835 |
| 16 | 0.907 | 0.880 |

Sequential halving is worse than uniform at every budget on a 5-arm ladder,
contradicting its synthetic validation (0.837/0.940/0.987 against
0.730/0.860/0.960 with 11 arms). Recommended only for many-armed screens.

---

## 5. Stochastic lead-time exploratory diagnostic

The step-0 protocol remained unsigned and defined an optimality-gap gate
against hindsight or a strong rollout benchmark. The executed analysis instead
measured variability cost against a fixed-mean lead. The first table preserves
Howard's reported values. PR #9 did not contain the original executable config
or rows, so the exact table cannot be reconstructed from the original
artifacts and is not a formal gate result.

Cost of lead-time variability for the tuned lead-time-aware planner, measured
exactly paired: the override draws the lead and then discards it, so both runs
consume an identical random stream. Lead distribution (0.1, 0.2, 0.4, 0.2, 0.1)
over 0-4 epochs, mean 2.0, variance 1.2, order crossing possible. 24
evaluation seeds per scenario; safety multiplier tuned on 10 separate
development seeds.

| Scenario | Gap | 95% interval | Seeds worse |
| --- | ---: | ---: | ---: |
| routing_abrupt_regime_shift | +0.113% | +/-0.064 | 20/24 |
| routing_compound_regional_stress | +0.160% | +/-0.172 | 21/24 |
| routing_nominal_history | +0.256% | +/-0.210 | 17/24 |
| **Pooled** | **+0.176%** | | |

A publication audit then reran that narrower variability estimand from an
explicit frozen config. This post-hoc reproduction was designed after the
original result was known, so it is neither an independent confirmation nor a
replacement for the unexecuted optimality-gap gate.

| Scenario | Reproduced gap | Normal 95% half-width | Seeds worse | Exact RNG end state |
| --- | ---: | ---: | ---: | ---: |
| routing_abrupt_regime_shift | +0.0824% | +/-0.0789% | 17/24 | 24/24 |
| routing_compound_regional_stress | +0.1559% | +/-0.1153% | 21/24 | 24/24 |
| routing_nominal_history | +0.1124% | +/-0.0542% | 21/24 | 24/24 |
| **Pooled** | **+0.1169%** | **+/-0.0479%** | **59/72** | **72/72** |

The pooled interval clusters the three scenario outcomes by their shared
evaluation seed (24 independent seed clusters); scenario intervals use their
24 seed-level outcomes directly.

The fresh development tuning selected safety multiplier `1.25`, rather than
the reported `1.0`. Thus the exact original numbers and tuning choice are not
reproduced. Direction and small magnitude are reproduced: stochastic leads
cost the lead-aware planner about one tenth of one percent relative to a lead
fixed at the same mean.

The draft protocol's 1% floor and the literature's 7.1% refer to heuristic
optimality gaps, not this variability-cost estimand. They therefore cannot be
used to classify this table. Supporting reported measurements, 12 seeds:

| Configuration | Cost | Relative |
| --- | ---: | ---: |
| No lead time, MDL-2 | 2137.4M | — |
| Stochastic lead, lead-blind MDL-2 | 2154.8M | +0.81% vs no lead |
| Stochastic lead, lead-aware MDL-2-LT | 2135.1M | -0.91% vs lead-blind (12/12 seeds) |

A correctly specified planner recovers essentially all of the loss, landing
within 0.11% of the no-lead cost in the reported diagnostic. This suggests,
but does not prove, that procurement-lead uncertainty leaves little headroom.

---

## 6. Evidence fingerprints

| Result root | Rows | Config SHA256 | Rows SHA256 |
| --- | ---: | --- | --- |
| `continuous_overtime_headroom_e2` | 2,592 | `ecd624963f49` | `d3a32e65c663` |
| `continuous_overtime_headroom_e2b` | 2,592 | `a9126e3ba180` | `352f9c9591e9` |
| `continuous_overtime_label_stability_e3` | 2,592 | `6bd2378f4b1c` | `dec632c3f050` |
| `continuous_overtime_critic_ranking_e4` | 12,960 | `c308fafd9825` | `bb854c9308e7` |
| `continuous_overtime_confirmatory` | 4,320 | `1c1e2d25daae` | `f42c2f07892f` |
| `routing_label_budget_study` | 8,640 | `cfac2cb97d36` | `d011d7850e9f` |
| `stochastic_procurement_variability_reproduction` | 72 | `e83eca64c0a5` | `9ef1bf7cb267` |

The final row is a post-hoc publication reproduction of the variability-cost
estimand. It supports only the direction and scale statement above; it does
not convert the unsigned optimality-gap draft into a preregistered result.

Prefixes shown; full digests are in each `summary.json`. Curated copies for the
earlier large studies live under `experiments/evidence/<result root>/`; the
compact 72-row publication reproduction is committed under its `results/`
root. Every row carries a live
provenance assertion that the scenario it claims is the scenario it ran in —
the defect that superseded the original F0/G0/G1 mechanism audits.

---

## 7. Specifications

| Directory | Contents | Status |
| --- | --- | --- |
| `specs/2026-08-29-continuous-overtime-control/` | requirements, plan, validation, held-out reservation, frozen protocol, results | complete, channel closed |
| `specs/2026-08-29-routing-label-budget-study/` | README, protocol, validation, results | complete, precondition violated |
| `specs/2026-08-29-stochastic-lead-time-regime/` | README, unsigned draft protocol, exploratory report, post-hoc reproduction | protocol deviation recorded; no formal gate decision |
| `specs/2026-08-29-g1-state-replication/` | README, protocol | **drafted, not authorized; requires the RTX 4090 host** |

---

## 8. Reusable instruments produced

| Module | Purpose | Tests |
| --- | --- | ---: |
| `evaluation/state_dependence_value.py` | Value of adapting to the situation, versus the best tuned constant, selected out of sample | 15 |
| `evaluation/label_stability.py` | Fresh-stream best-action and pairwise-sign agreement | 12 |
| `evaluation/ranking_feasibility.py` | Leave-one-seed-out learnability with per-fold floors and a state-blind comparator | 11 |
| `evaluation/sequential_halving_labeling.py` | Budget-efficient labelling; validated on synthetic ground truth, negative on short ladders | 11 |
| `evaluation/audit_overtime_headroom_e2.py` | Headroom screen with the code-enforced held-out reservation | 20 |
| `evaluation/routing_label_budget_pool.py` | Complete paired-CRN outcome pool for offline budget replay | 17 |

Test suite grew from 721 to 829 over this line of work.

---

## 9. Prediction scorecard

Recorded because the pattern is informative, not for its own sake.

| Prediction | Outcome | |
| --- | --- | --- |
| Overtime re-calibration would fail | Passed at +0.6446% | wrong |
| Routing labels would show instability at higher budget | 0.952 agreement at 32 worlds | wrong |
| Lead-time variability cost expected at 2-4% in an unsigned draft | reported +0.176% | exploratory estimate much smaller |
| Overtime learnability would fail | Failed | right |
| Held-out confirmation would pass state-dependence | Failed at +0.45% | wrong |

Four overtime/routing predictions were recorded within the authorized study
line; one was correct and the misses were optimistic about available headroom.
The lead-time expectation was written before implementation but remained in
an unsigned draft, so it is reported separately rather than counted as a
registered prediction.

---

## 10. Corrections made to this line's own record

| Claim | Status |
| --- | --- |
| "Paired comparisons weaken over the horizon as arms diverge" | **False.** Pairing is exact: identical RNG end-state, enrollment counts and patient ids. Corrected in three documents. |
| First headroom gate ("is there room to improve over the anchor?") | **Insufficient.** A constant policy answers yes; replaced with a state-dependence criterion. |
| Sequential-halving labeller | Validated on synthetics, **negative on 5-arm ladders**; scope narrowed. |
| Gradient-boosting nonlinearity check | First implementation was additive and could not represent interactions — the hypothesis under test. Caught by its own sanity check and rebuilt. |
| Lead-time "gate failed" | **Overstated.** The unsigned protocol specified an optimality-gap benchmark; the run measured fixed-mean variability cost. Reclassified as an exploratory diagnostic with no formal gate decision. |
| Lead-time environment wiring | Initially wired only the base environment; the patient environment overrides `step()`, so it was inert where it mattered. Caught by a two-configuration cost comparison returning identical values. |
