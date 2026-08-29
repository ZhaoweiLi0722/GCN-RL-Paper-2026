# Results

## Stage E2: overtime headroom screen — 2026-08-29

### Decision

The literal gate **passed**: `overtime_headroom_established`. Both non-nominal
scenarios cleared the prospective ≥30% state fraction.

**E3 is nevertheless NOT recommended on this configuration.** The screen
also produced a diagnostic that defeats the purpose the channel was built
for: the optimum sits at the ladder boundary in 26 of 27 states. Details and
recommendation below. Advancing to E3 as-is would measure a degenerate
problem.

### Execution

Implementation commit `9effc17`, run on the merged `bded504` tree.
2,592 rows = 27 states × 12 arms (anchor + 11 rungs) × 8 CRN replications
(3 discovery + 5 validation). Runtime 4m40s. All rows unique, finite, and
count-checked; every row passed the live-environment provenance assertion.

| Scenario | States | Material validated | Fraction | Discovery/validation best-arm agreement | Prospective success |
| --- | ---: | ---: | ---: | ---: | ---: |
| routing_nominal_history | 9 | 8 | 0.89 | 1.00 | 0.89 |
| routing_abrupt_regime_shift | 9 | 9 | **1.00** | 0.89 | 1.00 |
| routing_compound_regional_stress | 9 | 3 | **0.33** | 0.89 | 0.33 |

Gate: ≥30% of states in some non-nominal scenario with a validation-stream
saving ≥1M modeled units that is clinically noninferior. Both non-nominal
scenarios pass (1.00 and 0.33).

Median best-arm saving: 5.41M (abrupt shift), 0.45M (compound stress),
2.37M (nominal). Compound stress is the weakest regime and includes one
state with a negative best saving (−0.69M).

### The corner-solution problem

Mean validation cost relative to the MDL-2-OT anchor, averaged over all 27
states:

| Rung | vs anchor |
| --- | ---: |
| u_0.00 (no overtime) | +4.19M |
| u_0.20 | +2.66M |
| u_0.40 | −0.55M |
| u_0.60 | −4.53M |
| u_0.80 | −8.47M |
| u_1.00 | **−11.23M** |

Cost falls monotonically to the boundary, and `u_1.00` is the best arm in
**26/27 states** (the exception picks `u_0.80`).

The arithmetic explains it. With `initial_idle_bioreactors = 5` and
`max_overtime_fraction = 0.3`, the largest surge is 1.5 reactors per clinic,
so the marginal overtime cost never exceeds

```
15,000 + 2 x 5,000 x 1.5 = 30,000
```

while a single averted capacity shortage is worth 50,274, and an averted
patient loss is worth 500,000 under the calibrated weights. Marginal benefit
exceeds marginal cost across the entire admissible range, so the optimum is
always the upper bound.

**Why this matters more than the passing gate.** The overtime channel was
introduced to supply a *state-dependent interior optimum* — the smooth
"how much, given this state" trade-off that DDPG-class methods need and that
the specimen-routing channel cannot provide (see
`docs/online_rl_attribution_postmortem_and_followup_brief.md` §5.2). A corner
solution supplies headroom but not that geometry:

1. The optimal policy is the constant "always surge to the cap", which is
   exactly the `static_ot` tuned-scalar comparator the spec defined as the
   reference a learned policy must beat.
2. E3 label stability would pass trivially — the best action is `u_1.00`
   almost everywhere, so discovery and validation agree for a degenerate
   reason, not because the environment exposes a learnable decision.
3. E4 ranking, and any later learned policy, would be fitting a constant.

The screen therefore did its job: it caught a mis-calibrated environment
*before* any agent was trained against it. This is the gating protocol
working as designed, and it is reportable as such.

The cost weights responsible (`weight_overtime_linear = 15_000`,
`weight_overtime_quadratic = 5_000`) were flagged as unreviewed when the
screen was opened for review (PR #7), precisely because they had not been
sanity-checked against the real cost scale by anyone but their author.

### The decisive measurement: value of state-dependence

The corner solution prompted a second analysis of the same rows, comparing
three policies over the 27 states. The oracle picks the best rung per state
with knowledge of the realized outcome, so it is an upper bound no policy can
exceed:

| Policy | Total validation cost (M units) |
| --- | ---: |
| MDL-2-OT anchor (closed-form rule) | 12,283.3 |
| Best **constant** rung (`u_1.00`, the `static_ot` reference) | 11,980.0 |
| Per-state **oracle** (upper bound for any policy) | 11,979.4 |

- Overtime is worth **303.2M** over the anchor. The channel has real value.
- The in-sample oracle beats the best constant by only **0.67M (0.0054%)**.
- Selecting per-state arms **prospectively** — chosen on the discovery stream,
  scored on the held-out validation stream, with the constant chosen the same
  way — is **worse than the constant by 10.20M (−0.083%)**.

The prospective figure is the one that governs, and the gap between the two
matters. An oracle that picks each state's arm using the same means it is
scored on capitalizes on replication noise; it overstates what any real policy
can achieve. Selecting on independent data — the discipline Stage G1 was built
around — the per-state policy does not merely fail to beat a constant, it
**loses to one**. Discovery and validation agree on the best arm in 92.6% of
states, but that agreement is on `u_1.00` almost everywhere; where they
disagree, following the discovery signal costs more than ignoring it.

Only 2 distinct arms are ever optimal (`u_1.00` in 26 states, `u_0.80` in 1),
and the best arm is strictly interior in 3.7% of states.

**This is the finding that governs the study.** A state-dependent policy fitted
to real data on this channel is worth less than a one-line constant. There is
no budget for a learned policy to capture — the quantity is negative, not
merely small — so no algorithm, architecture, training budget, or graph
encoder can extract a defensible result. For scale, the loss is −0.083% where
the Stage F1 attribution noise floor was ±0.001–0.004%.

It is also the routing failure repeating in a new setting: a simple baseline
banks the value, and the learned component has nothing left to attribute. The
overtime channel was supposed to escape that, and as configured it does not.

Measure and gate implemented in `evaluation/state_dependence_value.py`
(15 tests, including one where pure noise produces a spuriously positive
in-sample oracle and a correctly negative prospective value). The gate refuses
a report that carries no prospective value rather than falling back to the
optimistic figure. Applied to these rows it classifies the channel
`channel_captured_by_constant_policy`; report at
`experiments/evidence/continuous_overtime_headroom_e2/state_dependence.json`.

### Recommendation (requires change control; not executed)

Do not advance to E3 on this configuration, and do not silently re-tune and
re-run — that would be post-hoc selection against an observed result. The
proper route is an appended change-control entry, under a new config name and
output root, in this order:

1. **Amend the E2 gate before re-calibrating anything.** Make the value of
   state-dependence the *primary* criterion, with the headroom criterion
   demoted to a necessary-but-insufficient precondition. Proposed threshold:
   `prospective_value_of_state_dependence_fraction ≥ 0.005` (0.5%), roughly
   100× the attribution noise floor, plus an interior-best-arm fraction ≥ 0.30.
   The criterion must be the prospective value, never the in-sample oracle:
   on this very screen the oracle reads +0.0054% while the honest
   out-of-sample value is −0.083%.

   The current gate asks "is there headroom over the anchor?" and a corner
   solution answered yes. The study needs headroom *a learned policy could
   capture that a tuned constant cannot*, which is a different question the
   gate never asked. An interior-optimum criterion alone is also insufficient:
   an interior optimum that sits at the same rung in every state is still
   captured exactly by a constant. Only the oracle-versus-best-constant gap
   measures the right quantity.

2. **Then re-calibrate and re-run**, with the amended gate deciding. Raising
   `weight_overtime_quadratic` toward ~20,000 puts the marginal cost at full
   surge near 75,000 — above the 50,274 shortage benefit, below the 500,000
   patient-loss benefit — which should move the optimum interior and make it
   track how much patient-loss risk each state actually carries. Widening
   `max_overtime_fraction` gives that optimum room to move between states.

3. **If no calibration clears the amended gate, reject the channel and say
   so.** Capacity surge may simply be an "almost always useful" lever whose
   optimum barely moves with state. That is a fast, cheap negative, and it is
   more valuable than tuning toward a passing number.

A caution for step 3: "find a channel that passes" must not become the
objective. If several candidate channels all show near-zero state-dependence
value, that is itself the result — evidence that this simulator's operational
decisions are heuristically saturated, which is a sharper and more defensible
claim than the current manuscript makes.

### Methodological note

The value-of-state-dependence measure should have been part of E2's design
from the start; its absence is a design defect in the screen as originally
specified, not a discovery enabled by running it. It is cheap (reuses rows
already collected, no training, no extra rollouts) and general: it is a
pre-training test for whether learning can possibly help on a given decision
channel. Applied to the original routing action space it would have predicted
the online-DDPG null in an afternoon rather than across Stages F1, G0, G1,
and H0/H1.

This makes it a stronger candidate contribution to the follow-up study's
gating-protocol framing than the overtime channel itself.

### Evidence

- Rows: `results/continuous_overtime_headroom_e2/headroom_rows.csv`
- Summary: `results/continuous_overtime_headroom_e2/summary.json`
  (carries `config_sha256`, `plan_sha256`, and `rows_sha256`)
- Config: `experiments/configs/continuous_overtime_headroom_e2.json`
- Implementation: `evaluation/audit_overtime_headroom_e2.py`

### Known limitation

Arms share a CRN seed and a common start state, but once arms diverge in
patient counts the per-patient deterioration draws desynchronize, so pairing
weakens over the remaining horizon. This matches the H0/G0/G1 precedent.
Tightening it would require per-stream RNG partitioning. It does not affect
the corner-solution finding, which is a monotone effect far larger than the
pairing noise.

---

## Stage E2b: re-calibrated headroom screen — 2026-08-29

### Decision

**PASSED both criteria: `state_dependent_headroom_established`. Stage E3 is
authorized.**

The re-calibration was executed under the approved change-control amendment,
with a new config name (`continuous_overtime_headroom_e2b.json`) and output
root. The E2 result stands as recorded; nothing about it was revised.

The author of the amendment predicted this run would fail. It did not, and the
evidence that it genuinely passed is stronger than a single threshold
crossing — see "Why this looks like signal" below.

### Calibration change

| Parameter | E2 | E2b |
| --- | ---: | ---: |
| `max_overtime_fraction` | 0.3 | 0.6 |
| `weight_overtime_quadratic` | 5,000 | 20,000 |
| Surge range (reactors/clinic) | 0 – 1.5 | 0 – 3.0 |
| Marginal cost crosses the 50,274 shortage benefit at | s\*=1.76 (outside range) | s\*=0.88 (**inside**, 29% of full surge) |

Everything else — scenarios, states, ladder, CRN streams, clinical rule,
headroom threshold — is unchanged from E2.

### Result

2,592 rows, 27 states, 4m42s. Headroom precondition passed in both non-nominal
scenarios (abrupt shift 1.00, compound stress 1.00; nominal 0.89).

| Policy | Total validation cost |
| --- | ---: |
| MDL-2-OT anchor (closed-form rule) | 14,059.5M |
| Best **constant** rung (`u_0.80`) | 13,772.3M |
| **Prospective** per-state (chosen on discovery, scored on validation) | 13,681.7M |
| In-sample oracle | 13,678.4M |

| Criterion | Measured | Gate | |
| --- | ---: | ---: | --- |
| Prospective value of state-dependence | **+0.6446%** | ≥ 0.5% | PASS |
| Interior best-arm fraction | **0.926** | ≥ 0.30 | PASS |

Totals are higher than E2 in absolute terms because overtime is now priced
roughly four times higher at the margin; the comparison that matters is
between policies within this calibration.

### Why this looks like signal, not noise

Four independent indications, and the first is the important one:

1. **The in-sample and prospective values nearly coincide**: +0.6677% versus
   **+0.6446%**, a gap of 0.023 points. In E2 the same comparison read
   +0.0054% versus −0.083% — the sign flipped, which is the signature of
   selecting on noise. Here, selecting each state's rung on independent data
   costs almost nothing relative to the optimistic bound, which is what a real
   effect looks like.
2. **Discovery/validation best-arm agreement is 96.3%** across 27 states. The
   Stage G1 gate that failed on the routing channel required 70%; the routing
   channel managed 54.5%.
3. **The optimum is genuinely spread**: 8 distinct rungs are optimal somewhere
   (`u_0.30` ×3, `u_0.40` ×6, `u_0.50` ×8, `u_0.60` ×1, `u_0.70` ×1,
   `u_0.80` ×1, `u_0.90` ×5, `u_1.00` ×2), interior in 92.6% of states. E2 had
   2 distinct arms and 3.7% interior.
4. **The closed-form anchor is beaten by a constant**, and the constant is
   beaten by state-dependence: 14,059.5M → 13,772.3M → 13,681.7M. The
   MDL-2-OT rule is not capturing the state-dependent structure, so this is
   not a case where a better heuristic trivially absorbs the gain.

For scale, the prospective value (0.64%) is comparable to the manuscript's
headline routing result (0.658% versus MDL-2), and roughly 150× the Stage F1
attribution noise floor.

### What this does NOT yet establish

The measured quantity is the value of choosing a rung **per state**, where the
27 states are known and each is scored under held-out replication noise. That
is not the same as a policy that **generalizes to unseen states** from
observable features. Two gaps remain, and they are exactly what the remaining
gates test:

- The per-state optimum does not track the obvious summary statistic. Grouping
  states by their best rung shows no clean ordering in the number of
  capacity-bound clinics (means 17.3, 19.0, 18.1, 19.0, 14.0, 13.0, 19.6,
  19.5 across `u_0.30` … `u_1.00`). Whatever drives the optimum is not a
  one-dimensional count, which is encouraging for a graph encoder and a
  warning against assuming a simple rule will do.
- The value could depend on realized future demand rather than on anything
  observable at decision time. **Stage E4's held-out-seed ranking is the test
  that separates those cases**, and it must pass before any actor is trained.

The per-state penalty for simply using the best constant is a median of 3.47M
(max 8.58M), so the effect is not carried by one outlier state.

### Recommendation

Proceed to Stage E3 (label stability) on this calibration, then E4. Stage E5
still requires its own specification and sign-off, and the prohibition on
training before E4 passes is unchanged.

Zhaowei's review of both the amendment and this result remains outstanding.

### Evidence

- Rows: `results/continuous_overtime_headroom_e2b/headroom_rows.csv`
- Summary: `results/continuous_overtime_headroom_e2b/summary.json`
  (config/plan/rows SHA256)
- State-dependence report:
  `experiments/evidence/continuous_overtime_headroom_e2b/state_dependence.json`
- Config: `experiments/configs/continuous_overtime_headroom_e2b.json`

---

## Stage E3: label stability on fresh streams — 2026-08-29

### Decision

**PASSED both criteria: `labels_replicate_on_fresh_streams`. Stage E4 is
authorized.** Training remains unauthorized.

### A circularity avoided

The spec defines the E3 gate as discovery/validation best-action agreement
≥70%. Stage E2b already reported that number — 96.3% — but computed from the
rows it selected on. Reporting it as a passing E3 would be circular: one
dataset cannot both fit the selection and test whether it replicates.

E3 therefore re-runs the identical 27 states, ladder, and environment under
CRN streams disjoint from every seed E2b used (`96400000` and `96500000`
families), and asks whether the same per-state optimum returns on data that
took no part in choosing it. Tests assert the seed families do not overlap and
that the environment, states, and ladder did not drift between E2b and E3 —
replication is only meaningful when the physics are identical.

### Result

2,592 rows, 4m46s.

| Criterion | Measured | Gate | |
| --- | ---: | ---: | --- |
| Best-action agreement (fresh discovery vs fresh validation) | **0.926** (25/27) | ≥ 0.70 | PASS |
| Pairwise cost-sign agreement over 1,147 material pairs | **0.990** | ≥ 0.80 | PASS |

**Cross-check across four disjoint seed families** (all E2b seeds pooled
versus all E3 seeds pooled):

| | Measured |
| --- | ---: |
| Best-action agreement | **1.000** (27/27) |
| Pairwise sign agreement | 0.992 over 1,151 material pairs |

The per-state optimum is identical across entirely separate seed families.
That is the strongest available evidence that the structure is a property of
the states rather than of one replication stream.

**Independent re-measure of the E2b headline.** E3's own rows, never used in
the E2b analysis, give a prospective state-dependence value of **+0.6182%**
against E2b's +0.6446% (interior fraction 0.926 in both). The E2b result
replicates on fresh data.

### Comparison with the routing channel

The same design applied to patient-indexed specimen routing failed at Stage
G1:

| | Routing (G1) | Overtime (E3) |
| --- | ---: | ---: |
| Best-action agreement | 0.545 | **0.926** |
| Pairwise sign agreement | 0.825 | **0.990** |
| Gate outcome | FAILED (70% required) | PASSED |

G1's lesson is preserved in the gate's structure: pairwise agreement stayed
high on the routing channel (82.5%) while the top-action choice replicated
only 54.5% of the time, so the two criteria are kept separate and the argmax
is primary. Policy improvement needs a reliable best-action choice, not a weak
average ordering. A regression test is pinned to G1's exact numbers and
asserts the gate rejects them.

### What remains open

Label stability across replication streams is not generalization across
states. E3 shows the per-state optimum is a stable target; it does not show
that target is predictable from features observable at decision time. The
optimum still does not track the number of capacity-bound clinics, so the
driver is not a simple count.

**Stage E4 (held-out-seed critic ranking) is the test that separates a
learnable signal from one that depends on realized future demand**, and it
must pass before any actor is trained.

### Evidence

- Rows: `results/continuous_overtime_label_stability_e3/headroom_rows.csv`
- Summary: `results/continuous_overtime_label_stability_e3/summary.json`
- Gate report:
  `experiments/evidence/continuous_overtime_label_stability_e3/label_stability.json`
- Config: `experiments/configs/continuous_overtime_label_stability_e3.json`
- Implementation: `evaluation/label_stability.py` (12 tests)

---

## Stage E4: held-out-seed ranking feasibility — 2026-08-29

### Decision

**FAILED the preregistered gate: `optimum_not_predictable_from_state`.
Stage E5 is NOT authorized. Training remains unauthorized.**

### Result

135 states (5 generation seeds x 9 decision epochs x 3 scenarios), 12,960
rows, 11-rung ladder. Leave-one-generation-seed-out ridge on 15 decision-time
network features.

| Criterion | Measured | Gate | |
| --- | ---: | ---: | --- |
| Pooled top-1 | **0.252** | ≥ 0.50 | **FAIL** |
| Worst-fold top-1 | **0.148** | ≥ 0.50 | **FAIL** |
| Pairwise accuracy | 0.754 | ≥ 0.70 | PASS |
| Worst-fold gain over state-blind | +0.111 | ≥ 0.05 | PASS |

Chance top-1 is 0.091; the state-blind predictor achieves 0.059. Per fold:

| Held-out seed | Fitted top-1 | State-blind | Gain | Pairwise |
| --- | ---: | ---: | ---: | ---: |
| 96600000 | 0.259 | 0.111 | +0.148 | 0.777 |
| 96600001 | 0.185 | 0.074 | +0.111 | 0.772 |
| 96600002 | 0.333 | 0.074 | +0.259 | 0.723 |
| 96600003 | 0.148 | 0.000 | +0.148 | 0.735 |
| 96600004 | 0.333 | 0.037 | +0.296 | 0.764 |

Every fold beats the state-blind predictor and every fold passes pairwise, but
no fold comes close to the 50% top-1 floor.

### Post-hoc diagnostic (not part of the gate)

Computed after seeing the failure, and labelled as such:

| Policy | Total cost | vs best constant |
| --- | ---: | ---: |
| MDL-2-OT anchor | 72,025.4M | |
| Best constant (`u_0.60`) | 70,721.4M | — |
| **Fitted model** | 70,574.6M | **−0.2038%** of anchor |
| Oracle (hindsight) | 70,257.3M | −0.6444% of anchor |

The fitted model captures **31.6%** of the available state-dependent headroom
and beats the best tuned constant by 0.20% of anchor cost, out of sample.
Median rung distance between prediction and truth is 1, and 53.3% of
predictions land within one rung.

So the model is not failing to learn; it is failing to pick the exact rung out
of eleven. Because the cost surface near the optimum is smooth — which was the
explicit design goal of the re-calibration — a near miss is cheap, and top-1
accuracy is a harsh proxy for the quantity that actually matters.

### Two limitations of this screen, both conservative

1. **The gate inherited argmax primacy from Stage G1 without rechecking that
   it fits this channel.** In routing, actions were integer lots on a jagged
   surface where a near miss was a different decision. Here the surface is
   smooth by construction, so the same criterion is stricter than the endpoint
   warrants. This is the same class of error as the E2 gate defect: measuring
   a proxy rather than the decision-relevant quantity.
2. **The learner is deliberately weak and destroys graph structure.** Fifteen
   network-level aggregates (sums, maxima, standard deviations over 20
   clinics) discard exactly the per-clinic spatial detail a GCN exists to use.
   A failure here bounds what a linear model on aggregates can do; it does not
   bound what a graph encoder on per-clinic features could do.

Both limitations mean the true learnability of this channel is at least as
good as measured, and plausibly better.

### Recommendation: stop here and obtain external review

The preregistered gate failed, so **E5 is not authorized and no actor may be
trained**. That stands regardless of the diagnostic above.

This is now the **second** time a screen has failed and analysis has surfaced
a reason the gate was mis-specified (E2's headroom-blind-to-geometry defect
was the first). Each amendment has been individually defensible and each was
executed under change control with fresh configs and output roots. But the
pattern itself is a warning: a sequence of individually reasonable amendments,
each made after seeing a failure, is a garden of forking paths, and it
converges on a passing result whether or not one exists.

The author of both amendments should not authorize a third. Zhaowei's review
of the E2b/E3 chain (PR #9) is already outstanding and is now the appropriate
decision point for all of:

- whether the argmax criterion should be replaced by a realized-cost criterion
  for a smooth channel, and if so what threshold;
- whether an E4b with per-clinic graph-structured features is a legitimate
  continuation or post-hoc gate-shopping;
- whether the accumulated amendment count already compromises the chain and it
  should be re-run end to end under a single frozen protocol.

The last option deserves serious weight. Everything measured so far is
reproducible and hash-recorded, so a clean re-run under one preregistered
protocol is affordable — roughly an hour of compute — and would answer the
forking-paths objection outright.

### Evidence

- Rows: `results/continuous_overtime_critic_ranking_e4/headroom_rows.csv`
- Ranking report:
  `results/continuous_overtime_critic_ranking_e4/ranking_feasibility.json`
  (config and rows SHA256)
- Curated: `experiments/evidence/continuous_overtime_critic_ranking_e4/`
- Config: `experiments/configs/continuous_overtime_critic_ranking_e4.json`
- Implementation: `evaluation/ranking_feasibility.py`,
  `evaluation/run_overtime_ranking_e4.py` (11 tests)
