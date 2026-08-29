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
- The value of *state-dependence* — the entire budget available to any
  state-dependent policy, learned or otherwise — is
  **0.67M, or 0.0054% of anchor cost**.

Only 2 distinct arms are ever optimal (`u_1.00` in 26 states, `u_0.80` in 1),
and the best arm is strictly interior in 3.7% of states.

**This is the finding that governs the study.** 0.0054% is the same order as
the online-attribution noise floor measured on this simulator (Stage F1
final-versus-frozen deltas were ±0.001–0.004%). A perfect oracle beats a
one-line constant by less than the noise. No algorithm, architecture,
training budget, or graph encoder can extract a defensible result from a
channel with that budget — the ceiling is below the measurement floor.

It is also the routing failure repeating in a new setting: a simple baseline
banks the value, and the learned component has nothing left to attribute.
The overtime channel was supposed to escape that, and as configured it does
not.

Measure and gate implemented in `evaluation/state_dependence_value.py`
(9 tests). Applied to these rows it classifies the channel
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
   `value_of_state_dependence_fraction ≥ 0.005` (0.5%), roughly 100× the
   attribution noise floor, plus an interior-best-arm fraction ≥ 0.30.

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
