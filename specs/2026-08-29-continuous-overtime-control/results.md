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

### Recommendation (requires change control; not executed)

Do not advance to E3 on this configuration, and do not silently re-tune and
re-run — that would be post-hoc selection against an observed result. The
proper route is an appended change-control entry authorizing an E2
re-calibration under a new config name and output root, holding the
prospective gates fixed:

1. Raise `weight_overtime_quadratic` so the marginal cost crosses the
   averted-shortage benefit inside the admissible range. A quadratic weight
   near 20,000 puts the marginal cost at full surge around 75,000 — above the
   50,274 shortage benefit, below the 500,000 patient-loss benefit — which
   should place the optimum in the interior and make it depend on how much
   patient-loss risk the state actually carries.
2. Optionally raise `max_overtime_fraction` so the ladder spans a wider range
   of surge levels, giving the interior optimum room to move between states.
3. Add an explicit interior-optimum criterion to the E2 gate itself: require
   that the best rung is strictly interior in a stated minimum fraction of
   states. The current gate measures headroom but is blind to geometry, which
   is why a corner solution passed it. This is a gate defect worth fixing
   regardless of the re-calibration outcome.

Recommendation 3 should be adopted whatever else is decided: as written, the
E2 gate can be passed by an environment that cannot support the study.

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
