# Value-TD: Development-Only Diagnostic and One Next Hypothesis

2026-10-08. Finite saved-data assignment complete. No new science, source edits,
manuscript edits, commit, push, or agent launch. Entry HEAD:
`85da13daa0726bfb368bb380426a4456b74af4c9`.

## Answer

The selected Value-TD recipe has a **condition-sensitive cost/service tradeoff,
not evidence of failed training or proven support overspending**. Nearly full
support volume accompanies patient benefits, while reagent expenses offset a
large part of the accounting gains. Recorded fit loss is not a reliable proxy
for the better controller. These are development associations, not causal findings.

Evidence: 72 paired development worlds/recipe (three blocks, three equally
weighted conditions, eight worlds/cell), plus 576 development fit receipts from
both existing learning rates: six jobs, 18,432 updates. Final/confirmation data
are excluded. Authority records supply scope only. MDL-2 remains posthoc on
consumed worlds, not new validation.

## Condition and Support Pattern

Selected recipe: `value_td-lr0-graph`, learning rate 0.0001. Savings are H8 minus
Value-TD; extra losses are Value-TD minus H8. Costs are synthetic units.
Intervals below reuse the saved descriptive bootstrap, with orientation reversed
where needed; percentages are recalculated using the H8 mean denominator.

| Development condition | Saving % | Saving M [95% descriptive interval] | Extra losses/world | Applied hours: TD / H8 | Cost / patient harmed worlds |
| --- | ---: | --- | ---: | --- | --- |
| No change | 0.9752 | 0.3624 [-0.1187, 0.9668] | -9.625 | 372.83 / 300.66 | 8 / 4 of 24 |
| Persistent change | 1.8845 | 0.7181 [-0.1161, 1.5577] | -8.250 | 378.00 / 322.90 | 6 / 5 of 24 |
| Fast fluctuation | 1.9587 | 0.8512 [0.0798, 1.6787] | -18.833 | 378.33 / 290.00 | 8 / 3 of 24 |
| Equal-condition overall | 1.6271 | 0.6439 [0.2067, 1.0791] | -12.236 | 376.39 / 304.52 | 22 / 12 of 72 |

Pooled block savings are positive (0.3406, 0.6520, 0.9389 M), but no-change block
0 is negative (-0.01484 M). Two condition intervals cross zero. This is
selection-exposed development evidence, not an independent screen pass or a
formal test of differences between conditions.

TD uses 98.02% of the 384-hour maximum bookable volume, versus H8's 79.30%.
This is whole-world volume, **not** a measured fraction of epochs at the cap.
Actions differ at 36.79 of 48 control boundaries on average. Extra applied hours
and cost savings have Pearson r=-0.0098, or r=0.0018 after centering within
block/condition. These descriptive associations provide no monotonic evidence
that adding or removing support improves cost, and no causal identification.

## Cost Decomposition

Mean selected-TD minus H8 cost, M units/world; positive means additional expense:

| Mutually exclusive group | Delta M |
| --- | ---: |
| Patient loss + expiry + urgency | -1.113472 |
| Reagent purchase + reagent shortage | +0.602111 |
| Bioreactor shortage | -0.131968 |
| Flexible/ordinary support + switching | +0.007273 |
| Remaining operating components | -0.007813 |
| Total | -0.643869 |

Purchase (+0.462743 M) and reagent shortage (+0.139368 M), not direct support
charges, are the main positive offsets. The grouped sum matches total cost to
roundoff; JSON retains all 14 components. This accounting decomposition does
not show that support caused reagent costs or that a reward/accounting defect
exists. Do not change reward weights to force a gain.

## Training Fit Receipts

All six development jobs retain 96 completed worlds and 3,072 finite fit updates.
Windows are the first and last 24 training worlds, ordered by saved update index;
each window contains eight worlds per condition. No checkpoint is selected.

| Existing learning rate | Mean recorded TD MSE: first / last window | Recorded norms above cap 5 | Development savings / extra losses |
| --- | --- | ---: | --- |
| 0.0001, selected | 35.5621 / 7.3081 | 99.80% | 1.6271% / -12.236 |
| 0.0003, unselected | 38.4677 / 4.0213 | 98.39% | 1.4296% / -8.125 |

Lower late-window loss at 0.0003 accompanies worse mean development outcomes.
Norms are pre-clipping; frequent clipping does not prove a causal weakness.
Sampled own-world TD targets change across worlds, so these losses do not measure
held-out calibration, action ranking, convergence, or a causal RL benefit.

## One Next Finite Comparison

**Hypothesis:** a single prospectively fixed attenuation of the learned TD
residual toward the zero-residual H8 anchor may improve the cost/service tradeoff
relative to unattenuated Value-TD. Compare these two controller variants using
the same sealed development models, retaining unchanged H8 as the strong reference.
This directly tests reliance on the learned correction; it is not an assertion
that lower support is better. Attenuation can also erase patient gains.

Separately authorize and freeze one nonzero attenuation below full weight,
model eligibility, finite budget, metrics and decision rule. No coefficient
grid, condition-oracle switch, new family, reward change or extra training.
Keep physics, public information and horizon fixed. No budget is invented here.

Require **new untouched paired worlds**, blocked by model/training seed, never
old final/MDL-2 worlds. Predeclare patient-loss guardrails against both controls;
report conditions, blocks, harms, components, hours and compute. Reusing three
development-selected models yields new-world evidence conditional on those
models, not independent-training-seed generalization. The original H8 screen
is unchanged. This hypothesis is not implemented, approved, launched or validated.

## Reproduction and Evidence

Run from this worktree:
`python3 reports/2026-10-08-value-td-next-step/read_development.py`.
Creates `diagnostic.json`, or compares without overwriting on later runs.
Stdlib only; no scientific imports/calls, new bootstrap, raw/archive/state audit.
Mixed-stage containers are immediately filtered. JSON enumerates receipt paths.

- `results/capacity_family_selection_20261006/terminal-readout/world-readout.json`: development metrics and additive cost components.
- `results/capacity_family_selection_20261006/terminal-readout/all-pairs-readout.json`: development paired rows and already-saved uncertainty only.
- `results/capacity_family_selection_20261006/terminal-readout/job-and-role-readout.json` and `verification.json`: completed development jobs and reused reconciliation receipt dated 2026-10-08T05:13:18.880895+00:00.
- `results/capacity_family_selection_20261006/payload/selection.json`: original development-only selection, unchanged.
- `results/capacity_family_selection_20261006/payload/updates/development_training-*-value_td-lr[01]-graph.json`: 576 fit receipt files; no confirmation receipts.
- `experiments/configs/capacity_family_selection_20261006.json` and `specs/2026-10-06-family-selection/protocol.md`: original recipe, cap, split and metric definitions.
- `src/rl/capacity_value_learner.py` and `src/rl/capacity_family_value.py`: read-only interpretation of logged MSE and pre-clipping norms, not outcome evidence or executed imports.

Files added in this assignment: `read_development.py`, `diagnostic.json`, and
`report.md`, all under `reports/2026-10-08-value-td-next-step/`. Reader execution
and scoped `python3 -m compileall -q .` passed. The default Python cache was
permission-blocked; rerunning with `PYTHONPYCACHEPREFIX="$PWD/__pycache__"`
confined generated bytecode to this report directory. No scientific tests were
rerun. Repository-wide integration and manuscript/Git remain coordinator-owned.
