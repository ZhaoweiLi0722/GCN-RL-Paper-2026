# GCN-TD3 budget probe plan

Date: 2026-07-13. Status: ready to run.

## Purpose

Howard's DDPG budget probe showed seed-dependent instability at 500k steps, so the
remaining decision gate is whether the more stable deterministic graph backbone,
`gcn_td3` with `facility_action` readout, improves at 300k/500k steps on the
nominal 20-clinic patient-condition environment.

This decides whether the manuscript can keep a "stable graph-DRL may become
competitive with adequate training" story, or should pivot to the honest negative:
graph encoding improves learned control over flat RL, but tuned heuristics remain
stronger under the tested regimes.

## Runner

Use:

```bash
PYTHONPATH=. python -m evaluation.td3_budget_probe
```

The runner is resumable. Each completed `(algorithm, steps, seed)` cell writes one
CSV under:

```text
results/td3_budget_probe/steps_<steps>/
```

Do not commit the raw `results/` directory unless the files are intentionally small.
After the run, commit a written summary and selected small summary CSVs under this
`specs/2026-07-13-td3-budget-probe/` folder.

## Smoke test

Before launching the long run:

```bash
PYTHONPATH=. python -m evaluation.td3_budget_probe --smoke
```

## Main experiment

Default matrix:

- algorithm: `gcn_td3`
- budgets: `300000`, `500000`
- seeds: `0`, `1`, `2`
- evaluation replications per seed: `20`
- reference heuristics: `mdl2`, `mdl1`, `iso`, `myo`, `umyo`

If the 3-seed result is close or ambiguous, extend to five seeds:

```bash
PYTHONPATH=. python -m evaluation.td3_budget_probe --seeds 0 1 2 3 4
```

## Decision gate

Use the best heuristic cost in each budget group as the reference. Let:

```text
gap = cost_iqm(gcn_td3) / cost_iqm(best_heuristic) - 1
```

- `gap < 0.20`, and 500k improves over 300k: undertraining remains plausible; consider
  escalating the stable graph-DRL story and updating the manuscript accordingly.
- `gap = 0.20-0.35`: ambiguous; extend to five seeds or add one 1M spot-check before
  changing the paper's claim.
- `gap >= 0.35`, or eligibility remains clearly below heuristics: treat this as a
  robust negative. Keep the contribution focused on the patient-condition testbed and
  graph-vs-flat RL improvement.

In all cases, do not weaken or remove the fair heuristic baselines just to make DRL win.
