# Phase 7 — Pilot Experiments (validation)

## Automated

Before each group's commit and once at the end:

```
conda run -n gcn-rl python -m compileall src evaluation tests
conda run -n gcn-rl python -m unittest discover -s tests
```

Required assertions (new tests, all smoke-scale — no full-budget training in the
unit suite):

- **Eval metrics (group 1):** patient rollout yields `eligibility_rate ∈ [0,1]`,
  non-negative `patients_lost` (= ineligible + expired), non-negative
  `material_wasted` and `at_risk_unserved`; base-env evaluation still emits the
  original column schema unchanged.
- **Aggregation (group 2):** IQM matches a hand-computed value on a known vector;
  a single outlier seed moves the plain mean but not the IQM (DDPG motivation);
  bootstrap CI brackets the point estimate and is reproducible under a fixed seed.
- **Configs (group 3):** every manifest config loads; each learned-agent config
  builds an env + agent and runs one step; graph agents receive a valid graph
  spec (state-dim matches the patient observation).
- **Two-stage runner (group 4):** `--smoke` run completes both stages on a small
  subset, writes a summary CSV containing the new patient-metric columns, and
  produces a non-empty ranking with a selected flagship.
- **Projection ablation (group 5):** repair magnitude ≥ 0 and = 0 for an
  already-feasible action; the ablation toggle yields two distinct eval configs.

All torch-dependent tests skip cleanly when torch is unavailable.

## Manual

- Confirm the flagship ranking is computed on **Stage B (20-clinic)** eligibility
  IQM, with cost then stability as tie-breakers — not on Stage A.
- Confirm every reported number traces to a logged run in `results/` (no
  hand-entered values); a partial run is labelled partial with the seed/agent
  coverage stated.
- Confirm DDPG-family rows show **all seeds + IQM**, never a single seed.
- Sanity-check the projection ablation: if the no-repair variant performs nearly
  as well, note it (projection is not masking non-learning); if it collapses,
  report the projection load honestly.
- Eyeball a `plot_results.py` figure: graph vs flat vs heuristic on eligibility
  and cost, with CIs.

## Definition of done

- Evaluator captures patient metrics; IQM/CI aggregation implemented; both
  tested. Pilot configs + two-stage runner built and smoke-validated end-to-end.
  V4 projection ablation + stability diagnostics implemented and tested.
- The thorough run is launched; **whatever completed** is aggregated with IQM/CIs
  and stability + projection reports. Partial coverage stated explicitly.
- `pilot-findings.md` records the ranking, the chosen flagship (with evidence),
  the temporal-encoder go/no-go, and stability/projection findings; the three
  decisions are mirrored in `tech-stack.md`.
- Full unit suite green; `compileall` clean; roadmap Phase 7 ticked and Phase 8's
  conditional set; changelog written; branch merged to `main` and pushed to
  `mine`.

## Explicitly deferred (not done here)

- Full experiment campaign / full scenario matrix / 500-rep evaluation (Phase 9).
- Building the temporal encoder (Phase 8) — gated on this pilot's decision.
- MILP/Gurobi heuristics.
