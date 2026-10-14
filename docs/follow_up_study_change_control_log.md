# Follow-Up Study Change-Control Log

This append-only log records work performed after the routing-primary locked
execution plan was closed. It is deliberately separate from
`docs/patient_indexed_specimen_routing_locked_execution_plan.md`: historical
lock files are evidence and must not double as a living project log.

Detailed numbers, configs, and evidence hashes are indexed in
`docs/follow_up_study_appendix.md` and the per-study `results.md` files.

## 2026-08-29: Overtime exploration and confirmation

- Stage E2b recalibrated the continuous overtime range and cost curvature.
  Prospective state-dependence was +0.6446% and the interior-optimum fraction
  was 0.926, so the exploratory headroom screen passed.
- Stage E3 used fresh random streams. Best-action agreement was 0.926 and
  pairwise sign agreement was 0.990, so the exploratory label-stability screen
  passed.
- Stage E4 failed the prospectively defined learnability gate: pooled top-1
  0.252 and worst-fold top-1 0.148, both below 0.50. No actor was authorized.
- Post-failure representation and nonlinear diagnostics did not reverse the
  learnability conclusion.
- A reserved-scenario confirmation measured +0.4500% state-dependence, below
  the 0.5% gate, and fitted top-1 0.267 versus a state-blind 0.311. Final
  classification: `channel_captured_by_constant_policy` and
  `optimum_not_predictable_from_state`. The overtime channel closed without
  policy training.

Evidence:

- `specs/2026-08-29-continuous-overtime-control/results.md`
- `results/continuous_overtime_headroom_e2b/`
- `results/continuous_overtime_label_stability_e3/`
- `results/continuous_overtime_critic_ranking_e4/`
- `results/continuous_overtime_confirmatory/`

## 2026-08-29: CRN correction and routing label-budget diagnostic

- Direct replay showed exact common-random-number alignment across action
  arms. Earlier text suggesting that patient trajectories desynchronized the
  random stream was corrected.
- The routing label-budget study collected 8,640 paired rollouts over 27 fresh
  MDL-2-anchored states. It did not reproduce Stage G1's 0.545 baseline:
  agreement was already 0.821 at the same 3-versus-5-world budget shape.
- Classification:
  `precondition_violated_cannot_adjudicate_g1`. The Stage G1 negative result
  remains unchanged; a valid budget explanation requires replication on the
  original Stage G1 frozen-pretrain states.
- Sequential halving was worse than uniform allocation at every tested budget
  on the five-arm ladder and is not recommended for short ladders.

Evidence:

- `specs/2026-08-29-routing-label-budget-study/results.md`
- `results/routing_label_budget_study/`
- `experiments/evidence/routing_label_budget_study/`

## 2026-08-29: Stochastic procurement exploratory diagnostic

- The step-0 protocol remained unsigned. It proposed an optimality-gap gate
  against hindsight or a strong rollout benchmark.
- A flag-gated stochastic-procurement implementation and lead-aware MDL-2-LT
  comparator were built. The implementation draws one lead per facility per
  epoch to preserve action-independent random-number consumption.
- The executed analysis reported +0.176% cost from lead-time variability
  relative to a fixed-mean lead. That is a different estimand from the drafted
  optimality gap and cannot be scored against its 1%/3% gate thresholds.
- The result is retained as an exploratory diagnostic. It did not authorize
  downstream screens or policy training and does not formally close the
  optimality-gap question.

Evidence status:

- `specs/2026-08-29-stochastic-lead-time-regime/results.md` records the report
  and protocol deviation.
- PR #9 did not include the original executable diagnostic config, raw rows,
  or machine-readable summary, so its exact `+0.176%` table and safety
  multiplier remain reported rather than reconstructed evidence.
- On 2026-09-01, a post-hoc publication reproduction added an explicit config,
  72 fresh paired rows, a machine-readable summary, and an artifact inventory.
  It selected safety multiplier `1.25` and measured pooled variability cost
  `+0.1169%` (seed-clustered normal 95% half-width `+/-0.0479%`), with stochastic lead worse
  on 59/72 pairs and exact RNG end-state equality on 72/72 pairs.
- The reproduction supports the direction and small-magnitude interpretation
  only. It was designed after the original result was known and is neither an
  independent confirmation nor the unexecuted optimality-gap gate.

Reproduction evidence:

- `experiments/configs/stochastic_procurement_variability_reproduction.json`
- `results/stochastic_procurement_variability_reproduction/`
- `evaluation/reproduce_stochastic_procurement_variability.py`

## Publication rule

The current manuscript may cite only results backed by committed executable
configuration and machine-readable evidence. Exploratory or incomplete
follow-up diagnostics belong in limitations or future work, not in headline
performance claims.
