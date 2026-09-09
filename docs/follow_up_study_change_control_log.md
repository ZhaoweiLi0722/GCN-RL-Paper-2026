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

## 2026-09-08: User-directed exception for the residual headroom screen

- Zhaowei explicitly authorized proceeding without waiting for Howard's
  review. Howard remains pending; no collaborator approval is asserted.
- The completed J2 evidence was reviewed: prospective value 0.06915%,
  optimistic within-library value 0.09356%, both below 0.5%. The one remaining
  bounded question is per-facility budget-neutral support, not more global
  tuning or a claim that online DDPG works.
- A separate branch/config and a hashed authorization addendum permit only
  the existing 54-state, 81-action R0-R2 screen. Scientific fields must match
  the original hashed config exactly. Original protocol, original config,
  pending Howard record, original result root, and PR #12 are unchanged.
- The new result root is
  `results/intertemporal_residual_allocation_headroom_development/r0_zhaowei_20260908`.
  Reserved development families 99700000/99800000/99900000 retain their
  original assignments; no recorded use was found before execution.
- Atomic output claiming and incremental CSV/status persistence protect
  partial evidence without changing evaluation mechanics. No resume/retry,
  downstream learner, new scenario, or formal-holdout use is authorized.
- Any outcome is collaborator-unreviewed development evidence. A pass only
  makes a separately approved ranking study eligible for discussion.
- Authority and exact bounds:
  `specs/2026-09-08-user-authorized-residual-screen/authorization.md`.
  This addendum and implementation must be committed before scientific use.

## 2026-09-09: User-directed residual screen completed

- Execution commit `400d64f764b4e96a0d93d0c4a2721c4f68b2f08b` completed with
  exit 0; 54 states, 81 actions, and 30,618 paired rollout rows were retained.
- Independent read-only recomputation and 17 inventory hashes passed. R0
  mechanics passed; R1/R2 failed. Prospective saving was 0.189611% and
  optimistic validation-selected saving 0.240906%; neither reached 0.5%.
  Material, clinically noninferior states were 0/54 for both selectors.
- This is evidence about a finite first-action residual library with frozen
  continuation, not a global optimality proof or an online DDPG result.
- The screen is closed without retry, tuning, training or formal confirmation.
  Howard's review is still pending; no joint approval or automatic downstream
  authorization is recorded. Original PR #12 and its result destination remain
  unchanged. No merge or push was performed in this execution step.
- Readout, audit, timing caveat and a design-only information-boundary memo:
  `specs/2026-09-08-user-authorized-residual-screen/results.md`.

## 2026-09-09: P0 information-boundary mechanics preparation

- Following the audited terminal residual-screen outcome, Zhaowei requested
  continuing with the proposed next design. Howard remains pending.
- Scope is new opt-in public-history forecast code, a bounded finite-tree
  planning diagnostic, and fixture-seed mechanics tests. Existing simulator
  code and all scientific evidence remain unchanged; Stage E stays closed.
- The finite-tree diagnostic uses two facilities, two deterministic tapes,
  four actions, six decisions and seed 123 only. It is not a newly approved
  stochastic scenario or a policy-training study. No fixture value may be
  cited as manuscript performance or online DDPG gain.
- P0 contract: `specs/2026-09-09-online-adaptation-mechanics/README.md`.
  The implementation/config/contract are committed before the full fixture.
- This step grants no training, new scientific seeds, formal evaluation,
  rerunning closed screens, changed gates, push, merge or proxy approval.

## 2026-09-09: P0 mechanics completed

- Commit `1d2fa7fe17a15fb48f4ba01119f80ba74a579a4b` completed the fixed
  six-step fixture with all 10,920 transitions; 108 focused tests passed.
- Independent saved-row reconstruction verified finite-grid clairvoyant,
  nonanticipative and shared open-loop values, policy replay, exact capacity
  timing/budget mechanics, and nine inventory hashes.
- These values are mechanics fixtures, not RL or research performance evidence.
  The contingent solution changes decisions without updating parameters,
  reinforcing the need for a competent frozen feedback comparator.
- P0 is closed. No automatic new scenario, performance campaign, DDPG training,
  formal evaluation or scientific approval follows it. Howard remains pending.
- Record: `specs/2026-09-09-online-adaptation-mechanics/readout.md`.
