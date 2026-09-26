# Submission readiness review — 2026-09-15

Reviewed branch: `e2b-run`, commit `0f9af1ddd297063a6a3cce8c630b8090b793fc91`.

**Recommendation: revise and audit before submission.** The existing evidence
supports a modest graph-aware policy improvement in the specified simulator.
It does not establish a material incremental benefit from online RL. A credible
application paper remains possible, but the current draft has unresolved
statistical, numerical, and scope inconsistencies. More training is not the
first priority.

This is a review, not a change to the locked execution plan. No experiment,
training, simulator evaluation, checkpoint selection, or evidence replacement
was performed. Suggested corrections and additional comparisons below are not
authorizations to reopen a closed campaign.

## 1. What the evidence actually supports

The following are the recorded formal estimates, not newly estimated effects.
The interval qualification in Section 3 applies to all reported bootstrap CIs.
Differences are candidate minus comparator; lower cost is better.

| Comparison | Objective difference | Reported 95% interval, million units | Interpretation |
| --- | ---: | ---: | --- |
| AFR-GCN-DDPG versus MDL-2 | −0.6585%; −17.690M | [−19.361, −15.725] | Modest favorable policy effect |
| AFR-Flat-DDPG versus MDL-2 | −0.3382%; −9.086M | [−10.587, −7.510] | Much of the benefit also exists without the graph encoder |
| AFR-GCN-DDPG versus matched flat | −0.3214%; −8.604M | [−11.065, −6.428] | Favorable matched architecture comparison |
| Final GCN versus frozen pretrained GCN | +0.00273%; +0.073M | Not provided in the compact formal comparison | No demonstrated online increment |

The formal design has five training seeds, four fixed scenarios, and 100
paired simulation replications per scenario. GCN cost is lower than MDL-2 in
all 20 seed-by-scenario cells, and the graph-minus-flat mean is favorable in
all five training seeds. These are substantive strengths. The 2,000 outcome
rows must not be described as 2,000 independently trained policies or 2,000
independent simulated worlds: the same scenario-specific worlds are reused
across training seeds.

The patient outcomes offer a more interpretable account of practical value:

| Outcome, GCN versus MDL-2 | MDL-2 | GCN | Change |
| --- | ---: | ---: | ---: |
| Modeled patients lost per network episode | 2,409.255 | 2,383.9925 | 25.2625 fewer, or 1.049% |
| Completion service | 48.5718% | 48.7524% | +0.1806 percentage points |
| Manufacturing ineligibility | 7.5963% | 7.1258% | −0.4705 percentage points; −6.194% relative |

These are simulation outcomes over the modeled 20-clinic, 52-epoch horizon,
not observed clinical benefits or evidence that 25 deaths would be prevented.
The distinction between percentage points and relative percentages matters.
Clinical rates may also have different denominators; retain their definitions.

Patient-loss cost supplies about 71% of the objective reduction. The reported
−12.631M patient-loss component is consistent with 25.2625 fewer lost patients
at the 500,000-unit weight. Thus the small total percentage is not simply a
numerical artifact, but neither can it be converted into dollars or described
as a large economic gain without calibration.

Sources: [formal final summary](../experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/formal/final_summary.json),
[frozen summary](../experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/formal/pretrain_summary.json),
[component summary](../experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/formal/cost_component_summary.json).

## 2. Why the original method story is difficult to defend

**The value is already present before online optimization.** Matched TD3
development evidence also favors GCN over its anchor and flat counterpart,
but final GCN-TD3 is approximately 0.00080% worse than frozen pretraining.
Support alignment, persistent-shift realignment, structured exploration, and
paired critic supervision did not establish an online gain. The F1 experiment
had a −0.001067% final-minus-frozen point estimate and a cost interval spanning
zero, with only one of three favorable seeds. Continuing the same search has
weak justification as a route to near-term submission.

This supports the tested offline-pretrained controller, not a causal claim
that every ingredient of pretraining is necessary. There is no matched formal
ablation here isolating the advantage filter from the rest of the teacher and
deployment pipeline. Nor is a frozen pretrained actor the same as a separately
optimized, simple non-RL baseline.

**The graph result has a narrower scope than topology attribution.** Parameter
matching is valuable, but GCN-versus-MLP alone does not isolate correct edges
from weight sharing, architectural inductive bias, or other representation
differences. The draft still lists no-edge, shuffled-edge, and edge-type
ablations as planned. State the matched architecture finding; do not claim a
demonstrated causal contribution from each edge type or generalization to new
clinic networks.

**Timing sensitivity is consequential.** Under immediate specimen availability,
the frozen GCN becomes 0.633% more expensive than MDL-2 and 0.255% more
expensive than flat. With one-epoch product return, it is 1.050% cheaper than
MDL-2. The reversal is comparable in magnitude to the main benefit. It bounds
deployment robustness and makes the one-week specimen-availability assumption
important to justify. Because policies were frozen, this is not proof that a
GCN retrained for lead zero would lose.

**Pooled clinical improvement is not universal clinical noninferiority.** In
the formal summary, GCN seeds 10 and 14 fail scenario-level clinical
noninferiority under abrupt shift. Their completion differences are roughly
−0.0714 and −0.1069 percentage points. All five GCN seeds pass the pooled
clinical check, and all 20 cells improve cost. Report both facts; aggregate
improvement does not justify a guarantee of no service deterioration.

Sources: [Stage E synthesis](patient_indexed_specimen_routing_stage_e_evidence_synthesis.md),
[F1 results](patient_indexed_specimen_routing_stage_f1_results.md),
[timing review](patient_indexed_specimen_routing_stage_a_review.md), and formal summary.

## 3. Submission blockers found in this review

### A. Audit the bootstrap against shared simulation worlds

`evaluation/evaluate_multiscenario_network_residual.py:649` constructs the
simulation seed from evaluation seed and scenario index, without a training
seed offset. `evaluation/evaluate_formal.py:133` resets with `seed + replication`.
Consequently, the same worlds are deliberately shared across trained policies.
The identical anchor mean across formal training runs corroborates this design.

However, `evaluation/aggregate_stats.py:180–187` samples training seeds and then
independently samples outcome rows inside each selected seed. It does not
preserve the shared world draw across seed rows, and it pools scenarios in the
inner resample rather than fixing their prespecified mixture.

This is a crossed training-seed/world design, not simply independent nested
replications. The current intervals need a statistical audit for the intended
estimand. When differences co-vary across policies exposed to the same world,
independent inner resampling can understate that common uncertainty. The size
and direction of the net interval change cannot be determined from marginal
summaries alone. This finding does not establish that the main effects become
nonsignificant.

Before submission, use immutable row-level outputs for a documented sensitivity
analysis that preserves cross-policy CRN dependence and fixed scenario weights,
with training-seed uncertainty handled explicitly. Keep existing reports
immutable; any replacement reporting requires the plan's correction/change
control process. The formal raw result root is absent from this checkout, so
this review could not recompute those intervals.

The methodological distinction is supported by
[Owen, *The pigeonhole bootstrap*](https://arxiv.org/abs/0712.1111), which studies
resampling with crossed row and column effects. Few-run RL inference also
requires explicit uncertainty reporting; see
[Agarwal et al.](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

### B. Reconcile an impossible combination of historical cost and stated weights

The manuscript's formal demand-prior-drift table (`main.tex:1069`) reports an
MDL-2 total objective of 1.268257 billion and 3,665.59 patients lost. The same
manuscript states a patient-loss weight of 500,000, held fixed for all policy
comparisons, and an undiscounted additive objective with nonnegative components.

Yet `3,665.59 × 500,000 = 1.832795 billion`, exceeding the reported total before
any operating, expiry, or urgency cost is added. These statements cannot all
describe the same run and accounting convention.

The current demand-drift config also has a 500,000 weight. The historical
results note repeats the table, but repetition does not resolve provenance.
Recover the executed snapshot and raw decomposition to identify whether the
table used older weights, different units, or another objective. Until then,
do not present it as part of a consistently calibrated comparison. This does
not invalidate the separately reconciled routing-primary component summary.

### C. Make the main text describe the executed routing protocol

The manuscript still contains multiple incompatible stages of the project:

- `main.tex:97`: RQ4 asks about a temporal encoder; a routing-primary temporal
  ablation is not supplied. RQ3 asks about changes in the optimal policy,
  although no optimal-policy benchmark is established.
- `main.tex:618`: the experimental overview describes older replenishment,
  network-resource, and conservative-TD3 studies, leaving the actual primary
  routing confirmation as a late addition.
- `main.tex:628–677`: the comparison hierarchy promises pure policies,
  supervised selectors, and graph ablations that the final routing results do
  not deliver. Case Study 1 explicitly remains pending (`main.tex:964`).
- `main.tex:826–847`: future-tense confirmation and possible budget-extension
  language conflicts with the completed 100-episode formal study and the locked
  post-formal development sequence.
- `main.tex:531–537`: generic DDPG equations omit the composed anchor/residual
  action at the target critic and do not describe the actual four-step
  anchor-relative target and regularization. As written, actor output is a
  residual while the critic notation is for a full action. Clarify coordinates
  and give the executed loss/target, with generic equations only as background.
- `main.tex:552–560`: generic checkpoint/scale selection language should be
  distinguished from the primary fixed episode-100 deployment protocol.
- The formal config enables learned corrections only for specimen transfer;
  reagent, capacity, and replenishment residual scales are zero. Broad claims
  about learning all four controls should distinguish framework capability
  from what the primary experiment actually tested.
- The literature section promises IQM reporting, whereas the primary paired
  summaries use arithmetic means. Describe the estimator actually reported.

Fix these by narrowing and reorganizing the text, not by launching everything
that an older draft promised. Move historical studies to a clearly labeled
appendix with their own model version and configuration.

### D. Complete reproducibility for the actual primary study

All **14/14** source hashes in the publication evidence map matched local files.
The formal point estimates and derived clinical percentages above also
reconciled. This is a strong compact evidence package.

It is not a substitute for the absent formal row-level results and checkpoint
archive. The current README describes an older benchmark and even calls GCN-DDPG
the ablation. Provide a primary-study entry point and an archive of the exact
configs, teacher, checkpoints, rows, manifests, and reporting commands. Make
the existing aggregate findings reproducible without retraining.

The target in the manuscript is EAAI. Its
[publisher description](https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976)
emphasizes novel AI applications and reproducible validation with public data.
My assessment is that an accessible simulation benchmark and clear application
insight would strengthen fit; private large artifacts and a generic GCN-DDPG
novelty story would weaken it. This is a fit assessment, not an acceptance
prediction or a claim that simulation studies are categorically excluded.

## 4. The follow-up studies should not be used to claim an impossibility result

The completed follow-ups justify stopping under their protocols. Several
interpretations in the follow-up notes exceed what was measured:

1. **Variability cost is not an optimality gap.** The stochastic procurement
   study compares one tuned heuristic under random lead times with that
   heuristic under fixed mean lead times. This is
   `J_stochastic(heuristic) − J_fixed(heuristic)`, not
   `J_stochastic(heuristic) − J_stochastic(optimal policy)`.
   The measured 0.176% cannot bound the best achievable learned improvement.
   The protocol originally requested a hindsight/strong-policy reference; the
   reported comparison does not supply it. A small perturbation effect is
   evidence of robustness to that perturbation, not proof of near-optimality
   or a structural sub-1% ceiling. Also, 0.176% is about 5.7 times below the
   1% gate; the notes' roughly 40-fold comparison is to 7.1%, not to that gate.
2. **A gate failure is not zero value.** Overtime's reserved-scenario
   state-dependence value is +0.4500%, below its 0.5% threshold but not zero.
   The fitted model's reported realized-cost gain is +0.1264% relative to the
   constant, despite worse pooled exact-rung accuracy. Report threshold
   failure without saying no state-conditioned method can be useful.
3. **Model failure does not establish missing information.** Limited ridge,
   aggregate/graph-feature, and boosted-tree screens do not prove that the
   optimum is not a function of observable state. Feature sufficiency, sample
   size, approximation, and finite-rollout labels remain possible explanations.
4. **The ranking constant uses held-out labels.** In
   `evaluation/ranking_feasibility.py:90–93`, the state-blind arm is selected
   from all states before folds are split. It is a retrospective reference,
   not a constant tuned only on each fold's training data. This tends to give
   the comparator an informational advantage for cost minimization; its effect
   on top-1 accuracy need not have the same sign. It does not repair the
   failed top-1 gate, but limits claims about fair deployable-baseline ranking.
5. **Exact CRNs do not eliminate Monte Carlo uncertainty.** A difference can
   be deterministic conditional on one shared world while varying across
   worlds. Finite averages of paired differences still estimate an expectation
   with sampling error. Label disagreement therefore does not prove that the
   expected optimal action is inherently unlearnable.
6. **State-level screens are not whole-policy bounds.** The overtime screen
   evaluates a finite scalar ladder and aggregates remaining-horizon costs
   across sampled states. Such totals may involve overlapping trajectory
   suffixes. A discovery-selected per-state lookup is neither a trained policy
   generalizing to new states nor a bound over all per-clinic controls and
   multi-step policy changes. A low estimate does not prove that no algorithm
   or training budget can improve performance.
7. **Observed near-zero deltas are not the noise floor.** F1's point effect is
   −0.001067%, but its reported interval corresponds approximately to
   [−0.0358%, +0.0264%]. Describing ±0.001–0.004% as its attribution noise
   floor confuses effect size with uncertainty. The 0.5% overtime threshold
   should retain its prespecified status without that unsupported rationale.

The original F0/G0/G1 scenario reconstruction correction must also remain
visible. Those rows are nominal-history diagnostics; they do not establish the
formerly assigned persistent-hotspot claims. The later label-budget study did
not reproduce G1's baseline state/policy context and cannot overturn its result.

Sources: [follow-up appendix](follow_up_study_appendix.md),
[overtime results](../specs/2026-08-29-continuous-overtime-control/results.md),
[procurement protocol](../specs/2026-08-29-stochastic-lead-time-regime/protocol.md),
[procurement results](../specs/2026-08-29-stochastic-lead-time-regime/results.md),
[corrected action-geometry review](patient_indexed_specimen_routing_ddpg_problem_redefinition_review.md).

These qualifications do not reopen any failed gate. They narrow the scientific
conclusions drawn from completed evidence.

## 5. Smallest defensible path to submission

1. **Resolve the reporting audit first.** Obtain immutable formal rows for
   dependence-aware uncertainty checks and a directly paired final-versus-
   frozen interval. Reconcile the historical cost/weight contradiction. Record
   corrections transparently, without changing policies or selecting results.
2. **Center the paper on the supported application contribution.** Lead with
   patient-indexed manufacturing coordination, the constrained simulator, and
   the graph-aware advantage-filtered residual controller. Preserve the name
   of the prespecified method in the experiment descriptions, while removing
   any implication that online RL caused the reported gain.
3. **Use three research questions.** Does the controller improve the anchor?
   Does the matched graph architecture improve flat control? What additional
   value, if any, comes from online updates, and how sensitive are the results
   to transport timing? Answer these with the existing evidence.
4. **Bring frozen pretraining into the main comparison.** Its absence as a
   prominent deployment alternative invites the obvious reviewer question:
   why incur the online phase if the frozen actor performs as well? Report
   observed differences and uncertainty, not statistical equivalence unless
   an equivalence margin and appropriate analysis are supplied.
5. **Report practical magnitude candidly.** Show total-objective effects,
   clinical levels and percentage-point changes, component decomposition, and
   scenario-level guardrail exceptions. Describe measured decision/training
   cost only where logs support it; do not infer a favorable cost-benefit ratio
   from a sub-1% weighted objective change alone.
6. **Package and freeze.** Reconcile title, abstract, methods, tables, README,
   claim map, and archive. Keep exploratory follow-ups separate; they should
   not obscure the primary result or become new central claims without review.

A possible title for discussion is **“Graph-Aware Residual Control for
Patient-Indexed Manufacturing Networks: Benefits and Limits of Online
Learning.”** This is a framing proposal, not a manuscript edit or a change to
the locked primary method.

If the intended claim remains a new RL method with a large, general performance
advantage, the current evidence is insufficient. A future study would need a
separately approved design with matched non-RL decision rules, justified control
authority, and independent confirmation. It should not be made a condition for
finishing this paper, nor should the environment be tuned merely to manufacture
a larger percentage gain.

## 6. Review scope and limits

This review examined the authoritative execution plan, current main manuscript,
primary compact evidence and claim map, historical results used in manuscript
tables, attribution and action-geometry reports, follow-up protocols/results,
and the relevant evaluation, bootstrap, and ranking code. It verified the 14
evidence-map source hashes and recalculated selected reported arithmetic.

It was not a full code audit, independent simulator replication, exhaustive
literature/novelty review, or clinical validation. Formal raw rows and tensors
were unavailable locally, so corrected formal CIs and a full run-level
reconciliation remain unresolved. No code was changed and no training or
evaluation campaign was run.
