# Integrated evidence and next study

## Integration and authorization

Zhaowei requested integration and new scenario exploration on 2026-09-28.
This authorizes the bounded synthetic pilot below; it is not Howard's sign-off
or authorization to reopen formal confirmation. The integration branch is
`codex/september-research-integration`, in a persistent project worktree.

Parents: our `codex/online-adaptation-information-contract` at `328806d` and
Howard's `e2b-run` at `461dbe4`. Original branches, historical outputs and
formal evidence remain unchanged. This integration is not a merge to main.

## Evidence ledger

| Evidence | Current interpretation | Trace |
| --- | --- | --- |
| Formal routing-primary GCN-DDPG: approximately -0.658% vs MDL-2 and -0.321% vs matched flat | Complete-controller benefits; isolated encoder value and online final-vs-frozen increment unestablished | Existing manuscript/formal evidence; September 29 graph/flat interface audit |
| Our local overtime reallocation screen: 0.189611% prospective validation saving, 0/54 material clinically noninferior states | Failed prespecified local first-action gate; not a global optimality certificate | `specs/2026-09-08-user-authorized-residual-screen/results.md` and independent audit |
| Our P0 finite-tree fixture | Feedback can matter without updating any policy parameters; software verification only | `specs/2026-09-09-online-adaptation-mechanics/readout.md` and independent audit |
| Howard restricted planner: reported -1.83% vs MDL-2 on 40 worlds; full planner: -6.64% on 12 worlds | Promising development reports of sequential control, not online RL; not independently reproduced here | `specs/2026-09-16-measurement-a-rollout-planner/results.md` |
| Howard GCN-TD3 on MDL-3: reported -4.329% vs MDL-2, pooled guards pass | Exploratory anchor transfer, not formal DDPG confirmation | `specs/2026-09-16-learned-vs-retuned-anchor-fresh-worlds/results.md` |
| Howard crossed-bootstrap audit | Shared worlds across training seeds require crossed rather than independently resampled within-seed inference | `docs/reporting_audit_2026-09-15.md` |

These studies have different comparators, action sets, horizons and seed
families. Do not pool their percentages or call their ratio the fraction of
the true optimum recovered. Our first-action failure and Howard's repeatedly
replanned controller can both be correct.

## Qualification of Howard's latest interpretation

Preserve his original reports verbatim, with this integration note beside them:

1. The planner's `make_matched_clone` conditions on each patient's exact age,
   survival and risk type. The neural observation has facility aggregates and
   histograms. Observation equivalence is not established by resampling latent
   health alone. Establish a common public interface before calling this an
   information-matched learned-vs-planner comparison.
2. The referenced planner artifacts/raw rows and decision logs are not present
   in the merged tracked tree. Numerical claims remain author-reported until
   the raw archive, execution hashes and row-level audit are available.
3. A feasible planner estimates achievable improvement, not the maximum in its
   action class. The 12-world planner and 200-world LP do not form a paired,
   statistical [6.6%, 39%] interval.
4. The privileged-vs-resampled planner comparison is not the value of all
   future information; it cannot explain all slack in the hindsight bound.
5. `prior_estimate` is not by itself a guarantee against future-regime access.
   P0 timing tests document the distinction from a public-history forecast.
   New pilot demand is stationary and demand forecasts are disabled.

## Next steps and boundaries

First run a prespecified **baseline feasibility pilot**, not DDPG: one
persistent idle-capacity outage and one persistent procurement-delay change,
each with a no-change control. Preserve clinical weights. Report MDL-2,
MDL-3, and inventory-position/lead-aware challengers; select any challenger
on discovery worlds only. Retain raw episode and step evidence.

This asks whether the simulator and stronger baselines respond meaningfully.
It does not estimate remaining RL headroom, establish clinical safety or
authorize another neural campaign. Dynamics, seeds and stop conditions:
`specs/2026-09-28-disruption-feasibility/protocol.md`.

Before training: agree operational calibration and observation timing; audit
repeated-decision headroom beyond the strongest adaptive challenger; establish
replicable rankings and learnability. Then separate belief adaptation from
weight updates with a 2x2 design, including frozen history-aware policy,
adaptive heuristic and system-identification/MPC with matched information and
query budgets. Preserve negative results, not just scenarios where RL wins.

Publication priorities do not depend on a positive RL result: recover raw
archives; correct crossed inference; reconcile equations with executed code;
justify clinical margins; test graph/transport dependence; keep formal,
development and mechanism-only findings separate. No promise of a positive
online effect or journal acceptance is made.

## Pilot completed

The bounded pilot and independent audit are complete. See
`specs/2026-09-28-disruption-feasibility/readout.md` for all rule comparisons,
clinical tradeoffs, the preserved smoke failure and the explicit change to
synthetic idle-storage limits. This does not advance a neural-training gate.
An aggregate-observation counterexample is now covered by
`tests/test_development_disruption.py`. The user elected local-only integration.

## Research roadmap after the pilot

The [online-adaptation roadmap](../../specs/2026-09-28-online-adaptation-roadmap/plan.md)
prioritizes an unknown persistent resource-response mechanism, with a strong
frozen history-aware policy, adaptive rules and system-identification/MPC as
comparators. The preceding equipment-removal pilot did not test that mechanism
and does not rule it out. The roadmap separates graph representation from
online weight updates, includes a conditional two-week decision schedule, and
preserves a GCN-focused manuscript route if online increments remain absent.
It is a planning document, not an executable protocol or a new training launch.

## Service-effort mechanics completed

The [isolated service-effort readout](../../specs/2026-09-28-service-effort-mechanics/readout.md)
records 24 deterministic transitions, 63 passing relevant tests and an
independent row/hash audit. Continuous progress with indivisible completions,
commitment delays and a receipt-only estimator are implemented outside the
patient simulator. This verifies mechanics only: a simple estimator identifies
the noiseless response, so it provides no evidence that online DDPG is needed.
No patient experiment, new neural training, remote push or main merge occurred.

## September 29: decision comparators checked

The [decision-fixture readout](../../specs/2026-09-29-service-effort-decisions/readout.md)
adds public-observation fixed/adaptive rules and fixed-model/identification MPC,
with recorded finite-grid diagnostic transitions and a separate evidence audit.
All costs are synthetic software-fixture units, not manuscript performance.

Under the persistent response change, identification MPC costs 22.25 versus
23.25 for fixed-model MPC. But a scenario-law-optimized open-loop sequence and
the optimal nonanticipative policy both cost 22.25. This fixture therefore does
not establish residual value for feedback beyond that schedule, much less for
online DDPG. The diagnostic has privileged model-law knowledge; its result
does not certify global optimality in the original patient simulator.

The run has 15 baseline episodes, 75 decisions and 4,092 exact-tree transitions;
31 audited hashes, 72 relevant tests and full compilation passed. No neural
training was launched, and the null diagnostic was retained without retuning.

The [primary-source engineering review](../../specs/2026-09-29-service-effort-decisions/engineering_basis.md)
supports investigating qualified-personnel scheduling, but not a calibrated
continuous overtime-to-biological-output law. Actual intervention constraints
and available measurements remain unresolved. Operational grounding and a
shared information interface come before patient-model integration or training.
This remains local-only work; no main merge or external publication occurred.

## September 29: queue boundary and information audit

The [queue readout](../../specs/2026-09-29-service-queue-boundary/readout.md)
records all eight prespecified synthetic cells. Coupled queues create feedback
value over optimized open-loop schedules, but each cell has a prespecified
non-neural comparator attaining its restricted optimum. This is not evidence
for online neural updates. The known-law optimum is itself a frozen history
policy, independently reproduced from logged edges.

The arrival factor proved redundant: batch/flow differ only by a fixed holding
charge, not decisions or service events. Completion-only observation retains
some feedback value, but has not been tested against practical baselines with
the same reduced information. These qualifications are part of the result,
not reasons to retune the completed packet or reopen formal testing.

Keep the adaptive-control baselines strong. Before a larger learning run,
establish the real operational decision and measurement contract. A possible
fast-policy-versus-expensive-planning contribution is distinct from online
learning and currently remains a hypothesis, not a measured result.

## September 29: formal raw cost evidence recovered

The [formal crossed-audit readout](../../specs/2026-09-29-formal-crossed-audit/readout.md)
closes the local raw-row availability gap for the existing DDPG cost contrasts.
Forty CSVs and 16,000 rows passed identity, pairing and historical-summary
reconciliation. Two summary hashes and ten historically recorded GCN-final
CSV hashes match; the other thirty CSVs have current hashes and summary
reconciliation, not historical byte-level proof.

The crossed 95% interval remains favorable for GCN vs MDL-2
[-19.806, -15.355] million and GCN vs flat [-11.284, -6.088] million objective
units. GCN final vs frozen remains inconclusive [+0.073 mean; -0.180, +0.313
interval], as does flat [-0.139 mean; -0.546, +0.278 interval]. All five
contrasts were independently checked by a multiplicity-weighted bootstrap.

This is a post-hoc sensitivity of already observed results, not new holdout
testing. The original primary intervals and clinical analysis are unchanged.
The manuscript now includes the labeled cost sensitivity. Historical reports
are retained, including the September 15 machine-specific availability note.

## September 29: executed-method reconciliation

The [method-contract readout](../../specs/2026-09-29-formal-method-contract/readout.md)
reconciles the manuscript with the historical training source and ten effective
configs, linked to the byte-verified training manifest. The primary method uses
separate actor/critic encoders, specimen-only residual corrections, request
actions in replay, four-step anchor-relative online returns, regularized
updates and fixed final/pretrain evaluation. Generic one-step DDPG equations,
shared-weight wording and validation-selected deployment prose were corrected.

This is a reporting improvement, not evidence of a new online gain. In
particular, the shaped training objective is not proven equivalent to the
evaluation cost, and a straight-through request gradient is not a derivative
of the discrete patient executor. The teacher/replay mixture and full graph
feature-parity checks remain the next independent audit packets. No historical
artifact or experimental outcome changed.

## September 29: teacher/replay audit findings

The [teacher/replay readout](../../specs/2026-09-29-formal-replay-contract/readout.md)
checks the byte-verified teacher cache, historical data-loading functions,
ten summaries and 1,000 training rows. Numeric replay reconstruction found
153/314 cached multi-step windows with discontinuous adjacent observations.
Offline absolute-reward and online anchor-relative targets share a critic;
the pre-online calibration helper also ignores multi-step discount metadata.
The teacher's action-group support exceeds the specimen-only actor support,
and cached label-generation horizon cannot be verified from echoed call settings.

Every run nevertheless performed 5,200 online critic and 2,600 actor updates.
Loss coefficients alone do not show which gradient dominated. These findings
limit the interpretation of the null result; they do not prove that fixing
the path will improve performance. The manuscript now discloses them. No
historical source, checkpoint, output, primary estimate or formal status changed.
Any corrected learning campaign needs its own development protocol and fresh
streams; it must not overwrite or retune on the completed formal comparison.

## September 29: completed local audit queue

The [graph/flat contract](../../specs/2026-09-29-formal-graph-contract/readout.md)
establishes parameter matching but not feature/head/gate parity. Earlier
"matched-encoder" wording in this note is narrowed to the complete-controller
comparison; the measured cost differences are unchanged.

The [qualified setup-support contract](../../specs/2026-09-29-qualified-support-contract/decision_contract.md)
names a candidate staffing lever without assuming that labor speeds biological
growth. Twelve operational inputs remain unresolved. The
[completion-count replay](../../specs/2026-09-29-completion-count-comparator/readout.md)
adds one fixed, reduced-information practical rule using only existing queue
edges. It reaches the restricted bound under changed response and nonbinding
downstream capacity; its bottleneck gap does not establish online-RL value.

The [manuscript evidence checkpoint](2026-09-29-manuscript-evidence-checkpoint.md)
consolidates claims, provenance, reproduction commands and remaining decisions.
This completes the finite routine queue, not the scientific work needed for
submission. No new patient experiment or neural training was launched.
