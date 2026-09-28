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
| Formal routing-primary GCN-DDPG: approximately -0.658% vs MDL-2 and -0.321% vs matched flat | Package and matched-encoder benefits; online final-vs-frozen increment unestablished | Existing manuscript/formal evidence |
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
