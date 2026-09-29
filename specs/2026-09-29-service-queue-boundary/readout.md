# Support-queue boundary: feedback value, not online-RL value

## Status and provenance

The eight-cell synthetic mechanism check completed on source freeze
`1c4ae58b9a9b76d5c4b1b688fb95fd45a313153b`, with no training or patient-model
execution. Raw evidence is in
`reports/2026-09-29-service-queue-boundary/run/`.
The independent audit passed: 60 baseline episodes, 300 decision rows,
16,368 diagnostic tree edges, 9,088 unique terminal closures and 59 hashes.
All booked work and commitments are settled and charged.

The original evidence audit did not independently implement the nonanticipative
optimizer. This was subsequently addressed with a separate shared-prefix-until-
disambiguation recurrence and policy-witness replay on the logged edges, commit
`425bb20a5b20bd08450477aaae2ed22a1257ef45`. It reproduced all eight optima without
calling the environment or original optimizer. A further measurement-contract
check, commit `92069931a044e83792795e1fd95bee346bdd498b`, also consumed only those
logged transitions. Both dated diagnostics remain alongside the run.

## All prespecified cells

Costs are synthetic objective units, not dollars or manuscript performance.
There are no independent empirical replications or confidence intervals.
"Optimum" below means four allowed actions for five decision epochs under the
known finite-world law, followed by the declared common terminal continuation.
It does not mean unrestricted continuous-action or patient-network optimality.

| Release / downstream / response | Balanced | Backlog | Adaptive backlog | Fixed MPC | ID-MPC | Feedback optimum | Best open loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch / nonbinding / nominal | 81.25 | 81.25 | 81.25 | 81.00 | 81.00 | 81.00 | 81.00 |
| Batch / nonbinding / changed | 109.00 | 103.25 | 103.25 | 106.375 | 106.00 | 103.25 | 107.75 |
| Batch / bottleneck / nominal | 109.25 | 109.25 | 109.25 | 102.25 | 102.25 | 102.25 | 102.25 |
| Batch / bottleneck / changed | 125.00 | 121.25 | 121.25 | 117.625 | 114.25 | 114.25 | 119.00 |
| Flow / nonbinding / nominal | 71.25 | 71.25 | 71.25 | 71.00 | 71.00 | 71.00 | 71.00 |
| Flow / nonbinding / changed | 99.00 | 93.25 | 93.25 | 96.375 | 96.00 | 93.25 | 97.75 |
| Flow / bottleneck / nominal | 99.25 | 99.25 | 99.25 | 92.25 | 92.25 | 92.25 | 92.25 |
| Flow / bottleneck / changed | 115.00 | 111.25 | 111.25 | 107.625 | 104.25 | 104.25 | 109.00 |

The response-change cases have 4.50 or 4.75 units of feedback value over their
best common open-loop sequence. With the downstream bottleneck, ID-MPC saves
3.375 relative to fixed-model MPC and attains the restricted optimum. Without
the bottleneck, the simple backlog rule is optimal and ID-MPC is worse.

Every cell has at least one *prespecified* non-neural baseline matching the
restricted optimum. This is a descriptive per-cell statement, not a validated
hybrid controller selected after looking at test outcomes. The exact
nonanticipative witness is itself a frozen history-to-action map: no policy
parameter is updated online. Therefore this packet establishes neither a
remaining online-RL gap nor a benefit of neural weight updates.

## Two important negative diagnostics

**Booked arrivals are redundant in this fixture.** Across 8,184 paired edges and
6,144 complete action sequences, batch/flow support and downstream events are
identical. Flow removes exactly 10 units of pre-release holding cost from every
sequence. We must not count the arrival variants as independent successful
mechanisms or describe this as evidence about unpredictable future demand.
No arrival times were changed after discovering this result.

**Exact work-progress measurements matter, but are not the whole effect.**
Removing exact remaining/delivered work while retaining completion events,
bookings, own commitments and observable costs leaves nonbinding feedback value
at 4.50. In the bottleneck cases the completion-only optimum is 115.00 (batch)
or 105.00 (flow), 0.75 worse than exact-progress feedback but still 4.00 better
than open loop. Under each recorded optimum, the two changed worlds first
become distinguishable at decision epoch 3.

This is another known-law diagnostic, not a completion-only practical-controller
trial. In particular, the full-progress ID-MPC cost of 104.25 cannot fairly be
compared to the completion-only bound of 105.00 as if they had equal information.
Actual availability and timing of these measurements remain unvalidated.

## Consequence for the next study

Do not launch a DDPG campaign on this solved fixture. Preserve it as a regression
case and a counterexample to equating feedback adaptation with weight learning.
Do not weaken the adaptive comparator to manufacture an RL gap.

The useful next question is operational and computational: which real support
decisions recur, what completion/progress information is actually available,
and at what network size does a fair planner become costly? The recorded MPC
uses roughly 12,000-17,000 internal transition queries per episode, whereas the
rules use none; these are query counts, not measured wall-clock latency.
A future fast frozen policy could have an amortized-computation contribution
even without online gains, but no learned-policy timing or accuracy result has
been obtained here. Any larger/noisy model needs its own ex-ante justification,
matched-information baseline, budget and held-out evaluation protocol.

Engineering calibration, patient integration, graph attribution, online RL and
clinical safety remain separate gates. The original Stage E evidence is not
changed by this synthetic packet. All results remain local; no remote push or
main merge occurred.

Final validation: 110 combined focused tests and full Python compilation
passed. The queue evidence audit was rerun after reporting edits and again
verified all 59 inventoried hashes. Its original limitation about independent
nonanticipative optimization is resolved by the separate recorded diagnostic,
not silently removed from the old audit result.
