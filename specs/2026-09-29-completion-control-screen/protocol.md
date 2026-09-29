# S2: bounded conventional-control feasibility screen

## Scope and fixed design

S1 mechanics completed with audited closure and no online-learning evidence.
Zhaowei requested continuation of the proposed conventional-control comparison.
This authorizes the bounded local synthetic screen below, not patient-model
changes, a neural campaign, formal confirmation or Howard's sign-off. Commit
protocol/config and implementation before recorded execution. Preserve S1 and
all earlier evidence. Do not change reward/scenario parameters after outcomes.

Reuse S1's two-site physics, eight booked jobs, release times, continuous request
bounds, one-interval delivery lag, fixed downstream work and all accounting.
For this pilot all controllers select from the same 15-point quarter-unit simplex
grid {(i/4,j/4): i,j>=0, i+j<=4}. Future continuous-action comparisons need
additional action-resolution checks; this grid is not a global optimality bound.

Three response families: unchanged (1,1); persistent unknown orientation of
(0.5,1.5) after interval 8; independently resampled orientation each interval
after 8. This last family replaces deterministic alternation **in this new study
only**, not in S1's records. Both changed families have the same marginal rate
distribution by design, but only the persistent one has sustained orientation.
The four persistent worlds balance orientations by world index. Controllers
receive no family, orientation, true rates, switch flag or private future tape.

Cross 3 families x 2 downstream capacities (1,4) x 4 fresh world indices x 2
splits (discovery, replication). Run all six controllers below in each of these
48 paired worlds: 288 complete episodes. Replication is fixed before discovery,
not an opportunity to select a method or alter parameters. Both splits are
development evidence, not the old formal holdout. Namespaced SHA256-derived
seeds separate actual completion uniforms, response changes and planning noise.
Methods share exogenous uniforms within a world, not their realized histories.
Do not treat methods, time steps, capacities or paired families as independent
replications; only four world indices per cell/split is a feasibility sample.

## Controllers and information

1. Booked-backlog rule: the unchanged S1 public availability rule.
2. Receipt-aware reservation rule: estimate next-interval support need. If two
   tasks are currently queued or a job is booked for next interval, need=1;
   if one task is queued, need=exp(-estimated_rate*pending_effort); otherwise 0.
   Request half a unit times need per site, projected to the shared grid by
   nearest squared distance, deterministic grid-order tie break. This is a fixed
   heuristic, not an optimized policy or a certified strong baseline.
3-6. Fixed-model and identification rollout-MPC, each with 16 or 64 Monte Carlo
   continuations per candidate. Both planners use identical code, grid, fixed
   continuation, noise and budgets. Only response estimate differs: nominal
   (1,1) versus the shared receipt filter's posterior mean. All methods maintain
   the same filter from their own receipts. Frozen policy later means fixed
   neural weights, not disabling this available system-identification interface.

The rollout planner optimizes one current request over all 15 actions, then
forecasts the public booked-backlog rule through **complete settlement**. It
replans at each actual decision; it does not minimize only immediate cost or
truncate a costly tail. Assumed response stays constant at the supplied estimate
within each forecast. It is a plug-in model, not an optimal belief-space planner.
This is a restricted rollout-MPC comparator, not evidence that stronger MPC
could not do better. Return all candidate expected costs and planning query/time
counts. Common planning uniforms are independent of actual uniforms. Within a
decision the 16-sample forecast is a prefix of the 64-sample forecast.

If no current support task and no next-interval booking exist, zero purchase
is known dominant and no rollout is needed. At interval 32 issue the common
zero request, then use the common booked-backlog continuation until all jobs,
downstream work and prepaid requests settle. Retain all closure cost. Fail at
the unchanged 256-step closure cap; never drop failed worlds or extend the cap.
Forecasts also fail on unsettled paths rather than treating truncation as free.

## Planner adequacy probes and outcome reporting

Before judging feasibility, retain the public states at epoch 9 of the booked
rule in discovery world 0, for all 3 families and both capacities (six probes).
At each state use the current estimated response to compare 16 vs 64 forecast
samples for depth 1, and depth 1 vs depth 2 at 64 samples. Depth 2 enumerates
all 225 two-request open-loop prefixes, then the same complete continuation.
It still is not a full contingent-policy optimizer. Report first-action agreement,
candidate costs and computation; do not pick whichever depth looks favorable.
Any disagreement prevents a claim of planning-budget/depth stability. Even full
agreement at six states is only a local diagnostic, not convergence proof.

Predeclared descriptive contrasts, per family/capacity/split: reservation minus
booked rule; fixed64 minus booked; ID64 minus fixed64; fixed64 minus fixed16;
ID64 minus ID16. Report all raw world pairs, mean/min/max differences and sign
counts in synthetic cost units. Negative means lower complete cost. No p-values,
power claim, operational benefit threshold, selected checkpoint or selective
sample extension. Record discovery and independent replication separately.

This comparison cannot establish headroom over a competent frozen neural policy,
nor online weight-update value. A lower ID-than-fixed complete cost attributes
benefit to ordinary identification/control. Negative, null or unstable findings
remain reportable. Before any DDPG launch, require adequate planning checks,
independent full-cost action-ranking replication, operational/synthetic study
scope and a competent matched-information frozen-policy comparator.

## Verification and bounds

Actual episodes always use the unchanged scalar S1 transition. Add a batched
forecast implementation to make the finite screen practical. Test every batched
state/cost against scalar trajectories, including FIFO ties, action delay,
downstream occupancy, arrivals, cost closure and an unfinished-path failure.
Audit saved actual rows by replay through unchanged S1 physics, recompute ledger,
filter and paired metrics, and verify hashes/identities. Test private-randomness
separation, sample-prefix matching, invalid actions, deterministic decisions and
input immutability. Run all related tests plus full Python compilation.

Record one matrix, not repeated searches. Recompute summaries from saved rows
and replay a bounded selected episode rather than rerunning all planning solely
for byte equality. A 900-second wall cap aborts the screen with partial records
and failed status; do not relaunch or retune automatically. This runtime limit
does not silently convert incomplete output into scientific evidence. No GPU,
teacher, historic checkpoint, neural optimizer, remote operation or publication
claim is part of S2. Formal Stage E and E1's missing site data remain unchanged.
