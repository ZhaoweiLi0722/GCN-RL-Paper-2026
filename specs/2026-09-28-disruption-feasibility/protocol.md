# Synthetic disruption feasibility pilot

Frozen before execution, 2026-09-28. Authorized by Zhaowei's integration and
scenario-exploration request. No proxy Howard approval or formal confirmation.

## Question and design

Can two new operational abstractions run consistently, and how much does a
stronger **non-learning** controller change cost and patient outcomes? This is
a baseline feasibility pilot, not an RL headroom gate, training campaign,
claim of near-optimality, or calibrated real-world result.

Use the routing-primary 20-site nominal configuration and existing patient
weights (500k/100k/25k), qualification graph and lifecycle. Disable demand and
supplier shocks, and demand forecasts, to isolate each new mechanism. Demand
remains stationary Poisson. All arms have the same changes and action authority.
The initial procurement delay distribution on delays 0/1/2/3 is .8/.2/0/0.

Pre-pilot mechanics amendment: the six-step test-seed smoke stopped at step 1
of its first no-change episode because the legacy per-site idle-stock clip
removed 5.6000005 reactor units. The failed smoke and its original composed
config are retained in `reports/2026-09-28-disruption-feasibility/smoke/`.
Before any discovery/validation outcome was generated, the new pilot's
`max_idle_bioreactors` was set to 210 per site, the entire initial fleet, in
all cells. This disables accidental inventory disposal; it does not create
equipment. It also relaxes a legacy local storage constraint and can change
control behavior. The pilot is therefore explicitly a **different synthetic
operating model**, not a robustness evaluation of the original formal system.
No historical environment code or evidence is changed. A realistic overflow
handling/storage constraint needs domain input before a full study. The
mechanics rerun uses `smoke-r2/`; no scientific run is being retried.

Four cells, with paired worlds shared across cells and policies:

1. Capacity no-change control.
2. Persistent capacity outage at decision 13: quarantine up to floor(50% of
   the initial fleet) at the first five facilities as units become idle.
   Production already underway is protected. Units are not destroyed or
   transferable while quarantined. Their holding cost remains charged.
3. Lead no-change control, deliberately identical to cell 1 for a reproducible
   negative control.
4. Persistent procurement change at decision 13 to .1/.2/.4/.3; orders already
   placed keep their original due dates. Delay 0 is available next decision;
   delay L is available at decision t+L+1 under the existing engine.

These ranges are illustrative, not empirically calibrated. Do not change them
after seeing pilot outcomes. The outage is idle-unit availability, not a claim
that half of all active treatments or throughput disappear instantaneously.

## Information and comparators

Current usable idle capacity is observed; future outage onset is not supplied
to policy code. Procurement due-date buckets are observed after ordering, an
explicit **acknowledged-ETA assumption**, not unknown in-transit delays. The
new distribution itself is not given to any policy; lead-aware rules retain
the original prior. Future work must include an estimator using public order
and receipt history, and an equally informed MPC, before claiming RL value.

Fixed challengers: MDL-2, lookahead-3 MDL, inventory-position/lead-aware MDL-2,
and the same lead-aware rule with 1.2 safety multiplier. All are deterministic
feedback rules, not online policy updates. Existing MDL-3 is mandatory even
if a clinical endpoint is worse. No planner or checkpoint is run here.

Four discovery worlds (203000000--203000003); eight fresh validation worlds
(204000000--204000007). A tracked-source/config seed search found no use of
these families before this spec. This cannot certify absence from unshared
external runs. Select the lowest mean-cost rule on discovery per cell, then
report its validation result. Report all rules; do not choose by validation
or hide adverse clinical endpoints. Expected 192 episodes, 9,984 steps.

## Endpoints and stopping

Use the existing 52-decision objective unchanged. Report modeled total cost,
completion, manufacturing ineligibility, patient loss, cost components, and
terminal waiting/in-production/in-transit patients and ordered inventory.
No new salvage value or terminal penalty is introduced. Incomplete terminal
outcomes are a limitation: no clinical noninferiority conclusion or training
advancement can follow this pilot. A settlement/terminal rule must be locked
for a full sequential-control study.

Report paired mean cost differences and individual-world direction; no
significance or safety claim from eight worlds. Stochastic-vs-no-change cost
is disruption burden, **not** an optimality gap. Rule-vs-MDL-2 difference is
an achieved rule improvement, **not** unexplained RL opportunity.

Stop on nonfinite metrics, broken identity, inconsistent arrivals/RNG pairing,
nonconserved reactor stock, or missing/duplicate rows. Preserve failures.
No retries, severity sweep, algorithm tuning, reward changes or neural training
are triggered by the result. Smoke uses test seed 123 and a separate output.

## Next gate, not authorized by this pilot

Information-interface parity; operational ranges and clinical-margin rationale;
strong adaptive comparator; repeated-decision headroom; independent ranking
replication; then a prespecified belief-update x policy-update experiment.
Frozen vs online must share initialization, observations, simulator-query and
interaction budgets. Costs of online identification and planning are recorded.
