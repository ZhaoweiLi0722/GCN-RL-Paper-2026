# Planner-Aligned Value Learning: One Integrated Comparison

Status: complete numerical PROPOSAL, NOT authorized or launched. Zhaowei's
`推进` authorizes implementing the endorsed direction, not these previously
unproposed scientific calls. The draft configuration remains
`scientific_execution_authorized=false`. Approval of this package will cover
necessary integration, committed locks, one run and readout without a second
routine launch question. It does not authorize automatic retries.

## Question And Six Arms

Can model-based TD value learning on the planner's candidate terminal states
improve actual simulated patient/cost performance beyond a competent MPC and
the already-trained frozen GCN value model?

| Role | Training | Planning at evaluation |
|---|---|---|
| plain_h8 | None | Existing 8-step MPC, no learned residual |
| existing_frozen | Reuse the completed five graph models unchanged | H8 plus saved value |
| observed_td | Continue the existing trajectory-based TD target rule | H8 plus continued value |
| planner_tail_td | TD on complete public-model forecast tails | H8 plus aligned value |
| planner_tail_mc | Direct return regression on the SAME forecast tails | H8 plus aligned value |
| plain_h16 | None | 16-step MPC with the same candidate sequence generator |

The added observed_td control prevents extra updates alone from being presented
as the new mechanism. All three trainable forks per block start with the exact
same historical graph weights, fresh Adam state, equal sampler seeds and zero
NEW update counters. Do not import historical Adam/RNG/counters through the old
full-state loader. Retain the unchanged ancestor for the frozen arm. Each ancestor
has 1536 historical updates, 3169 parameters and its existing provenance, including
mixed predictor training in block0. Reuse is not independent model replication.

Reward, costs, physical dynamics, public features, graph width/topology and action
support are unchanged. All H8 planners use the same corrected public predictor,
16 candidate sequences, three response quantiles and weights [.25,.5,.25]. H16
changes only planning horizon and explicitly spends twice the prediction queries.
It is a compute reference, not a matched-query comparator or an exact oracle.

## Data And Targets

Five blocks, each collecting 24 NEW plain-H8 reference worlds, eight per existing
condition, interleaved as condition=index%3, replicate=index//3. No learned arm
controls this data collection; generate it once and expose it to all learners.
Every world has 48 control and 16 fixed settlement steps. For world index j in
0..23, capture decisions j and j+24. Thus all 48 decision times are covered once
per block without outcome-based selection.

At each captured decision save all 16x3 forecast endpoints after the normal eight
prediction steps. Copy each endpoint and continue its existing public adaptive
rule until epoch48, then zero support to epoch64 with unchanged base operations.
Keep every tail-state feature, heuristic and float64 cost, including settlement.
There are 96 tails/world and 11520 overall; 374400 additional prediction steps,
not native patient worlds. Do not regenerate them under different learner policies.
Preserve root prefix costs and candidate/quantile identities for diagnostics.

These are approximate PUBLIC-MODEL returns, NOT observed counterfactual patient
returns or native ground truth. The tail policy is the specified adaptive rule,
not the learned MPC. This is model-based policy evaluation/amortized tail scoring.
It cannot by itself repair an inaccurate forecast model. Direct return regression
is Monte Carlo-style policy evaluation, not automatically a non-RL arm.

Observed TD uses all64 saved actual-state rows with the historical 8-step target
rule. Forecast TD and MC both use EVERY state in the same complete tails, not
only roots. Near-terminal rows anchor TD; otherwise all root bootstraps would
miss terminal supervision. For a state t with end=min(t+8,64):

`y_TD = (sum(cost[t:end]) + 1[end<64]*(H[end]+V_residual_frozen[end]) - H[t])/1e6`

`y_MC = (sum(cost[t:64]) - H[t])/1e6`

Freeze all cohort targets before its32updates. TD bootstrap calls are chunked
to at most64 examples. TD/MC use identical row ordering and sampler streams;
sample uniformly across the concatenated tail rows, so earlier/longer tails
receive more row weight by design. No claim of equal per-root weighting.
Use CPU float32 networks, float64 costs/target subtraction/MSE for ALL continued
arms, Adam0.0003, batch64, gradient cap5, gamma1,32updates/world and final-only
selection. Each of15 new models gets768 updates. The shared double-precision loss
is an explicit common numerical change from the historical float32 loss, not a
reward revision. No hyperparameter search, actor or additional initialization fit.

## One Schedule, New Tests And Caps

For each block import/check ancestor weights, collect the24 shared worlds with
saved tails, and update each fork serially32times per completed world. The ledger
switches between collection and fitting owners without resetting elapsed totals.
Save model/optimizer/RNG/frozen-target/sampling state after each complete fit,
raw eight-step boundaries and failure state, but do not authorize automatic resume.
Only after ALL15 new final models and five unchanged ancestor bindings are sealed,
open60 new test worlds (5blocks x3conditions x4replicates), each with all6 roles:
360 evaluation trajectories, zero test updates. Total480 NEW native trajectories.

Config: `experiments/configs/capacity_planner_tail_20261004.json`.
Exact caps, independently derived by `capacity_planner_tail_design.numeric_contract`:

| Quantity | Cap |
|---|---:|
| Native steps / operations including construction+reset | 30720 / 31680 |
| Value / actor optimizer calls | 11520 / 0 |
| Optimizer example presentations | 737280 |
| Neural forward calls, batch at most64 | 30000 |
| Native-decision planner calls / candidate-quantile rollouts | 23040 / 1105920 |
| Ordinary planning prediction steps | 9953280 |
| Shared training-tail prediction steps / forecast copies | 374400 / 11520 |
| All prediction steps | 10327680 |
| Live filter hypothesis transitions | 3072000 |
| New final model seals | 15 |

Forward cap includes11520 gradient calls,120 observed bootstraps,5880 tail-TD
bootstrap batches,11520 evaluation value calls and960 saved-training-root
diagnostic calls. MC has no bootstrap call; do not disguise equal update counts
as equal total compute. Log historical training cost separately from new compute
and amortized deployment cost. Shared collection/tails are charged once globally;
also report the data/query access each standalone method would need.

Wall caps: admission600s; reference collection/tails10800s; fitting3600s;
frozen evaluation14400s; analysis/archive1200s; failure preservation1800s;
global32400s (9h), INCLUDING IO/archive. Owners: reference world+tails240s;
H8 evaluation120s; H16 evaluation240s; fit60s; seal60s. No phase transfer or refund.
CPU4threads,1worker,4GiB RSS;4GiB raw plus4GiB archive,8GiB total/6000files.
Rough planning estimate6-8h is extrapolated from the preceding comparison, not a
new speed measurement or guarantee. All caps apply even if that estimate is wrong.
First counted reference world is the real preflight, with no extra smoke world.
One attempt; failure preserves evidence and closes the attempt, no automatic retry.

Prospective namespace `capacity-planner-tail-20261004-v1`; reference seed64000000,
evaluation64020000, plus1000*block+100*condition+replicate. All six roles share the
same evaluation world/tape; endogenous paths can diverge. Sampler seeds are
540270510/530/550/570/590, shared within each block; bootstrap64090001. Keyed
purpose streams reuse the established function. Before execution freeze input,
source/runtime and full streams, with a scoped local historical-manifest collision
check. Collision stops launch, never silently substitutes seeds. No old evaluation
worlds or formal holdout are used. Existing historical model selection is disclosed.

## Readout And Stop Decision

Primary candidate planner_tail_td must beat plain_h8, existing_frozen AND
observed_td separately under persistent change. For each: all five block mean
absolute savings positive, descriptive95% paired two-level bootstrap interval
above0 (2000 resamples, five blocks and within-block worlds), and no condition
mean extra patient losses. These are development screens, not multiplicity-adjusted
confirmation, clinical safety or a guarantee of publication. Report block and
world harm even if aggregate screens pass; no post-test sample expansion.

Report all15 role pairs under all3 conditions: absolute/paired-percent cost,
all cost components, losses/deliveries, requested/applied labor, action differences,
total training/prediction compute and decision latency distribution (median/p95/max
and hardware). Explain TD-vs-MC target evidence separately: beating frozen alone
does not prove TD is better than direct-return learning. Similar MC performance
supports model-based tail learning, not unique TD superiority. Longer-horizon MPC
can win; report its compute/performance trade-off honestly.

Use final models to score saved TRAINING roots once, recording candidate cost
ranking versus their recorded public-model tail returns, calibration and regret
within this candidate set. These are in-model training-data diagnostics, not an
independent performance test or new acceptance screen. No new diagnostic rollouts,
early stopping or parameter selection. Reconstruct actual test costs/patient
outcomes independently from raw records; test outcomes, not loss/ranking, decide.

Keep reward unchanged for THIS package. A later justified objective/accounting
amendment remains possible, not prohibited forever. Poor results do not authorize
reward search, more samples, epochs, new scenarios or a retry. Retain every result,
archive once and update manuscript claims honestly; the original Stage E stays
closed, E1 remains missing, synthetic assumptions remain explicit. This package
does not isolate GCN edges or demonstrate deployment-time online adaptation.
Local commits only; no remote/share/export action or Howard approval is implied.
