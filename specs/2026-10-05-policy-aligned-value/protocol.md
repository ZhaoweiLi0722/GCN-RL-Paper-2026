# Frozen-Parent MPC Continuation: One End-to-End Comparison

Status: complete prospective proposal, NOT scientifically executed. Preparation
follows Zhaowei's request to move directly to the next substantive comparison.
The new numerical package below is not the consumed planner-tail/recovery package.
The draft config retains scientific_execution_authorized=false. One package-level
approval covers committed locks, one run, training, evaluation and readout together;
there is no additional routine launch question and no automatic retry.

## Question And Amendment

Does learning public-model tail costs under a frozen, competent MPC controller,
rather than an adaptive rule, improve native simulated cost and patient outcomes
beyond the existing frozen value-MPC? The preceding completed comparison improved
over plain H8, but its increment over existing_frozen was not established and
TD was not distinguished from matched-data direct return regression. Its labels
continued an adaptive rule, not the deployed MPC controller. This is a concrete
policy-evaluation mismatch hypothesis, not evidence that fixing it will succeed.

Reuse the same five historical GCN ancestors, each with1536 historical updates.
Fork identical weights with fresh Adam/RNG/new-update counters into three methods:

| Role | New targets | Evaluation |
|---|---|---|
| plain_h8 | None | Ordinary H8 MPC |
| existing_frozen | None; historical ancestor unchanged | H8 plus historical value |
| adaptive_tail_td | 8-step TD under adaptive-rule continuation | H8 plus final value |
| policy_tail_td | 8-step TD under frozen-parent MPC continuation | H8 plus final value |
| policy_tail_mc | Direct full-return regression on the SAME MPC-tail records | H8 plus final value |
| plain_h16 | None | H16 ordinary MPC, twice the prediction queries |

The label controller is fixed at existing_frozen throughout collection. It is
NOT the changing student and is NOT exact policy evaluation of the updated
policy. This tests one approximate policy-improvement step. Public-model bias
remains possible; predicted tail returns are not native patient counterfactuals.
MC is Monte Carlo-style policy evaluation, not automatically a non-RL control.

Reward/cost weights, physical scenario, public inputs, graph/model width, action
support and settlement stay unchanged. Keep all original patient costs and labor
accounting. No reward tuning, new scenario, additional initialization, actor fit
or structure search. The inherited synthetic labor model remains uncalibrated;
E1 data remain missing. Historical block0 mixed-predictor provenance is retained.

## Shared Collection And Targets

Five blocks,24 new ordinary-H8 reference worlds each, eight per existing condition
(index%3), interleaved. Each world includes48 control plus16 settlement steps.
For world index i, roots are i and i+24. At each root select candidate
(i + 8 * root_slot)%16, where root_slot is0 or1, before looking at outcomes.
The public predictor runs the candidate's H8 prefix at response quantiles
0.1/0.5/0.9. Candidate support16 and quantile summary weights.25/.5/.25 are unchanged.

After the prefix, clone each endpoint into two continuations. One uses the
existing adaptive rule. The other replans using the frozen ancestor's H8 value-MPC
and a cloned public filter/lifecycle, updated only with hypothetical public
receipts. Prefix and tail operations preserve public history and patient identity.
After epoch48, no new flexible hours are requested; unchanged settlement runs
through64. No native hidden state, extra native branch or test observation enters
label generation. Shared float64 prefix cost/end-state bindings must match.

This yields6 tails per continuation policy per reference world,720 matched pairs
overall. Policy TD and MC use identical frozen-parent tail records, row ordering
and sampler stream. Adaptive TD uses separate matched-root adaptive records.
One candidate/root is a prospective compute allocation, not outcome selection.
It is sparser than the previous96-tail recipe; cross-study differences cannot
isolate continuation policy. The WITHIN-study adaptive-TD contrast does so subject
to model bias and its additional label-generation computation.

All states in the tails are used, not only endpoints. For end=min(t+8,64):

`y_TD=(sum(cost[t:end])+1[end<64]*(H[end]+V_frozen[end])-H[t])/1e6`

`y_MC=(sum(cost[t:64])-H[t])/1e6`

Freeze each cohort's targets before32updates. Gamma1; Adam.0003; batch64;
gradient cap5; CPU float32 network; float64 costs, residual subtraction and loss.
Uniform row sampling gives longer tails more row weight; no equal-root-weight
claim. Each method gets32updates/world,768/model,3840/method,11520 total,0actor.
All three arms have the same initialization and update count. TD requires extra
bootstrap forwards; equal data/updates do not imply equal total computation.

## One Serial Schedule And New Tests

For each block, bind/import its ancestor; collect24 shared reference worlds with
both continuation datasets; update the three forks serially after each complete
world; seal the three final models. All15 finals and5 unchanged ancestor bindings
must be sealed before opening the test worlds. No early stopping/model selection.

Then60 new test worlds (5blocks x3conditions x4replicates) each run all6 roles:
360 evaluation trajectories, zero evaluation updates. Shared keyed exogenous tapes
do not imply identical endogenous paths. Total120 reference+360 evaluation=480
native trajectories. The first counted reference world is the real preflight;
no extra smoke world. Save raw records, targets, updates, public tails, optimizer,
sampler/RNG and controller states at existing boundaries. Branch failure snapshots
mark possible partial primitives and do not promise automatic or bitwise recovery.

Config: experiments/configs/capacity_policy_tail_20261005.json.
Pure schedule/counter source: src/rl/capacity_policy_tail_design.py.

| Count | Maximum |
|---|---:|
| Native steps / operations including construction and reset | 30720 /31680 |
| Value / actor optimizer calls | 11520 /0 |
| Optimizer example presentations | 737280 |
| Neural forward module calls, batch at most64 | 36190 |
| Native planner decisions / candidate-quantile rollouts | 23040 /1105920 |
| Native-decision prediction steps | 9953280 |
| Nested continuation planner decisions / prediction steps | 12300 /4723200 |
| Shared candidate-prefix steps / paired continuation steps | 5760 /46800 |
| ALL prediction steps | 14729040 |
| Forecast root constructions / continuation clones | 720 /1440 |
| Native filter transitions / branch filter transitions | 3072000 /2916000 |
| ALL filter transitions | 5988000 |
| New final models / unchanged ancestor bindings | 15 /5 |

36190forwards include11520 fitting,850 TD-bootstrap,12300 frozen-parent label
planning and11520 evaluation. Nested planner work is charged separately from
native planning BEFORE dispatch, including every384-epoch decision reservation;
filter reservations are100transitions. Interrupted reservations are not refunded.
No post-fit ranking forward is planned; only saved paired tail costs are analyzed.

Wall caps INCLUDING IO/archive: admission600s; reference/tails23400s;
fitting3600s; evaluation18000s; analysis/archive1800s; failure preservation3000s;
global50400s (14h). Per-owner: reference world with tails600s; H8 evaluation180s;
H16 evaluation300s; eachfit60s; eachseal60s. No phase transfer or refund.
CPU4threads/1worker;4GiB RSS;4GiB raw+4GiB archive=8GiB combined;6000files.
The14h figure is a cap, not a measured runtime estimate. New nested-label runtime
is unmeasured. Evaluation has5h prospectively, addressing the known previous
4h timeout without altering or reopening its consumed attempt.

Streams: namespace capacity-policy-tail-20261005-v1; reference65000000 and
evaluation65020000 plus1000*block+100*condition+replicate; sampler
550270510/530/550/570/590 shared within block; bootstrap65090001. Before execution,
perform the existing scoped manifest collision check and commit source/runtime/
input/stream locks plus exact authority. A collision stops launch; no silent
reseeding. Old results and formal holdout are not independent tests for this run.

## Performance Decision And Stop

Primary policy_tail_td must beat plain_h8, existing_frozen AND adaptive_tail_td
under persistent change: for each comparator, all five block mean absolute cost
savings positive and paired two-level descriptive95% bootstrap lower bound>0;
also no condition-mean extra lost patients. Use2000resamples across five blocks
and their paired worlds. This is a development screen, not multiplicity-adjusted
confirmation, clinical safety or a guarantee of publication.

Report all15role-pairs x3conditions x5blocks, absolute/paired-percent costs,
components, losses/deliveries, patient/world harms, requested/applied labor,
action differences, decision median/p95/max latency, complete training/planning
counts and historical cost separately. TD/MC isolates target construction on
identical MPC-tail data, not compute equality. H16 is a stronger-compute reference,
not an oracle. Better loss or changed actions alone is not performance gain.

Read saved matched tail costs as an accompanying mechanism description only.
One candidate/root does not support a16-candidate ranking or best-action claim.
No extra model calls, diagnostic rollouts or preliminary diagnostic gate.
Independently reconstruct native costs/patient outcomes and verify the one local
archive with existing facilities. Preserve all positive, null and adverse results.

One attempt. Terminal failure preserves evidence and stops; no automatic repair/
retry, extension, additional epochs, seeds or reward search. Scientific negatives
close this package rather than trigger expansion. Complete the readout and honest
manuscript update; pause the same monitor. Later reward/objective amendments remain
possible with evidence and prospective scope, not as an automatic response to a
negative result. Stage E stays closed; no Howard approval, remote push/PR/merge,
external messages, Dropbox export or clinical/deployment claim is implied.
