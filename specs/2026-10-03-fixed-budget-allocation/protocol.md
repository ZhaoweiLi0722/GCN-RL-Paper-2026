# Fixed-Budget Capacity Allocation: One Performance Comparison

Prepared2026-10-03 from847f5d2. Zhaowei's `继续推进` accepts preparing the proposed
fixed-total direction after the completed negative trial. This new numerical
package still needs its one explicit execution decision. Once approved, necessary
implementation, committed locks, training, evaluation and readout proceed together
without another launch question. Companion: experiments/configs/fixed_budget_capacity_20261003.json.

## Question And Scientific Change

Can RL training improve **where a fixed amount of flexible support labor goes**,
relative to exact uniform allocation, without increasing expected patient losses?
Keep corrected ID-MPC as a strong external comparator. In the preceding completed
study two constrained models averaged only0.0977/1.2834 committed hours against
fixed8; all primary criteria failed. This motivates preventing total-resource
withdrawal, not assuming that under-allocation has a proven learning cause or
that the old reward weights were wrong. Old attempts remain closed and unchanged.

One fresh constrained learner per block; do not repeat the cost-only ablation or
rerun old failures. Inherit the old system, public inputs, graph/widths, physical
cost ledger and dual-critic objective by hash. No field data have appeared; E1
remains missing and the support-labor channel remains synthetic.

Replace only the actor's action family and corresponding exploration:

    h_i = 2 + tanh(z_i) - mean_j(tanh(z_j)), i=1..4.

In exact arithmetic the sum is8hours, with each site in[0.5,3.5] (closure).
The zero final actor layer yields exactly[2,2,2,2]. The map is permutation
equivariant but not invariant to a common shift of logits; do not claim otherwise.
This is a bounded redistribution family, NOT the entire0..4 capped simplex.
It intentionally excludes complete withdrawal from a facility. The bounds are
declared design choices, not clinically validated staffing requirements.

Use float32 network logits and a float64 map; cast tofloat32 only for existing
critic inputs. Native requested hours retainfloat64, with deterministic rounding
closure (final component from8minus the first-three sum; reject invalid range).
Absolute total residual <=1e-12hours and strict shared-budget non-exceedance;
never modify native projection or its tolerances. Report residuals and any native
projection change. Counterfactual gradient actions may have ordinaryfloat32
roundoff, covered by the unchanged critic-input validator.

Training adds iidGaussian sigma0.25 to logits BEFORE the map. There is no
exploration during evaluation. This is not the old post-clipping hours-noise
distribution, and the study cannot isolate a causal action-map effect against
historical results. Target-actor/actor-loss actions use the same noiseless map.
All64epoch worlds retain48control and16zero-action settlement epochs;8hours
applies only to control epochs, not the forced drain tail. Hours are commitments
with the existing two-step delay, not contemporaneously delivered labor.

Fixed total labor does not guarantee patient safety, useful allocation or equal
total financial cost. Different placements affect delay, expiration and resource
costs. Do not remove these outcomes or change their weights to obtain a gain.

## Learner And Comparators

Keep the prior exact fixed start, independent reward/loss critics, gamma1,
tau0.005, Adam actor1e-4/critics3e-4, batch64 and gradientcap10. Full cost reward
is negative settled cost/100000; terminal transition includes every tail cost
and loss. Per-condition multipliers start1, step0.01 times paired final excess
losses, bounded[0,20], updated after each complete training cohort. This remains
approximate Lagrangian DDPG, not guaranteed feasibility. No learning-rate/lambda
or architecture search. Three condition IDs enter replay/dual scheduling only,
not neural inputs. Strong frozen history/public-filter information stays intact.

For each of3fresh blocks, collect12fixed reference worlds (4/condition), warm
both critics256batches with actor frozen, preserve an initial full seal, fork
one constrained arm and collect the same12training world seeds. After each world
settles, update its multiplier once and perform48actor+two-critic updates from
32reference/32own rows of the current condition. Preserve replay/optimizer/RNG
and rollout state; snapshots confer no automatic retry permission.

After all3final models are sealed, compare three greedy frozen controllers on
36fresh paired worlds: trained constrained actor, exact same-start uniform fixed
allocation, and corrected ID-MPC. Uniform fixed is analytically identical to the
zero-head actor for every input, not a weakened learned imitation. MPC retains
its own originally defined resource decisions, including spending less than8;
do not force it into the new learner's action family to improve the comparison.
No BC, historical models, model selection, evaluation-time update or extra screen.

## Single Attempt And Budget

Streams: modelseeds520260501/502/503; shared reference/trainingbase62800000;
evaluationbase62820000; worldseed=base+1000b+100c+j (b,c0..2,j0..3).
Namespacefixed-budget-capacity-v1 with the existing keyed SHA256 substream rule.
Bootstrap62890001,2000resamples,blocks then paired worlds bycondition. Full local
derived-stream collision check is required before freeze; no silent replacement.

| Phase | Worlds | Native steps | Actor steps | Critic steps | Module forwards |
|---|---:|---:|---:|---:|---:|
| Fixed reference and shared warmup | 36 | 2304 | 0 | 1536 | 3840 |
| Constrained training | 36 | 2304 | 1728 | 3456 | 15552 |
| Three-controller frozen evaluation | 108 | 6912 | 0 | 0 | 1728 |
| Total | 180 | 11520 | 1728 | 4992 | 21120 |

6720neural optimizer calls/430080example presentations,36scalar multiplier
updates; no optimizer in evaluation.180native constructors+180constructor resets
plus11520steps=11880native operations; no extra resets/clones.36MPC worlds have
1728planning decisions/82944candidate-quantile rollouts/663552forecast epochs.
11520public-filter updates/1152000hypothesis transitions.3initial+3final seals,
3training forks and36evaluation model restorations. No extra smoke world: the
first reference world is retained as the budgeted real preflight.

CPUfloat32 network,4threads,1worker,4GiBRSS,1GiBraw+1GiBarchive,max1600raw files.
Global5400seconds INCLUDING recording/archival: admission300,reference/warmup600,
training1200,evaluation2400,readout/archive600,failureflush300. Perreferenceblock200,
pertrainingblock300,perevaluationcontroller/block/condition200seconds; allcaps
apply concurrently. Inherited internal phase IDs still saytwo_arm/four_arm for
compatibility only; the new contract has ONE trained arm and THREE evaluation
controllers. No budget transfer/refund. Failed attempt preserves evidence and
ends; no repair/retry, extra epochs or positive-result search under this package.

Preparation estimate30-60min; actual prior run was31.94min, with26.18min inMPC-
dominated evaluation. New run estimate30-45min under90min hardcap. These are
estimates, not new thresholds or performance promises. Remove optional work if
preparation overruns; do not add toy fitting gates.

## Readout And Closure

Primary: trained-vs-fixed in persistent change. Same development signal as the
previous trial: positive mean cost savings in all3persistent blocks; descriptive
95%savings interval above0; nonpositive mean extra lost patients in EACH condition.
Report all worlds, components, losses/deliveries, tail liabilities, committed
versus actually applied labor, native total residuals, action sizes and intervals.
MPC comparison is mandatory. Three seeds do not establish clinical safety or
independent confirmation. Better than fixed but worse thanMPC is a limited gain,
not best-controller superiority. Costdown/lossup remains an explicit trade-off.

Success would support RL training within a restricted fixed-resource family,
not full joint resource optimization, deployment-online adaptation or isolated
GCN contribution. Failure closes this package without budget/weight/action-range
search. Independent replication or a broadened action family requires a later
complete decision. Keep historical tests as development evidence, not confirmation.

## Engineering And Authority

Reuse the settled trajectory collector, loss-target/replay/restoration logic,
ledger, archive and corrected predictor. Add a versioned action/learner adapter,
one-arm schedule, raw fixed-budget check and exact execution binding. Only fake
entry and artificial forward/gradient tests before approval; no actual optimizer,
scientific model load/score or native patient construction. Preserve old sources.
Commit source/config/runtime/input/seed locks and exact approval before launch.
One complete approval covers all routine stages; no separate launch question.
Onlylocalcommits; no remote actions,Dropbox,holdout,StageE reopening or Howard
approval claims. The monitor staysPAUSED until actual authorized execution.
