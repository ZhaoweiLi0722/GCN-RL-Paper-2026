# Original Twenty-Site Value-TD: One Targeted Comparison

Status: proposed, NOT authorized or launched. The instruction to advance
authorizes additive implementation and this complete proposal, not its new
numerical budget. No historical experiment is reopened. This packet is intended
to close T1/T2 together, not repeat DDPG/TD3 or search for a favorable family.

## Fixed Problem and Reused Work

Use `patient_indexed_specimen_routing_benchmark.json` and its existing
`make_scenario_env_config` composition for the FOUR original primary scenarios.
Keep all environment fields, twenty facilities, geographic topology, physical
limits, specimen identity, reagent transfer, idle-bioreactor relocation, timing,
patient dynamics and costs unchanged. Twenty-one graph nodes, where applicable,
mean twenty facilities plus the EXISTING virtual capacity hub, not21sites.
Keep the original52-epoch endpoint. Do not introduce the four-site48+16support
task or a new settlement objective. Report outstanding obligations at cutoff;
the finite-window cost is not a patient's entire lifetime outcome.

Reuse old twenty-site graph/DDPG/TD3 results, the existing public-information
rollout machinery and all valid component checks. No rerun of headroom,
clean-critic, five-family, coefficient, teacher or old architecture campaigns.
No historical checkpoint or old formal world initializes or selects this study.

## One Method, Four Roles

1. **Original MDL-2:** operational baseline, with its original resource actions.
2. **Native-H8:** nonlearning, receding-horizon8-step rollout heuristic. This is
   an ORIGINAL-TASK implementation, not the four-site support-hour H8 model.
3. **Graph Value-TD:** the identical planner plus a learned terminal cost value.
4. **Flat Value-TD:** the identical planner, learning rule and feature tensor,
   with a global MLP in place of the GCN architecture.

All planning roles use the archived native14-option candidate generator:
MDL-2/MDL-3 anchors with specimen corrections, and MDL-2 reagent and combined
reagent/capacity corrections. All THREE transfer channels remain available.
Capacity corrections in this class are coupled with reagent corrections;
this is a finite existing action class, not unrestricted continuous network
control or independently optimized flow on every edge. Each requested action
is executed through the unchanged native feasibility/identity mechanisms.
The physical task is unchanged; this declared policy-class restriction must
remain visible in the methods and interpretation.

For every candidate, use2 common prediction-future starts, up to8native steps,
MDL-2 continuation after the first action, and identical information access.
Reuse only the nonprivileged conditional patient sampler and observable demand
estimator. No actual future events or true hidden patient attributes are used
as a planning oracle. Retain the sampler's approximation/tolerance limits; its
older information audit is not a claim of calibrated real-patient forecasting.
Save candidate requests, prediction starts, per-future rollout and tail costs,
selected index, sampler diagnostics and admitted/completed compute counts.

Score = expected native8-step cost + learned native cost-to-go at nonterminal
cutoffs. At actual episode end the tail is zero. The initial neural output is
exactly zero, so the frozen initial graph and flat policies are behaviorally
Native-H8 under the same planning random starts. Reuse that logical control,
not duplicate initial-policy trajectories. H8 is cheaper computationally because
it needs no neural calls; behavioral identity is not compute equivalence.

This is approximate rollout improvement: predicted first8actions follow MDL-2,
whereas the learned tail estimates the deployed receding-horizon policy. Do not
claim an exact Bellman-optimal planner or ignore this continuation approximation.

## Training and Information Matching

Train5 independent blocks x2 architectures x96complete own trajectories =960.
Each model sees24worlds from each original scenario in fixed round-robin order.
Pair graph/flat exogenous training starts within a block, not their trajectories
or observed targets. Both produce their OWN actions, costs and TD targets.
Five model-seed blocks are new; concrete allocations are pending collision
checking at freeze. Do not use old holdouts or test outcomes to change the recipe.

Use observed TD8 with gamma1, native positive full cost divided by1e9 only for
numeric units, one stop-gradient target snapshot per completed episode,
32Adam updates/episode, batch64, lr0.0001. Only current-episode transitions are
sampled. Zero initial output, no teacher labels, pretraining, reward shaping,
off-policy replay, checkpoint selection, early-success stop or hyperparameter
search. Final endpoints from ALL five blocks are evaluated after all models seal.
This tests RL training in simulation, not parameter adaptation in deployment.

Reuse the original graph feature transform and config-derived normalization for
BOTH architectures, including the same MDL-2-derived public features and time.
Graph widths[64,64] and flat[10,24] give7873and7859trainable parameters with the
current21x36tensor (0.18% gap, verified arithmetically/in regression). Widths were
chosen for parameter scale, not patient performance. Training exposure and
update counts match; architectures, weight sharing and FLOPs do not. This is
a graph-vs-global-MLP architecture contrast, not pure edge-only causality.

## Fresh Evaluation and Claims

Five blocks x4original conditions x12NEW worlds =240paired worlds. Evaluate
the4roles on each =960evaluation trajectories; zero evaluation updates.
Total main trajectories1920. No extra scientific smoke or qualification study:
necessary engineering tests use invented fixtures, and the first scheduled
training trajectory is also the real integration observation within the cap.

Report all6role pairs, all5block means, each condition, individual-world cost
and patient harms, completion/losses, transfer actions, operating-cost components,
cutoff obligations, planner counts and latency. Four conditions have equal
weight. Use2000descriptive hierarchical bootstrap draws: common block resampling
across conditions, independent world resampling within each condition/block.
Same replicate indices across conditions do not denote paired patients.

Predeclared support rule for EACH claimed contrast: mean saving >=0.5%, all5
block means positive, descriptive95%cost lower bound >0, mean additional patient
losses <=0 overall AND in every condition. Primary operational contrast is
graph/MDL-2; learning is graph/H8; architecture is graph/flat. This rule does not
guarantee outcomes, establish clinical noninferiority or adjust multiplicity.
Report all contrasts even on failure. Do not rename a favorable flat result as
proof of graph benefit. If only the operational/learning contrasts pass, narrow
the claim rather than search until the graph comparison is positive.

## Complete Single-Attempt Budget

Machine-readable scope: `proposal.json`. Pure arithmetic:
`src/rl/native_value_budget.py::expected_native_counts`.

| Quantity | Cap / exact complete-run schedule |
| --- | ---: |
| Main trajectories / native execution steps | 1920 / 99840 |
| Native prediction steps | 18251520 |
| Planning clones / candidate restores | 174720 / 2446080 |
| Planner continuation/candidate heuristic calls / direct MDL-2 calls | 15980160 /12480 |
| Shared feature transforms, including their MDL-2 feature calls | 1875840 |
| Value optimizer dispatches / training examples | 30720 / 1966080 |
| Neural forward dispatches / examples | 95040 / 3782400 |
| Evaluation optimizer dispatches | 0 |
| Shared prior construction survival evaluations | 680000 |
| Conditional posterior draws | at most500000000; separately metered |
| Total elapsed, INCLUDING IO | 72hours, not an ETA |
| Raw / archive / combined storage | 16 / 16 /32GiB |
| RSS / workers / threads / files | 4GiB /1 /4 /100000 |

The phase budgets are1h admission,42h training,22h evaluation,4h analysis/archive
and3h failure preservation. Owner caps:600s/trajectory including its fit and IO,
60s/fit,60s/model seal. Global and lower phase/owner bounds all apply. Unused
budget cannot increase sample size, updates or search. All prediction work is
charged: it is not free computation just because it occurs on clones.

The historical native planner took roughly10-20minutes/episode for a DIFFERENT
full-remaining-horizon workload. This is not a measured runtime for Native-H8.
No reliable new-run ETA exists before authorized native throughput is observed.
Use the72h ceiling as a stop condition, not a promised duration or completion.

One approval covers remaining necessary serial campaign integration, bounded
regression checks, committed source/input/runtime/seed locks, one launch,
training, sealing, fresh evaluation, independent saved-data readout and one
local archive. No redundant launch approval after that complete approval.
At failure or any exceeded bound, terminate and preserve the partial/negative
evidence. No repair-and-retry, resume, automatic follow-on, remote push,
manuscript rewrite, cloud export, collaborator message or schedule change.

## Current Implementation Boundary

The2026-10-10 exact-semantics acceleration is documented in
`acceleration-readout.md`: no redundant outer snapshot copy, batched identical
features, equivalent scalar reward checks and exact prior lookup indexing.
Keep the existing prior immutable and share one index across controllers;
lookup order, sampled particles and RNG consumption are unchanged. These reduce implementation
overhead, not study size or scientific operation caps. Local component timings
are not a measured native trajectory ETA. Original historical code is unchanged.

Delivered now: native shared feature encoder, original candidate/sampler adapter,
H8/value planner, full-length trajectory collector, dynamic graph/flat value
models, observed TD learner with strict state, integration functions and exact
budget arithmetic. No original scientific source/config has been modified.

Still required inside the proposed packet: wire these components to the existing
serial campaign admission/watchdog, fresh-stream registry, per-world writers,
all-model evaluation barrier, six-pair reader and one archive owner. Validate
the actual merged environment/runtime identities before launch. No production
entrypoint, seed allocation, freeze, authorization or launched process is claimed
by the fixture tests. These remaining operational bindings are implementation,
not a new scientific gate or another exploratory study.
