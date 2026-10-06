# Fixed-Benchmark Five-Family GCN+RL Selection

Prospective package, not a completed result. Zhaowei approved the complete
single package on 2026-10-06 with the literal reply recorded in approval-intent.json.
The user requested immediate advancement and one whole comparison, followed by
results-led selection. No family is preferred in advance. This package is larger
than, and does not reopen, any previous consumed experiment.

## Fixed Question And Task

Which of GCN-DDPG, GCN-TD3, GCN-SAC, GCN-PPO and observed-TD8 GCN value-MPC
achieves the best supported cost/service tradeoff within this finite budget?
Use the unchanged four-site synthetic support-hour task, 31 public belief
features, original three conditions, physical graph, full native cost, and
common MDL-2 routing/purchasing decisions. Primary baseline is always plain H8;
plain H16 is always the longer-planning compute reference. Do not compare
percentages from old full-routing studies as if they share this benchmark.

All policies see only public information. Each world has 48 decision steps and
16 settlement steps; no horizon extension. Actions satisfy four site caps of
4 hours and a shared 8-hour cap. All environment costs, patient losses, actual
hours, actions, latency and computational work are retained. Nothing here is
real-patient calibration, clinical safety or deployed online adaptation.

## Matched Finite Search

Development: five families x two learning rates (0.0001,0.0003) x three fresh
training blocks x 96 native training worlds = 2880 training trajectories.
Each block has 24 new development worlds, eight per condition. Evaluate all ten
configurations plus H8/H16, giving 864 development trajectories. Exogenous
tapes are shared within a block and stage; resulting action-dependent patient
paths are not necessarily event-level common random numbers.

Select exactly two distinct families and one learning rate per family using
development only. For each configuration calculate equally weighted condition
mean cost and extra patient losses relative to H8. Rank configurations with
mean extra losses <=0 ahead of those >0. Within the former rank by lower cost;
within the latter rank by fewer extra losses then lower cost. Ties resolve by
the declared method order and then learning-rate order. Keep each family's
best configuration, then the top two families. A selected family is a finalist,
not a demonstrated winner; even if all candidates fail H8, the least-bad two
remain eligible for the bounded independent comparison. No new recipe search.

Independent stage: five fresh blocks, two finalists, each graph and self_only,
96 worlds each =1920 further training trajectories. Graph and self_only use
identical initial trainable tensors, public features, widths, layers and seeds;
only neural adjacency differs. Physical links remain unchanged. Own trajectory
distributions may differ, which is part of this trained package ablation.
For each finalist also retain its block-specific frozen initial graph model.
After all models are sealed, evaluate 24 new worlds per block under at most
nine roles: two graph finals, two self_only finals, two graph initials, H8, H16,
and the five predeclared corresponding old conditional graph-TD models as one
historical-reference role. This is 1080 independent-stage trajectories.
Legacy models are never used to initialize or tune a new family.

## Training Recipes

All new models start from scratch, without teacher data, imitation or inherited
weights. Two message-passing layers, width32, CPU; 32 minibatch fit iterations
after each complete world, batch64. No update during evaluation or during a
training world's rollout. Training exogenous tapes and initial mean-actor
weights are paired where architecture permits, not an assertion that all
algorithms have identical parameter counts or computation.

Actor-critic rewards equal negative full native cost divided by 1e6, gamma1.
The last control transition includes all subsequent settlement costs and has
terminal=true. Off-policy replay is uniform over up to8192 rows. Each job
collects4608 control rows, so there is no eviction in this package. DDPG and
TD3 use target networks with tau0.005; TD3 has two critics, delay2 and clipped
target smoothing. SAC has two critics and fixed entropy temperature0.02 in
normalized reward units, not an additional temperature search. PPO uses its
current world's on-policy rows, GAE lambda0.95, stored latent actions and
pre-projection log probabilities; 32 minibatches with replacement. The common
deterministic feasible projection is part of the policy. SAC entropy is on
the proposal distribution, not a claimed density over many-to-one projected
actions. Exact constants and initialization are bound by the frozen source.

The value-MPC candidate is trained from scratch using observed TD8 targets
from its own complete native trajectories, holding its value fixed throughout
each world and fitting32 times afterward. Its zero residual initially recovers
H8. It has no cloned branches, forecast-tail collection or frozen-parent tail
labels. This is a NEW recipe, not a reproduction of the previous native-tail
method. Value-MPC versus actor-critic is a complete-controller comparison,
not an isolated backbone test. Random initial actors are learning controls,
not competent operational baselines.

## Final Decision And Uncertainty

No test-based retuning, checkpoint selection, extra seeds or new family follows.
The preferred family is fixed by development ranking; the second is a challenger.
Report all pairwise outcomes, per-condition and equal-condition aggregates,
every block and world harm. Paired worlds within each condition; bootstrap
training blocks jointly and worlds independently within each condition, 2000
resamples. Development has three blocks; the independent stage has five.

A practically supported cost candidate must save at least0.5% mean cost vs H8,
have positive cost savings in all five training blocks, a positive descriptive
95% bootstrap lower bound, and mean extra patient losses <=0. This research
threshold is chosen prospectively, not an established clinical/economic margin.
If both qualify, prefer a lower-cost candidate only when its head-to-head
descriptive interval excludes zero and it does not increase mean patient loss;
otherwise report no unique winner and retain the cost/service/compute Pareto set.
If neither qualifies, say so; lower mean cost alone is not a convincing win.
Patient-improving but costlier alternatives remain reported, not silently rejected
or rebranded as primary successes. Test selection has uncertainty; no multiplicity-
adjusted formal superiority, power or universal optimality is claimed.
Graph-final/self_only-final and graph-final/initial comparisons address separate
graph and learning contributions. An overall package win does not prove both.

## Maximum Single-Attempt Budget

6744 complete native trajectories;431616 native steps;445104 native operations
(construction and construction-triggered reset included); zero clones.
50429952 prediction steps and43161600 filter transitions at most. Maximum
153600 fit iterations,460800 optimizer dispatches and29491200 optimizer-example
presentations. Counter-wise upper bounds for actor/critic/value optimizers are
153600/307200/153600, subject to the lower shared460800 total; these are not
additive promises of actual use. Neural forwards capped at2013312, batch<=64.
Every training world has a400-forward/96-optimizer allowance, and every evaluation
world a48-forward allowance. Actual method-specific counts are reported; unused
allowance never buys extra updates, trials or worlds. H16 has twice the H8 model
query work. `numeric_contract` freezes phase-specific counter maxima.

Hard wall cap48hours including I/O, analysis, archive and handoff. Phase caps:
admission600s; development training43200s; development evaluation21600s;
independent training57600s; independent evaluation36000s; analysis/archive7200s;
failure preservation6600s. No phase transfer/refund. Per-job caps: H8world180s,
H16world300s, actorworld120s, fit60s, seal60s, including that job's recording.
32GiB combined disk (16raw+16archive),100000files,4GiB RSS,4threads,one worker.
Save incremental replay evidence instead of duplicating growing replay at each
world; seal complete model/optimizer/RNG/replay states at each job boundary.

Saved timing receipts suggest roughly31hours for the planner-containing
finalist path, including the historical reference. This is an extrapolation,
not a measured speed or promise for the new SAC/PPO implementation. The hard
cap is48hours; no automatic extension, repair retry or resume after failure.

## Authority And Delivery

One complete package approval covers necessary implementation, committed locks,
one supervised execution, selection, fresh retraining, sealing, evaluation and
readout; no routine per-phase permission requests. Before launch freeze source,
protocol/config, runtime, seed inventory and inherited inputs. Preserve successful,
failed and negative artifacts. Existing Dropbox authority covers additive copies
only under the designated Research Artifacts run directory, with destination
SHA256/bytes verification. Local copy, cloud sync and collaborator access are
separate states. No external messages, permission changes, push/merge, StageE,
old holdout, reward/scenario/architecture search or Howard approval claim.
