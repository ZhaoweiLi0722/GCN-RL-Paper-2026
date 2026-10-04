# Value-MPC Strong-Baseline And Representation Comparison

Status: engineering preparation authorized by Zhaowei's `继续推进` following
the four-controller recommendation. This complete numerical package has NOT
been approved. The original config remains scientific_execution_authorized=false.
No new scientific model loading, fitting, patient simulation or test access is
authorized until the exact package is approved and implementation/runtime/input/
seed locks are committed. That single approval covers preparation through final
readout; no redundant launch or per-stage approval is needed.

## Question And Fixed Method

Primary: does final GCN value-MPC improve on plain MPC in the persistent-change
condition, while reporting patient outcomes and both other conditions? Secondary:
does the graph-based representation improve on a parameter-matched global MLP?
The completed 20261003 experiment partly recovered a weak initial value model;
13.7493% updated-versus-initial is not strong-MPC superiority. The new main
comparator is plain MPC, not a deliberately weak initial learned model.

Reuse the corrected public predictor, 16 candidate sequences, three response
quantiles, eight-step planning horizon, public filter, all 31 node features at
four sites, patient support dynamics and original full cost weights. Neither
model sees latent native state, future tapes or privileged information. Uniform
allocation is a reference, not a query-budget-matched planner. The three MPC
controllers have the identical action support and model-query budget; learned
terminal scoring adds explicitly counted neural computation.

Both learned controllers use V(b)=H(b)+1,000,000*f(b), zero initial residual,
eight-step TD residual targets frozen before each cohort's updates, gamma1,
32 updates/cohort, batches64, Adam0.0003 and gradient cap5. Preserve all64 steps,
including16 settlement steps, terminal bootstrap0 and full patient liabilities.
No actor optimizer, reward change, architecture search or extra training screen.
Graph: existing width32 two-layer GCN with mean pooling and32-wide readout.
Flat: flatten the SAME four-site31-field input, hidden widths21/23, ReLU, scalar
head. Trainable parameter gap must be <=1%; exact arithmetic is recorded in the
integration readout and model metadata. No inactive padding parameters. These
are different inductive biases; the comparison does not isolate edges alone.

## One Serial Schedule

Five independent blocks, ten fresh learned models. Per block:

1. Collect24 plain-MPC initialization worlds, eight per condition. Each settled
   world's identical saved features and costs train BOTH graph and flat models
   for32 updates each. This shares data, not targets: TD bootstraps depend on
   each current model. Save both full optimizer/RNG states and initial seals
   after768 updates/model. This uses120 native worlds across all blocks, not240.
2. Restore each exact initial seal and train24 new worlds/model under its own
   value-MPC,32 updates/world, reaching1536 total updates/model. Use paired
   exogenous tapes across architectures. Endogenous states and continuation
   data may diverge; do not claim matched continuation data or event-level CRN
   after divergence. Both receive equal environment and update budgets. Total
   continuation worlds240; final seals10. No checkpoint selection or convergence
   claim: the fixed budget is not evidence of full convergence.
3. Only after ALL ten final seals exist, open60 new test worlds
   (5blocks x3conditions x4replicates), each with plain MPC, final graph value-MPC,
   final flat value-MPC and uniform allocation. Exactly240 frozen evaluations,
   zero optimizer updates, no test-driven parameter changes. Initial models are
   preserved, not additional test arms. This package does not identify deployment
   online updating versus a sufficiently trained frozen policy.

Total600 native trajectories:120 shared initialization+240 continuation+240test.
Each learned architecture gets240 training-cohort exposures and7680 value updates.
Shared initial collection costs are accounted once, transparently; the standalone
data budget per learned model is still48 cohorts. No historical model is reused.

## Exact Caps And Streams

One attempt, no retry or automatic resume. 38,400 native steps plus1,200
construction/reset calls=39,600 native operations;15,360 value optimizer steps,
zero actor;33,120 neural forwards, maximum batch64;25,920 planner decisions,
1,244,160 candidate-quantile rollouts,9,953,280 prediction epochs and3,840,000
filter transitions. Initial/final seals10 each. Debit before dispatch; no refunds,
phase transfers, extra smoke worlds or testing until every model is sealed.
The first counted initialization world is the real preflight, not an extra trial.

Phase wall-clock caps: admission600s, shared initialization/fits7200s,
continuation10800s, evaluation10800s, readout/archive1200s, failure preservation
1800s; global32400s (9h), including recording/IO/archive. Per-world120s,
per-fit60s, per-seal60s. CPU float32,4 threads,1 worker,4GiB RSS,
4GiB raw+4GiB archive=8GiB combined disk,6000files. No external compute.
Prior measured throughput suggests roughly5-7h, not a guarantee or permission
to exceed any cap. Failure consumes this attempt; preserve evidence, no repair
and rerun without a separately approved scope.

Namespace capacity-value-comparison-20261004-v1. Initial63000000,
continuation63010000, evaluation63020000 plus1000*block+100*condition+replicate.
Graph model seeds530270510/530/550/570/590; flat530270520/540/560/580/600;
each replay RNG uses model+1. Bootstrap63090001. Derived public world-purpose
streams use the existing keyed seed function. Intentional role pairing is
explicit; unique world allocations must pass a scoped local historical-seed
conflict check at freeze. Collision prevents launch, not silent substitution.
Do not use the old seen test worlds or formal holdout.

## Prespecified Readout And Decision

Independent reader reconstructs all600 complete raw costs/components and patient
identities/outcomes, checks expected files, fit inputs,32-update receipts, model
seals and paired cohort/tape identities. It computes six contrasts: graph/plain,
flat/plain, graph/flat, graph/uniform, flat/uniform and plain/uniform, for ALL
three conditions and five blocks. Report requested/committed/applied hours,
losses/deliveries, action changes, data/compute consumption and wall-clock cost.

Primary development signal for graph versus plain MPC: every persistent block
has positive mean cost savings, the descriptive two-level paired-bootstrap95%
cost interval is above0, and no condition has positive mean extra patient losses.
Use2000 bootstrap resamples of FIVE independent blocks and within-block worlds,
not the old helper hard-coded to three. Retain paired-world percent savings,
absolute costs and losses; do not mistake a pooled-ratio percent for paired mean.
Graph/flat gets the same separately labeled secondary screen. These descriptive
screens are not multiplicity-adjusted formal confirmation, clinical safety or
equivalence; disclose all negative blocks and cost/patient trade-offs even if
aggregate screens pass. No new sample-size choice after looking at outcomes.

If graph beats plain but not flat, report learned-value benefit without isolated
graph advantage. If neither learned method robustly improves on plain MPC,
report the negative comparison and stop this package, not more epochs, seeds,
reward changes or searching for a favorable condition. Positive findings still
need external validity discussion, not promises of EAAI acceptance.

This is simulation TD policy learning within value-augmented MPC, not DDPG,
full TD-MPC, deployed online adaptation or real clinical benefit. Five blocks
remain a limited development sample. Existing ring topology, approximate public
belief/forecast distribution shift and uncalibrated synthetic support labor
remain limitations. E1 field data remain missing and Stage E remains closed.

Archive once with member hashes, retain originals and all previous failures.
Only local commits are permitted. No push/PR/merge, Dropbox export, messages,
Howard approval claims or automatic follow-on experiment. The unchanged old
sources/results must not be edited to implement this new scope.
