# Matched GCN / Self Attribution: One Local Attempt

## Direct Authority And Question

On 2026-10-15 the user accepted the matched GCN/Self design and directly requested
starting this experiment: 嗯，可以，直接开始实验吧。就是我想说的意思是什么，就是我们现在已经证明我们在成本上，我们这一轮单独IR是已经是有增益的，但是现在这个来自GCN的这个贡献，反而就是不明显。但是之前的结论是GCN的作用一直都是很明显的，所以我是在confuse这一点，你明白吗？然后现在只需要把这个问题解决了就好。

This accepts equal shared H8 and preservation of matched Flat as auxiliary
evidence. It supersedes the preparation-only state, not any closed historical
attempt. Question: does explicit neural neighbor aggregation add useful cost
and patient performance within the current planning-assisted RL method?
Do not guarantee a positive GCN result or describe the prior cost gain as proof
of patient safety. This package is additive, distinct from Original B, Revised B,
Family, Candidate-Advantage v1, Localv2 and the completed twelve-fit diagnostic.

## Minimum Comparison And Invariants

Reuse all six sealed Localv2 Graph/Flat final models. Train only three Self
models, one per existing block, on the exact corresponding original 96 training
worlds and full future/planning streams. Each model has the same three rounds,
1024 updates per round, 64 pairs per update, Adam1e-4, 21x41 public/candidate/H8
features, Graph64/64 backbone, raw projection, flattened readout and10881
parameters. Self changes only neural message adjacency to identity, never the
physical edges, transfers or planner. Actual initialization must match each
Graph zero-update checkpoint's trainable tensors, optimizer and private RNG.
Do not load a Graph trained checkpoint into Self or silently use seed0.

Training totals288 new main trajectories,3456 branch tails maximum,1728 paired
returns maximum,864 selected decision states,9216 optimizer updates,
589824 optimizer pair examples,3 new final models. All six imported finals
remain frozen. Nine endpoints must be bound before evaluation.

Evaluate72 NEW paired development worlds (3blocks x4conditions x6replicates)
across Graph,Self,Flat,structuredH8,MDL2:360 complete zero-update trajectories.
No old evaluation world or checkpoint selection. The same fresh world panel
supplies every reported contrast, including Graph/Flat; never splice prior
test results. The original20sites,52steps,all three transfers,replenishment,
patient identity,dynamics,physical costs,rewards and fixed target are unchanged.
There is no H16 arm, new prior, extra model/algorithm, auxiliary objective,
hyperparameter search or blanket Graph/Flat retraining.

This is a matched Self ablation with reused trained comparators, not three new
independent training samples. Post-update policies and data can diverge, so it
is not a same-on-policy-dataset contrast. Self still has global readout/public
inputs and the shared physical planner; it is not decentralized or topology-blind.

## Fixed Limits

Retain the previous Localv2 resource envelope without expansion:
96h including all IO,raw32GiB/archive32GiB/combined64GiB,100000files,
aggregateRSS4GiB,max6CPUworkers each1thread. Only3 independent training chains;
up to6 evaluation workers. Existing phase limits: admission1h,training60h,
evaluation22h,analysisarchive6h,failurepreservation7h. Individual main training
world with all native tails and IO3600s,branch/evaluation/fit1200s,seal120s.

New-operation ceilings are exactly in experiments/configs/native_gcn_self_20261015.json:
127656 native steps,26982144 prediction steps,240912 planning clones,
3372768 restores,23850288 heuristic actions,3744 directMDL2 actions,
120456 candidate calls,125928 advantage forward calls/2813616examples,
124344 feature transforms,648 main environments,3456 branch environments,
245016 total environment constructions including all clones.
Prior generation,actor/temperature/evaluation optimizers are0;
posterior draws<=1800000000. Initial/round/final checkpoints3/9/3.
Dynamic candidates and skipped one-candidate states can use fewer variable
operations; actual and admitted counters remain distinct, no refunds.

Fresh seed identities are fixed once:72 x(2native/planning +104prediction)
plusbootstrap/global=7634. Full inherited training and corresponding Graph model
seed identities are separately labeled reuse and not subjected to false fresh-
allocation claims. Only fresh identities must be disjoint from historical seed
files and each other. No silent collision replacement, reseeding or extra draw.
The new manifest includes the complete inherited prediction tables and source
packet digest. No old allocation is refunded.

AC power and verified process-bound caffeinate idle/system sleep assertions are
required. They cannot prevent manual/lid shutdown. Existing jobs are untouched.
Single new result root results/native_gcn_self_20261015; no retry/resume, automatic
repair, extra training/evaluation, model promotion or automatic follow-on.
Any failure preserves receipts and the one local archive, never overwrites old
data or reopens a consumed attempt.

## Prospective Reporting

Primary:Graph/Self message passing,Graph/H8 learned correction,Graph/MDL2
operational performance. Auxiliary:Graph/Flat practical architecture.
All10paired contrasts and all blocks,conditions,world harms are retained,
irrespective of sign. A lower cost with increased patient loss is not unconditional
success. Three-block descriptive intervals have limited uncertainty resolution;
no corrected significance, clinical noninferiority/safety or publication guarantee.
No manuscript/cloud/external messages,remoteGit or automation changes.

The existing runner performs raw accounting, all comparisons and a unique local
archive automatically. Monitoring never repeats scientific calculations.
Preparation uses only changed-path construction/config/mock checks and full
compileall; no numerical toy/smoke or extra audit campaign. After committed locks
and actual host admission, launch once without another launch-only approval.
Estimated running time10-13hours is based on prior completed trajectories, not
a promise or the96hour cap. Refresh only from actual complete worker boundaries.
