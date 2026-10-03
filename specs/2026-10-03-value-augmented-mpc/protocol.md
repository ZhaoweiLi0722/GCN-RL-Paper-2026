# GCN Value-Augmented MPC: One Prospective Comparison

Status: direction and implementation approved on 2026-10-03; the complete
288-world numerical execution package has been asked once and awaits its exact
reply. No model loading, fitting or patient simulation is authorized by this
draft. After exact approval, commit implementation and source/runtime/input/seed
locks, then execute preparation-to-readout once without another launch question.

## Question And Method

Does eight-step TD value learning improve an otherwise identical public-input
MPC, and does continued simulation learning improve the same frozen initial
value model? This is simulation RL training attribution, not parameter adaptation
during deployment. The fixed-budget DDPG attempt remains closed. We do not use
its seen evaluation worlds for training, selection or independent confirmation.

Reuse the corrected recovery2 public predictor and its 16 candidate sequences,
three response quantiles and eight-step horizon. The predictor is an approximate
model, not a native-simulator clone. All MPC arms receive identical public
information, candidate support, predictor and planning budget. Total hours can
vary up to eight; this is the existing MPC action family, not the prior restricted
fixed-total actor. No historical causal comparison between action maps is claimed.

The value is H(b)+1,000,000*f_theta(b), where H is the existing terminal patient
heuristic and f is a fresh GCN residual initialized to zero. This is a residual
parameterization of ONE total cost-to-go, not a second terminal cost added to
a learned full return. For complete settled 64-step trajectories, the fixed
eight-step TD target for the residual at t is:

sum(cost[t:u])/1,000,000 + (H(b_u)/1,000,000 + f_old(b_u))*1[u<64]
- H(b_t)/1,000,000, where u=min(t+8,64).

All 32 updates after a cohort use these same pre-update targets and only that
cohort's 64 rows, sampled with replacement in batches of 64. No off-policy replay
from earlier controllers, no actor optimizer, no imitation targets, no change to
the full undiscounted cost or its patient-loss weights. Fit the residual by MSE,
Adam lr0.0003, gradient norm cap5; two32-wide GCN layers and32/1 readout. The
existing public ring graph is used, with no isolated graph-effect claim.

The SAME31-field node schema describes an observed public belief and each
forecast terminal belief: patient queues/stages/ages/survival, inferred remaining
work intervals, available and in-transit resources, committed labor, inferred
response and time remaining. The current belief uses the public filter's median
response; forecast beliefs use the selected response quantile. Neither accesses
latent native health, true remaining work or future tapes. Approximate belief
compression and forecast-to-observed distribution shift are limitations. Preserve
all16 settlement transitions; bootstrap at epoch64 is exactly zero.

## Schedule

Three independent model blocks. Each collects24 fresh initialization cohorts
(8 per condition) under plain MPC, doing32 TD updates after each. Seal the initial
value model. Continue from that exact model/optimizer/RNG state for24 fresh
cohorts under value-MPC, with weights fixed inside each cohort and32 updates
afterwards. Seal the final model. Repeat serially for all blocks before test access.

Test36 fresh worlds (3blocks x3conditions x4replicates), each with plain MPC,
initial-value MPC, updated-value MPC and fixed uniform allocation:144 evaluations.
Restore the correct sealed model per world; zero test optimizer updates. No
checkpoint selection: initial and final only. Conditions are unchanged no-change,
persistent unknown response change and fast fluctuation. The initial/final
contrast includes additional continuation experience and compute, not equal-
budget algorithm superiority. A matched-data alternative-learning ablation is
not included. Support labor remains uncalibrated synthetic; E1 is still missing.

## Bounds And Streams

One attempt:288 trajectories,18,432 native steps plus576 construction/reset
operations=19,008 native operations;4,608 value optimizer steps, zero actor steps;
11,664 neural forwards with maximum batch64;12,096 planning decisions,
580,608 candidate-quantile rollouts,4,644,864 prediction epochs and1,843,200 filter
transitions. Initial/final seals3each. Every scientific call is charged before
dispatch; reservations and failed calls are not refundable or transferable.

Phase caps: admission300s, initial learning3600s, continuation3600s, frozen
evaluation5400s, readout/archive900s, failure preservation600s. Global14,400s
includes all I/O. Each trajectory90s, each cohort-fit30s, seal60s. CPU float32,
4compute threads,1worker,4GiB RSS,2GiB raw+2GiB archive=4GiB combined disk,
2,500 files maximum. Protocol/config specify all bounds; no extra smoke world.
The first initialization world is the counted real preflight and remains data.

Seeds: initial62900000, continuation62910000, test62920000 plus1000*block+
100*condition+replicate; model seeds530260510/520/530, replay RNG seed=model+1;
bootstrap62990001. Derived
purpose streams use the existing keyed-seed utility under a fresh namespace.
Freeze a scoped local historical seed-conflict audit; collision means no launch,
not silent seed substitution. No formal holdout, old test reuse or external data.

## Readout And Stop

Primary: updated versus initial-value MPC in the persistent condition. Require
positive mean savings in every independent block and a descriptive paired
block-bootstrap95% savings interval above zero, plus no positive mean extra
patient losses in any condition. Report all costs/components, losses/deliveries,
requested/committed/applied hours, all blocks/conditions and secondary plain-MPC
and uniform contrasts. An initial-value improvement over plain MPC establishes
value learning under its initialization data, not continued-learning gain.
Report compute and additional data; do not infer clinical safety or equivalence.

Use raw64-row ledgers and patient IDs for independent reconciliation, preserve
all outcomes. Store tapes, public inputs, intermediate states, model/optimizer/RNG,
targets/update receipts and budget ledger. Archive once with member hashes.
Failure consumes the attempt; preserve evidence and do not auto-repair/retry.
No reward/model/scenario searches, test-based selection, extra epochs, holdout,
StageE reopening, remote push/PR/merge, Dropbox or external messages. No Howard
approval is represented. Manuscript changes remain prospective until results exist.
