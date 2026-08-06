# Experiment Plan

## Fixed Factorial Design

The learned pilot has four treatment cells:

1. GCN plus routing;
2. matched flat plus routing;
3. GCN plus no routing;
4. matched flat plus no routing.

Routing and no-routing MDL-2 are evaluated as CRN anchors. Training seeds are
`0, 1, 2`; final and frozen-pretrain checkpoints use the same 100-replication
holdout streams. Main routing uses one-epoch specimen transport and immediate
modeled product return. Lead-zero and return-one scenarios are sensitivities,
not alternative main analyses.

No hyperparameter may be selected from smoke, pilot holdout, or sensitivity
outcomes. Any changed scientific setting requires a new config name, output
root, and preregistration.

## Stage A: Mechanics and Headroom

Run focused tests and the deterministic regional bottleneck gate. It must show:

- one or more specific, legal, conserved patient routes;
- no identity failure and identical exogenous RNG state;
- nonzero MDL-2 routing;
- more completed patients than the paired no-routing mechanics control.

This gate validates mechanism and headroom only. It is not a performance result.

## Fresh Teachers

After Stage A passes, independently generate routing and no-routing teacher
caches under the new routing output root. Neither arm may import any legacy
teacher, normalization artifact, pretrain, policy checkpoint, or training state.

## Stage B: Five-Episode Smoke

Run GCN and matched flat in isolated processes and distinct algorithm output
directories for each arm. Required checks are:

- exactly five episode rows per run;
- finite cost and loss diagnostics;
- positive online update count after replay warmup;
- finite, nonzero actor drift from frozen pretrain;
- legal, nonzero routing in the routing arm and zero routes in control;
- expected actor and atomic full-state checkpoints;
- no identity, CRN, NaN/Inf, OOM, CUDA fallback, or duplicate-process anomaly.

Smoke results cannot be used to tune the pilot. Any anomaly stops escalation.

## Stage C: Matched Pilot

Pilot execution requires explicit `-ApprovePilot`. Run each algorithm/seed in a
new process, preserving the locked order in the runner. Each run has 100 online
episodes and full atomic training-state plus actor checkpoints every five
completed episodes. A partial run is resumed only by a separate, explicit
reviewed command; the locked runner neither repairs nor auto-resumes it.

Only after all twelve learned runs pass output, update, finite-loss, drift,
parameter-gap, and route-arm checks may evaluation begin.

## Evaluation and Attribution

Report pooled and per scenario with paired hierarchical bootstrap intervals:

1. routing GCN versus routing flat;
2. routing GCN versus routing MDL-2;
3. GCN, flat, and MDL-2 routing versus their no-routing counterparts;
4. `(GCN - flat)_routing - (GCN - flat)_no-routing`;
5. final versus frozen pretrain in each learned treatment cell.

Metrics include total cost, completion service, patients lost, manufacturing
ineligibility, waiting/transit expiry, turnaround time, route count/distance/time/
cost, blocked requests, residual use, updates, finite losses, and actor drift.

The holdout is never used for retraining or deployment selection. Sensitivity
evaluations reuse the trained routing policies and the main holdout seed.

If routing is beneficial but the graph interaction is not established, the
only permitted summary is:

> Specimen routing is beneficial, but graph-specific DRL advantage is not established.

Expansion beyond this pilot is prohibited unless all identity, conservation,
CRN, nonzero-routing, checkpoint, CUDA, and finite-metric gates pass.
