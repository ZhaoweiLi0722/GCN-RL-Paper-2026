# Experiment Plan

## Scientific Role of Routing

Patient-indexed, identity-preserving, pre-manufacturing specimen routing is part
of the admissible operating model. It is not the treatment whose benefit this
study is designed to prove. The primary comparisons therefore hold routing
enabled and ask whether graph-aware residual control improves on a
parameter-matched flat residual controller and the MDL-2 anchor.

No-routing is retained only as a deterministic mechanics control and as an
optional, separately approved supplementary ablation. The locked default runner
does not generate a no-routing teacher, train a no-routing policy, evaluate a
no-routing learned policy, or include routing-versus-no-routing interaction
claims.

Training seeds are `0, 1, 2`. Final and frozen-pretrain checkpoints use the same
100-replication holdout streams. Main routing uses one-epoch specimen transport
and immediate modeled product return. Lead-zero and return-one scenarios are
sensitivities, not alternative main analyses.

No hyperparameter may be selected from smoke, pilot holdout, or sensitivity
outcomes. Any changed scientific setting requires a new config name, output
root, and preregistration.

## Stage A: Mechanics and Headroom

Run focused tests and the deterministic regional bottleneck gate. It must show:

- one or more specific, legal, conserved patient routes;
- no identity failure and identical exogenous RNG state;
- nonzero MDL-2 routing;
- more completed patients than the paired no-routing mechanics control.

This gate validates implementation mechanics and the existence of routing
headroom only. It is not a formal routing-effect estimate and cannot support a
manuscript claim that routing itself is beneficial.

## Fresh Routing Teacher

After Stage A passes, independently generate one routing-enabled teacher cache
under the new routing output root. Neither learned architecture may import any
legacy teacher, normalization artifact, pretrain, policy checkpoint, or training
state. GCN and matched flat use the same reviewed teacher-generation protocol.

## Recovery 2 Execution Transport

Recovery 2 changes no scientific parameter. Every phase is launched through
`start_patient_indexed_specimen_routing_phase.ps1`, which starts an independent
background PowerShell process, redirects stdout and stderr to distinct new
timestamped launcher logs, writes a one-time phase claim, and immediately
returns the detached PID. A thin detached wrapper writes the final process exit
code to a new status JSON. The long-running process therefore does not inherit
the Codex foreground command output pipe.

The sequence remains Validate, Teachers, Smoke, Pilot, Evaluate. A phase may be
claimed only once, its predecessor must pass, and any nonzero exit stops the
campaign without retry. Read-only monitoring may inspect process trees, GPU
state, logs, checkpoints, hashes, and output metadata; it may not alter the
process or any artifact.

Recovery 2 later stopped during the routing state probe with a Python
`AttributeError` after 325 states. It produced no teacher cache, and no Smoke,
Pilot, or Evaluate stage started. That root is immutable partial evidence.

## Recovery 3 Split Execution

Recovery 3 moves all CPU-dominant teacher work to macOS and reserves the RTX
4090 PC for CUDA work. This changes execution transport and artifact paths, not
the scientific configuration.

The Mac sequence is strictly serial at the phase level:

1. run the state probe as independent rollout shards;
2. merge shards in canonical global rollout/step order;
3. run the routing teacher as independent replication shards;
4. merge teacher rows and caches in canonical CRN order;
5. require the preregistered headroom decision to advance;
6. create one frozen bundle with manifest and SHA-256 sidecar.

Individual shards may run concurrently because their global seeds, rollout or
replication indices, and look-ahead decision offsets are fixed before launch.
Each Mac phase has a one-time claim, per-process stdout/stderr/status files, and
a terminal phase status. A failed phase is not retried in the same namespace.

The PC sequence is `Validate`, `ImportTeacher`, `Smoke`, `Pilot`, `Evaluate`.
`ImportTeacher` verifies the exact commit, config hash, source hashes, bundle
hash, canonical artifact set, finite values, complete rows, and gate decision
before atomically creating the routing teacher directory. The PC never runs a
formal teacher phase in Recovery 3.

## Stage B: Five-Episode Smoke

Run routing-enabled GCN and matched flat in isolated processes and distinct
algorithm output directories, both at seed 0. Required checks are:

- exactly five episode rows per run;
- finite cost and loss diagnostics;
- positive online update count after replay warmup;
- finite, nonzero actor drift from frozen pretrain;
- legal, nonzero routing;
- expected actor and atomic full-state checkpoints;
- no identity, CRN, NaN/Inf, OOM, CUDA fallback, or duplicate-process anomaly.

Smoke results cannot be used to tune the pilot. Any anomaly stops escalation.

## Stage C: Matched Routing Pilot

Pilot execution requires explicit `-ApprovePilot`. Run GCN seeds 0/1/2 followed
by matched-flat seeds 0/1/2, with every algorithm/seed in a new process. Each of
the six learned runs has 100 online episodes and full atomic training-state plus
actor checkpoints every five completed episodes. A partial run is resumed only
by a separate, explicit reviewed command; the locked runner neither repairs nor
auto-resumes it.

Only after all six learned runs pass output, update, finite-loss, drift,
parameter-gap, and nonzero-route checks may evaluation begin.

## Evaluation and Attribution

Evaluate four primary routing scenarios with 100 paired CRN replications for
each training seed. Report pooled and per-scenario paired hierarchical bootstrap
intervals for:

1. routing-enabled GCN versus routing-enabled matched flat;
2. routing-enabled GCN versus routing-enabled MDL-2;
3. final versus frozen pretrain for GCN;
4. final versus frozen pretrain for matched flat.

The MDL-2 routing anchor is evaluated, not trained. Lead-zero and return-one
sensitivities reuse the trained routing policies and the locked holdout seed.

Metrics include total cost, completion service, patients lost, manufacturing
ineligibility, waiting/transit expiry, turnaround time, route count, distance,
time, cost, blocked requests, residual use, updates, finite losses, and actor
drift. Route activity is a mechanism diagnostic, not evidence that routing is a
novel contribution.

The holdout is never used for retraining or deployment selection. If GCN
improves on MDL-2 but not on matched flat, the permitted summary is:

> Under patient-indexed specimen routing, GCN residual control improves on
> MDL-2, but an advantage over matched flat residual control is not established.

Expansion beyond this pilot is prohibited unless all identity, conservation,
CRN, nonzero-routing, checkpoint, CUDA, and finite-metric gates pass. A formal
no-routing campaign requires a separate scientific justification and explicit
approval; prior no-routing Recovery outputs remain immutable historical
evidence and are not retrained by default.
