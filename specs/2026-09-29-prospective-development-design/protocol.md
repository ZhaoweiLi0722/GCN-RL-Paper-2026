# N5: prospective development design, not a training launch

## Scope and evidence review

Zhaowei requested continuation after N4 and the recommendation to fix the small
development design before training. This packet executes only a metadata/model
shape preflight: read the hash-locked N4 schema, construct frozen CPU networks,
count components and check neural-operator isolation. No environment step,
optimizer update, performance evaluation, scientific seed allocation, historical
cache migration, remote action or formal-holdout query is authorized here.
Commit protocol/config and implementation before the recorded preflight.

N4 verified real collection/target dataflow, but graph/flat counts differed:
3,669 versus 2,409. M2 found historical replay/target inconsistencies; G1 found
feature/gate asymmetry. The solved queue fixture does not establish a remaining
RL opportunity. E1 lacks operational task, measurement and calibration evidence.
This protocol neither repairs nor overrides those facts.

## Fixed count and operator check

Use the unchanged N4 schema, not its costs. Fix graph hidden width 64; search
integer flat widths 1..256 using counts only. Require every actor/critic/gate
count and their total to differ by <=1%, with graph count as denominator.
Select minimum worst component gap, then total gap, then smaller width. Record
every candidate. No outcome-driven resizing or unused padding parameters.

Verify analytical counts against instantiated modules and that every parameter
participates in a backward graph. The latter is differentiation only, with no
optimizer step; it does not imply nonzero/useful gradients. Exclude separate
target copies consistently. These widths concern only N4 engineering dimensions,
not an unbuilt operational scenario.

For isolated message-passing attribution, physical and self-only models must
have identical architecture, weights, features, physical metadata, heads, gates
and parameter counts. Only neural adjacency changes. A matched flat model still
differs in representation/depth: equal counts do not isolate message passing.

## Proposed development design

The comparison structure is fixed for preparation, but this is **not an
execution-ready scientific preregistration**. Unknown choices stay null in the
JSON and block launch. Do not import the toy units, widths, gamma=.9 or random
soft gate from N4 into a study by default.

### Attribution and comparators

Primary: same independently pretrained graph policy, tensor-equal frozen/online
forks before change exposure. Actor, critic, targets, gate, scaling, offline data
and initial belief are equal. Only the online arm updates actor/critic and their
targets. Hold any certified pretrained gate fixed in both. Do not simultaneously
redesign the gate/teacher or weaken the frozen policy.

Both policies receive the same causal history/belief/estimator interface. Frozen
means fixed neural weights, not fixed actions or no system identification. Each
arm sees its own realized outcomes in paired exogenous worlds, not the other's
trajectory or hidden mechanism. Exploration costs count in online deployment.
Unchanged worlds are mandatory retention controls. Unknown persistent changes
must have operational justification, not be chosen because they favor RL.

Include a competent adaptive rule and online identification + MPC with equal
measurements, censoring, actions, booking constraints and query accounting.
Their tuning data and compute must be disclosed. Known-dynamics/clairvoyant
solutions are diagnostic bounds, not deployable competitors. Secondary claims
are physical-versus-self-only message passing and common-information graph-
versus-flat representation. Do not pool architectures to rescue the primary
graph online-versus-frozen contrast.

### Proposed reward and learner contract

Use negative complete step cost, gamma=1, one-step replay and a finite, fully
settled episode for the first corrected pilot. This aligns the learning target
with undiscounted total cost; one-step TD still bootstraps long-term value.
Freeze one positive scale from pretraining units for every arm. No absolute/
relative mixing or importing uncertified historical caches as trajectories.

Settle all accepted work, paid commitments, loss/expiry and closure costs.
Terminal means absorbing with zero outstanding liability; a collection cutoff
is not terminal. If a common closure controller takes over at an action-window
cutoff, its actual actions/returns define continuation and bootstrap, not the
learned actor after that cutoff. The old horizon flag does not prove closure.
If a continuing task is needed, revise this objective before data collection.

New features require a newly verified, adequately pretrained frozen baseline;
old checkpoints are not automatically compatible. Actor gradients follow the
declared deployed request; target Q is detached; critic parameters are fixed
during actor differentiation. Gate parameters stay fixed, and the treatment of
proposal-dependent gate gradients must be locked. Shared action bounds and
feasibility/latency remain operational constraints, not outcome-tuned penalties.
Optimizer, rates, schedules, replay, exploration and target sync need a lock
and exact-resume acceptance before any scientific launch.

### Budget, endpoints and stops

Exact sample count, fresh seeds/streams, update/step/wall-time caps and precision
requirements remain unallocated until units and a practical effect margin are
defined. Start with a variance/feasibility development pilot, not a claimed
powered confirmation. No best-checkpoint selection or favorable-seed extension.

Primary endpoint: complete cost_online - cost_frozen, including adaptation and
exploration. Negative is better. Set an absolute operational margin before data
collection; require uncertainty supporting improvement beyond it plus declared
clinical/service guardrails. A p-value alone is insufficient. Report native cost,
adaptation/post-adaptation cost, retention, clinical components, violations,
transition/model queries, offline training and measured online latency.

Independent pretraining seeds and exogenous change worlds are replication axes.
If worlds are crossed across seeds, account for both shared axes in uncertainty.
Do not count steps/checkpoints as independent replicates. Predetermine intervals,
missing-run handling and secondary multiplicity. No invented avoidable-gain
percentage without a defensible bound. Retain negative and failed runs; material
protocol revisions get a new version, not an overwritten unfavorable result.

## Work order and stopping boundary

1. Finish this count/operator preflight, without environment rollout.
2. Separately verify learner gradients, replay/optimizer/RNG state, frozen-arm
   invariance and interruption/resume in bounded engineering fixtures.
3. Resolve E1 inputs; implement a defensible scenario and freeze information,
   physical lever, liability closure and units.
4. Screen action leverage, safe headroom, independent label replication and
   learnability; accept the frozen/history policy and adaptive comparators.
5. Commit actual model/reward/budget/stream/analysis locks and scientific launch
   authorization. Only then execute the small development pilot.

Stop this packet at the preflight/readout. Later steps are dependencies, not
permissions inferred from model counts. Preserve Stage E. A further online null
result leaves a package-level manuscript with explicit limits, not a reason to
hide negatives or manufacture a positive scenario. No publication guarantee.
