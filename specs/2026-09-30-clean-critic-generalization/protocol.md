# R6: clean frozen-continuation critic generalization

## Authority and question

On 2026-09-30 Zhaowei explicitly approved the bounded question: four training
and two independent test trajectories per frozen model (18 total), three new
critics, at most 1,000 updates each, 113,256 simulator steps and two hours,
single attempt. Actor, gate, reward, scenario and formal holdout stay unchanged.
No Howard sign-off is inferred. R3 conditional headroom is motivation, not
training data or evidence of online improvement. Stage E stays closed.

Can a fresh critic rank map-certified requests in unseen trajectories under
one fixed continuation policy? This is supervised Monte Carlo value learning,
not online DDPG, a full-episode learned controller, or a clinical safety trial.

## Immutable inputs and prospective split

Use R4 replacement GCN actors/gates 60/61/62 through the exact R5 input lock.
Historical F1 weights were not recovered; these baselines' full-episode
competence relative to MDL-2 remains unmeasured. Retain all old results/source.
Use unchanged `routing_nominal_history`, horizon52, observations561/actions80,
and the complete per-policy effective config. No old critic/replay is loaded.

For each policy, collect four training and two test frozen trajectories.
Take exactly t=0,13,26,39, including tied/inactive states. No replacement or
selection by outcome. Train/test parent trajectories are disjoint globally,
even across policies; different future draws on the same parent are not a
generalization split. No pooling of different continuation policies in one
unconditioned critic. This yields48 training and24 test states.

Allocate fresh NumPy streams from the first96 SHA256 bits of the literal
config namespace, shifted16 bits; enumerate actor, role(train then test),
trajectory, parent then step/draw (eight perstate). All594 starts distinct.
Scan committed prior source/config/docs for namespace/init-seed collisions;
external unrecorded streams remain unknown. Exact starts enter the audit.

Requests: frozen, MDL-2-first, specimen -0.05,+0.05,-0.10,+0.10. Retain all six
labels, including requests not certified by the existing inverse map witness.
Critic fitting/selection uses only certified support; certification uses public
state/action mapping, never future costs or realized execution aliases.
Certification does not prove neural learnability or global reachability.

## Labels and fit, fixed before collection

Each state/request has eight paired future RNG starts, then the frozen policy
continues through t52. Reuse only exact post-state/RNG/cost/event equivalent
tails through the unchanged R3 collector. Keep first-step trace for every row.
Target A_mu(s,a) = -1e-9 * mean(C_remaining(s,a;mu)-C_remaining(s,mu(s);mu)).
Absolute incurred-cost components, finite endpoint and all clinical outcomes
are unchanged. Gamma1, no terminal salvage, no TD target, no teacher labels,
no old replay or four-step reward, no reward tuning. Obligations retained.

One fresh existing GCN critic architecture per policy: inherit graph feature
construction, action quantization and layer shapes from effective config;
reset every Linear parameter under the declared independent initialization
seed. No architecture search. Predictions are Q(s,a)-Q(s,mu(s)), exactly zero
for frozen. Model input is existing public observation and request only; no
trajectory ID, seed, hidden snapshot, label or future execution identity.

Train full-batch for exactly1000 Adam updates, lr0.0003, gradient norm cap5.
MSE has equal state weight and equal certified-action weight within each state.
Scale labels by training-only equal-state RMS, floored at1e-9; multiply outputs
by the same saved scale for reporting. This rescales optimization units, not
reward priorities/ranking. No additional feature normalization, early stopping,
best-epoch selection, calibration, hyperparameter retry or clinical filtering.
Save fresh initial and final critic, separate Adam and Python/NumPy/CPU/MPS RNG
states. Actor/gate equality and empty legacy replay/optimizers checked afterward.
Do not promise bitwise GPU recovery or describe supervised steps as DDPG updates.

## Sealed test and comparators

For each policy: training states/labels -> fixed fitting -> test parent states
-> sealed predictions and all comparator choices -> test counterfactual labels.
Checkpoint, training-data digest and public-test-input digest enter the seal.
No further optimizer step or baseline selection after opening test labels.
Per-policy serial processing cannot pool labels across policies.

Comparators: frozen; MDL-2-first with public unsupported fallback to frozen;
best constant request family selected on training means only, using the same
public fallback; critic argmax within certified support, first-index ties.
The old `leave_one_seed_out_ranking` constant selector is not reused: its
global label-based constant choice is unsuitable for this prospective split.
No retrospective clinical veto can turn a harmful selection into a benefit.

Report train fit and all24 test states, pairwise accuracy (exact label ties
excluded; predicted ties are not correct), support counts, paired costs and
clinical deltas vs all three comparators, and optimistic sampled-oracle regret
as a diagnostic, not a deployable baseline. Aggregate state/draw results to
each test trajectory first and report all six trajectories and each policy.
Dependent states/draws are not independent sample-size multipliers. Conditional
eight-draw intervals are descriptive; six trajectories are not powered evidence.

## Decision and boundaries

Only a provisional follow-up discussion is warranted if cost beats frozen and
the training-chosen constant on all six test trajectories, ranking accuracy
exceeds0.5 for every policy with comparable pairs, and no test trajectory has
an adverse mean direction in losses, completions, service level or manufacturing
loss rate vs frozen. This demanding triage rule is not statistical significance
or clinical noninferiority. Report each failed component and all null/adverse
rows. Regardless of result, **no automatic actor training** or actor proposal
changes. New data, objective changes or expansion need separate approval.

## Execution and preservation

Commit source/config/protocol first; relevant unit/integration tests use only
synthetic numeric inputs (no new patient episodes) and compileall must pass.
The unchanged R3 collector already passed exact cloned-step engineering checks.
One exclusive output root, no duplicate process. MPS with fallback0 required.
Caps: 18*52 parent steps +18*6*8*(52+39+26+13)=113256 actual steps;
3456 logical outcomes;7200 seconds including fitting. Aliasing may save steps.
Missing states/nonfinite values/lock mismatch/timeout terminate the attempt;
preserve failure, no automatic repair-and-retry. Save exact code archive,
runtime, locks, effective config, raw trajectories, snapshots, paired traces,
predictions, checkpoints and terminal status. Independently verify raw costs,
selection/split chronology, digests and unchanged actors. Archive all members
and verify versioned Dropbox local copy; cloud sync/access must be separate.
No remote Git action, messages, new scenario, holdout, or reopening Stage E.
