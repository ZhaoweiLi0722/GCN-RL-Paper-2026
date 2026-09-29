# R3: fixed-window value contract and proposed data-only pilot

Status: design and synthetic regression checks only. No scientific execution.
Following the explicit 52-step endpoint question and R2 readout, Zhaowei said
"continue". Use that endpoint for this protocol draft, not as permission for
new simulation, checkpoint inference, fitting or an online campaign. Howard's
sign-off is not asserted. Stage E and all historical results remain closed.

## Why this is different from another G1 rerun

R2 shows that mean costs and exact top-1 accuracy answer different questions.
It does not establish clinical safety, true optimality or online actor gains.
The archive also evaluates an important different continuation:

- G1 constructs the configured anchor and passes it to
  `mean_rollout_metrics_after_action` in
  `evaluation/audit_ddpg_legal_action_ranker_feasibility.py`.
- `evaluation/run_gcn_residual_sweep.py:rollout_metrics_after_action` applies
  the candidate once, then calls that baseline at every later step.
- `src/models/gcn_ddpg.py:update` instead bootstraps through the target actor,
  gate and critic action projection. Historical rewards also have the M2
  absolute/relative inconsistency; a continuation difference is not its only
  issue and is not proven to have caused the old online null.

Thus an MDL-2-followup advantage is not automatically Q under the deployed
actor. Keep old diagnostics intact. A prospective value check must name the
continuation policy as well as its reward, horizon and first executed action.

## Fixed objective and information

Use unchanged original simulator cost weights and dynamics. No new scenario,
overtime lever, shaping, clipping or outcome-based scaling. The proposed
objective is the undiscounted sum of costs incurred during steps 0..51.
Reward is negative absolute step cost, with one common scale 1e-9 and gamma=1.
The scale inherits the historical units; it is not tuned for fit or performance.
One-step replay still bootstraps future value; it is not a myopic objective.

For a continuation policy mu fixed before collecting data:

    Q_mu(t,s,a) = E[-1e-9 * sum(cost_k, k=t..51) | first a, then mu]
    y_t = -1e-9 * cost_t + (1 - objective_terminal_t) * Q_mu(t+1,s',mu(s'))

The endpoint is objective-terminal after step 51: its future value is zero by
definition of this finite window, **not because pending patients/inventory are
settled**. An interrupted collection before then is truncation, retains value
and is not a completed outcome. Do not continue past native simulator `done`.
Preserve outstanding patients, queues, on-order resources and pipeline counts
at the cutoff as separate diagnostics. There is no full-lifecycle welfare claim.

Remaining time must be available identically to both future study arms. The
historical observation includes t/H and the graph transform broadcasts it;
verify the actual runtime config and checkpoint feature contract, not only the
current source default. Do not add latent state or future outcomes to a policy.

## Preserve a competent baseline

Proposed provenance: **all** F1 control GCN frozen-pretrain actors, seeds 60,61,62,
selected as the existing G1 mechanism chain, not by their outcomes. Retain the
historical actor, gate, thresholds, feature processing and executed projection.
No actor/gate fitting. The N4-N7 random common-input prototype is not a substitute:
it has a different head/layout and an engineering soft gate. Do not transplant
old weights into it or silently use a different seed, role or final checkpoint.

Before any future scientific execution require each original checkpoint,
effective config, summary and manifest, a whole-file hash, source-version lock,
strict tensor-key/shape match and actor/gate output-parity check. The historical
`load_actor` supports partial gate transfers on mismatch; this pilot must reject
that path, not regard a successful load as proof of parity. Old critic, Adam,
replay and uncertified teacher labels must not initialize a clean objective.

The former temp worktree is absent. Its relative F1 tree is absent from this
worktree, and targeted filename searches in the two documented persistent
result roots found no F1 seed60-62 pretrain files. The coauthor draft points to
an RTX host, but this location is unverified; no remote access was attempted.
Missing files are a data dependency, not evidence they are globally lost.

## Small proposed pilot, not authorized to run

This is a **value-label feasibility pilot**, not critic fitting, corrected DDPG,
an online/frozen performance comparison, or a repeat of the 156-state G1.

| Setting | Proposed fixed choice |
| --- | --- |
| Scenario | Explicitly `routing_nominal_history`, reconstructed from benchmark plan; assert built scenario matches label |
| Baselines | F1 frozen-pretrain GCN seeds 60,61,62; no favorable-seed selection |
| State collection | One fresh 52-step frozen-policy trajectory per actor, no exploration |
| Decision steps | 0,13,26,39 for every actor, yielding 12 dependent diagnostic states |
| First requests | Frozen actor request, MDL-2, specimen corrections +/-0.05 and +/-0.10; maximum six unique executed choices |
| Feasibility | Preserve constraints/projection; enumerate and record any candidate outside current gated actor reachability as diagnostic-only, never count it as actor-achievable headroom |
| Deduplication | Merge behaviorally identical executed choices, preserve requested-label aliases; do not add replacements or force six distinct choices |
| Continuation | Same frozen actor and fixed gate on its own subsequent observations for every first choice; reset any policy state per clone |
| Future draws | Eight discovery and eight independent validation RNG initializations per state; no adaptive expansion |
| Primary reference | Frozen actor's first executed action followed by that same actor |
| Secondary reference | MDL-2 first action followed by the frozen actor, explicitly NOT a full MDL-2 rollout |
| Maximum continuation records | 3 actors x 4 steps x 6 actions x 16 draws = 1,152 |
| Maximum environment transitions | 156 trajectory steps + 3 x 6 x 16 x (52+39+26+13) = 37,596 |
| Other caps | No optimizer updates, no retraining, one serial job, 3,600 s wall limit, no automatic retry or extension |

Three state-collection seeds plus 12 x 16 continuation RNG starts require at
most 195 fresh numeric seeds; common within-state/action CRNs do not increase
that seed count. Their ranges remain **unallocated**, pending a repository-wide
collision check against the formal and prior streams before authorization.
Identical RNG initialization is not proof that action-dependent simulator draws
remain event-aligned. Record this coupling limitation; do not call it identical
exogenous patient worlds without a separate event-stream check. Do not alter
the simulator RNG mechanism as an undeclared repair.

## Raw evidence and analysis fixed before collection

For every state/action/draw retain absolute cost components, step count,
clinical metrics and cutoff obligations, not just averaged labels. Record
source hashes, actual scenario, public state hash, action request/projection,
executed-event identity, actor/gate hashes, continuation ID, RNG starts, t/H,
and explicit objective termination vs collection truncation. Store chronological
single-step rewards with lineage. Assert first-step reward equals negative
component cost once; raw counterfactuals are independent one-step records, not
adjacent transitions to concatenate. Actual cloned rollout paths may have their
own verified trajectory IDs. Never insert a full return as an immediate reward.

Before seeing validation, choose an action using discovery mean cost with the
first-index tie rule, and record its discovery clinical outcomes. Analyze ALL
chosen actions on the validation block, including clinical failures; never
filter unfavorable cases after validation. Report cost differences against
the frozen action and secondary reference, clinical harms, zero-effect actions,
per-seed signs and draw-level uncertainty of conditional means. Comparisons
across actions share RNG starts; keep that pairing when computing conditional
intervals. State/step rows are dependent and not independent episode replicates.
Any interval is conditional on the sampled states/actors, not generalization
evidence. A validation argmin is an optimistic diagnostic, not the policy.

No gate authorizes training from this pilot. Eight draws and three actors are
an intentionally capped feasibility sample, not a powered benefit/safety test.
If values are noisy, actions not reachable, or clinical constraints unresolved,
report inconclusive/negative without buying more samples. Practical margins,
clinical noninferiority and a fitted-critic validation gate need a subsequent
prospective decision, never retroactive thresholds that make this pilot pass.

## Current engineering checks and stop boundary

Use existing `validated_returns` in synthetic tests to verify 52-step exact
backward returns, one scale, gamma=1, objective-terminal masking, early
truncation and rejection of independent-counterfactual concatenation. Include
a hand-constructed two-step example whose action ranking reverses when only
the continuation policy changes. This is a logical example, not PRM evidence.
No environment, historical checkpoint or optimizer is loaded by these tests.

Finish this design/test packet locally. Do not automatically execute this pilot.
Next dependency: locate the three checkpoints/configs and verify their source
identity; then commit implementation/stream locks and obtain explicit approval
for the capped data-only pilot. Any failure is preserved without automatic
repair/relaunch. Never use the formal holdout or reopen Stage E.
