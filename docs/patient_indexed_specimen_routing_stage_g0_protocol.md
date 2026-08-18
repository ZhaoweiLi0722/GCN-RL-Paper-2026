# Stage G0 DDPG Actor-Projection Transfer Audit

## Purpose

Stage G0 is a read-only post-F1 mechanism audit. It asks where a learned DDPG
signal is lost between the critic and the integer patient-lot action executed
by the environment. It does not update a model, select a checkpoint, tune a
threshold, reuse the formal holdout, or authorize another training campaign.

The audit distinguishes three failure points:

1. the paired critic does not rank independently valued legal actions;
2. the critic ranks them, but its local straight-through gradient points to a
   different action; or
3. the actor moves continuously, but projection and quantization erase the
   change before execution.

## Immutable inputs

- F1 control and candidate GCN checkpoints at pretrain, episode 25, and final;
- training seeds 60-62, mapped prospectively to persistent hotspot clusters
  1-3;
- the completed F1 comparison and Recovery 2 status; and
- both 151-file F1 training trees, verified before and after the audit.

The F1 control and candidate pretrain actors must produce identical raw and
executed actions. Any mismatch invalidates the audit.

## Fixed diagnostic design

- Use the frozen control-pretrain policy to generate one deterministic
  trajectory per assigned scenario.
- Inspect steps 0, 6, 12, 18, 25, 31, 37, 43, and 51: 27 fixed states total.
- At every state, compare MDL-2 with specimen corrections `+/-0.05` and
  `+/-0.10`. Deduplicate only actions that execute identically after projection.
- Estimate each first action with identical CRNs and MDL-2 continuation at a
  four-step and remaining-episode horizon, using three replications.
- Use fresh live-trajectory base seed 96900000 and rollout base seed 98000000.
  The resulting 165 diagnostic streams are unique and disjoint from all prior
  execution, development, bootstrap, and formal streams.
- Evaluate every checkpoint critic, its straight-through gradient at the
  deployed policy action, the raw actor output, the projected policy action,
  and the final executed-action identifier on exactly the same states.

The full audit produces 270 independently valued legal-action rows and 1,620
checkpoint transfer rows. CPU is intentional because no training occurs and
the purpose is deterministic checkpoint diagnosis.

## Prespecified metrics

For the remaining-episode horizon, report:

- fraction of states with at least `$1,000,000` legal-action headroom;
- critic top-1 accuracy against the independently rolled-out best legal action;
- straight-through gradient top-1 accuracy;
- deployed policy top-1 accuracy and legal-candidate match rate;
- candidate-minus-control critic accuracy;
- control/candidate raw actor and projected-policy distances;
- fraction of states whose executed actions differ; and
- conditional quantization-collapse fraction among states where the candidate
  raw actor actually differs from control.

Episode 25 and the four-step horizon are trajectory diagnostics, not alternate
selection criteria.

## Hierarchical decision gate

1. If material headroom is below `0.10`, classify
   `insufficient_legal_action_headroom` and close this extension.
2. Otherwise, candidate final critic accuracy must be at least `0.40` and at
   least `0.05` above control. Failure classifies
   `paired_critic_did_not_generalize`; legal-action normalization or margin
   ranking would need a new design review before any actor change.
3. If the critic passes but exceeds gradient accuracy by at least `0.10`,
   classify `straight_through_gradient_bottleneck`. This alone justifies design
   review of critic-ranked legal-action selection.
4. If critic and gradient transfer pass, but executed control/candidate actions
   differ in no more than `0.20` of states while at least `0.50` of genuine raw
   actor changes collapse, classify `actor_projection_transfer_bottleneck`.
   This alone justifies design review of a quantization-aware,
   advantage-weighted actor update.
5. Every other pattern is `actor_transfer_inconclusive`.

No classification authorizes training automatically. A passing mechanism gate
would support one separately frozen action-aligned DDPG protocol. A failed or
inconclusive gate ends further online-attribution development and leaves the
Stage E manuscript conclusion unchanged.

## Implementation

- Audit module:
  `evaluation/audit_ddpg_actor_projection_transfer.py`
- Locked config:
  `experiments/configs/patient_indexed_specimen_routing_ddpg_actor_projection_transfer_g0.json`
- Focused tests:
  `tests/test_ddpg_actor_projection_transfer_g0.py`

