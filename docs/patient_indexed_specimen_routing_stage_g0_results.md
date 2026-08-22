# Stage G0 DDPG Actor-Projection Transfer Results

> **Scenario reconstruction correction (2026-08-18).** This post-hoc audit
> reconstructed each run from the stored pretraining reference `env`
> (`routing_nominal_history`) rather than its multiscenario online assignment.
> The rows and hashes remain immutable nominal-history diagnostics, but the
> former persistent-hotspot interpretation is superseded. Stage E formal
> evaluation and Stage F1 training/evaluation are unaffected. No corrected full
> G0 rerun has been performed; see
> `patient_indexed_specimen_routing_ddpg_problem_redefinition_review.md`.

## Decision

Stage G0 completed on the locked implementation commit `6031b21`. All 27 fixed
states retained five distinct executed legal actions, all 270 rollout rows and
1,620 transfer rows were unique and finite, and both immutable F1 training-tree
hashes matched before and after the audit.

The locked classification is `paired_critic_did_not_generalize`.
`action_aligned_ddpg_design_justified` is false, and no training or formal
confirmation is authorized.

## Primary remaining-horizon result

Material legal-action headroom was present in 17/27 states (`62.96%`). The
candidate final critic nevertheless selected the independently rolled-out best
legal action in only 3/27 states (`11.11%`), compared with 2/27 (`7.41%`) for
control. The incremental gain was one state (`3.70` percentage points), below
both the absolute `40%` accuracy gate and the `5`-point candidate-gain gate.

The candidate straight-through gradient selected the best legal action in
4/27 states (`14.81%`). Because the critic itself failed first, this gradient
result cannot justify critic-ranked deployment or an actor redesign.

## Secondary transfer result

The candidate raw actor differed from control at all 27 final states, but the
mean raw difference was only `1.35e-4`. Correction composition and projection
reduced the mean deployed continuous difference to `9.90e-6`, retaining about
`7.34%` of the raw displacement. Only 2/27 executed integer-lot actions changed;
25/27 genuine actor changes (`92.59%`) collapsed before execution.

Projection and quantization are therefore a real downstream bottleneck, but
they are not the first failed link. Making actor changes larger or bypassing
quantization before the critic ranks legal actions would amplify an unreliable
signal.

## Why the paired target did not transfer

The F1 candidate was supervised on an exact-CRN four-step behavior-versus-MDL-2
advantage. G0 found that the best four-step and remaining-episode legal actions
agreed in only 14/27 states (`51.85%`), falling to `33.33%` and `44.44%` in
hotspot clusters 2 and 3. Four-step optima were heavily concentrated on the
`+0.10` specimen correction (16/27), whereas remaining-horizon optima were much
more heterogeneous.

This horizon mismatch appears directly in the checkpoint diagnostics:

| Checkpoint | Candidate four-step top-1 | Candidate remaining top-1 |
| --- | ---: | ---: |
| Pretrain | 6/27 (22.22%) | 4/27 (14.81%) |
| Episode 25 | 12/27 (44.44%) | 5/27 (18.52%) |
| Final | 10/27 (37.04%) | 3/27 (11.11%) |

The candidate checkpoints contained more short-horizon structure by episode
25, but matched control followed nearly the same pattern, so this change cannot
be attributed to the paired term. Neither arm retained a useful long-horizon
legal-action ranking. Moreover, the paired term represented only `0.245%` to
`0.343%` of total critic loss across candidate seeds. A sparse
behavior-versus-anchor MSE target was therefore both weak relative to Bellman
training and only partially aligned with the endpoint objective.

## Scientific implication

The evidence does not support more vanilla online episodes, a larger actor
learning rate, or immediate critic-ranked action selection. The only remaining
mechanistically justified design review is critic-first:

1. supervise all legal actions at each selected state, not only the sampled
   behavior action versus MDL-2;
2. include a long-horizon target aligned with the evaluation endpoint;
3. normalize advantages within state and optimize pairwise margins so the
   auxiliary objective has controlled scale; and
4. require held-out-seed legal-action ranking before changing the actor or
   execution rule.

This can remain a DDPG-based method, but it would need to be reported as a
projected-action or legal-action-aligned GCN-DDPG extension rather than vanilla
DDPG. The Stage E manuscript conclusion remains unchanged unless a separately
frozen development gate later passes.

## Evidence

- Config SHA256:
  `bfa5aeb1a7248cf35ffbd67caaaa13bd8f259a6ecd9d9b4bdb72e3fa145920ed`
- Summary SHA256:
  `738bf04c8f5716e284a2f3006fe0a102516e83ae81dc1b87bbd37c1513f82524`
- Rollout rows SHA256:
  `8a8ef60b5de6e3dbf9952b3879a83a26dd6357bc501bc1872e5f0a8d1f6df989`
- Transfer rows SHA256:
  `f3873b8f0c7fb261453b804d37d226546d18281e65a8d43432095a3885af94b6`
- Curated directory:
  `experiments/evidence/patient_indexed_specimen_routing_ddpg_actor_projection_transfer_g0/`
