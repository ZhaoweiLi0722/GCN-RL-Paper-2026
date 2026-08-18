# Stage F1 Paired Online DDPG Attribution Results

## Locked question

Stage F1 asked whether exact within-state four-step counterfactual-advantage
supervision of the online DDPG critic could turn the already verified legal
specimen-action coverage into a favorable final GCN policy. Control and
candidate were tensor-identical at episode 0, shared the same structured
exploration, and differed only in
`online_paired_advantage_critic.enabled`.

This was development evidence. It did not reuse the formal holdout and could
not launch formal confirmation automatically.

## Execution integrity

- Recovery 1 completed 12/12 serial training jobs, 1,200 online episodes, 240
  five-episode policy checkpoints, and 12 full states on MPS with CPU fallback
  disabled.
- Recovery 2 reused all training artifacts read-only and completed the two
  prespecified 36-run checkpoint curves: 3,600 learned and 3,600 MDL-2 anchor
  rows in total.
- Routing and online updates were nonzero, persisted metrics were finite, exact
  cloned first-step rewards reproduced, and no formal stream was launched.
- Recovery 2 status SHA256:
  `0097b4c84f9f9ab2403eccd670543b640109657eb03a86764d423c5a0496d380`.
- Comparison SHA256:
  `8f4967115c584900969a7dba93e29da5ab70dfc65caa27d81423805be85c010a`.
- Immutable control/candidate training-tree SHA256 values are respectively
  `eb276c4402d766124080f1eddbfa1c365ff5a32d8baedb0204ff39e23c5041e8`
  and
  `af861462f59e0917a7de4ad3311e02ccbf244e8ccbf6587ecfd0dac4e231cbb3`.

## Prospective result

For the primary GCN algorithm, candidate final minus its frozen pretrain had a
mean total-cost difference of `-21,297.91` (`-0.001067%`; lower is better),
with paired 95% CI `[-715,370.98, +527,374.97]`. Only seed 61 was favorable;
the prespecified requirement was at least two favorable seeds plus an interval
wholly below zero. Clinical noninferiority passed, but the primary online-gain
gate failed.

Candidate final minus matched control final was `+4,080.57` (`+0.000205%`),
with paired 95% CI `[-95,056.15, +95,036.33]`. There were 131 exact total-cost
ties among 150 pairs. The candidate therefore also failed the prespecified
no-regression condition versus control.

The locked classification is
`close_online_ddpg_attribution_extension`. Stage F1 does not establish a
performance contribution from online DDPG updates and does not authorize
formal confirmation.

## Mechanistic reading

The paired target was not completely ignored. Across GCN candidate seeds, its
persisted sign accuracy ranged from `0.709` to `0.786`. However, the paired
loss contributed only `0.245%` to `0.343%` of the total critic loss. Candidate
and control remained nearly indistinguishable in pretrain-reference actor MSE
and actor-parameter drift.

These diagnostics support a narrower conclusion than "DDPG cannot work": the
additional critic signal was sampled and partly learned, but it did not produce
materially different deployed behavior. The unresolved transfer path is:

1. critic ranking on the actual legal integer-lot actions;
2. straight-through gradient agreement with that ranking; and
3. survival of actor changes through correction gating, projection, and
   quantization.

Stage G0 audits that path read-only before any further algorithm change is
considered.

