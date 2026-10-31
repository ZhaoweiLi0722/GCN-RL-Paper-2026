# Common Joint-Action Recovery: Completed, Screen Failed

Verified 2026-10-30 at 13:24 UTC. The authorized recovery and both required
archives are complete. The scientific process, archive process and bound power
guard have exited. Execution completed successfully; the performance screen did
not pass. These are different conclusions.

## Completed Scope

Retained 1,433 complete training tails and completed only the seven missing tails.
The 72 training roots now have 1,440 complete tails. Three abandoned admissions
and all old charges remain. Six fits completed 1,024 updates each, sealing six
models. Assessment completed 24 new roots with four futures each, 480 candidate
tails, 288 arm-future records and all three planned comparisons. No new seeds,
old-model changes, H8 calls or extra experiments were introduced.

## Results

Differences below are left minus right. Patient and service figures are average
counts over 96 paired future outcomes, clustered within 24 roots and three blocks.
Negative cost and patient-loss differences are better; positive completions are
better. The reference is the common frozen Local32 policy, not H8.

| Comparison | Cost | Patient loss | Completed services | Pending services |
| --- | ---: | ---: | ---: | ---: |
| Joint GCN - matched non-graph | +0.0184% | +0.98 | -0.83 | -0.15 |
| Joint GCN - reference | -0.1086% | -5.69 | +5.96 | -0.27 |
| Joint non-graph - reference | -0.1270% | -6.67 | +6.79 | -0.13 |

GCN has higher cost and patient loss than matched non-graph in all three blocks
and all four conditions. Both learned selectors improve pooled outcomes against
the reference, but both worsen patient loss and completions in block 0. Pending
services also worsen in some conditions. All three fixed screens fail. The full
block, condition, future-half and paired results remain in
`results/native_gcn_common_joint_action_20261030_recovery1/payload/comparison.json`.

This tests one joint action followed by the same frozen continuation. It is not
a new full-policy result, an improvement to old Graph+H8, or evidence of clinical
safety. It provides no independent GCN advantage in this DEVELOPMENT synthetic
screen. Close this configuration without extending it into a long experiment.

## Preservation

- Original failure archive: 284,822,007 bytes, 5,230 members, all verified.
  SHA256 `c4a7f5cfcdae800579d23f5feed8dd9fcef7fcd0ddb160dd028fc7023b346629`.
- Recovery outcome archive: 101,542,963 bytes, 1,401 members, all verified.
  SHA256 `12e5550d5245258c523783499f125437273bc7d84e60d775ec10c4fa33bc914b`.
- The final read-only outcome hash and size match the existing full-member
  verification receipt. No archive was rebuilt. The original failed root remains
  unchanged. Final terminal/archive receipts are external post-archive records.

See adjacent `terminal-readout.json` for identities, counts and receipt paths.
No manuscript, remote, goal, automation or external-message changes were made.
