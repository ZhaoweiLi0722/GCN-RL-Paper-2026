# Fixed-Budget Allocation: Completed, No Stable Training Gain

2026-10-03 UTC. The authorized single attempt completed normally in 1,633.21
seconds (27.22 minutes), with supervisor and child exit code 0. It is now closed;
no retry, extra training, reward tuning or new experiment follows automatically.

## Result

**The trained policy did not meet any of the three prespecified benefit criteria.**
Positive changes below are worse for the trained policy. Percentages are means
of paired-world percentages, not percentages of pooled mean costs. Each row
covers 12 matched worlds across three independently trained blocks.

| Condition | Comparator | Mean cost change | Mean extra simulated patients lost |
|---|---|---:|---:|
| No change | Same-start uniform fixed | +0.2462% | +0.7500 |
| Persistent change (primary) | Same-start uniform fixed | +1.8183% | +4.1667 |
| Fast fluctuation | Same-start uniform fixed | +1.4309% | +0.4167 |
| No change | ID-MPC | +4.2187% | +4.1667 |
| Persistent change | ID-MPC | +3.2592% | -0.1667 |
| Fast fluctuation | ID-MPC | -0.6333% | -23.0833 |

Persistent-change cost increases versus uniform by block were +3.8039%,
+1.8926%, and -0.2416%; extra losses were +2.50, +8.25, and +1.75. Thus even
the block with lower cost had more lost patients. The descriptive 95% interval
for absolute cost savings is [-1,441,883.59, +389,180.54] synthetic cost units;
the extra-loss interval is [-0.91875, +8.41875]. Both cross zero: unfavorable
sample means are not proof of clinical harm, nor evidence of safety/equivalence.

The three false criteria are: positive persistent savings in every block;
positive lower endpoint of the persistent savings interval; and nonpositive
mean extra losses in every condition. Fast-fluctuation means favor trained over
MPC, but not over the same-start uniform policy. They do not establish an RL
increment over that primary control. Only three independent training blocks
support these development observations.

## What Changed

This was actual training: 1,728 actor updates, 4,992 critic updates including
warmup, and 36 scalar multiplier updates. Greedy allocations differed from the
uniform policy at 1,726/1,728 evaluated control boundaries. Yet redistribution
did not yield stable better outcomes. Trained site requests were approximately
1.662 to 2.603 hours. Both trained and uniform committed and eventually applied
384 hours per world; total resource withdrawal cannot explain this contrast.

In the persistent condition, the main additive cost differences versus uniform
were bioreactor shortage +456,651.87, patient loss +208,333.33, expiry +120,000,
reagent purchases -77,318.99 and reagent shortage -36,043.55. These describe the
ledger, not a proven causal mechanism or proof that reward weights are wrong.

The same-start comparison tests learning within the restricted fixed-total
allocation family. MPC can use different total hours and a wider site range;
it is a performance comparator, not a matched action-map ablation. This run
does not prove a historical action-map improvement, globally optimal uniform
allocation, isolated GCN benefit, or deployment online adaptation. The support
labor model remains uncalibrated synthetic and E1 operational data remain absent.

## Completion And Evidence

- 36/36 reference worlds, 36/36 training worlds, 108/108 frozen evaluations.
- 11,520 native steps, 11,880 native operations, 6,720 optimizer calls, 21,120
  forwards, 663,552 planner epochs and 1,152,000 filter transitions.
- Three initial and three final seals; all final seals precede test access;
  zero optimizer updates during evaluation. All phase/owner/global caps passed.
- Independent saved-data calculation: 108 files, 6,912 rows, 72 paired contrasts;
  all patient counts match exactly. Largest numeric difference is 7.45e-9
  synthetic cost units from summation roundoff, within recorded tolerances.
- Current packet, 531 source files, five input locks and runtime entry hashes
  match. Existing 670-file payload hashes and archive bytes match their receipt;
  no new archive was made. No failures or stderr; no related runner remains.
- Prior 23 necessary zero-update tests and compileall are reused for unchanged
  code. This handoff made no model, optimizer or simulator calls.

Run: `results/fixed_budget_capacity_20261003`. Execution commit
`28e8987db9e07a2b3a2fc3d4b2d762ef9bca0e79`; packet
`174c1d00adaecdad79f4aac43b0aef3b7f69768fd5fb3e005a856ec4766d75d1`.

See `reports/fixed_budget_capacity_20261003_independent_readout.md` for all block
contrasts and interpretations, its companion `independent_raw_check.json` for
row-derived calculations, and `reports/fixed_budget_capacity_20261003_terminal_evidence.json`
for counts, integrity and process checks. Original comparison, inventory and
archive receipt remain unchanged. No Dropbox export, cloud-sync or Howard-access
claim is made.

## Handoff

The same `gcn-rl` automation is PAUSED, retained in the list. The one approved
attempt is consumed; all earlier attempts, Stage E and formal holdout stay closed.
Do not add another epoch, constraint weight or action-map variant by default.

The next decision is whether to close this capacity/DDPG recipe and develop a
distinct, bounded scientific proposal, rather than continue tuning on these
seen tests. No new recipe or reward change is approved here. Preserve the
paper's previously supported claims and report these negative attribution
results without presenting training completion as a performance contribution.
