# Time-baseline comparison: diagnostic improvement, no greedy-policy gain

## Decision

**Close the tested baseline route.** The complete 216-episode evaluation found zero incremental raw cost or patient-outcome benefit from time-baseline PPO versus own frozen, current PPO, BC-CONTINUE or R4. All four comparisons have identical requested and executed actions on all 1,872 paired test steps. The prespecified >=1% equal-block cost reduction with favorable cost direction in all three blocks was not met: each block's primary effect is exactly zero. No extra baseline, seed, epoch, fit gate or retry is justified by this attempt. The persisted analysis classifies this as `limited_negative_or_inconclusive`, not an engineering failure. [Evaluation][evaluation]

The intervention improved scalar baseline/advantage diagnostics but did not improve the evaluated greedy controller. The cost advantage versus full MDL-2 is inherited R4-like behavior shared with frozen and BC, not an RL increment or an isolated GCN gain. This result does not establish that the reward is wrong, that useful action headroom is absent, or that PPO cannot learn under other justified designs.

## Complete comparison

Six controllers, three independently trained blocks (60/61/62), 12 fresh paired worlds per block, 52 steps per episode. Differences below are left minus right; negative cost is favorable. Costs are unscaled simulator cost units. Intervals are the persisted 10,000-draw hierarchical paired block/world descriptive 95% bootstrap intervals, not confirmatory inference. No historical tests were pooled and no formal holdout was used. [Protocol][protocol] [Evaluation][evaluation]

| Contrast | Mean raw cost difference | Equal-block cost change, % [95% interval] | Changed requests / 1,872 | Changed executions / 1,872 |
| --- | ---: | ---: | ---: | ---: |
| Time baseline - own frozen (primary) | 0 | 0 [0, 0] | 0 | 0 |
| Time baseline - current PPO (estimator effect) | 0 | 0 [0, 0] | 0 | 0 |
| Time baseline - BC-CONTINUE (RL attribution) | 0 | 0 [0, 0] | 0 | 0 |
| Time baseline - R4 (context) | 0 | 0 [0, 0] | 0 | 0 |
| Time baseline - full MDL-2 (context) | -14,867,150.12 | -0.564443 [-0.814411, -0.286772] | 1,664 | 1,653 |
| Own frozen - R4 (inheritance check) | 0 | 0 [0, 0] | 0 | 0 |

For the five null contrasts, cost, completions, losses, expiry losses, waiting patient-steps and terminal-active differences are zero in **every** paired world, not merely canceling in the mean. Their per-block cost effects are [0, 0, 0]; no adverse patient block direction is recorded. A degenerate [0, 0] interval describes these identical observed traces, not population equivalence or clinical noninferiority. Runtime measurements are not identical and are not patient outcomes. [Evaluation][evaluation]

### Inherited cost/patient trade-off versus full MDL-2

| Block | Raw cost difference | Cost change % | Completions | Losses | Expiry losses | Waiting patient-steps | Terminal active | Changed requests / 624 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | -14,758,942.23 | -0.558556 | +10.500 | -23.417 | -12.500 | -4.833 | +12.917 | 553 |
| 61 | -9,211,740.47 | -0.346685 | -0.667 | -20.167 | -11.000 | +55.167 | +20.833 | 607 |
| 62 | -20,630,767.66 | -0.788088 | +12.667 | -26.667 | -15.250 | -83.250 | +14.000 | 504 |
| Equal-block mean | -14,867,150.12 | -0.564443 | +7.500 | -23.417 | -12.917 | -10.972 | +15.917 | 1,664 total |

Adverse directions must remain visible: terminal active rises in all three blocks; block 61 also has fewer completions and more waiting patient-steps (and lower completion service level). There is no adverse mean loss or expiry-loss direction in any block. Across blocks, the terminal-active difference has interval [11.278, 20.917], completion difference [-1.778, 15.611], and waiting patient-step difference [-90.778, 78.918]. Thus favorable mean cost and losses are not clinical superiority. [Evaluation][evaluation]

The five R4-like controllers each average 2,624,218,461.72 cost, 2,929.194 completions, 2,311.917 losses and 708.694 terminal-active patients per test episode; full MDL-2 averages 2,639,085,611.84, 2,921.694, 2,335.333 and 692.778 respectively. Waiting is summed end-of-epoch queue occupancy, not completed-patient mean waiting time. Expiry losses are a subset of patient losses, not an additional disjoint population. Terminal-active patients are unresolved obligations, not deaths or free benefit. Action differences against MDL-2 compare closed-loop traces, not same-state counterfactuals. [All episodes][all-episodes] [Reward memo][reward]

## Estimator diagnostics

The separate raw-target report verifies raw-reward/collected-value binding, unchanged critic targets and whole-trajectory exclusion across 48 PPO updates (24 per arm, 208 rows/update). The episode-analysis field `target_receipt_diagnostics: not_measured` refers only to that reader's scope; the separate target report supplies the following diagnostics. Values are arithmetic means of eight saved update statistics per block, in squared once-scaled reward units, before advantage normalization. [Targets][targets]

| Block | Current PPO applied-baseline MSE | Time-baseline applied MSE | Current PPO raw advantage variance | Time-baseline raw advantage variance |
| --- | ---: | ---: | ---: | ---: |
| 60 | 1.779948 | 0.043429 | 0.690162 | 0.043429 |
| 61 | 1.808024 | 0.070205 | 0.705860 | 0.070205 |
| 62 | 1.892433 | 0.044488 | 0.717575 | 0.044488 |
| Mean of 24 updates | 1.826802 | 0.052707 | 0.704533 | 0.052707 |

The treatment's collected-value MSE remains 1.817627, distinct from its applied time-baseline MSE of 0.052707: **this is an estimator change, not a repaired critic**. Mean time-mean-advantage variance falls from 0.673125 to approximately 1.63e-33. Zero cross-episode mean at each time is an algebraic property of this leave-one-episode-out construction, not learned predictive accuracy. These arm summaries use their own training trajectories, not identical counterfactual returns. Gradient variance was not measured; the target verifier explicitly does not reconstruct the segment-digest payload. Four-trajectory estimation, residual state variation and useful decision headroom remain uncertain. Cleaner scalar advantages did not translate into different test actions or patient benefit. [Targets][targets] [Protocol][protocol]

## One next objective decision

**Recommend deciding whether the intended objective covers care obligations after step 52, and, if so, adopting a prospectively calibrated stage/risk-based residual terminal liability `C_T(s_T)` as the next objective-design direction.** Add its charge once as `-1e-9 * C_T` only in a separately approved future protocol; retain old-objective cost and all raw patient outcomes for comparison. This recommendation is motivated by unresolved terminal obligations and the inherited MDL-2 trade-off, not proof that terminal valuation caused the PPO null. [Reward memo][reward]

Missing calibration is explicit: intended clinical/economic horizon; expected remaining resource use and patient risk by stage, age/expiry risk and state; monetary/clinical valuation and its uncertainty; and reconciliation with already charged purchases, operations, patient harm and material waste. Do not price every active patient as a death or reward premature loss for removing backlog. No terminal coefficient, success threshold or patient tolerance is selected. The current finite-window objective is declared behavior, not a demonstrated implementation defect; overlapping harm/material charges are not proven duplicate valuation. Time-only potential shaping would revisit the baseline idea under this full-episode return contract. Proceed with this one objective-definition/calibration decision, not another baseline campaign or a numerical reward experiment. [Reward memo][reward]

## Execution and evidence boundary

Execution commit `45a5e16cd8e26364edaacb3aab885d7e4203621c`; frozen packet `5279726f8fc069905ec1097fc961c13136c25241136a99383edae78bcd3c6300`. All 522 episode records are present: 18 preflight, 288 training, 216 test. Accounting reports exactly 27,216 environment calls (27,144 trajectory + 72 clone) and 1,920 optimizer calls (1,152 actor + 768 critic). The completed sequence records 33 sections and 12 sealed learned artifacts. Scientific terminal receipt: completed, exit 0; child elapsed 5,155.572 seconds, no forced kill or supervisor error. The closure receipt says payload unchanged. These facts reuse persisted receipts, not a new audit or simulation. [All episodes][all-episodes] [Compute][compute] [Completed][completed] [Terminal][terminal] [Closure][closure]

Compute accounting was written during verification and does not include final archival clocks. The completed receipt records 247.906 seconds for raw verification and 549.032 seconds for payload archival. Child elapsed is not parent-through-final-archive elapsed. **Both local archives are verified and final closure is complete.** Payload: 3,595 members, 983,284,303 bytes, SHA256 `3b4508ee154296f0df0064407ae82542c03f9393b4fb6a76393d7517ee180408`. Launcher: 777 members, 1,366,103,243 bytes, SHA256 `b9656338a8a8c83ba0ffb376d08b1051c0c86b3b3ec029be9d35ea8caeb48bed`. Both persisted receipts report verification by reading every archive member. The coordinator separately confirmed exec session 35201 exit 0, zero stderr bytes and both known PIDs absent in host `ps` at 07:17Z; this delegate did not repeat that process check. Final launcher receipt read at 07:17:56Z. No Dropbox export, cloud-sync verification or collaborator-access claim. [Payload archive][payload-archive] [Launcher archive][launcher-archive]

This readout used final persisted reports, source/protocol reads and the coordinator's terminal handoff. It did not rerun verification/bootstrap, load or score a model, simulate, optimize, change reward/source/tests, alter scheduling/Live/workflow, launch or stop a job, commit, or take external action. Report-only JSON syntax validation passed; source tests/compileall remain coordinator-owned. This finite readout is complete; coordinator owns integration. Machine-readable companion: [decision.json](decision.json).

[protocol]: ../../specs/2026-10-02-time-baseline-comparison/protocol.md
[reward]: ../../specs/2026-10-02-time-baseline-comparison/reward-pivot.md
[evaluation]: ../../results/dynamic_candidate_time_baseline_20261002/payload/independent-verification.json
[targets]: ../../results/dynamic_candidate_time_baseline_20261002/payload/independent-target-verification.json
[all-episodes]: ../../results/dynamic_candidate_time_baseline_20261002/payload/all-episode-verification.json
[compute]: ../../results/dynamic_candidate_time_baseline_20261002/payload/compute-accounting.json
[completed]: ../../results/dynamic_candidate_time_baseline_20261002/launcher/completed.json
[terminal]: ../../results/dynamic_candidate_time_baseline_20261002/launcher/terminal.json
[closure]: ../../results/dynamic_candidate_time_baseline_20261002/launcher/closure.json
[payload-archive]: ../../results/dynamic_candidate_time_baseline_20261002/launcher/archive-receipt.json
[launcher-archive]: ../../results/dynamic_candidate_time_baseline_20261002/closure-archive-receipt.json
