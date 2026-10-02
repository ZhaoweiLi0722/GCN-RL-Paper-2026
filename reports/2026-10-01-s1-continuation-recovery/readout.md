# Continuation Recovery Readout

## Finding

The complete three-block development comparison found **no incremental greedy
PPO benefit over its own frozen initializer or continued R4 imitation**.
Both paired cost differences are exactly zero in every evaluated world, with
saved 95% bootstrap intervals [0, 0]. PPO, own-frozen, BC-CONTINUE, and R4
produced identical evaluated requests, executed actions, and patient outcomes.
PPO versus full MDL-2 reduced modeled cost by 0.521671389%, but increased
terminal-active burden in all three blocks. This is an inherited R4-like
cost/patient trade-off, not a gain attributable to PPO continuation.

All 387 scheduled episodes and 1,152 optimizer calls are recorded, including
the complete independently verified 180-episode test matrix. Scientific
comparison completion is separate from preservation: the coordinator confirmed
**both scientific completion and required local archives**. The closure receipt
was saved at 2026-10-02T03:02:18Z, the owned parent exited 0, and no related
process remained by 03:03Z. The complete record is
[closure-index.json](closure-index.json). This readout adds no gate, retry, or
execution authority; the approved attempt is consumed.

## Fixed Cost Contrasts

Source: [independent-verification.json][verification], `outcomes` and
`analysis.contrasts`. Each contrast is first controller minus comparator,
paired within block/world. Cost differences and intervals below are in
**million modeled objective units**, not dollars. Each block contains twelve
paired worlds; the summary gives equal weight to the three blocks.

| Fixed contrast | Block 60 delta | Block 61 delta | Block 62 delta | Equal-block delta | Saved cost 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| PPO minus own-frozen (primary) | 0 | 0 | 0 | 0 | [0, 0] |
| PPO minus BC-CONTINUE | 0 | 0 | 0 | 0 | [0, 0] |
| PPO minus R4 | 0 | 0 | 0 | 0 | [0, 0] |
| PPO minus full MDL-2 | -15.807765115 | -7.235232035 | -17.468346345 | -13.503781165 | [-19.667724360, -5.553843712] |
| Own-frozen minus R4 | 0 | 0 | 0 | 0 | [0, 0] |

Relative cost is `mean_b(100 * mean_world(left_cost - right_cost) /
mean_world(right_cost))`, not a ratio formed after pooling denominators.
PPO minus full MDL-2 is -0.593747984%, -0.281424615%, and -0.689841569%
in blocks 60/61/62, respectively; its equal-block change is **-0.521671389%**
with saved relative 95% interval **[-0.755552405%, -0.213307447%]**.
The other four relative changes and their intervals are exactly 0% and
[0%, 0%]. All relative denominators are positive; the saved analysis reports
zero undefined relative bootstrap draws.

The primary >=1% reduction screen is **not met**, with **0/3 strictly
cost-favorable blocks**. The saved decision is
`limited_negative_or_inconclusive`. A favorable secondary comparison cannot
replace the primary contrast or establish an RL increment.

## Patient Trade-offs

These are equal-block mean paired differences per episode. Positive losses,
waiting, or terminal-active burden are adverse; negative completions are
adverse. All four zero-cost contrasts also have exact zero patient differences
in every world, not merely cancellation in their means.

| Contrast | Completions | Losses | Expiry losses | Waiting patient-steps | Terminal active |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO minus own-frozen | 0 | 0 | 0 | 0 | 0 |
| PPO minus BC-CONTINUE | 0 | 0 | 0 | 0 | 0 |
| PPO minus R4 | 0 | 0 | 0 | 0 | 0 |
| PPO minus full MDL-2 | +5.222222 | -20.888889 | -12.972222 | +0.833333 | +15.666667 |
| Own-frozen minus R4 | 0 | 0 | 0 | 0 | 0 |

For PPO minus full MDL-2, saved 95% intervals are: completions
[-4.972222, 13.222222], losses [-29.416667, -10.471528], expiry losses
[-19.694444, -6.416667], waiting patient-steps [-52.973611, 60.725000], and
terminal active [12.055556, 19.277778]. The zero contrasts have [0, 0]
intervals for these endpoints.

| PPO minus full MDL-2 | Completions | Losses | Waiting patient-steps | Terminal active |
| --- | ---: | ---: | ---: | ---: |
| Block 60 | +8.833333 | -23.250000 | -5.833333 | +14.416667 |
| Block 61 | -2.916667 | -12.416667 | +17.916667 | +15.333333 |
| Block 62 | +9.750000 | -27.000000 | -9.583333 | +17.250000 |

Terminal burden increases in every block; block 61 additionally has fewer
completions and more waiting. The mean terminal-active difference decomposes
into +8.000000 waiting, +9.250000 specimen transit, -1.583333 production,
and zero finished-return patients. Enrolled-patient differences are zero.
Unresolved terminal patients are neither completed care nor deaths, and
their obligations extend beyond the unchanged 52-step objective. Waiting
patient-steps sum end-of-epoch queue occupancy; they are not mean waiting
time among completed patients. Lower finite-window cost therefore does not
establish overall service improvement or clinical noninferiority.

## Actions and the Primary Null

Summing the saved **per-world contrast** action differences gives the following
counts, each over 36 paired worlds x 52 steps = 1,872 aligned step positions.
These are descriptive closed-loop differences, not same-state counterfactual
effects. No raw action reconstruction was repeated for this readout.

| Contrast | Requested-action changed steps | Executed-action changed steps |
| --- | ---: | ---: |
| PPO minus own-frozen | 0 | 0 |
| PPO minus BC-CONTINUE | 0 | 0 |
| PPO minus R4 | 0 | 0 |
| PPO minus full MDL-2 | 1,663 | 1,655 |
| Own-frozen minus R4 | 0 | 0 |

For full MDL-2, requested/executed changed-step totals by block are 572/568,
610/608, and 481/479. Summed requested L1 difference is 875.429730710;
integer specimen-request L1 is 51,143. Executed L1 totals are 28,456 specimen
transfers, 8,301.600359581 reagent transfers, 131.999996960 capacity transfers,
and 10,957.802034527 replenishment units; these channel-specific quantities
must not be added into one physical measure. All these L1 quantities are zero
for the other four contrasts. Divergent trajectories can change non-specimen
actions without changing the within-state candidate-bank restriction.

The coordinator's [saved decision diagnostic](saved-decision-diagnostic.json),
reproducible with [inspect_saved_decisions.py](inspect_saved_decisions.py),
provides a separate **post-hoc descriptive explanation**, not a new endpoint.
Across 108 bound event files and 1,872 shared test states, actor observations
and candidate banks match across own-frozen/PPO/BC; all three always choose
the R4 class and submit identical requests. Both trained roles have nonzero
recorded probability-vector changes on all 624 states per block. Mean R4
probability changes from frozen [0.26455, 0.25674, 0.28723] to PPO
[0.19052, 0.19023, 0.21740], but R4 remains the greedy winner everywhere
(minimum PPO probability margin >0.0124). Training PPO sampled non-R4 classes
1,288, 1,324, and 1,215 times out of 1,664 decisions per block (73-80%).

Thus this is neither zero recorded updates nor zero sampled non-reference
coverage: probabilities changed while the evaluated greedy winner remained
unchanged. The diagnostic does not prove absent headroom, entropy causality,
a wrong reward, or benefit from stochastic deployment. It is reused here,
not rerun; no tensors or models were loaded.

## Coverage and Budget

[All-episode verification][all-episodes] contains exactly 387 outcome records:
15 preflight, 96 PPO training, 96 BC training, and 180 final evaluation.
The final matrix has 180 unique `(block, role, world_index)` keys, all fifteen
block/controller cells containing indices 0-11, each a complete 52-step test
episode. It covers 36 world starts and **three trained blocks**, not 180
independent trained policies. Historical initialization and qualification
episodes are excluded from these counts and from the cost comparisons.

| Phase | Episodes | Trajectory calls | Clone calls | Actor updates | Critic updates | Recorded seconds / cap |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Runtime/input binding | 0 | 0 | 0 | 0 | 0 | 3.285 / 600 |
| Same-start preflight | 15 | 780 | 60 | 0 | 0 | 121.236 / 1,200 |
| PPO continuation | 96 | 4,992 | 0 | 384 | 384 | 1,176.901 / 4,500 |
| BC-CONTINUE | 96 | 4,992 | 0 | 384 | 0 | 587.376 / 2,700 |
| All-model seal | 0 | 0 | 0 | 0 | 0 | 0.691 / 300 |
| Final evaluation | 180 | 9,360 | 0 | 0 | 0 | 876.788 / 4,500 |
| Saved-data verification/analysis | 0 | 0 | 0 | 0 | 0 | 172.497 / 1,200 |
| Local archive verification | 0 | 0 | 0 | 0 | 0 | 368.310 / 1,200 |
| Supervisor/terminal closure | 0 | 0 | 0 | 0 | 0 | 40.547 child portion; parent supplement completed within 900 |
| Scientific-call totals | 387 | 20,124 | 60 | 768 | 384 | Child elapsed 3,349.973; post-parent-exit observed upper bound 3,516.931 |

[Compute accounting][accounting] matches **20,184/20,184 environment calls**
and **1,152/1,152 optimizer calls**, with zero test optimizer calls. Each
training arm/block used 32 episodes and 1,664 trajectory calls; PPO used
128 actor + 128 critic updates and BC used 128 actor updates. Frozen received
no continuation training. Actor-update and interaction budgets match, not
total computation. The [model seal manifest][seals] records all nine learned
artifacts; the completed phase sequence precedes final evaluation.

Time entries through sealing/evaluation come from compute accounting;
verification/analysis time comes from [status 000050][status], and payload
archive time from [completed.json][completed]. The [supervisor][supervisor]
reports 3,349.973290 seconds through child exit, not final elapsed time after
the separate closure archive. The coordinator's later same-clock observation
puts complete parent exit within 3,516.931 seconds (58.62 minutes), an upper bound
including observation delay rather than an exact exit duration. The authorized sum of phase caps is 17,100
seconds and the independent global cap is 17,400 seconds (290 minutes),
including startup, verification, archive, and closure. No extra time or calls
are authorized by an unfinished closure archive. The same packet limits historical
loads to 3 initializers + 3 R4 policies and construction to 387 episodes +
3 zero-step layouts; the coordinator confirmed matching operation receipt counts
after parent exit, without loading any model or building an environment.

## Reconciliation and Uncertainty

This readout independently joins the 180 saved per-episode `outcomes` by
block, controller, and world index, subtracts the fixed pairs, and recomputes
block/equal-block means and the prescribed mean-of-block-ratios percentage.
It separately sums `paired_worlds[].worlds[].action_differences`. Every stored
per-world scalar difference and block-mean difference matches exactly; all
equal-block cost deltas match exactly. No substantive discrepancy was found.
Only floating-point accumulation differences occur: up to 4.768371582e-7
objective units in comparator means, 5.684341886e-14 in non-cost equal-block
summaries, and 1.110223025e-16 percentage points in the nonzero relative cost.
They do not affect any displayed result or interpretation.

Intervals above reuse the saved 10,000 hierarchical paired block-then-world
bootstrap draws (three blocks, twelve worlds per sampled block, original seed
5103747212375785473, PCG64, linear quantiles). No bootstrap, raw/hash-chain
audit, scientific scoring, test, model, environment, or optimizer execution
was repeated. The existing independent verification supplies raw-record and
pairing validation; this report adds bounded saved-record arithmetic only.

**A [0, 0] bootstrap interval here reflects identical observed greedy
action/outcome traces**, so resampling those zero paired differences remains
zero. It does not establish population equivalence, absence of learnable
headroom, or near-optimality. Three trained blocks give low-precision
descriptive development evidence in one scenario; there is no independent
confirmation or new formal holdout. Common world starts do not guarantee
event-level common randomness after actions diverge. The study isolates
neither GCN message-passing value nor superiority over a corrected DDPG
comparator. Simulator-trained PPO with frozen greedy evaluation is not
deployment-time parameter adaptation.

## Preservation and Closure

Coordinator closure check at 2026-10-02T03:03:01Z supersedes the earlier
archive-in-progress snapshots. Compact index: [closure-index.json](closure-index.json).

| Item | Evidence-backed state |
| --- | --- |
| Scientific phases and saved verification | Complete; terminal receipt records `scientific_completion_verified=true`, `status=completed`, exit 0 |
| Local payload archive | Archive receipt records `local_archive_verified=true` and every-member verification: 2,426 files, 697,297,439 archive bytes |
| Terminal/supervisor outcome | Supervisor passed, exit 0, `reason=child_exited`, no error or forced kill; elapsed 3,349.973290 seconds |
| Closure supplement | Successful [closure archive receipt][closure-archive]: 588 files, 497,640,409 archive bytes, every member verified; owned parent session exited 0 |
| Final process/clock state | No matching Python remains; parent completion observed within 3,516.931 seconds from the same monotonic origin |
| Dropbox/off-device/cloud/collaborator preservation | Not performed or claimed by this local-only attempt |

The [payload archive receipt][archive] records SHA256
`d2838aa5b491ccced3fc158cca116176943e2a30d0e14d72a885f2ff58f5b320`.
[Closure.json][closure] records unchanged payload and no automatic follow-on;
the separate [closure archive receipt][closure-archive] satisfies the terminal
receipt's preservation requirement. Launcher archive SHA256 is
`98999db5ee040413c28cfc19397e896bc8e798d594e0b34b2e22686bbc1da1b4`.
Receipt identities are retained in the closure index; member verification was
performed by the bounded campaign, not repeated for report writing. Verified
local archives are not off-device backup. Parent/child [claim][claim] /
[child claim][child] record 51957/51969,
execution commit `63c2383d33f5795f22ec1173bd75222a30a63a4a`, and packet
`8f0698cc8bdf41725e07ed78b0a856695473588cdcbbb817d90fa9d8c0223107`.
These are recorded identities, not a fresh process-liveness assertion.

This single approved attempt supports a bounded negative continuation result
and an inherited baseline trade-off. It authorizes **no automatic retry,
extra training, reward search, stochastic-policy appendix, or replacement
seeds**. The [methods guide](methods-and-interpretation.md) and
[approved continuation protocol][protocol] retain the design; this file is
the canonical results readout. The coordinator owns the completed closure and
Live/workflow/plan changes. The post-hoc descriptive script passed its actual
bound-record assertions and three artificial helper checks; full repository
compileall passed. The frozen scientific implementation remains unchanged
from its 120-test acceptance.

[verification]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/independent-verification.json
[all-episodes]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/all-episode-verification.json
[accounting]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/compute-accounting.json
[seals]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/model-seals.json
[status]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/status/000050.json
[completed]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/completed.json
[archive]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/archive-receipt.json
[supervisor]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/supervisor.json
[terminal]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/terminal.json
[closure]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/closure.json
[closure-archive]: ../../results/dynamic_candidate_continuation_recovery_20261001/closure-archive-receipt.json
[claim]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/claim.json
[child]: ../../results/dynamic_candidate_continuation_recovery_20261001/launcher/child-claim.json
[protocol]: ../../specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-protocol.md
