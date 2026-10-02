# Cohort-Objective Comparison: Complete, No Incremental RL Benefit

Recorded 2026-10-02T11:42Z. Recovery2 completed normally with exit code 0.
PIDs 79988/80002 were absent after termination. The same `gcn-rl` automation
is PAUSED and remains visible. No follow-on experiment or retry was started.

## Answer and Decision

Including post-window patient and resource consequences in the PPO training
objective did not improve the evaluated policy over its own frozen initializer,
window-only PPO or continued imitation. Each of 36 paired worlds had identical
raw cost and final-state hashes against all three primary controls and R4.
Each primary contrast had zero changed requested actions across 1,872 greedy
prefix decisions. This is an exact observed behavioral null, not merely a
failed significance test or an imprecise mean.

The prespecified decision is `close_one_shot_route`: the positive-cost screen
failed; the observed loss-direction screen passed only because differences were
zero. No extra epochs, samples, objective weights or replacement baseline are
authorized. This does not prove that all RL is ineffective, that R4 is globally
optimal, or that any possible reward revision must fail.

## Paired Results

The independent reader verified 216 evaluations: six controllers on 12 worlds
within each of three training blocks. Costs cover the common 63-step endpoint
(52-step enrollment prefix plus 11-step fixed tail). Values are equal-block means
of paired differences; negative cost/loss differences favor cohort PPO. The
three blocks, not 216 controller episodes, are independent training replicates.

| Cohort PPO minus comparator | Cost change | Patient losses per cohort | Changed prefix requests |
| --- | ---: | ---: | ---: |
| Own frozen initializer | 0.000000% | 0 | 0 / 1,872 |
| Window-only PPO | 0.000000% | 0 | 0 / 1,872 |
| BC-CONTINUE | 0.000000% | 0 | 0 / 1,872 |
| R4 | 0.000000% | 0 | 0 / 1,872 |
| Full MDL-2 | -0.474826% | -21.555556 | 1,603 / 1,872 |

All three primary contrasts have zero cost, loss, completion and waiting
differences in every block; their descriptive paired intervals are [0, 0].
This finite observed equality is not a population equivalence claim.

Against MDL-2, block cost differences were -0.169996%, -0.625833% and -0.628649%.
The equal-block absolute cost difference was -13,058,259.567836 simulator cost
units; the descriptive 95% hierarchical paired interval for the relative change
was [-0.800847%, -0.126348%]. All five non-MDL-2 controllers share this benefit:
it is inherited R4/frozen performance, not added RL or isolated GCN attribution.
No clinically calibrated monetary interpretation is established.

## Endpoint and Trade-Offs

Per-cohort means across 36 paired worlds:

| Outcome | Frozen / both PPO arms / BC / R4 | Full MDL-2 |
| --- | ---: | ---: |
| Raw 63-step cost | 2,764,672,750.716592 | 2,777,731,010.284429 |
| Original 52-step cost | 2,546,767,147.690549 | 2,564,551,047.920767 |
| Tail cost | 217,905,603.026043 | 213,179,962.363662 |
| 52-step completions | 2,898.972222 | 2,888.111111 |
| 52-step losses | 2,242.750000 | 2,268.805556 |
| Final completions | 3,300.416667 | 3,278.861111 |
| Final losses | 2,526.388889 | 2,547.944444 |
| Final completion service level | 0.567170 | 0.563447 |
| Waiting patient-steps | 26,624.694444 | 26,593.861111 |
| Turnaround, simulator time units | 5.667983 | 5.580913 |
| Post-patient-resolution tail cost | 179,143.846565 | 180,375.191400 |
| Retained reagent stock, sum | 509.755555 | 514.338888 |
| Retained bioreactor stock array, sum | 128.983333 | 121.361111 |

The inherited policy reduces final simulated patient losses by 21.56 per cohort
and increases mean completion service level by 0.37226 percentage points against
MDL-2, but increases waiting by 30.83 patient-steps and turnaround by 0.08707 time
units. Block 60 expiry losses increase by 1.16667 despite lower total losses.
These adverse directions remain part of the result. At fixed enrollment,
completion and loss changes are complements, not independent confirmations.

Lower prefix cost is partly offset by higher tail cost: approximately
-17.784 million and +4.726 million cost units, respectively. All patients resolve
at tail step 8 in every evaluation; final active count is zero. Resources remain
without terminal credit. Source resource clipping is preserved, not claimed
to conserve material. Full primitive components, loss causes, per-facility
stocks, flows and clipping records remain in the raw verification artifact.
This fixed endpoint is not a lifetime economic valuation.

## Execution and Preservation

Execution `e453037`; implementation
`63058e6b4d9d30bf7f3ec5aa64e111f89a8b0e30`; packet
`61ea95b9b4c7b4cde8152e28df4a54ed2929c383c78ad2f29557e0838e4e3ae8`.
Seven real-saved-metadata/mock-entry tests and full compileall passed before
freeze. No scientific source changed during execution. Earlier valid tests,
training-target verification and failed-run archive were reused, not repeated.

Recovery reused 11 complete evaluations and 12 sealed models, adding exactly
205 evaluations / 12,915 environment calls / zero optimizer calls. The original
17 partial calls remain spent and preserved. Admission counts matched 3 layouts,
3 R4 loads, 12 model loads and 205 episode builds. Prior training remains 9 jobs /
288 cohorts / 1,920 optimizer calls; it was not repeated. These are the original
test worlds, not independent confirmation.

Supervisor elapsed time was 2,133.496322 seconds (35.56 minutes); evaluation used
1,709.170479 seconds, raw verification 109.232680 seconds and payload archival
306.155437 seconds. Normal exit, no forced kill, no error, stderr empty. Existing
runner closure checked source/input bindings and unchanged payload. Outputs:

`results/dynamic_candidate_cohort_objective_20261002_evaluation_recovery2/`

| Artifact under that root | SHA256 |
| --- | --- |
| `payload/comparison.json` | `b4c993de4d48fa111ea5d9f9f2b06d2eee250042924e13fd4cc27db58f44ff60` |
| `payload/independent-verification.json` | `65677d8d1db44b6cbadacb4115ce988ed6f1e7ad233dbcfaa0265bae9fd1c337` |
| `payload/compute-accounting.json` | `aaacce2f4da281d467c412eddbe90c6ea9e5e2612257387c25d89db39eff5453` |
| `payload/evaluation-index.json` | `759c69350391b96c70bfbcf22a2bfb208923e1cce26637a1b4842b7b29703a05` |
| `archives/completed-payload.tar.gz` | `58df171a520e56b01ec617ea06991fc2dd0336f8a2fd1d13deb3f1ba651ab748` |
| `archives/completed-launcher.tar.gz` | `d656af136876a243f150479a6b8614400ea24cb4e3704fad97d772cf1e0123ce` |

Payload archive: 2,637 files / 745,774,457 compressed bytes. Launcher archive:
645 files / 1,170,279 compressed bytes. Both receipts confirm reading and hashing
every member; original payload unchanged. This is local preservation, not
Dropbox export, cloud sync or Howard access. Historical evidence remains intact.

## Manuscript-Ready Scope

In a prespecified three-block development comparison, extending the training
objective to include standardized post-enrollment consequences did not change
the greedy policy relative to the same frozen initializer, window-only PPO,
or continued imitation. All 36 paired worlds produced identical costs and final
states for these controllers. Their common cost advantage over MDL-2 therefore
cannot be attributed to additional RL training. The advantage also involved
waiting-time and expiry trade-offs. These findings bound this particular
objective amendment and restricted candidate policy; they do not test
deployment-time parameter adaptation or establish global optimality.

## Next Decision

Close this exact training/objective route. Do not extend it or turn retrospective
reward reweighting of identical trajectories into an RL gain. The immediate
paper decision is whether to center the supported graph-aware policy package
and report the bounded RL null as a limitation, with any further RL scenario
study separated into a new prospective scope. This comparison alone neither
isolates GCN nor removes the need to trace graph claims to matched ablations.
No new experiment is currently authorized. A further study must pose a different
decision-changing hypothesis and have one complete finite approval package,
not another audit or baseline-fitting loop. Stage E remains closed and the
formal holdout untouched; no coauthor approval is implied.
