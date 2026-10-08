# Five-Family Fixed-Benchmark Terminal Readout

2026-10-08. **Scientific execution completed successfully; neither finalist
passed the frozen joint screen. No winner.** This is a completed scientific
negative/inconclusive result, not an engineering abort. The one approved
attempt is consumed; no retry, resume, extra fitting or follow-on is authorized.

## Authority and completion

Zhaowei's literal approval `批准完整单次包` covered the full single48h/32GiB
package. Science46b078f0fa13809dacbe307bbfaac6d3c82b1282;
freeze104ddff436571dab44440644168317b315c8db0b;
executionf5772e824632248be458d42e487f757477100575;
packet40fd80177913db5b6009bcccf4a87703a17bcdc3fd5caf2fc184f3739b009cc9.
No Howard approval is inferred.

`results/capacity_family_selection_20261006/launcher/terminal.json` records
child/supervisor exit0 and scientific_completion_verified=true. The claimed
PIDs92809/92821 are absent at the terminal process check. No failure markers;
stderr/stdout/detached logs are0bytes. Actual runtime117071.166s=32.520h,
including automatic analysis/archive. No scientific runtime remains.

| Phase | Completed / target | Owner seconds |
| --- | ---: | ---: |
| Development training | 2880 / 2880 | 26918.613 |
| Development evaluation | 864 / 864 | 14810.410 |
| Fresh finalist training | 1920 / 1920 | 42086.668 |
| Independent evaluation | 1080 / 1080 | 32830.630 |
| Admission | complete | 5.803 |
| Automatic analysis/archive | complete | 418.389 |

All30development and20fresh training jobs completed96worlds and3072minibatch
iterations each. Each of4800world fits has32iterations. Fifty initial and50final
model seals are retained; all20fresh finals were sealed before independent
evaluation. Both evaluation phases used zero optimizer updates. No clones,
old holdout, StageE, reward/scenario search or budget transfer occurred.

Actual totals:6744trajectories;431616native steps;445104native operations;
153600fit iterations;95232actor+153600critic+67584value=316416optimizer
dispatches;20250624example presentations;1016448module forwards;
50429952prediction steps;43161600filter transitions. The value count includes
18432PPO-value and49152value-TD updates. Per-item ceilings are not added to
claim actual consumption. The frozen limits remain unchanged.

## Development selection is not a winner

Five families, two learning rates(.0001/.0003), three blocks,96training worlds
per configuration/block,24development worlds per block and12roles.
Patient-first/cost selection chose value_td(.0001) and SAC(.0003), using only
development data. Value-TD's development saving was1.6271% with12.2361fewer
losses/world; SAC's was-1.3898% with5.375fewer losses/world. The full development
family rankings, both learning rates and all66pairs are retained, including
eliminated DDPG/TD3/PPO. No family had a preselected preference.

The fresh stage used5blocks x2families xgraph/self_only x96worlds. Matched
initial tensors and seeds differ only in adjacency, not physical dynamics;
own-policy training trajectories can diverge. Final independent evaluation
used120new worlds x9roles, including H8, H16, evaluation-only historical
conditional graphTD and both graph-initial controls. The value-TD recipe is
zero-residual observed TD8 on own complete trajectories, not native-tail
replication. Actor/critic planning and sample-reuse costs remain charged.

## Frozen decision

Savings are reference cost minus candidate cost. Percentages are ratios of
overall means, not means of world ratios. Costs are synthetic objective units,
not calibrated currency. Extra losses are candidate minus reference per world.

| Candidate vs H8 | Savings M [descriptive95%] | Savings % | Extra losses [descriptive95%] | Positive blocks | Screen |
| --- | --- | ---: | --- | ---: | --- |
| Value-TD graph-final | 0.337067 [-0.026348,0.677834] | 0.8509 | -11.875 [-14.3669,-9.2415] | 4/5 | fail |
| SAC graph-final | -1.386049 [-2.135340,-0.807385] | -3.4991 | 3.2583 [-3.5167,12.0538] | 0/5 | fail |

The screen requires all four conditions:mean saving>=0.5%,all five blocks
positive,cost lower bound>0,and mean extra losses<=0. Value-TD fails the
all-block and interval conditions. SAC fails every condition. Neither is
eligible; favorable value-TD-versus-SAC head-to-head outcomes cannot override
failure against the primary strong baseline. No unique winner or universal
best-controller claim follows.

Value-TD block savings M:[0.362718,0.654785,0.400996,0.338851,-0.072017].
SAC:[-1.252384,-1.265931,-2.630104,-0.663817,-1.118008].
Value-TD costs more in51/120worlds and loses more patients in21/120;
SAC costs more in91/120and loses more in75/120. Lower average loss is not
world-level noninferiority or clinical safety.

| Condition | Value-TD savings % / extra losses | SAC savings % / extra losses |
| --- | --- | --- |
| No change | 0.7906 / -12.025 | -4.3217 / 5.025 |
| Persistent change | 0.2495 / -5.925 | -5.5921 / 10.225 |
| Fast fluctuation | 1.4561 / -17.675 | -0.8017 / -5.475 |

All three value-TD condition-specific cost intervals cross zero. All conditions
are equally weighted. The2000frozen bootstrap resamples share training-block
draws across conditions, but draw worlds independently within each condition.
Same replicate numbers across conditions are not paired patients. Intervals
are descriptive, not multiplicity-adjusted formal significance or power.

## Graph and training increments

| Candidate / reference | Savings M [descriptive95%] | Savings % | Extra losses |
| --- | --- | ---: | ---: |
| Value-TD graph / self-only | 0.013048 [-0.114957,0.161164] | 0.0332 | 0.1500 |
| SAC graph / self-only | -0.261264 [-1.273655,0.546489] | -0.6414 | 0.5250 |
| Value-TD final / graph-initial | 0.337067 [-0.024162,0.676015] | 0.8509 | -11.8750 |
| SAC final / graph-initial | 0.005197 [-0.881935,0.716527] | 0.0127 | 3.3833 |

Neither graph/self-only nor final/initial contrast establishes a cost benefit.
Value-TD's zero-residual initializer reproduces H8 cost/loss outcomes; its
contrast-specific bootstrap interval differs slightly from the separately
seeded primary H8 screen. Random SAC initialization is not a strong operational
reference. A whole-controller comparison does not isolate a pure backbone
causal effect, and observed updates alone do not establish an RL contribution.

H16 is secondary:0.7674%saving versus H8,[-0.078649,0.707722]M cost interval,
-6.2losses/world. The fixed legacy role gives0.0814%,[-0.566420,0.524803]M,
-4.7917losses/world. Their mixed historical provenance remains explicit; they
are not newly selected winners. Old full-routing results are not in this table.

Mean applied flexible hours/world:H8 304.128,H16 338.726,value-TD graph376.859,
SAC graph328.479. Mean whole-trajectory elapsed seconds:37.023,67.980,36.988,
4.147respectively. These include rollout I/O and are not separately measured
per-decision latencies. Full requested/committed/applied actions and waiting/
turnaround epoch reports remain in raw records; no unrecorded latency is invented.

## Evidence and preservation

Canonical frozen output:`results/capacity_family_selection_20261006/payload/`
`comparison.json`, `completion.json`, `selection.json`, `artifact-inventory.json`,
all raw trajectories, summaries, models, tapes,4800after-fit states and receipts.
Supplemental saved-data reader:`reports/2026-10-08-family-selection/read_saved.py`;
generated results:`results/capacity_family_selection_20261006/terminal-readout/`.
The reader imports no scientific project modules and makes zero environment,
forward or optimizer calls. It reads6744raw trajectories and reconciles them
with saved summaries/comparison, checks receipt counts/model-state hashes, and
retains all66development plus36final pairs with world/block/condition outcomes,
action differences, labor, delay summaries and compute. Original automatic
validation of all4800saved Adam states and every archive member is reused.
See `verification.json`, `world-readout.json`, `all-pairs-readout.json`, and
`job-and-role-readout.json`; no new statistical search or scientific gate.

Single existing archive:`archives/payload.tar.gz`,4813583255bytes,
SHA256`bfe956a533ead6c27108c570cf6fe705763368fb9ec53dec9eb2470ab7001ec1`,
25790members. No rearchive. Additive Dropbox delivery includes this archive,
manifest, authority, launcher evidence, readout and actual manuscript source;
destination-byte verification is separate from cloud sync and collaborator
access. See `dropbox-handoff.md` and its delivery receipt for final state.

The actual paper's abstract, new five-family results subsection and discussion
are updated with these outcomes. No local TeX engine is available; PDF rebuild
and rendered layout are unverified, and old PDF previews must not be mistaken
for this updated source. Reuse23necessary scientific tests/fullcompileall;
the full compileall was also rerun successfully after adding the saved-data
reader; the completed23scientific checks were not needlessly rerun. No manuscript
submission or human coauthor approval is claimed.

E1 remains missing. Synthetic-only evidence does not establish real-patient
calibration, clinical safety, deployment-online adaptation or publication
readiness. Failed screens and unfavorable controls are retained. Closure ends
the authorized chain and pauses the same visible `gcn-rl` monitor; any further
scientific work requires a new explicit scope and approval.
