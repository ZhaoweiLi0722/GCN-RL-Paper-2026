# GCN Value-MPC: Completed Development Comparison

2026-10-04 UTC. The single authorized attempt completed normally at
01:36:06 UTC, exit0/child0, in8658.41seconds (2h24m18s). No related runner
remained at the terminal process check. This is a completed simulation
comparison, not another readiness check or permission for follow-on training.

## Decision

The prespecified continuation-training signal is met. Continued eight-step TD
learning improves the frozen initial GCN value model in this experiment.
It does not yet establish robust superiority over plain MPC, a separate GCN
effect, deployment-time online adaptation, or real-world clinical benefit.

The primary persistent-change comparison, updated versus initial-value MPC,
has mean paired cost savings13.7493% and55.4167 fewer simulated patient losses
per world. Mean absolute savings are5363862.59 synthetic objective units;
the saved descriptive95% interval is[3785278.11,6682369.54]. All three blocks
have positive savings, and mean extra losses are negative in every condition.
All three prospective criteria therefore pass. The matched persistent worlds
change507/576 control-epoch action vectors; changes are not themselves the
benefit criterion.

## Full Comparator Context

Each condition has12 matched worlds:3 independent model blocks x4 replicates.
Savings are comparator minus evaluated controller; positive means lower cost.
Percentages average paired-world percentage savings, not ratios of pooled means.
Loss changes are evaluated controller minus comparator, in simulated patients
per world. Intervals below are the prespecified saved descriptive paired
block/within-block bootstrap95% intervals, not formal confirmation or safety bounds.

| Evaluated vs comparator | Condition | Cost savings, % | Descriptive95%, % | Extra losses/world |
|---|---|---:|---:|---:|
| Updated vs initial value | No change | 11.8585 | [4.8409,19.1375] | -49.9167 |
| Updated vs initial value | Persistent change | 13.7493 | [9.9857,17.0320] | -55.4167 |
| Updated vs initial value | Fast fluctuation | 7.9253 | [3.6403,13.0944] | -64.0833 |
| Updated vs plain MPC | No change | 2.3329 | [0.8303,3.8467] | -8.7500 |
| Updated vs plain MPC | Persistent change | 2.4573 | [-1.1702,5.4154] | -5.1667 |
| Updated vs plain MPC | Fast fluctuation | 0.5937 | [-3.1272,3.9924] | -12.0000 |
| Updated vs fixed uniform | No change | 4.6046 | [2.2646,6.6942] | -5.8333 |
| Updated vs fixed uniform | Persistent change | 4.1144 | [1.4547,6.2640] | -6.6667 |
| Updated vs fixed uniform | Fast fluctuation | -0.7170 | [-3.8209,1.8965] | +6.1667 |
| Initial value vs plain MPC | No change | -11.7842 | [-20.8882,-3.3950] | +41.1667 |
| Initial value vs plain MPC | Persistent change | -13.5086 | [-18.0624,-8.0819] | +50.2500 |
| Initial value vs plain MPC | Fast fluctuation | -8.6344 | [-15.3469,-3.3955] | +52.0833 |

Primary persistent-change block detail:

| Block | Absolute savings | Paired savings, % | Extra losses/world |
|---|---:|---:|---:|
| 0 | 6280597.43 | 15.3689 | -67.75 |
| 1 | 5872590.23 | 15.0879 | -70.00 |
| 2 | 3938400.12 | 10.7912 | -28.50 |

Important limits are not hidden by the successful primary screen:

- Initialization alone worsens cost and patient loss versus plain MPC in every
  condition/block mean. The13.75% continuation result partly recovers from a
  poor initial learned value; it is not a13.75% advantage over strong MPC.
- Updated versus plain MPC has favorable aggregate means, but persistent and
  fast-fluctuation cost intervals cross zero. Persistent block2 has2.25 extra
  losses/world despite positive mean cost savings. No universal dominance.
- Against uniform allocation, fast-fluctuation means worsen both cost and loss.
  Fast block2 has17.25 extra losses/world; no-change block2 saves cost but has
  7.25 extra losses/world. Lower aggregate cost is not a clinical safety claim.
- Continuation includes24 additional cohorts/block and2304 additional updates.
  Training data, compute and collection policy change together. This is not a
  matched-data or equal-budget comparison of alternative learning algorithms.
- Only3 independent model blocks underpin descriptive intervals. Support labor
  and patient dynamics remain synthetic and uncalibrated; E1 field data remain
  missing. Forecast/belief approximation remains a limitation. No formal holdout
  was used; Stage E stays closed.

## Operational Readout

The direction of recorded resource use changes materially, but this does not
identify a causal mechanism. Persistent-change mean applied flexible hours/world
are93.2256 for initial value,354.6667 for updated value,352.9133 for plain MPC,
and384 for uniform. Updated versus initial increases flexible-labor cost by
29523.69 objective units/world, while reducing patient-loss cost by2770833.33;
other cost components account for the remaining net saving. Updated versus
plain MPC changes429/576 persistent-condition action vectors, with mean
985067.94 objective-unit savings. Full component differences remain in the
original comparison, including negative and trade-off outcomes.

## Completed Scope And Preservation

- Initial72/72 and continuation72/72 training worlds; frozen evaluation144/144
  (36/controller); all288 worlds contain64 rows, including16 settlement epochs.
- Value updates4608/4608, actor0;144 target sets and432 full saved states.
  Three initial768-update and three final1536-update model seals precede tests.
  Evaluation has zero optimizer updates.
- Native steps18432/18432; native operations19008/19008; forward11664/11664;
  planner epochs4644864/4644864; filter transitions1843200/1843200.
  Global and phase counters stay within all frozen caps.
- Phase seconds: admission4.60, initial2447.31, continuation2450.98,
  evaluation3721.75, readout/archive33.48. Total8658.41 is below14400.
- Stderr/stdout/detached log are empty; no failure/overrun marker. Terminal
  exit0 and no matching Python remain. No additional scientific calls by the
  terminal monitor. Existing20zero-update tests and compileall are reused.
- Reused the existing archive, without recreating it. Its1492 members match
  both recorded SHA256 values and current payload bytes;1491 inventory entries
  also match. Total run-root storage454890618bytes, within4GiB.
- Local archive verified only: no Dropbox export, cloud-sync confirmation or
  Howard-access confirmation. Original protocol, sources, models, raw records
  and earlier failures remain unchanged.

The coordinator independently streamed all raw rows using standard-library
JSON/gzip arithmetic, not the campaign analysis helper. Checks covered exact
phase/block/condition/replicate seeds,64 sequential epochs, component-cost and
negative-reward sums, unique persistent patient identities, terminal-state
irreversibility, reconstructed loss/delivery counts, settled obligations,
requested/applied hours, four-role cohort/tape equality, and every reported
paired mean/block difference and action-change count. Archived member and
source/input hashes were checked separately. Existing bootstrap intervals were
read from the locked analysis output and their implementation inspected; no
new bootstrap or model/environment execution was performed.

One finite independent interpretation agent (Fermat,
01a10495-81bd-7471-aa32-b0eddc973112) confirmed the primary/secondary distinction
and initialization weakness from saved results; it completed and was closed.
The earlier efficiency advice was reused, without a new gate.

## Evidence And Next Decision

- Execution commit:654ff0f5a4f07116f6fe1ba3fdef22e17aa907c4.
- Frozen implementation:0fa338b6b1eced4468db3279029a0c7219f1f694;
  scientific implementation:ae5fe4bcc4ca6390ac2ff5be05f218bf822b75e1.
- Packet:50fab0f56d52a8cb07629f59156dc9e5e6a882bc32dbb3f354f93d1af9f4770d.
- Original comparison:results/capacity_value_mpc_20261003/payload/comparison.json,
  SHA256bf8870a6a7fcf93748abaa8cf3723a8a5a89c2e2f81d17cb7d3edb456b26a883.
- Raw rows:results/capacity_value_mpc_20261003/payload/raw/.
- Inventory:results/capacity_value_mpc_20261003/payload/artifact-inventory.json.
- Archive:results/capacity_value_mpc_20261003/archives/payload.tar.gz,
  SHA25688a1aaa542369defaa25a508aa25a71a75bf83265a5465b14859e2b84b464274.
- Terminal:results/capacity_value_mpc_20261003/launcher/terminal.json.

Close this attempt and pause the same monitor. The manuscript should retain
historical negative actor-update results and add this bounded TD-value result.
The next scientific decision is whether to authorize a separate, fully specified
confirmation focused on the final value-MPC versus plain MPC and a sufficiently
trained frozen-value comparator, with graph attribution addressed separately.
No new package, sample expansion, reward adjustment or training is authorized
by this readout; the successful primary screen does not trigger automatic work.
