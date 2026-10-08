# MDL-2 Fixed-Support Supplement: Completed

2026-10-08. One authorized attempt completed successfully (exit0) in
112.676843291s including analysis and the single archive. All120new trajectories
and7680raw epoch records were reconciled;360previously verified trajectories
were reused without model loading, inference or retraining. This is a posthoc
comparison on120already consumed worlds, not new independent validation.

## Fixed rule and authority

MDL-2 routing/purchasing and ordinary4hours/site/epoch are unchanged. Book
[2,2,2,2]extra support hours every epoch0..47, with the original2epoch lag;
book zero new hours during epochs48..63 and settle all obligations. Each world
therefore books and receives384extra hours. Idle booked labor and all native
costs remain charged. All four roles share the original feasible resource limits,
physics, tapes and three conditions, but actual support usage differs.

Literal user approval: `是的`, bound to protocol.md and approval-intent.json.
Implementation commit:c6f95a1092696b7aa42a4aecf0e932ecefce7f31.
Execution commit:4e4141d642588dc88851632614882e3d71aecd8a.
Frozen file SHA256:902d9be3604466c52b19b2c603c9141ae0594c2a62ec6894759ab27b0bd474f9.
Original626sources plus8new files and484inputs (480tapes/summaries plus4authority/
readout inputs) were verified at entry and completion. Original results and
model seals remain unchanged. PID33599/PPID30450 and the full runner command
were observed with new progress boundaries; the execution tool then returned0.

## Four-role outcome

All entries are per-world means across5blocks x3conditions x8worlds. Cost is
the native synthetic objective, not calibrated dollars. Losses/deliveries are
simulated patient counts, not clinical observations. Percent savings uses the
ratio of equally condition-weighted mean costs, not the mean of world ratios.

| Role | Mean cost | Saving vs fixed MDL-2 | Patient losses | Deliveries | Applied extra hours |
|---|---:|---:|---:|---:|---:|
| MDL-2 fixed2 | 40,140,047.42 | reference | 157.1250 | 130.5917 | 384.0000 |
| H8 | 39,611,487.34 | 1.3168% | 163.3250 | 124.3917 | 304.1278 |
| Value-TD graph-final | 39,274,420.79 | 2.1565% | 151.4500 | 136.2667 | 376.8592 |
| SAC graph-final | 40,997,535.97 | -2.1362% | 166.5833 | 121.1333 | 328.4793 |

H8 reduces mean cost but increases mean losses relative to fixed MDL-2; it is
not uniformly superior. Value-TD has favorable mean cost and loss differences
against fixed MDL-2. SAC is worse on both means. Fixed MDL-2 is not a deliberately
weakened version of H8: it buys more support and exhibits a cost/service tradeoff.

## Paired uncertainty and harms

Positive cost saving favors the candidate; negative extra losses favor it.
Intervals are descriptive95% hierarchical bootstrap intervals,2000draws,
joint training-block resampling across conditions and independent world
resampling within each condition. They are not multiplicity-adjusted tests.

| Candidate vs fixed MDL-2 | Cost saving [95% interval] | Extra losses [95% interval] | Positive cost blocks | Cost-harmed worlds | Patient-harmed worlds |
|---|---|---|---:|---:|---:|
| H8 | 528,560.08 [202,162.53, 892,393.55] | 6.2000 [3.7915, 8.6421] | 5/5 | 43/120 | 70/120 |
| Value-TD | 865,626.63 [549,502.26, 1,180,690.95] | -5.6750 [-7.3504, -3.9998] | 5/5 | 34/120 | 26/120 |
| SAC | -857,488.56 [-1,645,302.26, -277,870.72] | 9.4583 [3.0748, 18.1167] | 0/5 | 88/120 | 104/120 |

Value-TD block-mean cost savings are639029.74,1209886.69,810832.77,792895.06,
875488.88. Corresponding extra losses are-5.5833,-8.0417,-4.6667,-4.7083,-5.3750.
These block averages do not imply that every world or block-condition improves.

| Condition | H8 saving / extra losses | Value-TD saving / extra losses | SAC saving / extra losses |
|---|---|---|---|
| No change | 3.0274% / +3.150 | 3.7941% / -8.875 | -1.1635% / +8.175 |
| Persistent unannounced change | 1.9300% / -0.875 | 2.1747% / -6.800 | -3.5542% / +9.350 |
| Fast independent fluctuation | -0.9766% / +16.325 | 0.4937% / -1.350 | -1.7862% / +10.850 |

Under fast fluctuations, Value-TD cost saving has interval[-443287.12,865050.39]
and extra losses[-5.15,2.475], both spanning zero. Its favorable overall result
does not establish a reliable gain in that condition. All6pairs,15block-condition
cells/pair,5blocks/pair and individual world differences/harms are retained in
payload/comparison.json, including the unfavorable results.

## Original comparison preserved

Value-TD vs H8 remains0.8509% mean savings, cost interval[-26348.35,677834.28],
4/5positive cost blocks and-11.875extra losses. SAC vs H8 remains-3.4991%
savings and+3.2583extra losses. Neither passed the original frozen H8 screen.
The original pair objects, interval seeds and numerical results are retained.
There is no retrospective replacement of that primary endpoint or new winner.

The supplement's primary operational comparator is fixed MDL-2; H8 remains
the original primary and the supplement's strong comparator. Comparator choice
followed inspection of the previous test. These data cannot prove separate
message-passing or RL contributions; prior graph/self-only and final/initial
cost intervals still cross zero. No E1 calibration, clinical safety, deployment
or publication-readiness claim follows from this synthetic whole-controller test.

## Cost, time and compute

All14native cost components, support requests/commitments/application, recorded
waiting/turnaround metrics, world/block/condition values and evaluation compute
are retained in the machine readout. Delay means are means of epoch-reported
metrics, not reconstructed patient-level average waits.

Mean whole-world elapsed seconds: fixed MDL-2 0.8505; H8 37.0225; Value-TD
36.9878; SAC4.1467. These include I/O and were not a synchronized hardware
benchmark; they are not per-decision latency or total training costs.

New actual counts:120constructions+120construction resets;5760control and1920
settlement steps=7680native steps;7920native operations. Actor/critic/value and
total optimizer steps,fit batches,example presentations,neural forwards,
planner calls/model steps,filter transitions and clones are all0.
Reused H8/Value-TD each have2211840planner model steps; Value-TD/SAC each have
5760neural forwards; each reused role has768000filter transitions. These are
evaluation counts only; original training costs are not erased or charged anew.

Validation reused:12focused tests passed and fullcompileall passed before
launch; original23tests and environment checks reused. No extra native smoke,
fit, retry, resume or model tuning was added.

## Preservation and next decision

Single archive:results/capacity_fixed_reference_20261008/archives/payload.tar.gz.
998members,56875091bytes,SHA256
02ae5667583b42182f2e8dec9a621e247e4de354619035ba3c6dd0b1908e8fa7.
Every member was read and verified at creation, and source inventory remained
unchanged. No old experiment was rearchived. This narrative is outside the
immutable scientific payload and accompanies its archive/manifest/authority.
Destination-copy verification is recorded separately in dropbox-handoff.md;
cloud sync and collaborator access must not be inferred from a local copy.

Supported next direction: prioritize Value-TD over the current SAC recipe for
a separately scoped improvement study, retain fixed MDL-2 and H8, and address
robustness under rapid variation. The observed subgroup is hypothesis-generating,
not a new tuning/selection set. Diagnose causes on training/development data;
any changed method needs prospective design and new untouched evaluation worlds.
No performance tuning or further scientific execution is authorized here.
The old gcn-rl monitor remains PAUSED; no external messages or remote Git.
