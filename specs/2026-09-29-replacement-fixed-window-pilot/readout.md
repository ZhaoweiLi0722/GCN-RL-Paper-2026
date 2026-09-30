# R3 replacement fixed-window pilot: limited local headroom, not online benefit

## Verified execution

On 2026-09-29 local time, committed source
`195d0eb3f7d095c3f53c211a42bcdbac8fd1e67a` completed the approved single attempt.
The distinct MPS engineering smoke used24 steps and reproduced all six pairs
of two-step clones exactly. Production completed12 states/1,152 logical
continuations in600.139 seconds, exit0, using24,348 actual simulator steps.
There were zero optimizer updates, no second scientific attempt, and no reward,
scenario, cost, sample-size or formal-stream change. No collector remains.

Twelve new/81 related tests, full compileall and diff checks passed. The
stdlib-only verifier independently reconciles costs, chronological lineage,
physical/RNG-state action aliases, selections and validation contrasts. All24
R4 payload hashes and83 pre-existing source locks match. Three replacement
actors and actual nominal-history environment dataclasses were hash-locked.
These are not recovered F1 weights and do not establish baseline competence.

Production roots:

- `results/replacement_fixed_window_pilot_20260929/`
- `results/replacement_fixed_window_engineering_20260929/`
- `reports/2026-09-29-replacement-fixed-window-pilot/verification.json`

## What was estimated

At four times on each frozen actor's fresh trajectory, choose one of six first
requests from8 discovery futures, then evaluate that fixed choice with8 new
RNG starts. All later actions use the same frozen actor/gate. The primary
contrast is against its frozen first action, not a trained policy comparison.
The secondary reference is MDL2-first/frozen-followup, not a full MDL2 rollout.

Costs are incurred through t51, with outstanding obligations saved. States
within a trajectory and overlapping remaining horizons are dependent. The
draw-paired t7 intervals below are conditional, descriptive and unadjusted for
12 comparisons; do not label their signs as confirmatory significance.

Shared RNG initialization is not guaranteed event-aligned after actions change
draw consumption. ID-keyed transport losses retain the original identity;
their probability is0 in this effective config, so that mechanism does not
contribute random loss here. The results still condition on sampled patient
states rather than represent population or clinical-safety inference.

## All prespecified selections

Negative cost/lost-patient deltas are favorable. Cost percent is the paired
validation mean difference divided by the mean frozen continuation cost for
that same state. It is **not** an episode-wide deployable-policy percentage.
Intervals and cost differences use millions of original simulator cost units.
`Map` means the fixed deployment-map witness passed, not network learnability.

| Actor | t | Discovery choice | Map | Cost delta % | Cost delta, M [conditional interval] | Lost patients delta |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | 0 | frozen | yes | 0 | 0 [0,0] | 0 |
| 60 | 13 | specimen +.10 | uncertified | -0.180 | -3.905 [-9.797,1.988] | -9.000 |
| 60 | 26 | specimen -.10 | yes | -0.730 | -10.789 [-16.175,-5.403] | -8.875 |
| 60 | 39 | specimen -.10 | yes | -0.243 | -2.250 [-4.174,-0.327] | +1.375 |
| 61 | 0 | frozen | yes | 0 | 0 [0,0] | 0 |
| 61 | 13 | specimen +.10 | yes | -0.432 | -8.728 [-14.846,-2.610] | -9.500 |
| 61 | 26 | specimen -.05 | yes | -0.132 | -2.178 [-4.672,0.316] | +6.500 |
| 61 | 39 | specimen -.05 | yes | -0.571 | -4.790 [-8.822,-0.757] | -3.250 |
| 62 | 0 | frozen | yes | 0 | 0 [0,0] | 0 |
| 62 | 13 | specimen -.05 | yes | +0.000114 | +0.003 [-6.665,6.670] | -1.125 |
| 62 | 26 | specimen -.10 | uncertified | -0.216 | -3.307 [-8.901,2.287] | +4.750 |
| 62 | 39 | MDL2-first | yes | -0.502 | -4.683 [-6.324,-3.041] | -6.125 |

All t0 choices execute identically. Across192 state/draw groups there are48
groups with1 distinct execution,32 with5, and112 with6. Exactly272 logical
continuations share verified physical/RNG-equivalent tails, saving13,248 steps.
Requested aliases retain their own first-step records; no sample was dropped.

Eight of the nine noninitial selected choices have lower mean validation cost;
five have descriptive cost intervals wholly below0. Three cost-lowering choices
increase mean patient losses:60/t39,61/t26 and62/t26. Two selected corrections
are uncertified by the action-map witness (60/t13,62/t26); each proposal is
within the box but reconstruction differs by0.10. This is not a proof of global
unreachability and does not justify removing/changing the gate automatically.

The three non-anchor correction states60/t26,61/t13 and61/t39 have map witnesses,
cost intervals below0, lower mean losses, more mean completions, higher mean
completion service and lower mean manufacturing-loss rates. Mean cost decreases
are0.730%,0.432%,0.571%; mean completion changes are+6.75,+14.0,+4.5. This is a
**descriptive subset of all12 rows**, not a retroactive safety/efficacy gate.
It contains only two of the three baseline actors. The other interval-below0
clinically favorable state,62/t39, simply selects MDL2-first; that benefit cannot
be attributed to a learned residual or online update.

Per-actor equal-state mean cost deltas are -4.236M,-3.924M,-1.997M. These mix
dependent remaining horizons and are descriptive signs only, not independent
episode replications, an aggregate percent gain, or a learned policy value.

## Decision

1. Preserve the narrow positive finding: there is sampled local, fixed-policy
   cost headroom. Do not infer that the pretrained controller is globally
   optimal or that every action channel has no useful signal.
2. Do not claim online DDPG benefit. A simulator-evaluated discovery selector
   chose these actions; no critic learned this ranking and no actor was updated.
   Original paper/formal conclusions stay unchanged.
3. A next study should first ask whether a **clean critic can generalize these
   fixed-window absolute-cost rankings** using public observation, time and
   feasible action, with the frozen policy as continuation. It needs a new
   prospective protocol, explicit data/compute caps, full-trajectory-disjoint
   training/evaluation states and a clinical admissibility rule decided before
   outcomes. Splitting only future draws on these same12 states is not a
   state-generalization test. No new fitting or collection is authorized here.
4. If a critic passes that independently specified gate, consider a separately
   bounded actor-update versus frozen comparison with equal information and
   adaptive-control baselines. Do not jump directly to another long DDPG run.
5. Do not tune rewards to manufacture a gain. Current cost permits some
   cost/clinical tradeoffs, which need a domain/scientific decision about
   constraints or objective priorities. This pilot does not identify cost
   weights to change. Its positive rows do not make that issue disappear.

Missing original actors, changed continuation definition and the new finite
objective confound comparison with historical G1. This pilot cannot identify
which historical mismatch caused the online null, cannot prove a horizon repair
works, and cannot support a general novelty/clinical-safety claim for the paper.

## Reproduction and preservation

Do not rerun the collector into these immutable roots. Independent verification:

```bash
python -m evaluation.verify_replacement_fixed_window_pilot \
  results/replacement_fixed_window_pilot_20260929 \
  --output /a/new/local/verification.json
```

Source/config/runtime/input identities, selected full states, all first post-step
states, route events, raw step costs, clinical metrics, terminal obligations,
selection locks and24 progress boundaries remain in the raw trees. The archive
and Dropbox receipt are recorded separately after verification. A local sync
folder copy does not establish off-device cloud sync or Howard's access.
