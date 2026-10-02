# Paired-cohort recovery: behavior changed, primary outcomes worsened

2026-10-02. Independent saved-result synthesis only; no new scientific calls.

## Answer and decision

Simulator-label-assisted one-step improvement changed the greedy policy's
closed-loop behavior, but did not improve the complete patient-cohort comparison.
Against BOTH own-frozen and continued-BC, block 60 was exactly unchanged;
blocks 61 and 62 had higher mean cost and more patient losses. This is no longer
merely an unchanged-greedy-policy result: movement occurred without the required
performance benefit. It does not establish that RL generally fails.

The original protocol requires strictly lower mean cost in every block for both
primary contrasts, with no observed block-level increase in patient losses.
Both conditions fail. The saved decision `close_one_shot_mechanism` is therefore
consistent with the preregistered screen. **Close this one-shot recipe without
promotion to independent confirmation or another baseline/epoch variant.**
No new experiment, search, gate, or automatic follow-on is recommended here.

## Primary comparisons

There are 216 complete 63-step evaluations: six controllers, three trained
blocks, twelve paired worlds per block. Own-frozen and continued-BC have
identical requested/executed traces and cost/patient outcomes in all 36 worlds,
so the following paired-cost differences apply separately to BOTH contrasts.
Counts are mean differences per evaluated cohort, not individual-patient risks.
Cost is in the simulator's raw objective units, not calibrated clinical value.

| Block | Mean cost difference | Relative cost | Losses | Completions | Waiting patient-steps | Expiry losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 0 | 0% | 0 | 0 | 0 | 0 |
| 61 | +423,034,113.55 | +14.8065% | +656.00 | -656.00 | +700.42 | +558.08 |
| 62 | +15,650,587.95 | +0.5439% | +29.75 | -29.75 | -76.75 | +15.17 |
| Equal-block mean | +146,228,233.83 | +5.1168% | +228.58 | -228.58 | +207.89 | +191.08 |

Relative percentages average the three block-specific percentages; they are
not the percentage ratio of pooled mean costs. Block 61 costs and losses both
worsened in 12/12 paired worlds. Block 62 costs worsened in 10/12 and improved
in 2/12; losses worsened in 10, tied in one, and improved in one. Block 60 tied
in all twelve. Neither an isolated favorable world nor a mean across blocks
can replace the required all-block screen.

## Behavior and patient tradeoffs

In the 52-step decision prefix, requests differed from either primary control
on 0/624 steps in block 60, 624/624 in block 61, and 464/624 in block 62.
Executed actions differed on 0, 612, and 462 steps, respectively. Thus 1,088
of 1,872 prefix requests and 1,074 executed actions changed. Paired-cost selected
the reference class on 624, 0, and 238 prefix decisions across the three blocks;
own-frozen and BC selected it on all 624 per block. These are observed
closed-loop trajectories, not same-state causal comparisons after divergence.

The losses are not offset by a mean cost saving. Block 61's cost increase
includes +328.00 million patient-loss cost and +66.00 million expiry cost,
despite -16.27 million reagent-purchase cost. Its mean waiting-expiry losses
rise by 629 while transit-expiry losses fall by 70.92. Block 62 saves 5.10
million in tail cost, but adds 20.75 million in prefix cost and still loses
29.75 more patients. All terminal active counts are zero, so the comparison
includes settled cohorts rather than leaving unresolved patients out.

Average turnaround among completed patients falls by 0.202 and 0.107 simulator
time units in blocks 61 and 62, respectively. With fewer completions and more
losses, that changing completed-patient population is not evidence of overall
patient benefit. The block-62 waiting reduction likewise does not override its
loss/expiry deterioration. Waiting patient-steps measure occupancy, not the
waiting time experienced by each completed patient.

Saved cohort PPO and R4 match the primary controls' observed actions and
cost/patient outcomes. Against full MDL-2, paired-cost has +4.4496% equal-block
mean relative cost and +199.31 losses per cohort; that secondary comparison
does not rescue either primary failure or establish equal-budget PPO superiority.

## Precision and scope

The saved 10,000-draw paired block-then-world descriptive 95% intervals are
[0, +14.4488%] for mean relative cost and [0, +648.45] for mean losses against
either primary control. Three trained blocks provide weak population-level
inference; these are not clinical noninferiority/superiority tests. Failure of
the fixed development screen does not require a significance claim.

Training labels use simulator state and fixed-reference continuation at 36
snapshots from only twelve context worlds. This replacement learning package
is not model-free learning, online deployment adaptation, an isolated GCN
effect, or proof that the reward is wrong or that no headroom exists. Paired
seeds also do not ensure event-aligned randomness after action divergence.

## Evidence and boundary

Source: `results/paired_cohort_improvement_20261002_recovery1/payload/independent-comparison.json`
(216 outcomes; SHA256
`18f149fb4e01056648f5348311b43f7bbb8a0cfd7eb046ecf8c96c9fe6763d35`).
Decision rule: `specs/2026-10-02-paired-cohort-improvement/protocol.md`,
"Complete comparison". I independently recomputed the primary block means
from saved per-world outcomes and recounted prefix requested/executed changes;
they agree with the saved analysis. Bootstrap intervals are reported from the
saved prescribed analysis, not newly resampled here. Primitive raw-cost
reconciliation, current locks, process status, and archive closure remain the
coordinator's responsibility. Presence of this comparison is not a claim that
launcher/archive closure has completed. No models, simulations, optimizer
steps, retries, archives, remote actions, or source changes were performed.
