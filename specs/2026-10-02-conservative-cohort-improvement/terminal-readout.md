# Conservative Two-Round Cohort Improvement: Terminal Readout

2026-10-02. The single approved experiment completed with exit code 0.
Both original processes, supervisor 12670 and child 12683, exited. This is a
completed negative scientific comparison, not an engineering abort. The
prespecified decision is `close_two_round_mechanism`; no automatic follow-on.

## Result

Both primary contrasts, paired-cost minus same-start frozen and paired-cost
minus BC-CONTINUE, are exactly zero in all 36 paired worlds across blocks 60/61/62.
Cost, loss, completion, expiry, waiting and turnaround differences are zero.
Each contrast has 0/1872 changed prefix requests and 0/1872 changed executed steps;
all 36 final simulator states match. R4 produces the same actions and outcomes.
The result concerns observed greedy behavior, not proof that trained weights or
probabilities stayed unchanged, nor proof of global policy optimality.

| Block | Cost change vs frozen | Cost change vs BC | Mean loss change, both |
| --- | ---: | ---: | ---: |
| 60 | 0% | 0% | 0 |
| 61 | 0% | 0% | 0 |
| 62 | 0% | 0% | 0 |

The development screen required at least 1% equal-block mean relative cost
reduction, at least 2 cost-improved blocks and no block-level mean loss increase
for both primary contrasts. Both produce 0% reduction and 0/3 improved blocks.
Only the observed loss-direction condition is met; the complete screen fails.

Against full MDL-2, the shared frozen/R4/BC/paired behavior gives:

| Block | Relative cost change | Mean loss change | Mean completion change |
| --- | ---: | ---: | ---: |
| 60 | -0.678738% | -31.750000 | +31.750000 |
| 61 | -0.234781% | -15.250000 | +15.250000 |
| 62 | -0.262079% | -12.916667 | +12.916667 |
| Equal-block mean | -0.391866% | -19.972222 | +19.972222 |

These are inherited advantages, not incremental benefit from this training.
They accompany 47.611111 more waiting-patient-steps per cohort and 0.092076 more
turnaround-time units among completed patients. Mean prefix cost savings of
17.044197 million model cost units are partly offset by 6.084849 million higher
tail cost. All directions, including the waiting/turnaround trade-off, remain
in the record; no clinical superiority or noninferiority claim is supported.

## Execution and Evidence

- Execution commit: `880defba211bf0e444bb8781c8be5735fb0cabe0`.
- Frozen implementation: `c9d5290690ee6b49fa67984e982fda6bfa32fd4b`.
- Packet: `c571729b094dd02e6034434273d04fc72c23c9db31a0d2f0e69fdc6c01bd0d66`.
- Preflight 3/3; collection 12/12 cohorts and 36/36 states; 800 actual branches
  complete under the 864-branch bound, with aliases reducing the count.
- Paired 6/6 and BC 6/6 round-fits; 768/768 actor updates, 0 critic updates.
- All 9 final models sealed before tests; 180/180 evaluation cohorts complete.
- Environment 46,362/49,626 calls; elapsed 12,973.902 seconds of 28,800 allowed.
- No stderr, episode failure, forced kill, retry or unapproved follow-on.

The independent coordinator readout reparsed all 180 cohorts / 11,340 steps directly
from raw JSON/JSONL, verified 1,080 referenced files, recomputed primitive costs,
registry outcomes, paired differences and action changes without invoking the
campaign comparator, model inference or the simulator. Maximum per-step
floating-point difference between primitive cost sums and logged cost was
1.49e-8 model cost units. All 180 terminal patient registries are settled.

Readout: `reports/2026-10-02-conservative-cohort-independent-readout.json`.
Saved comparator SHA256:
`42a55d858344a10b2a81cab956c158912dc06d53cc83f028ec63b3a624d240a5`.
Terminal/hash/archive handoff:
`reports/2026-10-02-conservative-cohort-terminal-verification.json`.

All 482 current source/input/scope locks match. Existing payload archive
(10,137 files, 2,572,706,004 bytes) and launcher snapshot archive (853 files,
7,357,544 bytes) were independently read back member by member; archive-byte
hashes, manifests, receipts and unchanged source-tree hashes match. No new
archive was created. The launcher snapshot deliberately excludes final closure
and supervisor/terminal receipts; those remain at the original run root.
No Dropbox export, cloud-sync verification or Howard-access claim. The same
`gcn-rl` automation is tool-confirmed PAUSED and retained, not deleted.

One finite independent agent (Euler) interpreted the saved comparator and
confirmed the null primary contrasts, secondary trade-offs and decision. It
did not duplicate raw-file verification, create a gate or run science; it was
closed after delivery. Prior efficiency guidance was reused. Code unchanged;
the 46 engineering tests and prior compileall remain applicable. This handoff
validates only documentation JSON/consistency and performs no scientific rerun.

## Interpretation and Next Decision

Defensible statement: under this fixed two-round configuration, simulator-label
assisted actor training produced no incremental greedy-action, cost or patient-
outcome benefit over same-start frozen and BC controls. Three training blocks
support descriptive inference; 180 cohorts are not 180 independent training runs.
This is not evidence of statistical equivalence, isolated GCN or KL effects,
model-free algorithm superiority, deployment online adaptation, or all RL being
ineffective. The result does not establish an incorrect reward or a causal
explanation for the unchanged greedy choices. No training-loss improvement is
reported as patient benefit.

Close this two-round recipe without more epochs, KL variants or added samples.
Preserve the current paper's truthful attribution and report this null result.
The next scope decision is whether to authorize preparation of a separate
dynamic-capacity/continuous-resource study aimed at an identifiable adaptation
benefit. No new protocol execution, reward revision, scenario, model search,
formal holdout, remote action or Dropbox export is authorized by this readout.
Stage E stays closed; no Howard approval is asserted. Pause the same `gcn-rl`
monitor after terminal handoff; preserve its visible record rather than delete.
