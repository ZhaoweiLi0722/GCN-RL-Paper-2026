# Planner-Tail Recovery1: Completed Comparison

2026-10-05 UTC. The approved remaining-only package completed with supervisor
and child exit code 0 in **140.53 seconds**. This completes the original sample,
not a new independent replication. The original timeout remains a failed attempt.

## Execution And Preservation

- User's exact confirmation and bounded context: [approval-intent.json](approval-intent.json).
- Execution commit: `f9f7a40407e926bb3063322ab78b149b8307b20c`.
- Packet: `a31517a9a37aa15301832245f5c21031662c92dc9d6f6531bf6dd9a7f54bf6f3`.
- Root: `results/capacity_planner_tail_20261004_recovery1`.
- New work: 54 native steps (38 control, 16 settlement), 960 diagnostic forwards,
  29,184 planning epochs, 5,400 filter transitions, zero optimizer updates,
  zero new world starts. No completed environment steps or training replayed.
- Complete total: 120 reference plus 360 evaluation trajectories; 30,720 native
  steps, 31,680 native operations, 11,520 inherited value updates, zero actor
  updates, 15 final models plus five ancestor bindings, 30,000 forwards.
- The interrupted 768-epoch forecast reservation remains consumed; its
  replacement is separately charged. Cumulative ordinary planning epochs are
  9,954,048, plus 374,400 training-tail prediction steps. No budget refund.
- Original first ten raw rows preserved byte-for-byte; decompressed-prefix SHA256
  `b218afb3627557e8e41a553ec343f46745cef599385365c3c95d88fe4007666f`.
- No failure markers; stdout/stderr empty. Supervisor 8547 and child 8561 have
  exited; no matching scientific Python process at the final process check.

Independent JSON/gzip analysis reconciled all 480 complete raw trajectories and
30,720 rows: epochs 0..63, component cost sum, reward sign, summary cost, final
patient losses and settled status. No discrepancies. All 15 contrasts have
three conditions, five blocks and 20 paired worlds per condition. The companion
[saved readout](terminal-saved-data.json) retains every contrast, interval, block
saving and harm count; no incomplete-world imputation was used.

## Scientific Result

Persistent-change results below use mean paired-world percentages; absolute
savings are comparator minus forecast-tail TD, in millions of objective units.
Extra losses are TD minus comparator. Intervals are the prespecified descriptive
five-block/two-level bootstrap intervals, not multiplicity-adjusted confirmation.

| Comparator | Savings M [95%] | Mean paired savings | Extra losses/world |
|---|---:|---:|---:|
| Plain H8-MPC | 0.948 [0.432, 1.484] | 2.562% | -12.55 |
| Existing frozen GCN | 0.246 [-0.181, 0.680] | 0.679% | -3.40 |
| Same-data observed TD | 1.324 [0.365, 2.430] | 3.671% | -13.80 |
| Same-tail MC | -0.025 [-0.252, 0.169] | -0.261% | -1.15 |
| Plain H16-MPC | 0.513 [-0.110, 1.089] | 0.924% | -6.15 |

**The primary conjunction fails.** TD improves over H8 and observed TD in all
five persistent-change block means, but only four blocks improve over frozen
and that cost interval includes zero. TD versus MC is not distinguished. This
supports a favorable learned-controller comparison with H8 in persistent change,
not an established new-learning increment over the existing frozen controller
or a uniquely beneficial TD target mechanism.

No-change/fast-noise mean TD/H8 savings are 1.401%/1.450%, with cost intervals
including zero. TD/frozen is -0.142%/+0.224%; neither supports a robust increment.
The completed secondary fast-noise TD/H16 contrast favors TD by 1.089 M
[0.398, 1.759] M, 2.346% mean paired savings and 11.50 fewer losses/world.
Its five block mean savings are positive, but it does not rescue the primary.

Harms remain visible: persistent TD/H8 has higher cost in 3/20 worlds despite
favorable block means; TD/frozen has more lost patients in 5/20. Fast-noise TD/MC
has +0.45 mean extra losses despite a positive mean absolute saving. Persistent
TD/H8 uses 53.60 more applied flexible hours/world and differs at 757/960 control
boundaries: lower patient-loss/expiry cost trades against more reagent and labor
cost. Median decision latency is 0.699 s for TD, 0.701 s for H8 and 1.414 s for
H16; H16 uses twice the planning queries. Full negative and positive cells are
in the companion JSON, not selected away.

## Mechanism And Next Direction

The independent [diagnostic decision](diagnostic-decision.md) uses only the
already-saved training-root data: target-best agreement rises from 34.58% for
frozen to 42.08% for TD and mean in-model regret falls in all five blocks.
However, TD/MC select the same candidate at 230/240 roots. These are public-model
adaptive-continuation labels, not native patient counterfactual ground truth.
MC is model-based direct-return policy evaluation, not necessarily a non-RL arm.

The next decision-changing hypothesis is **continuation-policy alignment**:
would a terminal estimate of the deployed receding-horizon policy's own future
cost produce an incremental native benefit over `existing_frozen`, where an
estimate under a different adaptive continuation did not? The present ranking
gap motivates the question but does not establish causality. Keep the objective
and patient accounting unchanged; the current evidence does not justify reward
weight search, more epochs of the same objective or outcome-driven expansion.
This is one proposed question, not a launched or numerically approved experiment.
Future authorized packages should execute their routine preparation, training,
evaluation and closure continuously without another launch confirmation.

The study remains synthetic and uncalibrated against E1 operational data. Five
historical model initializers and block-0 mixed predictor provenance are retained.
There is no isolated GCN edge attribution, deployment-time online adaptation,
clinical-safety claim, or guaranteed publication outcome.

## Archive And Handoff

The runner verified every archive member: 2,987 files, 471,059,815 compressed
bytes. Archive SHA256 was independently rechecked:
`f2ee81d7542c78ac296595a849c41c9f42085572670fc6bf933419ad0b26423e`.
The complete comparison SHA256 also matches:
`df3585761af0036dbaeb163977813991b5822f6c12fb910b0c99eead89b1152d`.
Evidence is in `launcher/terminal.json`, `payload/comparison.json`,
`payload/artifact-inventory.json`, `payload/completion.json`,
`archive-receipt.json` and `archives/payload.tar.gz` under the new root.
No second archive, Dropbox export, cloud-sync or Howard-access claim.

Only status/readout/manuscript files changed after execution. Reuse the 29
passing focused engineering tests and full compileall; no new scientific source
or dependency changes. Manuscript source checked, PDF not compiled because no
TeX engine is installed. Godel's finite diagnostic task completed and closed;
prior efficiency advice reused. The finite package is consumed and `gcn-rl`
remains PAUSED; no background training, automatic retry or new experiment.
