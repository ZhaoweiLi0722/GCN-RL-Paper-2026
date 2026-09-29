# N5 readout: parameter control and separate attribution questions

## Outcome

The count-only preflight passes on the locked N4 engineering input schema.
The small-development comparison design is documented in `protocol.md` and
the companion JSON, with unresolved launch evidence left explicit. This is not
an executed learning study or a fully calibrated scientific preregistration.

| Component | Graph width 64 | Flat width 68 | Absolute gap / graph |
| --- | ---: | ---: | ---: |
| Actor | 6,512 | 6,472 | 0.6143% |
| Critic | 6,565 | 6,529 | 0.5484% |
| Gate | 7,280 | 7,288 | 0.1099% |
| Total | 20,357 | 20,289 | 0.3340% |

Among all 256 flat widths, 68 is the only width satisfying both every-component
and total <=1% requirements. All candidates are retained in the report. The
old width-8 graph family has no eligible common flat width under these strict
limits; the tests retain that negative result rather than relaxing the bound.
Selection used module shape/count only, not rewards or performance. Graph width
64 and search range were fixed in the committed protocol before recording.

Analytical counts match actual instantiated modules. Every counted parameter
participates in a backward graph; no unused padding was added. Some gradients
can be zero, including upstream gate gradients under zero-logit initialization.
This is not an actor-loss/optimizer correctness test or a useful-gradient claim.
All weights remained unchanged, gradients were cleared, and global torch RNG
was preserved. Counts exclude target copies and the engineering models are
frozen outside the bounded differentiation check.

Physical/self-only graph models have identical weights, architecture and raw
numeric inputs. Only neural adjacency differs. The flat outputs are invariant
to changing that adjacency. This creates the software basis for an isolated
message-operator contrast; it does not establish any performance difference.
Graph versus flat still changes representation/depth, despite controlled counts.

## Protocol decisions

- Primary question: same pretrained graph policy with online actor/critic
  updates versus its tensor-equal frozen fork. Both retain causal history and
  the same estimator interface; frozen does not mean nonadaptive actions.
- Secondary questions: physical versus self-only messages, then a common-input
  parameter-controlled graph/flat comparison. Do not conflate these effects.
- Required practical comparators: competent frozen history policy, adaptive
  rule and online identification + MPC, with equal information and disclosed
  data/model-query/latency budgets.
- Proposed objective: negative complete step cost, gamma=1 and one-step TD for
  a finite fully settled episode. Numeric reward scale, liabilities, actual
  task and its closure still need a lock. This proposal is not applied to N4
  or historical evidence; a continuing task requires a different preregistered
  objective before collection.
- Primary cost includes exploration/adaptation; unchanged worlds test retention.
  Independent pretraining seeds and change worlds, not time steps/checkpoints,
  are the replication axes. Practical margin, uncertainty, guardrails and caps
  must be specified before the retained outcomes are collected.

The widths above are not a proposed clinical-model size or a repair of historical
613k-parameter agents. A different task/input/action schema requires its own
count/operator check. Do not load old weights into these new heads.

## Evidence and validation

- Protocol/config commit: `a5ad5ff`.
- Execution commit: `9cc32621b33debd02f8e650117247b33fcfd71b8`.
- Report: `reports/2026-09-29-prospective-development-design/preflight.json`.
- Report SHA256: `4e4caf4018e44301add56f79f790cfcc8ca4ab6e7ff49fb9dab6134958109a2c`.
- Immutable input: N4 report hash
  `67c35b480a7ca647f791e6a401660eb4ced8d413ef7c3421ac68b7fab1ec3c61`.
- Seven audited source hashes reconcile; repeated recorded runs are byte-identical.
- Fifteen new tests; the combined 95-test input/replay/head/historical-contract
  suite passes. Full Python compilation and diff checks pass.
- Zero environment steps, optimizer updates, new performance evaluations,
  scientific seed allocations, remote changes or formal-holdout queries.

The preflight cannot authorize a launch: all five execution permissions must
be explicitly false. Eight named missing evidence requirements remain in the
output. Replacing nulls with an unverified "approved" string does not pass
them or grant execution permission.

Reproduce from the persistent worktree using a fresh destination:

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_prospective_design
"$PY" -m evaluation.audit_prospective_design \
  --config experiments/configs/prospective_development_preflight_20260929.json \
  --output /private/tmp/prospective-design-new-check.json
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Reports include execution commit metadata, so whole-file repeat equality
requires the same source commit/runtime. Never overwrite the archived report.

## Next work

The next independent engineering task is the actual learner contract: actor
gradients through deployed requests, detached bootstrap targets, fixed critic
during actor updates, frozen-arm invariance, replay/optimizer/RNG serialization
and exact interruption/resume. Counting parameters and head backward participation
do not discharge that task. Keep it a separately bounded mechanics verification,
not a performance campaign.

Scientific launch remains gated by the operational E1 bundles, complete objective
and closure, replicated safe leverage/learnability, competent frozen/adaptive
baselines, actual task/model/gate lock, fresh streams/budgets/analysis and launch
authorization. These are real missing inputs, not coding defaults. The current
manuscript's numerical results and online-null conclusion are unchanged.
