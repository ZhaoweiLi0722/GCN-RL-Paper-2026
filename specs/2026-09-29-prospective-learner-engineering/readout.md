# N6 readout: DDPG update-boundary mechanics

## Outcome

The opt-in CPU float32 kernel passes bounded synthetic update and fresh-process
resume checks. This is software acceptance, not a learning-performance study.
The existing environment, historical agents/results and N4/N5 source are unchanged.

| Case | Continuous | Interrupted + fresh-process remainder | Result |
| --- | ---: | ---: | --- |
| Graph64, physical messages | 6 updates | 3 + 3 updates | Exact |
| Graph64, self-only messages | 6 updates | 3 + 3 updates | Exact |
| Flat68, physical metadata | 6 updates | 3 + 3 updates | Exact |

Each recorded matrix executes 36 kernel updates / 72 Adam steps. A second matrix
reproduces all seven recorded JSON files byte for byte and all nine checkpoint
payloads tensor for tensor. Thus the two matrices together execute 72 kernel
updates / 144 Adam steps; unit tests perform additional bounded updates. None
is an independent scientific replication. No simulator step was taken.

For each case, uninterrupted versus restarted execution agrees on every sampled
index, loss, bootstrap target, gradient norm, final request and complete semantic
state hash. This includes Adam first/second moments, target networks, replay
contents/provenance, wraparound position and owned sampling RNG. After inserting
12 records into a capacity-8 ring, the cursor is 4. All optimizer counters are 6.
The fresh workers each exit 0 with empty stderr.

Both actor and critic actually change, with finite nonzero gradient norms in
this fixture. Gate, target gate and the frozen reference do not change. The
reference evaluates the same invented inputs without receiving optimizer steps;
this is not a trained, competent frozen baseline. Initial gate logits are zero,
so a separate fixed nonconstant-gate numerical test checks its input derivative.

## What the implementation verifies

- The critic uses the typed first submitted request, not a later executed action
  or current actor proposal. Targets use the endpoint's own state/anchor and the
  same deployed target-policy transform; bootstrap is detached.
- Actor differentiation uses the bounded residual, fixed sigmoid gate and final
  clipping, numerically identical to N4 deployment. A finite-difference check
  verifies the nonconstant gate's proposal dependence. Fixed gate parameters do
  not imply a detached proposal input. The configured policy differentiates that
  input; the optional detached alternative is explicitly tested, not silently used.
- The critic is fixed during actor differentiation. Only actor/critic targets
  receive one Polyak update per successful update. Zero, partial and full tau
  agree with an independent parameter formula.
- Failed actor steps after the critic step, nonfinite loss/gradient and nonfinite
  post-step parameters roll back modules, moments, counters, replay and RNG.
  Frozen mode, insufficient replay and update caps reject without mutation.
- Checkpoint loading uses tensor/primitive-only loading, a payload checksum and
  exact semantic/model/optimizer manifests. Malformed state is validated on a
  private copy. A zero-update checkpoint cannot disguise changed initial weights.
  Saving refuses to replace an existing file. Legacy checkpoints are not migrated.

This is deliberately a small isolated update kernel, not a new registered agent
or campaign framework. Existing neural heads, matched-input views and typed-return
adapter are reused. The kernel owns its clone, Adam instances, typed-window ring
and sampling RNG. The old checkpoint API would require unrelated legacy replay,
noise and environment members; no dummy objects were introduced to emulate them.

## Evidence

- Protocol/config commit: `1822c1f996e3c0ff62927d079bd3e78b93a93d07`.
- Recorded execution commit: `1b5ebc21b32a86d64834ce14ddbf451e39961208`.
- Report: `reports/2026-09-29-prospective-learner-engineering/check.json`.
- Report SHA256: `bff185e60ae0d57d155082088731fbc40245f6b0c309a5349de4af6c4fd45bca`.
- Ten source hashes and both immutable N4/N5 input hashes reconcile.
- Eighteen new unit tests; 113 combined focused tests pass, as do full Python
  compilation and diff checks. The recorded run and repeat passed on first execution.
  The post-run test addition covers deliberate nonfinite-gradient rollback; no
  implementation or recorded outcome was changed after seeing results.
- Runtime: Python 3.9.6, torch 2.8.0, NumPy 2.0.2, CPU, one torch thread. CPU is
  the explicit engineering device, not a fallback from a required MPS run.

`checkpoint_inventory.json` records an additional read-only direct tensor
comparison, file sizes and hashes. The nine small `.pt` files remain locally
under the persistent report directory (about 2.7 MB total directory footprint).
Existing repository policy ignores `.pt`; JSON diagnostics/inventory are tracked,
and the binary files are not force-added. A fresh clone can regenerate them with
the fixed script. Binary container hashes need not match between saves; the
tensor/primitive payloads do. No remote push, PR, merge or coauthor message occurred.

Reproduce using a new destination, never the archived report directory:

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_prospective_learner
"$PY" -m evaluation.check_prospective_learner \
  --config experiments/configs/prospective_learner_engineering_20260929.json \
  --output /private/tmp/prospective-learner-new-check
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Full-report byte equality requires the same commit/runtime, because execution
metadata is embedded. The recorded repeat used execution commit `1b5ebc2`.

## Remaining boundary

This checkpoint is only an update-boundary kernel checkpoint. It does not include
an environment, partial episode/collector window, exploration process, GPU RNG
or asynchronous sampler. CPU agreement does not establish MPS/CUDA equivalence.
The next engineering integration must test those live collection/resume states
under an explicitly bounded protocol before a campaign can rely on them.

The fixture inherits only feature dimensions/order from N4; every value and reward
is invented, under new schema/reward IDs. Its specimen-only residual scale is not
a proposed qualified-support staffing action. No clinical meaning or operating
calibration is inferred from it. This does not repair the historical campaign or
justify changing the manuscript's online-null conclusion.

N5's scientific launch dependencies remain open: operational evidence and costs,
complete episode settlement, independently replicated safe headroom/learnability,
competent frozen/adaptive controls, the actual task/model/gate lock, fresh streams,
budgets, analysis and explicit launch authorization. The optimizer/resume requirement
is only partially discharged: this kernel passed; full live-collector resume has not.
Do not mark all eight requirements passed or start a performance campaign here.
