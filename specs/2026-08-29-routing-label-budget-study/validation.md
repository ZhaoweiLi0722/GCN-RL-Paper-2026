# Validation

## Test module (fixed name): `tests/test_routing_label_budget_study.py`

Written and green before the pool collection runs.

### Config discipline

- All state, world, and analysis seeds are drawn from the `98xxxxxx` family
  and are rejected if they collide with: the formal holdout (`91100000`), dev
  CRN streams (`94000000`/`94100000`), any exploratory overtime family
  (`96xxxxxx`), the reserved confirmation set (`97xxxxxx` — enforced by the
  existing code-level guard), or the G1 families (`101000000`, `103000000`,
  `107000000`).
- The config declares `enable_specimen_routing: true` and must NOT enable
  overtime control or the dormant production throttle.
- The reserved scenario `routing_regional_drift` is rejected.

### Action family

- Exactly five arms per state; the four shifted arms differ from the anchor
  only in the specimen net-flow block, by exactly ±0.05/±0.10 raw units,
  clipped to [−1, 1].
- The distinct-execution filter drops a state unless all five arms produce
  distinct executed integer routings there; a fixture state where two arms
  collapse to the same execution is asserted to be dropped.

### Pool integrity

- Pool completeness: every surviving (state, arm, world) triple appears
  exactly once; rows are finite and unique; per-row scenario provenance
  assertion (live environment interrogation, as in every screen since the
  Stage E2 guard fix).
- CRN exactness spot-check: for one state and one world, two different arms
  finish with identical RNG bit-generator state and identical cumulative
  enrollment — the verified pairing property, asserted rather than assumed.

### Offline analyses

- Agreement estimator: on a synthetic pool with a known dominant arm and zero
  world noise, `A(k) = 1.0` for all k; with heavy world noise and two
  near-tied arms, `A(1) < A(32)`.
- Disjointness: the two groups in every split share no world seeds.
- Sequential-halving replay: consumes only pool worlds, never exceeds the
  stated budget, and on the noise-free synthetic pool picks the dominant arm.
- Reading rule: unit tests pin all three classification branches, including
  the boundary (`A(32) = 0.70` → underpowered branch; `A(32) = 0.599` with
  flat tail → fundamental branch).

## Run discipline

- `PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .`
  plus the module above, green, before pool collection.
- Pool rows and the analysis summary are written with SHA256 digests beside
  them, in a new output root
  (`results/routing_label_budget_study/`); the curated summary is copied to
  `experiments/evidence/routing_label_budget_study/`.
- The classification line is printed by the analysis script from the reading
  rule, never hand-assigned.
- On completion (any branch), a change-control entry is appended to the locked
  execution plan recording the classification and its consequence — including
  the `inconclusive` branch, which must be recorded rather than silently
  extended.
