# Validation

## Local Validation Commands (Stage E1)

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
python3 -m unittest tests.test_overtime_control_equivalence
python3 -m unittest tests.test_overtime_control_contract
python3 -m unittest tests.test_overtime_heuristics
```

Test modules are created in E1; names above are fixed by this spec.

## Mandatory test content

### `test_overtime_control_equivalence` (invariant 1)

- Construct the current 20-clinic patient environment and the flag-off extended
  environment with identical seeds; run ≥3 full episodes with identical action
  sequences (noop, random, and MDL-2 policies).
- Assert bit-identical observations, rewards, cost components, info dicts, and
  terminal RNG state at every step. Any tolerance-based comparison fails review;
  the comparison is exact.

### `test_overtime_control_contract`

- Flag-on: `action_size == 5n`; observation and `node_features` widths grow by
  exactly the documented per-facility block; `noop_action` maps to `u_ot = 0`.
- Cost hook: `overtime_cost` present in `_operating_cost_components`, equal to
  the closed-form convex expression **on the committed surge** (nonzero for
  `u_ot > 0` even in epochs where the integer production count is unchanged),
  and included in the additive total (decomposition sums to the reported cost).
- Continuity: for a fixed state, committed surge and overtime cost are monotone
  and continuous in `u_ot` on a dense grid; marginal cost is strictly positive
  for `u_ot > 0`.
- Action mapping: raw `-1` maps to `u_ot = 0` and raw `+1` to `u_ot = 1`;
  `base_capacity` is `initial_idle_bioreactors`, not `max_idle_bioreactors`.
- Fleet conservation (borrowed-capacity accounting): drive an episode with
  overtime binding (production > physical idle) and assert at every step that
  `idle + in_process - outstanding_overtime` equals the physical fleet
  (adjusted for executed capacity transfers); assert completing overtime lots
  repay the outstanding counter instead of inflating the idle pool, and that
  the counter is nonnegative and reaches zero after a quiet tail.
- Fatigue (flag-on): decay recursion matches the spec; flag-off leaves no
  fatigue trace in state or cost.
- Decision B dormant: `enable_production_throttle=False` yields no third block
  and no behavior change, and `enable_production_throttle=True` raises a
  validation error naming this spec's activation requirement.

### `test_overtime_heuristics`

- MDL-2-OT reduces to plain MDL-2 when `max_overtime_fraction = 0`.
- uMYO-OT surges only under its urgency condition.
- Static-OT with `u_ot = 0` equals plain MDL-2 exactly under fixed seeds.
- Sanity gate for E2 readiness: in a stressed fixture (demand shock exceeding
  idle capacity), at least one OT heuristic strictly beats plain MDL-2 on the
  fixture's paired episodes. If no heuristic can use the channel, no screen may
  claim an agent could.

## Screen validation (Stages E2–E4)

- Every screen run asserts, per output row, that the constructed scenario
  matches the declared scenario (the F0/G0/G1 reconstruction defect guard).
- Discovery and validation CRN streams are disjoint by construction and
  hash-logged; config, summary, and row-file SHA256 digests are recorded in the
  results document before interpretation.
- Gates are evaluated by script, not by inspection; the pass/fail line is
  printed with the measured value and the prospective threshold side by side.
- All immutable inputs (frozen checkpoints, teacher artifacts, prior result
  roots) are hash-checked before and after each screen; any mismatch aborts.

## Documentation obligations

- Each completed stage appends its result to a `results.md` in this directory
  (created at first result) with the same evidence-hash format used by the
  Stage F/G/H reports.
- A completed E2–E4 chain (pass or channel-closed) triggers a change-control
  entry in the locked execution plan recording the outcome.
