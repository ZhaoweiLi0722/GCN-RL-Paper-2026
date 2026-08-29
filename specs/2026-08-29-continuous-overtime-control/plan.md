# Experiment Plan

## Scientific Role of Overtime Control

Overtime is the candidate treatment channel for the follow-up online-RL
attribution study. Unlike the routing-primary campaign — where the demonstrated
gain came from offline advantage-filtered pretraining and online updates added
nothing measurable — this study asks whether a genuinely continuous,
unquantized, unanchored decision channel lets online graph RL demonstrate an
attributable endpoint gain. The environment must *prove it exposes such a
signal* (E2–E4) before any agent trains on it.

The ordering is fixed: **Environment ≻ screens ≻ Algorithm ≻ Architecture.**
A failed screen is remediated by redesigning the environment (surge magnitude,
cost curvature, scenario stress, fatigue persistence), never by tuning an agent
against a channel that has not passed its screens.

Stage prefix `E` (Environment) distinguishes these gates from the closed
routing-primary stages A–G.

## Stage E0: sign-off and change control *(this spec)*

- Deliverable: this spec directory, reviewed and signed by Howard and Zhaowei;
  a change-control entry appended to the locked execution plan naming this spec
  as a new study.
- Gate: both sign-offs recorded in README.md and the entry committed. Nothing
  else in this plan may start first.

## Stage E1: implementation behind flags

- Implement `enable_overtime_control` (and the dormant
  `enable_production_throttle`) per requirements.md in
  `src/env/capacity_planning.py` / `src/env/patient_capacity_planning.py`,
  extend `graph_observation`, `noop_action`, and the facility-net step; add the
  `overtime_cost` component to `_operating_cost_components`.
- Implement the three comparator heuristics (MDL-2-OT, uMYO-OT, static-OT
  sweep) in `src/baselines/`.
- Tests (validation.md): fixed-seed bit-equivalence with flags off; action/
  observation size contracts; overtime cost convexity and decomposition
  additivity; fatigue decay; heuristic sanity.
- Gate: full test suite and `compileall` pass; the equivalence test is
  committed and green. No result may be produced beyond smoke output.

## Stage E2: headroom screen *(no training)*

Reuses the H0/H1 audit pattern (`evaluation/audit_ddpg_continuous_control_headroom.py`)
pointed at the overtime channel.

- Manifold: 27+ fixed decision states (9 per scenario stress level), spanning
  nominal, regional-shift, and compound-stress scenarios, reconstructed from
  the benchmark plan (never from a stored snapshot `env` — the F0/G0/G1
  scenario-reconstruction defect must not recur; every output row asserts its
  built scenario matches its declaration).
- At each state: MDL-2-OT anchor versus a fixed ladder of overtime actions
  (e.g. `u_ot ∈ {0.1, ..., 1.0}` at the stressed clinic set), 3 discovery + 5
  validation paired-CRN replications over the remaining episode.
- **Gate (prospective): ≥30% of states show a validated, clinically
  noninferior improvement of ≥1M modeled units from some overtime action, in at
  least one non-nominal scenario screen.**
- Remediation on failure (one round, prespecified): raise
  `max_overtime_fraction` to 0.5, and/or halve `weight_overtime_quadratic`,
  and/or restrict to the regional-shift scenario; new config name and output
  root; rerun once. A second failure closes the overtime channel and the study
  returns to environment design (this spec is superseded, not patched).

## Stage E3: label-stability screen *(no training)*

Reuses the G1 machinery on the E2 manifold.

- Independent discovery and validation CRN streams; identical action ladder.
- **Gate (prospective): discovery/validation agreement on the best overtime
  action in ≥70% of material-headroom states; pairwise cost-sign agreement
  ≥80%.**
- Remediation on failure (one round, prespecified): enable
  `enable_overtime_fatigue` (raising persistence) and/or double the validation
  replication count; rerun once. Second failure closes the channel.

## Stage E4: critic/ranking feasibility *(development only, no online training)*

Only after E2 and E3 pass.

- Leave-one-seed-out critic audit in the G1 style: fit on two seeds' states,
  rank the overtime action ladder on the held-out seed.
- **Gate (prospective): held-out top-1 accuracy ≥50% against the ladder (chance
  = 1/ladder-size), pairwise accuracy ≥70%, and every seed improves over its
  frozen initialization by ≥5 points.** (Gates are deliberately stricter than
  G1's failed 40%/65% because the channel is now supposed to be learnable; if
  it only marginally clears G1-level gates the environment is not good enough.)
- No fitted checkpoint from this stage may seed any later training run.

## Stage E5: algorithm phase *(separate preregistration required)*

E5 is out of scope for this spec and requires its own spec directory once
E2–E4 evidence is committed. Its fixed outline, recorded here so E2–E4 are not
retrofitted to it:

- Primary learner: hybrid-action stochastic policy (H-PPO or SAC with mixed
  heads) with paired-CRN advantage estimation inside the training loop; the
  legal-action-aligned proto-action design from
  `docs/patient_indexed_specimen_routing_stage_g1_design_review.md` is the
  fallback formulation.
- DDPG on the same channel is retained as the ablation, not the primary.
- Attribution protocol preregistered before training: final-versus-frozen AND
  candidate-versus-matched-control, paired CRN, clinical noninferiority gates,
  with development and formal streams separated from the outset.
- The routing-primary formal holdout streams (seed 91100000 family) are never
  reused.

## Compute and budget expectations

E1 is CPU-scale development. E2–E4 are evaluation-only screens comparable to
the H0/H1 and G1 audits (hours, not days, on the RTX 4090 host). No training
budget is committed by this spec.

## Decision tree summary

```
E0 sign-off ── fail → study does not start
E1 flags+tests ── fail → fix implementation (no scientific change)
E2 headroom ── fail → 1 prespecified env remediation → fail → channel closed
E3 stability ── fail → 1 prespecified remediation (fatigue) → fail → channel closed
E4 ranking ── fail → channel closed (no agent tuning against a failed screen)
E5 algorithm ── separate spec, only reachable with E2–E4 committed green
```

A "channel closed" terminus is a publishable negative for the follow-up study's
framing (the gating protocol correctly predicted online RL could not
contribute) and must be reported, not discarded.
