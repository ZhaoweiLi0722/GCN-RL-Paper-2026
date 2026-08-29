# Requirements

## Scientific Name and Scope

The mechanism is **continuous overtime capacity control**: a per-facility,
per-epoch continuous decision that temporarily expands effective manufacturing
capacity at a convex, prespecified cost. It is the study treatment channel for
the follow-up online-RL attribution question. A secondary, optional mechanism —
the **production-rate throttle** — is specified here for completeness but is
OFF by default and may only be activated by a separate change-control entry.

The scientific question is NOT "does overtime help" (it trivially can). It is:
**does this environment now expose a continuous decision channel on which online
graph RL can demonstrate an attributable endpoint gain that heuristics and
offline distillation do not already capture?** The channel must therefore pass
the E2–E4 screening gates (plan.md) before any agent trains on it.

## Design rationale (traceability to the audit evidence)

Each requirement below traces to a failure documented in
`docs/online_rl_attribution_postmortem_and_followup_brief.md`:

- Failure 1 (quantized-gradient death) → the overtime channel must execute as a
  float end-to-end: no projection, rounding, or integer-lot quantization may sit
  between actor output and physical effect.
- Failure 2 (unstable counterfactual labels) → per-decision consequences must be
  large and persistent enough to pass a ≥70% label-stability gate; the optional
  fatigue state exists to raise persistence if the memoryless form fails E3.
- Failure 5 (strong anchor banks the gain) → no look-ahead anchor is defined on
  the overtime dimension. The learned policy controls overtime directly;
  heuristic overtime rules exist as comparators, not as anchors.

## Decision A (primary): overtime capacity `u_ot`

- **Action block**: one additional `facility_net` block of `n` continuous
  values. Raw actor output in `[-1, 1]` maps affinely to `u_ot ∈ [0, 1]`.
  `action_size` becomes `5n` when `enable_overtime_control` is true.
- **Effect**: effective idle capacity this epoch becomes
  `idle_effective = idle_bioreactors + u_ot * max_overtime_fraction * base_capacity_i`,
  entering the existing production expression
  `production = min(specimens_eligible, idle_effective, reagents)`.
  Overtime capacity is transient: it never enters the bioreactor shift register,
  never persists to the next epoch, and cannot be transferred.
- **Cost**: `overtime_cost_i = weight_overtime_linear * s_i + weight_overtime_quadratic * s_i^2`
  where `s_i = u_ot_i * max_overtime_fraction * base_capacity_i`. Both weights
  are strictly positive so the optimum is interior and state-dependent. The
  component is added to `_operating_cost_components` under the key
  `overtime_cost` and reported in every additive decomposition.
- **Continuity invariant**: the mapping from `u_ot` to executed surge and to
  `overtime_cost` is continuous and piecewise-smooth. Production itself remains
  integer in patient lots; the requirement is that the *capacity bound and its
  cost* are continuous, so marginal overtime always has a nonzero, sign-correct
  cost gradient even when it does not change the integer production count that
  epoch.
- **Optional persistence (fatigue)**: behind `enable_overtime_fatigue`, a
  per-facility state `f_i ← decay * f_i + u_ot_i` that scales up the marginal
  overtime cost, `weight_overtime_linear * (1 + fatigue_cost_scale * f_i)`.
  Default OFF; may be switched on without a new spec only as the prespecified
  E3 remediation (plan.md).

## Decision B (secondary, default OFF): production-rate throttle `u_prod`

- One additional block of `n` values mapping to `u_prod ∈ [0, 1]`, scaling the
  production expression: `production = min(...) * u_prod`, rounded by the
  existing integer-lot mechanics.
- Because rounding re-introduces the failure-1 geometry when per-epoch
  production is small, Decision B may enter the study only if the E2 headroom
  screen is run separately for it and passes; otherwise it stays disabled.
  `u_prod = 1` is the required default.

## State and observation

- Per-facility features appended when `enable_overtime_control` is true:
  current `u_ot` (previous epoch's executed value), fatigue `f_i` when enabled,
  and `max_overtime_fraction * base_capacity_i` (static surge headroom).
- The identical feature block is appended to both the flat observation and the
  graph `node_features` so graph and flat policies remain representation-matched.
- Observation and action sizes change ONLY when the enabling flags are true.

## Extended heuristic field

The comparator field must be extended before any learned policy is evaluated on
the new channel, or every comparison is a strawman:

- **MDL-2-OT**: MDL-2 plus a myopic overtime rule — surge when projected
  eligible-waiting exceeds idle capacity and the marginal modeled patient-loss
  risk exceeds the marginal overtime cost. Closed-form threshold, no simulation
  at decision time.
- **uMYO-OT**: the existing urgency-surge heuristic mapped onto the new
  continuous channel.
- **Static-OT sweep**: constant `u_ot ∈ {0.0, 0.1, ..., 1.0}` on top of MDL-2;
  the best constant is the "tuned scalar" reference the learned policy must
  beat for any interesting claim.

## Invariants and prohibitions

1. **Bit-identical legacy behavior**: with `enable_overtime_control` and
   `enable_production_throttle` false, every trajectory, cost, observation,
   RNG draw, and hash is identical to the current environment under fixed
   seeds. This is a mandatory regression test, not a code-review judgment.
2. **No contamination**: no existing result root, checkpoint, teacher artifact,
   CRN stream, or provenance file is modified. All new outputs live under a new
   output root (`results/continuous_overtime_control/` or successor).
3. **No anchor on the treatment channel**: no MDL-based lookahead may set or
   regularize `u_ot` for a learned policy. Heuristic overtime rules are
   comparators only.
4. **Patient identity rules unchanged**: overtime affects capacity only; it
   cannot create, split, pool, substitute, or reassign specimens or patients.
5. **Clinical gates unchanged**: completion service level, patients lost, and
   manufacturing ineligibility retain their existing noninferiority definitions
   and margins.
6. **No training before gates**: no RL or distillation run may consume the new
   channel until E2 (headroom), E3 (label stability), and E4 (critic ranking)
   have passed and their results are committed (plan.md).
7. **Cost-weight sensitivity**: any manuscript-facing economic statement about
   overtime requires the prespecified sensitivity sweep over
   `weight_overtime_*` before interpretation.
8. **Preregistration discipline**: no screening threshold, config value, or
   gate may be changed after observing outcomes; a changed setting requires a
   new config name, output root, and an appended change-control entry.

## Configuration surface (names fixed by this spec)

```
enable_overtime_control: bool = False
max_overtime_fraction: float = 0.3          # E2 remediation may raise via change control
weight_overtime_linear: float > 0
weight_overtime_quadratic: float > 0
enable_overtime_fatigue: bool = False
overtime_fatigue_decay: float in (0, 1)
overtime_fatigue_cost_scale: float >= 0
enable_production_throttle: bool = False    # Decision B, requires separate activation
```

Initial numeric values for the cost weights are set in the E1 implementation
config and justified against the cost decomposition (base operating cost is
~46% of MDL-2's 2,686M objective) so that a full surge at all clinics is
material but not dominant; the chosen values are frozen before E2 runs.
