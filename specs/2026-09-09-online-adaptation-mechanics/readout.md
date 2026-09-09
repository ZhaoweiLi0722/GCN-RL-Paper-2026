# P0 Mechanics Readout

Completed 2026-09-09. This is a software/mechanism verification record, not
research performance evidence. No trained policy, scientific seed family,
new benchmark scenario, clinical validation or formal holdout was used.

## Completed

- Added an opt-in public-history forecast with explicit observation/issue time,
  fixed-versus-adaptive estimate modes, finite-value checks and exact restore.
  It is not wired into frozen training or evaluation paths.
- Verified that legacy scheduled-wave forecasts can differ with latent future
  regime multipliers even when the current observed demand is identical, for
  both `effective_rate` and `prior_estimate`. The new public-history forecast
  stays identical until different public observations arrive. This is a timing
  test, not a claim of a bug in a legitimately announced-schedule setting.
- Added a bounded finite-tree solver that groups full public histories before
  selecting actions. A hand-computable case gives cost 0 with clairvoyance,
  cost 1 under nonanticipation, and cost 2 for shared open-loop decisions.
- The fixture adapter uses the existing patient simulator with two deterministic
  demand tapes, six decisions, four overtime actions and test seed 123 only.
  All 10,920 transitions were retained. There is one root information set.

## Fixed Patient-Simulator Fixture

| Diagnostic | Expected fixture cost units |
| --- | ---: |
| Best sequence knowing the future world in advance | 2,964,928.225 |
| Best public-history policy on the declared finite grid | 3,067,475.425 |
| Best shared open-loop sequence | 3,146,448.325 |

The public-history policy must choose the same action at the common root.
After different arrivals are observed at epoch one, its second action can
differ. The independently reconstructed best paths were:

- Left-inflow tape: left, balanced, balanced, balanced, balanced, balanced.
- Right-inflow tape: left, right, balanced, balanced, balanced, balanced.

This is an exact finite-grid calculation given the fixture's known two-world
distribution and deterministic simulator transitions. It is not an exact
continuous-action optimum, not a calibrated stochastic operational result,
and not a deployable policy evaluated on independent held-out worlds. No
clinical noninferiority or confidence-interval claim is attached to these
fixture cost values.

## Important Attribution Lesson

The contingent policy is a fixed history-to-action table. It changes actions
after seeing information but never updates learned parameters. Therefore its
advantage over an open-loop sequence demonstrates feedback in this fixture,
not an online DDPG contribution. A future study needs a competent frozen
history-aware controller and an adaptive planner, not just a fixed-action
baseline. Unknown demand by itself does not establish a need for parameter
updates; an appropriately informed frozen policy can react to it.

The fixture does not rescue or replace the earlier R1/R2 failure. That screen
tested a different state distribution and finite first-action library around
a strong graph-forecast comparator. Pooling the two or presenting these values
as a newly demonstrated RL gain would be invalid.

## Verification

- Execution commit: `1d2fa7fe17a15fb48f4ba01119f80ba74a579a4b`.
- Config SHA256: `3fc80ef48fb527723694011a42a01c6264b6ed205d4a0ba1a6772a9ede5c4e81`.
- 108 focused tests passed: 21 new information/tree tests and 87 existing
  authorization, residual, overtime and intertemporal regression tests.
- Full `python -m compileall -q .` passed using the existing venv and the
  separate temporary bytecode cache.
- The six-step fixture completed with exit 0 and no error output.
- [Independent audit](audit_fixture.py) reads only the saved transitions. It
  imports no simulator or planner and launches no rollout. It independently
  recomputes the three values, policy replay, complete Cartesian-product
  coverage, budget, lead time, persistence and all nine inventory hashes.
  Maximum capacity-mechanics discrepancy was exactly 0.0.
- Source/config bytes in that inventory match the execution commit. Original
  simulator files are unchanged. The earlier residual audit was also rerun
  read-only to check preservation of its 17 input/output hashes.

Evidence: [fixture summary](../../reports/2026-09-09-online-adaptation-mechanics/fixture/summary.json),
[transition rows](../../reports/2026-09-09-online-adaptation-mechanics/fixture/transitions.jsonl),
[inventory](../../reports/2026-09-09-online-adaptation-mechanics/fixture/inventory.json),
[independent audit output](audit.json).

Recompute without any new simulation:

```bash
python3 specs/2026-09-09-online-adaptation-mechanics/audit_fixture.py
```

## Remaining Scope

P0 is complete; no scientific-performance campaign is started by passing it.
The next design needs a justified unknown process, a reviewed public-information
adapter for every comparator, calibration and matched adaptation/query budgets.
See [next-study decision sheet](next_study_decision_sheet.md). Howard's review
is pending; there is no proxy approval, push or merge.
