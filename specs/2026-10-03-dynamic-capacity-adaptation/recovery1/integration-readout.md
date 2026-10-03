# Precision Correction: Ready For One Frozen Trial

## Delivered

The saved public boundary reproduces the original error at epoch2,
candidate6/quantile0.1/horizon7. The low-level fixture also fails in the legacy
routine: 8 + 8.881784197001252e-16 rounds to 8, erasing strict positive slack.
Signed compensated sums retain the residual, and midpoint service now uses
the same cumulative arithmetic as the feasibility test. No tolerance is added.

Additive implementation:
- src/baselines/capacity_completion_control_recovery1.py
- src/rl/capacity_pilot_recovery1_execution.py
- experiments/scripts/run_dynamic_capacity_recovery1.py

The planner-only version inherits the unchanged public estimator, lifecycle,
adaptive solver and forecast setup. It binds the corrected step/decision path
without patching old globals. Original scientific source/result bytes are not
overwritten. The existing real serial runner, native environment, learner,
budgets, independent raw analysis and archives are reused.

## Necessary Regression

Fourteen focused zero-neural/zero-native tests pass in 2.020s: exact rational
prefix oracle, archived public-input fixture, original eight-step horizon,
patient/resource conservation, repeated decisions, filter isolation, restore
format and the actual new child builder. The saved public decision now finishes
all384forecast epochs with finite scores and unchanged live filter state.
Heisenberg delivered the disjoint five-test horizon suite and was closed;
existing efficiency advice was reused, without an additional gate.

Full compileall passed with PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache.
The default macOS Python cache is sandbox-inaccessible; the first compilation
attempt failed on cache permissions, not source syntax. No scientific optimizer
or native patient episode ran during these checks. Tests are not RL gains.

## Execution Boundary

User reply `继续推进` is recorded with the previously asked complete correction
scope. Commit this implementation and authority, freeze separate current source,
runtime, input and consumed-failure bindings, commit the effective authorization,
then launch once into the new recovery1 result root. The original draft remains
unauthorized and unchanged; do not launch the old entry. Preserve original
budgets/seeds/design and no-second-retry rule. No further toy fit or historical
audit precedes this comparison. Three offline seals must precede all evaluation.

## Executed Admission

Implementation56a85d2 frozen under packet
1e6a59ff5f83d2c11e61a5a047aff3fff0d376b3b9e72402739841a2a379186c;
effective locks committed18ee090decf3c1473097464737b698a0241f5297.
Supervisor29491/PPID1 and child29504/PPID29491 were independently observed
with matching full commands. At03:42:33UTC,88native steps and one complete
teacher trajectory were saved; optimizer updates and evaluation remain zero.
The original failing boundary was crossed, but this is not a performance result.
Same30minute monitor is tool-confirmedACTIVE for this single running attempt.
