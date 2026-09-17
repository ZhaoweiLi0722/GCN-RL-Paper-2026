# Measurement A: information-matched rollout planner (interim, restricted class complete)

Date: 2026-09-16. Branch `e2b-run`. Development pilot of Measurement A in
`specs/2026-09-15-optimality-gap-measurement/plan.md`, on the development
seed stream 99.7M, world-paired with every other policy evaluated this week.
No training. No formal-holdout seed. No existing artifact changed.
Tool: `evaluation/information_matched_rollout_planner.py`.

**Status.** Restricted class: 36 of 40 planned worlds complete (nominal,
abrupt shift, regional drift 10 each; compound stress 6). Full class and the
privileged-information diagnostic: not yet run. Repeated attempts were killed
by the operating system for memory pressure caused by other applications
(swap at 18.5 of 19 GB, a browser helper alone at 7.8 GB); the runner is
resumable and the remaining jobs are queued in the commands below.

## What the planner is

At each weekly decision it enumerates candidate actions, simulates each to
episode end on six common-random-number worlds under the MDL-2 continuation
policy, executes the candidate with the lowest mean cost (ties to the anchor),
and replans. The restricted class is exactly the learned residual's action
class: the MDL-2 anchor plus the four G1 specimen options (±0.05, ±0.10 on
the centred specimen pattern). Settings were fixed before any evaluation.

## Information audit (the proposal's mandatory step)

Rollout worlds are deep copies of the live environment in which everything a
policy cannot observe is replaced:

- every waiting, in-transit, and in-production patient's latent health
  parameters are resampled from the enrollment prior conditional on observable
  age, current survival, and risk type (risk-type counts are in the
  observation), and survival is recomputed from the resampled draw;
- true demand rates, regime multipliers, and active shocks are replaced by the
  observable 12-epoch arrival-history estimate blended with the prior;
- remaining durations of regional supplier disruptions are cleared;
- future draws use a fresh generator per rollout world.

Verified audit figures come from the smoke run (3 decisions, 2 worlds): 580
patient resamples, **zero** kept their true health index, mean absolute
mismatch between recomputed and observed survival 0.0003. The 36 evaluation
episodes ran the identical code path, but their aggregate audit counters and
per-decision logs were not persisted: the runner wrote them only at the end
of a batch and every batch was killed by the operating system for memory
before reaching that point. The runner now writes both files after every
episode; the pending runs will carry the full audit. Nothing in the planner
reads a quantity the learned policy could not read.

## Result: within the learned policy's own action class, the achievable improvement is about three times what the learned policy captures

Paired on the same 36 worlds, versus executed MDL-2:

| policy | Δcost | 95% CI | wins | Δcompletion pp | Δmfg-inelig pp | Δlost | guardrail C/I/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--: |
| **planner, restricted class (information-matched)** | **−1.81%** | **[−2.21, −1.44]** | **35/36** | +0.31 | −0.58 | **−31.4** | ✓ ✓ ✓ |
| AFR-GCN-TD3 residual (3-seed mean) | −0.67% | [−0.87, −0.48] | 32/36 | +0.15 | −0.45 | −25.8 | ✓ ✓ ✓ |
| MDL-3, no learning | −4.37% | [−5.40, −3.33] | 32/36 | +1.01 | +0.41 | −14.5 | ✓ ✗ ✗ |
| GCN-TD3 on MDL-3, zero-shot | −4.40% | [−5.45, −3.37] | 32/36 | +0.86 | −0.07 | −16.9 | ✓ ✗ ✗ |

Per scenario, planner versus MDL-2: nominal −2.35%, abrupt shift −1.98%,
regional drift −1.16%, compound stress −1.71% (6 worlds).

Direct contrasts:

- planner minus learned residual: −1.15% [−1.50, −0.81], 32 of 36 worlds;
- planner minus MDL-3: +2.68% [+1.67, +3.70]; the planner cannot touch the
  coverage lever and does not compensate for it;
- the planner loses the fewest patients of any policy measured this week,
  31 fewer than MDL-2 and 17 fewer than MDL-3, because it selects among the
  same routing options the residual has but selects them better.

On the 36 worlds the per-scenario guardrail is not uniform (the abrupt-shift
and regional-drift scenarios show the same manufacturing-ineligibility
sensitivity every routing policy shows), but the pooled formal rule passes on
all three margins.

## Reading, against the proposal's table

*"Planner materially beats frozen GCN inside the residual class → the current
learning procedure leaves usable improvement uncaptured."* That row applies.
Under identical observations and identical action authority, a policy that
spends 30 simulations per decision recovers 1.8% where the distilled residual
recovers 0.7%. The gap is not information and not action space; it is the
quality of the decision rule that the offline procedure produced. This is the
first positive headroom measurement for the routing channel in the project,
and it is consistent with the label-stability finding: the signal exists but
is noisy, so a policy that averages six fresh continuations per decision
extracts more of it than one that had to learn a fixed mapping from noisy
labels.

What it does not say: that a learned policy can reach 1.8%. The planner is not
deployable at 10 minutes per episode. The right use of the number is as the
target and as the teacher: a residual distilled from *this* planner's
decisions, rather than from single-rollout advantage labels, is the obvious
next candidate. The planner logs every decision with all candidate means;
those logs will be available from the pending runs onward (see the note
above on why the completed 36 episodes did not persist them).

## Pending

1. Restricted class, compound stress reps 6–9 (4 episodes).
2. Full class (MDL-2 and MDL-3 anchors, specimen and transfer options), 3
   worlds per scenario: does the planner capture coverage and routing at once,
   and does it pass the ineligibility margin doing so?
3. Privileged variant (no resampling, true rates), 3 worlds per scenario: the
   value of information, i.e. how much of the perfect-information bound's
   slack is information rather than control authority.

Commands (resumable):

```bash
python3 evaluation/information_matched_rollout_planner.py --candidate-class restricted --replications 10 --worlds 6 --workers 2 --resume --output-root results/measurement_a_planner/restricted
python3 evaluation/information_matched_rollout_planner.py --candidate-class full --replications 3 --worlds 6 --workers 2 --output-root results/measurement_a_planner/full
python3 evaluation/information_matched_rollout_planner.py --candidate-class restricted --privileged --replications 3 --worlds 6 --workers 2 --output-root results/measurement_a_planner/restricted_privileged
```

Each episode takes 7 to 10 minutes on one core; the process needs about
1 GB per worker and is killed on this machine while other applications hold
its memory.
