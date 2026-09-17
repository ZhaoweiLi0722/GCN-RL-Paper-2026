# Measurement A: information-matched rollout planner

Date: 2026-09-16. Branch `e2b-run`. Development pilot of Measurement A in
`specs/2026-09-15-optimality-gap-measurement/plan.md`, on the development
seed stream 99.7M, world-paired with every other policy evaluated this week.
No training. No formal-holdout seed. No existing artifact changed.
Tool: `evaluation/information_matched_rollout_planner.py`. Artifacts: per-
episode rows for both classes, full-class decision logs (every decision with
all candidate means), audit counters, and the joined comparison.

**Status.** Restricted class complete (40 worlds, 10 per scenario). Full class
complete at its reduced budget (12 worlds, 3 per scenario). The privileged-
information diagnostic is running; it will be appended when it finishes.
Runs were interrupted repeatedly by the operating system reclaiming memory
held by other applications; the runner persists every episode, so nothing
was lost, but the full class was cut from 10 to 3 worlds per scenario.

## What the planner is

At each weekly decision it enumerates candidate actions, simulates each to
episode end on six common-random-number worlds under the MDL-2 continuation
policy, executes the candidate with the lowest mean cost (ties to the anchor),
and replans. Settings were fixed before any evaluation.

- **Restricted class** is exactly the learned residual's action class: the
  MDL-2 anchor plus the four G1 specimen options (±0.05, ±0.10 on the centred
  specimen pattern). Five candidates.
- **Full class**: the restricted set, the same five built on the MDL-3 anchor,
  and ±0.05 reagent-transfer and ±0.05 combined-transfer options on MDL-2.
  Fourteen candidates. This is the proposal's "existing full controls".

## Information audit

Rollout worlds are deep copies of the live environment in which everything a
policy cannot observe is replaced: each waiting, in-transit, and in-production
patient's latent health parameters are resampled from the enrollment prior
conditional on observable age, current survival, and risk type, and survival
is recomputed; true demand rates, regime multipliers, and active shocks are
replaced by the observable 12-epoch arrival-history estimate blended with the
prior; remaining supplier-disruption durations are cleared; future draws use
a fresh generator per world. Persisted audit counters (the last resumed
batches, 5 episodes, 1.13 million patient-world resamples; an earlier
persisted batch of 8 episodes showed the same): **zero** resamples kept the
true health index; mean absolute survival mismatch 0.0008. The smoke run
(580 resamples) gave the same picture. Nothing in the planner reads a
quantity the learned policy could not read.

## Result 1, restricted class: inside the learned residual's own action class, the achievable improvement is about three times what the learned policy captures

Forty paired worlds, versus executed MDL-2:

| policy | Δcost | 95% CI | wins | Δcompletion pp | Δmfg-inelig pp | Δlost | guardrail C/I/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--: |
| **planner, restricted** | **−1.83%** | **[−2.19, −1.48]** | **39/40** | +0.35 | −0.65 | −34.4 | ✓ ✓ ✓ |
| AFR-GCN-TD3 residual (3-seed mean) | −0.69% | [−0.87, −0.51] | 36/40 | +0.18 | −0.53 | −27.4 | ✓ ✓ ✓ |
| MDL-3, no learning | −4.35% | [−5.28, −3.39] | 35/40 | +1.08 | +0.29 | −21.9 | ✓ ✗ ✗ |
| GCN-TD3 on MDL-3, zero-shot | −4.42% | [−5.41, −3.43] | 35/40 | +0.97 | −0.24 | −25.5 | ✓ ✓ ✗ |

Per scenario, planner versus MDL-2: nominal −2.35%, abrupt −1.98%, regional
drift −1.16%, compound stress −1.79%. Planner minus learned residual:
−1.15% [−1.49, −0.82], 34 of 40 worlds. Planner minus MDL-3: +2.64%; it
cannot touch the coverage lever. Candidate mix (persisted episodes): +0.10
specimen 31%, anchor 23%, +0.05 20%, −0.10 16%, −0.05 11%.

## Result 2, full class: the planner captures coverage and routing together, exceeds MDL-3, and passes the clinical gate that MDL-3 fails

Twelve paired worlds (3 per scenario), versus executed MDL-2:

| policy | Δcost | 95% CI | wins | Δcompletion pp | Δmfg-inelig pp | Δlost | Δstarted | guardrail C/I/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--: |
| **planner, full** | **−6.64%** | **[−7.98, −5.22]** | **12/12** | **+2.34** | **−0.44** | **−129.5** | +141 | **✓ ✓ ✓** |
| GCN-TD3 on MDL-3, zero-shot | −4.82% | [−6.38, −3.16] | 11/12 | +1.10 | −0.17 | −30.5 | +71 | ✓ ✗ ✗ |
| MDL-3 | −4.79% | [−6.24, −3.26] | 11/12 | +1.22 | +0.31 | −29.6 | +98 | ✓ ✗ ✗ |
| planner, restricted (same 12 worlds) | −1.59% | [−1.98, −1.19] | 12/12 | +0.16 | −0.44 | −23.0 | −8 | ✓ ✓ ✓ |
| AFR-GCN-TD3 residual | −0.67% | [−0.90, −0.45] | 12/12 | +0.17 | −0.46 | −29.2 | −5 | ✓ ✓ ✓ |

Per scenario, full planner versus MDL-2: nominal −9.22%, abrupt −7.17%,
regional drift −4.82%, compound stress −5.79%. Full planner minus MDL-3:
−1.94% [−2.83, −1.10], 11 of 12. Minus the best learned configuration:
−1.91% [−2.87, −0.92], 11 of 12.

Three things in this table matter beyond the headline number.

1. **It is more than the sum of its parts.** MDL-3 alone gives −4.8%; the
   restricted planner alone −1.6%; the full planner −6.6%. Choosing epoch by
   epoch between the two coverage levels and the routing options is worth
   more than fixing either.
2. **It passes the manufacturing-ineligibility margin that MDL-3 breaches.**
   MDL-3 raises that rate by 0.31 pp; the full planner, which uses MDL-3's
   action in 52% of its decisions, lowers it by 0.44 pp, because it declines
   the higher-coverage action in the states where it would push frail
   patients into manufacturing. It starts 141 more patients per episode than
   MDL-2 and loses 130 fewer, four times any other policy's reduction.
3. **The decision mix is genuinely state-dependent.** MDL-3-based candidates
   52%, MDL-2-based 48%; the plain MDL-2 anchor only 5%; reagent-transfer
   corrections 13%; the +0.10 specimen option, on either anchor, 34%.

## Reading, against the proposal's table

Two rows apply at once. *Planner beats frozen GCN inside the residual class →
the current learning procedure leaves usable improvement uncaptured*: 1.8%
versus 0.7% with identical observations and identical action authority.
*Improvement appears (much larger) under broader existing controls → the
residual action restriction is a material limitation*: 6.6% versus 1.8%.

The optimality-gap question now has an implementable upper bound to go with
the certified lower bound. MDL-2 is at least 6.6% from the best achievable
policy under its own information and controls; the certified relaxation says
at most 39%. The learned method's problem is no longer whether room exists.
It is that offline distillation from single-rollout labels recovered a tenth
of the room a six-rollout planner recovers at decision time.

## What this does not say

- The planner is not deployable at 10 to 20 minutes per episode. Its value is
  as the achievable benchmark and as a teacher: a residual distilled from the
  full planner's decisions, with authority over coverage and routing, is the
  obvious next learned candidate, and the decision logs in `artifacts/` are
  the training data.
- Twelve worlds for the full class is a pilot budget. The direction is
  unanimous and the intervals exclude zero by wide margins, but the
  per-scenario numbers rest on three worlds each and the guardrail verdict on
  twelve paired differences.
- Decision B keeps MDL-2 as the anchor of record; nothing here changes that.
  It does change what a reviewer will ask: a non-learning planner with the
  same information beats every learned configuration by a factor of three to
  ten on the paper's own metric.

## Pending

The privileged variant of the restricted planner (true latents, true rates)
is running. Its gap over the matched restricted planner is the value of
information on this channel and separates how much of the certified bound's
slack is information rather than control authority.
