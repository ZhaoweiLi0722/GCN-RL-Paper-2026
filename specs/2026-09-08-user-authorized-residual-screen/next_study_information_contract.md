# Information Boundary for a Possible Online-Adaptation Study

Design memo, 2026-09-09. The completed residual screen failed R1/R2; this memo
does not alter that result, reopen the current campaign, or authorize a new
scenario, simulator modification, training run, or formal evaluation.

## Verified Behavior

The source scenario config uses scheduled referral waves,
`demand_forecast_source = effective_rate`, a four-step forecast horizon, and
5% or 10% forecast noise depending on scenario. In `_sample_demand_forecast`,
scheduled-wave forecasts use future-step regime and scheduled-referral
multipliers, then add noise. Changing to `prior_estimate` changes base rates
but does not remove those future multipliers.

The graph-forecast heuristic combines forecast, waiting patients and twice
the at-risk count, then applies graph smoothing and bounded proportional
allocation. Its other action channels also read environment-provided arrays.

Sources at the unchanged PR #12 commit:

- [Scenario contract](https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026/blob/118f9978f50433d25de1e392e242cd1655101b43/experiments/configs/intertemporal_shared_capacity_episode_prescreen.json)
- [Forecast implementation](https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026/blob/118f9978f50433d25de1e392e242cd1655101b43/src/env/capacity_planning.py#L1628)
- [Graph-forecast allocation](https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026/blob/118f9978f50433d25de1e392e242cd1655101b43/src/baselines/heuristics.py#L521)

This is not automatically leakage in an announced-referral problem. It would
be an invalid information boundary if the claim were adaptation to an
unannounced future change already present inside the forecast. Code reading
alone does not establish that forecast information caused the negative RL
attribution result.

## Contract to Specify Before Implementation

Separate the latent demand/service process, information actually available to
the operator, and forecasts/beliefs computed from that permitted information.
Known calendars can remain public. Unannounced regime identity, future
multiplier and future patient outcomes cannot be inputs to a deployable
policy, forecaster or heuristic. Extra-information diagnostics must be labeled
and excluded from ordinary baseline comparisons.

Use one operationally justified persistent unknown process and one feasible
intertemporal control channel. Justify both independently of a desired RL
result. Equalize observations, history and model access across methods,
including direct access to environment attributes.

## Mechanics Fixtures Before Scientific Seeds

- Identical public histories and calendars, but different latent future
  regimes, must produce identical current forecasts and policy inputs.
- Replacing an unobserved future outcome cannot change a forecast issued now.
- Newly available observations may update beliefs; paired fixtures must expose
  and verify that timing.
- Heuristics and planners must have no extra latent information via simulator
  fields. Check scenario IDs, reset seeds and metadata for side channels.
- Checkpoint restore must recover histories, beliefs and update counters exactly.

These are proposed fixtures, not evidence of operational or clinical validity.

## Attribution and Baselines

| Arm | Forecast/belief adaptation | Policy parameter updates |
| --- | --- | --- |
| A | No | No |
| B | Yes | No |
| C | No | Yes |
| D | Yes | Yes |

The primary online-policy contrast is D versus B; both have updated information.
C versus A is secondary. Paired initialization and exogenous streams are
required, but state/action visitation will diverge once policies differ and
must not be described as identical training data. Any counterfactual simulator
queries need explicit accounting and budget-matched comparators.

Include an adaptive heuristic, rolling-horizon planner/MPC, and a strong frozen
history-aware policy with the same information. State adaptation by a frozen
recurrent policy is not policy-parameter learning. If belief updates or MPC
capture the gain, do not relabel it as online DDPG contribution.

## Feasibility Before Training

1. Specify the operational case, observation timing, costs, constraints and
   generalization question before changing the simulator.
2. Build a small faithful solvable case with repeated sequential decisions.
   Distinguish an exact optimum or certified bound from a finite action-library
   best value or a heuristic planner. Do not call all of them an oracle bound.
3. Check that a meaningful, reproducible gap remains after a strong adaptive
   comparator, and that available observations contain the signal needed to
   exploit it. Freeze cost normalization and guardrails before measurement.
4. Prespecify held-out process families, seed groups, adaptation budget, full
   episode endpoints, failure reporting and stopping rules. Freeze the method
   before independent confirmation; do not keep changing scenarios after each
   negative result.
5. Only then consider bounded online DDPG training. This design makes its
   contribution identifiable; it cannot guarantee a positive contribution or
   journal acceptance.
