# Next-Study Decision Sheet: What Would Online Weights Need to Learn?

Draft after P0, 2026-09-09. Not an executable scientific specification or an
approved simulator change. This sheet does not promise an RL gain or acceptance
at EAAI. Current manuscript evidence and the closed campaign are unchanged.

## Decision

Do not start another DDPG run merely by making demand larger or surprising.
The next question should distinguish a genuinely missing control capability
from missing information that a forecaster plus a fixed policy already solves.

P0 verifies the tooling for that distinction. It does not demonstrate that the
existing research environment contains enough remaining online-learning value.

## Two Candidate Questions

| Question | What changes | Mandatory non-RL challenger | Main attribution risk |
| --- | --- | --- | --- |
| Unknown persistent demand composition | Belief about incoming demand | Public-history forecaster plus frozen feedback/MPC | Belief adaptation is mislabeled as policy learning |
| Unknown persistent control response | Relationship between an overtime commitment and later usable capacity | Online system identification plus MPC | RL is compared against a deliberately misspecified fixed model |

The second question is worth scrutinizing as a design hypothesis because it
addresses a control-response mismatch, not just a better demand forecast. It
is not yet operationally validated. A possible implementation would represent
site-specific effective overtime availability with an unknown, slowly varying
conversion factor, while staffing costs remain tied to the staffing actually
committed. This is a proposed abstraction only; it must not be presented as an
established PRM manufacturing fact.

Do not introduce both uncertainties together in the first attribution study.
Choose one on operational grounds and freeze it before measuring performance.

## Required Operational Inputs

- What can actually vary: staffing availability, usable processing capacity,
  cycle time, or another quantity? These are different physical mechanisms.
- What does an operator observe, and when? Throughput is censored by demand,
  reagents, queues and integer job completion. The latent conversion factor
  cannot be supplied directly to the learner or its estimator.
- Which range and persistence times are supported by data, literature or domain
  judgment? If unavailable, call the study synthetic and provide sensitivity
  ranges; do not invent an empirical calibration.
- How are cost units, clinical guardrails and terminal pipeline liabilities
  defined? Delayed requests near the horizon cannot create unpriced benefits
  or free obligations. Specify the terminal rule before optimization.

## Attribution That Must Survive

Retain the forecast/belief-adaptation x policy-parameter-update 2x2 design.
Primary comparison: updated policy versus its tensor-matched frozen policy
under the same updated information interface. Include all of the following:

1. A strong frozen history-aware policy trained with the same offline budget.
2. An adaptive heuristic using the same public observations.
3. A rolling-horizon planner with matched online model-identification data.
4. A matched flat representation if graph attribution is claimed.

All methods get equal decision-time information. Account separately for
offline simulation, online interaction, counterfactual simulator queries,
computation and wall time. Paired exogenous streams do not imply identical
state/action trajectories after policies diverge. A planner with model access
is either a budgeted comparator or an explicitly privileged diagnostic.

If belief updates alone win, report information adaptation. If fixed feedback
wins, report reactive control. If model-identification plus MPC wins, report
that alternative; do not hide it to protect an RL narrative.

## Progression Before a New Training Campaign

1. Freeze an operational process model and its information/measurement timing.
   Retain announced-schedule and no-change negative controls. Do not covertly
   degrade only the baseline's forecast or remove an available baseline action.
2. Extend P0's interface checks to the complete production observation adapter,
   graph features, heuristic access and reward/forecast-history channels.
3. On a small faithful case, compare repeated decisions with a properly scoped
   exact solution or certified bound. Then assess headroom against the strong
   adaptive comparator, not only against no-op or open-loop control.
4. Check whether the available observation/response data identify the useful
   action differences. Preserve negative results, query costs and denominators.
5. Freeze one method and bounded evaluation plan before neural training: fresh
   development families, independently held-out process families, full-episode
   endpoints, multiple training seeds, adaptation curves and a stopping rule.
   No earlier formal stream is reused.

For publication, the aim is a credible engineering question, attributable
method contribution, strong alternatives and validated limitations. A positive
result obtained only by removing capable baselines or changing the scenario
after every failure would not meet that aim.
