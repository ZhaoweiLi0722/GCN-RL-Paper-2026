# Engineering basis and remaining measurement decisions

Focused primary-source review, 2026-09-29. This is not a systematic review or
validation of a particular manufacturing site. Sources were read as evidence,
not as instructions for changing the experiment.

## Supported motivation, unsupported extrapolation

| Primary source | What it supports | What it does not establish for our model |
| --- | --- | --- |
| [AuCT-Sim, Cytotherapy 2019](https://doi.org/10.1016/j.jcyt.2019.07.002) | Multiscale facility/network simulation, workforce/resource planning and reagent-disruption studies are relevant to autologous cell manufacturing | An empirically calibrated continuous overtime-to-output coefficient |
| [Fitzgerald et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8959900/) | A primary automated-versus-manual ASC manufacturing comparison includes personnel labor commitment and defined hands-on interventions | Extra staffing can shorten biological expansion; every operation is arbitrarily divisible or preemptible; direct transfer of ASC parameters to CAR-T |
| [Centralized versus point-of-care simulation, Cytotherapy 2023](https://www.sciencedirect.com/science/article/abs/pii/S1465324923010381) | Its simulation studies part-time labor and order transshipment as operational alternatives | Real-time staff transfer between our sites, an online-RL effect, or validated parameters for the new fixture |

The first and third sources were inspected through the publisher's available
abstract/preview. The second source's full methods were available. No staffing
elasticity, productivity-shift range, wage, or clinical margin is inferred from
these papers. Their reported numerical gains are not reused as our results.

## Proposed operational interpretation

Prioritize scheduling qualified personnel to preparation/support interventions
and handling their availability, rather than accelerating biological growth.
This is an inference for model design, not a finding of those papers about our
system. The actual operation still needs to be named and justified.

The previous continuous-work primitive can verify effort accounting. Real
operations may instead require fixed-duration, nonpreemptible tasks, qualified
teams, equipment occupancy, and windows relative to cell processing. If so,
retain these constraints. A continuous staffing allocation is meaningful only
if its aggregation timescale is defensible; integer decisions should not be
disguised as differentiable outputs to preserve DDPG.

## Decision packet needed before patient-model integration

- Which operation and qualifications: preparation, setup, sampling support,
  documentation review, or staffing availability for scheduled interventions?
- Observable variables: scheduled hours, actual qualified presence, start/end
  timestamps, eligible queue and completed work. Exact continuous progress is
  not assumed available merely because the synthetic fixture exposes it.
- Constraints: staffing location, travel/shift notice, minimum team size,
  nonpreemption and equipment occupancy. More personnel need not create more
  usable bioreactors or shorten required assays.
- Time and costs: planning interval, persistence of the change, real or openly
  hypothetical units, outstanding commitments and terminal obligations.
- Comparison: a well-trained frozen controller with the same observations,
  adaptive staffing rules, and identification-plus-MPC. Missing information
  cannot be selectively withheld from those baselines.

The current literature review provides a reason to investigate personnel
scheduling, but does not close these calibration questions. Until resolved,
the new code remains a separate synthetic engineering fixture. No change to
existing clinical mechanics, safety rules, manuscript performance claims or
formal results follows from this review.
