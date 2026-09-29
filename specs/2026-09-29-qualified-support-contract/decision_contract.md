# Qualified setup-support decision and measurement contract

E1 design packet, 2026-09-29. Local only. **Not an executable experiment spec,
calibrated manufacturing model, or authorization to reopen the closed paper
campaign.** No new trajectories, training, patient-model changes or holdout
queries were used. Unknown numerical inputs remain null in the companion
[measurement contract](measurement_contract.json).

## Decision made by this packet

Investigate **booking site-local qualified technician availability for pre-run
setup and material-connection tasks before automated processing**. Treat each
SOP-defined task type separately. This is a narrower candidate than a generic
overtime-to-production multiplier. Its purpose is to reduce avoidable waiting
for eligible work, not to speed biological expansion or shorten required tests.

This choice is a design hypothesis. We have not established that this is the
binding bottleneck, that additional qualified staff can actually be booked, or
that its changing response leaves a benefit beyond feedback rules and MPC.
Do not implement a new scenario until these points are resolved.

The learning question is whether past task/availability outcomes improve future
staffing commitments after a persistent change, beyond an equally informed
frozen controller and ordinary adaptive control. A frozen history-conditioned
policy can adapt its actions without updating its weights; that is an essential
baseline, not an online-RL result.

## Source-to-assumption boundary

| Primary evidence | Limited support | Unsupported extrapolation |
| --- | --- | --- |
| [Fitzgerald et al., 2022, automated ASC production](https://pmc.ncbi.nlm.nih.gov/articles/PMC8959900/) | Methods describe setup/connections and defined operator interventions; the labor analysis distinguishes manufacturing and other personnel commitments | Labor estimates are not our timestamped staffing data. Biological process monitoring does not measure remaining manual work. ASC parameters do not automatically transfer to CAR-T |
| [Centralized versus point-of-care simulation, Cytotherapy 2023](https://doi.org/10.1016/j.jcyt.2023.08.007) | Part-time labor and order transshipment are studied as operational alternatives | No demonstrated continuously variable staff-to-output law, immediate staff mobility, or online-learning benefit in the proposed channel |
| [AuCT-Sim, Cytotherapy 2019](https://doi.org/10.1016/j.jcyt.2019.07.002) | Facility/network resource planning is an appropriate simulation subject | Does not supply a calibrated staffing-response shift or evidence that our proposed channel has learnable headroom |

Access: Fitzgerald full methods; Cytotherapy papers publisher abstract/preview
and bibliographic records. These sources justify an operational question, not
the numerical model. The constraints below are conservative proposed modeling
rules pending site/SOP review, not claims that every process uses identical
rules. No numerical result from these papers is imported as our performance.

## Action, timing and constraints

At a documented booking cutoff, choose qualified staff availability by site and
future service window, from the actually bookable roster. A request is distinct
from accepted and delivered staffing. Record all three. Notice, cancellation
terms and minimum booking units must come from operational evidence.

- Keep task dispatch/priority fixed and shared across controllers initially.
  Adding a task-routing or sequencing decision requires its own declared arm.
- A task starts only with its required qualifications, minimum crew, materials,
  equipment and permitted window. Keep equipment and people occupancy explicit.
- Default to nonpreemptible tasks and no labor-dependent shortening of their
  validated procedure. Relax only with documented operation-specific support.
- Additional staff can cover eligible parallel tasks only if equipment and
  workspace permit. An idle or equipment-blocked period is not a productivity
  label. Do not count all paid hours as productive hours.
- Staff are site-local. Cross-site mobility requires a real shared pool,
  qualifications, notice and travel constraints. A common abstract budget alone
  does not establish a physically shared workforce or a network-level benefit.
- Use integer crews/slots if that is the actual decision. Continuous staff-hours
  need a defensible aggregation timescale and execution mapping. Do not force
  this channel into DDPG through an unjustified differentiable surrogate.
- Keep biological processing, required testing/release, patient identity and
  clinical acceptance rules outside the action space and unchanged.

Candidate unknowns are task-duration/availability relationships conditional on
known task, site and roster information. Persistent variation is a hypothesis to
check in data, not permission to hide known schedules or inject arbitrary noise.
Task mix and selective staff assignment can confound apparent response changes;
observational correlations alone do not identify the causal effect of staffing.

## Measurement interface

The companion JSON is a data dictionary and missing-input register, **not** a
simulator configuration or implemented observation adapter. Its records have an
event time and a controller-availability time. At decision time `t`, all
controllers may use only versions with `known_at <= t`. Later corrections must
not overwrite the earlier view in a replayed evaluation.

| Record | Minimum contents | What the controller must not obtain early |
| --- | --- | --- |
| Task booking | Pseudonymous task/batch/site IDs, SOP version, task type, booking creation and revisions, scheduled release/window | Future unbooked demand, later schedule revisions |
| Staff commitment | Request/cutoff, service window, qualification class, requested/accepted staffing, notice/cancellation terms | Actual future attendance or realized service |
| Eligibility and resources | Recorded material readiness, qualifications/crew requirement, observed equipment/space occupancy and blocking reason | A hidden simulator flag explaining the true bottleneck |
| Task events | Observed eligibility/start/completion/cancellation, event and log-availability times, associated staff and equipment classes | Exact remaining work, future completion timestamp, latent productivity regime |
| Delivered staffing | Actual qualified presence and time allocation when recorded | Final payroll/timesheet values retroactively treated as known |
| Costs and outcomes | Committed versus settled staffing cost, cancellation charges, observed waiting and final closure obligations | Future clinical outcomes or final economic costs as immediate observations |

Use aggregate qualification/resource classes, not employee names, patient names
or directly identifying clinical data. Pseudonymous identifiers still require
appropriate data access; this packet does not authorize collecting records.

**Primary candidate view: task events and recorded staffing, without exact
continuous progress.** Even event timeliness must be verified. The existing
synthetic completion-only projection retains detailed stages, resource countdowns
and step costs; it is not proof those quantities are available in practice.
Bioreactor sensors may measure process conditions without revealing completion
of a manual task or its remaining labor.

For learning, unfinished tasks are censored observations, not zero-performance
labels. Empty queues, missing materials and downstream blocking must be
distinguished when observable, otherwise kept as estimation uncertainty. A
completion count alone cannot separate all these causes. Don't give only the
neural controller a privileged estimator or future-corrected records.

## Objective and attribution

Before a new screen, freeze units, meaningful benefit, precision requirement,
staffing/cancellation/switching costs, and handling of uncompleted work. Charge
accepted commitments and outstanding obligations; horizon truncation cannot
erase a task or make a costly action appear free. If a common closure policy is
used, disclose and sensitivity-check its contribution. Report waiting, paid
hours, completion and constraints separately from a weighted total. Uncalibrated
costs remain synthetic units, not dollars or clinical benefit.

Required matched comparisons in any later approved study:

1. Same initial trained policy frozen versus online-updated, with identical
   feasible actions, observations, reward availability and compute accounting.
2. Frozen history/belief-conditioned policy using the same estimator interface;
   include its offline training data/compute, not just its deployment speed.
3. Adaptive staffing/dispatch rule and online identification plus MPC, with
   appropriate censored-data treatment and their own realized histories.
4. Known-dynamics/clairvoyant calculations labeled diagnostic bounds only;
   privileged future knowledge is not a deployable competitor.

Use independent future change episodes for validation; account for resets of
weights, optimizer, replay and beliefs. Count transition/model queries and
measure wall time separately. Graph contribution needs feature/head/gate parity
in addition to parameter counts, as established by the separate G1 audit.
No advantage of the current whole package identifies either graph-only or
online-update contribution in this new channel.

## Bridge from the existing fixture

| Frozen software object | Useful software evidence | Gap before operational use |
| --- | --- | --- |
| `BookedJob` and queue stages | Identity, release and closure accounting | SOP task types, qualification and equipment eligibility, observed log delays |
| `QueueObservation.pending_hours` / `PublicServiceReceipt` | Request delay and paid-effort conservation | Real notice, acceptance, staffing granularity and presence |
| Divisible `remaining_work` and response multiplier | Checks float accounting in a synthetic primitive | Not a validated model of nonpreemptible setup procedures |
| Fixed mandatory downstream service | Checks that labor cannot bypass a downstream bottleneck | Actual timing/resource coupling and task-specific obligations |
| `completion_events` projection | Quantifies the value of hiding progress in a recorded finite tree | Still reveals synthetic stages/countdowns/costs; no operational validation |
| Known two-world change law | Finite diagnostic benchmark | No field evidence for response values, probabilities, change time or prior |

The existing eight-cell result remains: simple methods attain its restricted
full-progress optimum; reduced-progress observation leaves a practical comparator
question. Do not change that fixture or add noise to manufacture an RL gap.
The independent C1 packet can analyze its recorded evidence while domain inputs
for this proposed operational channel remain missing.

## Inputs that cannot be supplied by coding

Three bundles are needed from a process owner or adequately documented source;
these consolidate the existing domain questions rather than creating another
approval loop:

1. **One actual SOP-defined task and bookable lever:** qualification, minimum
   crew, nonpreemption, equipment/space, staffing notice/granularity, and whether
   a site-local or shared pool really exists.
2. **One de-identified event-log example/data dictionary:** bookings and their
   revisions, task/resource events, requested/accepted/actual staff, and when
   each record becomes visible. Include unsuccessful/blocked/unfinished tasks.
3. **Units and economics:** service/booking interval, cost and cancellation
   terms, delay objective, terminal obligations and an operationally meaningful
   benefit/precision target. Establish the empirical or explicitly hypothetical
   range/persistence of changes before selecting scenarios.

Missing values must not silently fall back to the convenient two-site fixture.
No raw clinical or employee records should be sent without appropriate access.
If this task has no bookable leverage or measurable response, close the channel
and report that limit. The independent manuscript and comparator work can proceed.

## Readiness decision

E1 documentation is complete; operational feasibility and calibration remain
**unresolved**, and new patient-model execution is **not ready**. Next gate is
domain evidence, then a separately approved, committed prospective protocol for
mechanics, headroom, information/identifiability, adaptive comparators and only
then any bounded learning run. Neither literature nor this packet establishes
positive RL headroom or a publication outcome.
