# Support queues: bounded mechanism extension

Date: 2026-09-29. User instruction: continue local work step by step. No Howard
sign-off, patient experiment, formal confirmation, remote push, or main merge
is inferred. This is a separate synthetic mechanism study, not a modification
of the closed paper campaign. Freeze source and this config before its recorded
comparison; retain all eight cells and do not tune after inspecting outcomes.

## Why this extension, and what is not established

The preceding static-work fixture omitted arriving work and downstream delays.
Its best open-loop schedule matched the optimal public-feedback policy. That
does not settle the question in a recurring, coupled workflow. This extension
tests those two missing mechanisms as a small factorial boundary check, not a
search over cost weights until DDPG wins.

The operational hypothesis is labor-limited preparation/support followed by a
mandatory fixed-duration workstation. A primary manufacturing comparison
[Fitzgerald et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8959900/)
documents defined manual interventions and personnel commitment; it does not
calibrate our task duration, labor elasticity, queue layout, or costs. The new
support stage remains a divisible, preemptible synthetic task. Site-specific
qualifications, actual intervention windows and progress measurement need
domain validation before patient integration. No extra labor shortens biological
growth or testing in the existing simulator. See the earlier
[engineering qualifications](../2026-09-29-service-effort-decisions/engineering_basis.md).

## Committed factorial scope

Two sites each receive four named, unit-work jobs. Compare releases all at zero
versus booked releases [0,0,2,3], and a nonbinding eight-slot downstream station
versus one shared slot. Its service lasts two intervals per job, indivisible,
nonpreemptible, mandatory and unaffected by effort. The capacity contrast changes
total resources; it is a bottleneck sensitivity, not an isolated graph ablation
or an equal-budget comparison between network topologies.

Cross these four mechanisms with no response change and two equally likely
persistent response changes, nominal (1,1) until epoch two, then (0.5,1.5) or
(1.5,0.5). The same eight jobs, action grid, effort costs, nominal prior and
holding charge are used everywhere. All numbers are declared synthetic units,
not calibrated costs or clinically justified thresholds. No stochastic seeds
or prior scientific CRNs are used, and there are no independent replications.

Decisions: five intervals, four allocations (idle, left, right, balanced), a
one-interval commitment delay and a shared one-hour budget. After those five
decisions, drain already purchased effort, then use the same balanced-if-work
continuation for every method until all jobs complete downstream and all
commitments settle. Every elapsed holding, labor and switching cost is charged.
No terminal write-off, salvage value, free outsourcing or test bypass occurs.
Closure must finish within 64 additional intervals or the run fails. The
diagnostic optimizes only the first five decisions plus this fixed continuation,
not the unrestricted completion-horizon problem.

## Information and controllers

The common public interface includes named-job states, remaining support work,
pending effort, downstream occupancy and the entire booked release schedule.
Booked release times are not hidden future demand. Unknown response factors,
world identity and future change time never enter practical controller inputs.
Exact noiseless support-progress measurements are an optimistic measurement
assumption; no real-site availability is claimed. Only uncensored receipts can
update the shared response estimator. Every controller obtains its own receipts.

Compare fixed balanced effort, backlog feedback, adaptive backlog feedback,
fixed-model MPC and identification MPC. The MPC implementation, action set,
prediction horizon, nominal process and closure are identical; only response
estimation differs. Record planning transition queries. No neural policy,
offline policy training, online policy-weight update or GCN is run.

Using the same finite-world diagnostic as before, compute best open-loop,
best nonanticipative public-history policy and clairvoyant costs. The diagnostic
knows the finite response law, unlike the practical constant-response MPC.
It is a privileged model benchmark, not a fair deployment competitor. Its
nonanticipative policy is a fixed history-action map and requires no online
weight update. Even a strict feedback benefit would not establish online RL.

## Verification and decisions

Before recorded comparison, test FIFO support conservation, indivisible
completion, downstream occupancy and duration, delayed action effect, booked
arrival timing, no response leak, full terminal work and prepaid-hour closure,
bounded failure, immutable input state and valid actions. Check all row identities
and source/artifact hashes. An independent row auditor must recompute cost,
timing, dispatch, work and stage progression without calling the transition.
Independently enumerate open-loop and clairvoyant values from logged edges.
The nonanticipative solver remains regression-tested, not independently replaced.

Record raw baseline decisions, whole completion episodes and compressed exact
tree transitions. Expected 60 baseline episodes, 300 decision rows, and 16,368
exact tree edges. All signs, positive/null/adverse, are retained. Report absolute
synthetic cost differences; do not infer statistics from these deterministic
worlds or reuse the percentages in the manuscript's results table.

Interpret sequential control, public-information use and parameter updating
separately. If ID-MPC solves the restricted problem, record that cheap-control
result. If feedback has value but ordinary comparators fail, diagnose why before
training. Domain calibration and prospective, replicated action rankings remain
prerequisites for a patient-model DDPG campaign regardless of the observed sign.
The next step is selected from those gates, not an automatic neural launch.
