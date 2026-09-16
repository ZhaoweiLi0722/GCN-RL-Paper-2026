# Measuring remaining policy improvement: design proposal

Date: 2026-09-15. Based on commit `0f9af1ddd297063a6a3cce8c630b8090b793fc91`.

Status: **design only; not an execution authorization or frozen protocol**.
The routing-primary campaign remains at Stage E. No new simulator rollout,
training, optimization experiment, or formal-holdout reuse was performed.

## Question and estimand

Measure how far MDL-2 and the frozen pretrained graph controller are from the
best policy allowed under the same environment, objective, observations, and
operational constraints. Distinguish this from the observed benefit of online
DDPG updates.

For a baseline B, let J_B be its expected full-episode cost and J* the minimum
expected cost over the specified admissible policy class. The desired gap is
G_B = J_B - J*, or 100 G_B / J_B as a percentage of baseline total cost.

An exact stochastic optimum for the full 20-clinic model is not presently
available. A useful alternative is a bracket:

    L <= J* <= J_P
    J_B - J_P <= G_B <= J_B - L

Here L is a mathematically valid lower bound on optimal expected cost, and P is
a fixed, feasible policy using only allowed decision-time information. Include
B among the candidate feasible policies. These inequalities concern population
expectations; empirical estimates require uncertainty intervals and independent
evaluation after policy selection.

A better feasible policy proves available improvement. Failure to find one
does not prove near-optimality. A sufficiently tight lower bound can certify
near-optimality. A loose lower bound leaves the question unresolved.

## Separate action restrictions from learning limitations

Analyze two nested operational classes, with the same information contract:

| Class | Allowed controls | Question |
| --- | --- | --- |
| Restricted residual | Specimen-only corrections within the formal execution envelope; other controls follow the same anchor | Is useful improvement available within the proposed controller's operational scope? |
| Existing full controls | All controls already allowed by the current simulator, with original physical limits | Does the residual restriction itself exclude valuable decisions? |

The restricted class should include the executed baseline policy exactly.
Extract its support from checkpoint/config composition, correction gating,
scaling, and integer projection; do not equate the historical five-option
diagnostic ladder with every action the actor can execute.

Keep capacities, demand, patient dynamics, costs, timing, and automatic production
unchanged. Adding overtime, changing weights, or allowing new production
scheduling decisions would measure a different problem.

A full-control cost lower bound is also a valid, possibly looser bound for the
restricted class. Failure of a five-action search cannot certify optimality
over either larger class.

## Measurement A: achievable improvement using an information-matched planner

Build a bounded rollout/search comparator, using existing candidate generation,
feasibility projection, and rollout infrastructure where appropriate. At each
decision, compare legal actions using simulated alternative futures, execute
one action, and replan. Evaluate the resulting policy over complete episodes.
Use the remaining episode horizon for candidate valuation where feasible; any
truncation needs an explicit terminal-value treatment.

Freeze planner settings on separate development data. Compare MDL-2, frozen
GCN, fixed final GCN, and the planner under identical evaluation worlds. Report
planner-versus-MDL-2 and planner-versus-frozen differences separately. Include
clinical guardrails and computational cost; an expensive feasible policy is a
useful achievable benchmark even if it is impractical to deploy.

**Information audit is mandatory.** Current lookahead utilities deepcopy the
live simulator, including patient records, before reseeding future draws.
Reseeding does not remove latent patient attributes already sampled. A planner
using hidden health/deterioration variables, true unknown demand parameters,
the live future RNG stream, or patient details unavailable to the policy is not
an information-matched comparator. Such access must be removed or replaced by
sampling conditional on allowed observations/history; otherwise label the
result a privileged-information diagnostic, not achievable policy headroom.

Finite simulation/search does not inherit an unconditional policy-improvement
guarantee. A negative planner result remains inconclusive about optimality.

## Measurement B: certified optimistic cost bound

Construct a perfect-information optimization benchmark that knows exogenous
future arrivals, disruptions, and patient trajectories. Because it has more
information, its minimum cost is no greater than that of any implementable
policy under the original information restriction.

An exact deterministic formulation or a provably optimistic relaxation can
provide a per-world lower bound. Average these bounds over independently drawn
worlds, with statistical uncertainty, to bound expected optimal cost.

Required validity checks:

- Preserve or relax, never silently tighten, identity, routing, resource,
  transit, production, return, and expiry constraints.
- Preserve the objective or prove that the relaxed objective is no larger
  than the original cost for every mapped feasible trajectory.
- Demonstrate that each original feasible trajectory maps to a feasible
  solution of the relaxation. This is the central lower-bound argument.
- If automatic production is relaxed into freely chosen scheduling, explicitly
  identify it as an optimistic relaxation, not an executable comparator.
- Use a solver-certified lower bound or proven global optimum. A feasible
  hindsight solution from a heuristic search is not a certified cost lower
  bound merely because it has future information.
- A relaxation may drop clinical guardrails to remain optimistic; state this.
  Feasible comparator policies must still meet the chosen guardrail contract.
- Retain the original episode boundary and treatment of unfinished patients.
  Changing terminal accounting changes the estimand.

A large perfect-information gap does not prove that an observable-state policy
can capture that gap. It may reflect the value of future or hidden information.
If the bound is too loose, report that limitation. Tighter information-relaxation
bounds are a possible later method, not a promised cheap implementation.

This construction follows the established distinction between feasible-policy
values and information-relaxation bounds; see
[Brown, Smith, and Sun (2010)](https://scholars.duke.edu/publication/762575).

## Bounded development sequence proposed for review

1. **Formulation and information audit.** Document observation versus simulator
   state, executed action support, objective terms, and the mapping establishing
   the lower bound. Inventory source snapshots and archived policies.
2. **Small-instance verification.** Use a finite, explicitly discretized
   two-clinic test problem with a horizon long enough for manufacturing and
   transport consequences. Verify any exact solution by enumeration where
   feasible, replay feasible plans, and test the bound against known policies.
   Exactness applies only to this reduced problem, not automatically to the
   original continuous/stochastic model.
3. **Computational feasibility pilot.** On fresh development-only worlds, assess
   whether a valid bound is tight enough and a full-episode planner affordable.
   Record solver bounds, search budgets, and wall time. If bounds remain loose,
   declare the proposed certificate uninformative; do not report near-optimality.
4. **Freeze a separate audit protocol before measurement.** Lock the four
   existing scenarios, source/checkpoint hashes, new nonoverlapping seed ranges,
   policy selection rule, solver limits, evaluation sample size, uncertainty
   method, and practical-effect threshold. Determine compute feasibility before
   committing to a sample size; do not promise full-scale exact optimization.
5. **Evaluate once on fresh worlds.** Preserve shared-world dependence across
   policies and training seeds, fixed scenario weights, and paired comparisons.
   Report simultaneous uncertainty for the gap bracket rather than combining
   unrelated marginal 95% intervals as if jointly valid.

Practical materiality must be chosen from the intended decision and computational
burden, not from the observed result. Any illustrative 0.5% or 1% threshold is
not automatically a scientifically or clinically justified cutoff.

## Reading the result

| Evidence | Supported interpretation |
| --- | --- |
| Planner materially beats frozen GCN inside the residual class | The current learning procedure leaves usable improvement uncaptured |
| Improvement appears only under broader existing controls | The residual action restriction is a material limitation |
| Upper confidence bound on remaining gap is below the prespecified practical threshold | Baseline is near-optimal to that tolerance for the stated model, policy class, and scenario mixture |
| Large clairvoyant gap but no better information-matched planner | Cannot distinguish information limits from inadequate planning/learning |
| No better planner and a loose lower bound | Unresolved; not evidence of no room |

Report both MDL-2's total gap and the frozen GCN's remaining gap. Report graph
versus flat and online final versus frozen as separate attribution questions.
Do not sum improvements from isolated overlapping state probes into an episode
gain or treat a lead-time perturbation cost as an optimality gap.

## Existing resources and execution boundary

Reusable starting points are `evaluation/network_residual_headroom.py`,
`evaluation/audit_frozen_specimen_route_headroom.py`,
`src/rl/residual_options.py`, and the frozen publication summaries. Existing
historical teacher gains used different action/configuration settings and do
not supply the proposed routing-primary certificate.

No certified full-horizon optimality benchmark was found in the reviewed
evaluation modules. The formal routing-primary raw/checkpoint root is absent
from this checkout; its archived inputs must be made available before a
comparison involving those exact policies can run.

The [locked execution plan](../../docs/patient_indexed_specimen_routing_locked_execution_plan.md)
requires explicit approval of a scientific amendment, an appended change-control
entry, and a committed plan update before a changed experiment launches. The
proposal above is a new diagnostic study; it must not reuse the formal holdout
or overwrite any existing artifacts. Complete and approve its execution
protocol before launching either the simulator pilot or final measurement.
