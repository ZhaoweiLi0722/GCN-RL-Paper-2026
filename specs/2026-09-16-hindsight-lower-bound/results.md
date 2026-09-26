# Certified perfect-information lower bound on optimal cost (Measurement B pilot)

Date: 2026-09-16. Branch `e2b-run`. This is the computational-feasibility
pilot (step 3) of `specs/2026-09-15-optimality-gap-measurement/plan.md`, run
on the development seed stream 99.7M. No training, no formal-holdout seed, no
existing artifact changed. Tools: `evaluation/hindsight_lower_bound.py`
(coarse, network-level) and `evaluation/hindsight_lower_bound_site.py`
(per-site, per-patient-profile; the one reported here). Artifacts: per-world
rows, run summary, and the joined comparison under `artifacts/`.

## Construction

Per world, exogenous randomness is fixed and known: arrivals per clinic per
epoch, supplier availability per clinic per epoch, and every patient's health
attributes drawn at enrollment. All three were verified policy-independent by
comparing traces under MDL-2, MDL-3, and a uniform-random policy on the same
seed (identical). Survival is a deterministic function of age and attributes,
so each patient has a hindsight-known first ineligible age a*; this predicted
all ~51,000 outcomes of eight simulated episodes with zero violations.

A linear program over patients grouped by (origin, arrival epoch, min(a*, 9))
keeps only constraints the simulator provably implies and only costs every
trajectory provably pays: start windows (ages 0–5), one reagent per start,
three-epoch bioreactor occupancy, per-site bioreactor and reagent stocks with
transfers on the simulator's edges and exact per-edge delays, one specimen
route per patient with one-epoch transit, purchase caps under supplier
availability, site idle and reagent caps with disposal, per-facility net
transfer caps, per-site bioreactor- and reagent-shortage penalties, loss and
material-waste penalties for expired, ineligible, and doomed-start patients,
and reagent purchase cost. Relaxed: transfer, holding, and urgency costs;
integrality; reagent and slot use by doomed starts.

Validity checks, every world: the relaxed objective evaluated on the
simulator's own MDL-2 trajectory never exceeds the simulator cost (it captures
about 99.4% of it), and the LP optimum never exceeds that relaxed value.

## Result: the bound is valid and too loose to locate the optimum

200 worlds (50 per routing scenario), HiGHS interior point, about 80 seconds
per world for the two objectives.

| scenario | certified LB | MDL-2 | gap bound, MDL-2 | MDL-3 | GCN-TD3 on MDL-2 | GCN-TD3 on MDL-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal history | 1,527.5M | 2,524.6M | 39.5% | 35.9% | 39.2% | 35.8% |
| abrupt regime shift | 1,219.0M | 2,159.1M | 43.5% | 40.1% | 43.3% | 40.1% |
| regional drift | 1,391.3M | 2,374.4M | 41.4% | 39.5% | 41.0% | 39.5% |
| compound regional stress | 2,375.9M | 3,615.9M | 34.3% | 32.2% | 33.7% | 31.9% |
| **pooled** | **1,628.4M** | | **39.0%** | **36.3%** | **38.6%** | **36.2%** |

What is certified: no policy of any kind can beat 1,628M on average on these
worlds, so MDL-2's optimality gap is at most 39% and the best configuration
measured so far (MDL-3 plus the GCN residual) is within 36% of optimal. That
does not distinguish among the policies, whose mutual differences are 0.2 to
4 percent.

## Why it is loose, and why a linear relaxation cannot fix it

Adding the three constraints suspected of carrying the slack (site idle cap
with disposal, site reagent cap with disposal, per-facility net transfer caps)
moved the bound by 0.03% on a test world. The slack is structural:

1. **Information.** The optimizer knows every future arrival and every
   patient's health in advance. This is the information-relaxation gap the
   proposal anticipated; removing it needs Brown–Smith–Sun style penalties,
   not more constraints.
2. **Control authority the environment does not grant.** The simulator forces
   every site to start its most-urgent waiting patients whenever a slot and a
   reagent exist (`production = min(waiting, idle, reagents)`, queue sorted by
   survival). The LP may decline to start a patient whose manufacturing would
   fail and may hold a slot for a better-timed arrival. No implementable
   policy has that authority, so the bound is over a strictly broader control
   class than the one the paper's policies operate in. The forced-start rule
   is a non-convex equality (a minimum) and cannot be encoded in an LP lower
   bound.

Per the proposal's reading table: *no better information-matched planner and a
loose lower bound → unresolved; not evidence of no room.*

## The number that is useful: a certified floor on patient loss

Re-solving each world with the objective set to lost patients only gives a
certified minimum loss under the same relaxation, that is, the fewest patients
even a clairvoyant scheduler with start-selection authority could lose:

| scenario | arrivals per episode | certified minimum loss | MDL-2 | GCN on MDL-3 |
| --- | ---: | ---: | ---: | ---: |
| nominal history | 5,817 | 1,568 (27.0%) | 2,224 (38.2%) | 2,176 (37.4%) |
| abrupt regime shift | 5,353 | 1,198 (22.4%) | 1,856 (34.7%) | 1,822 (34.0%) |
| regional drift | 5,676 | 1,412 (24.9%) | 2,112 (37.2%) | 2,124 (37.4%) |
| compound regional stress | 6,997 | 2,558 (36.6%) | 3,363 (48.1%) | 3,311 (47.3%) |
| **pooled** | | **28.3%** | **40.1%** | **39.6%** |

Of the roughly 40% of arriving patients the network loses, at least 28 points
are a structural consequence of capacity relative to demand and are beyond
the reach of any decision policy. At most 12 points are addressable by
control at all, and the full spread among every policy evaluated in this
project, from the executed MDL-2 to the best learned configuration, is 0.5
points. This is the sentence the motivation section needs: control matters
at the margin of a system whose losses are dominated by capacity, and the
paper should say so before it says anything about learning.

## Not established

- How far any implementable policy is from the implementable optimum. That
  requires Measurement A (an information-matched, forced-start-respecting
  planner) or a penalized information relaxation. Neither was attempted.
- The compound-stress bound uses the same 50 worlds as the others but its
  scenario has more arrivals and a higher absolute floor; no cross-scenario
  inference is intended.
- These are development-stream worlds. A formal statement would rerun the
  same code on a preregistered stream after the protocol is approved.
