# Stochastic Procurement Lead Times (Step-0 Draft)

> **STATUS: DRAFT — NOT AUTHORIZED.** A new study, not an amendment to any
> closed campaign. It proposes changing the environment so that the incumbent
> heuristic is *structurally* misspecified rather than merely suboptimal.

> **Execution record.** An implementation and exploratory variability
> diagnostic were subsequently run while both sign-offs below were still
> pending. That diagnostic is retained for transparency, but it did not
> execute the optimality-gap gate defined in the draft protocol and is not a
> preregistered gate result. See [results.md](results.md) for the deviation and
> the narrower conclusion supported by the evidence.

## Why this exists

Every screen so far says the same thing: this simulator's decisions are
heuristically saturated. The measured ceilings are all sub-1% — the formal
result is 0.658% over MDL-2, and the overtime channel's entire
state-dependent budget was 0.45% on held-out data. That is not a weak method;
it is a well-specified heuristic operating in a regime it suits.

The operations-research literature quantifies the way out. In one paper's
three testbeds the best heuristic's own optimality gap was **0.8%** under lost
sales, **4.1%** under perishability, and **7.1%** under random lead times, and
the learned method's margin scaled almost exactly with it (−0.9, −4.0, −4.8
percentage points). **The size of an achievable learned-policy win is bounded
by how misspecified the incumbent heuristic is.** Chasing a bigger number by
better learning is therefore the wrong lever; changing the regime is the right
one.

Our environment already has perishability (6-epoch specimen shelf life,
2-epoch finished product). What it does not have is **lead-time uncertainty**:
reagent procurement is effectively deterministic and one epoch long.

## The latent hook

`src/baselines/heuristics.py` already reads

```python
reagent_lead = int(getattr(env.config, "reagent_purchase_lead_time", 0))
```

and folds that lead into its order-up-to horizon — but **no such field exists
anywhere in `src/env/`**, so the `getattr` silently returns 0. The planner
knows how to buffer for a procurement lead time; the simulator has never given
it one. That makes this regime change materially smaller than it appears, and
it means the heuristic can be made lead-time-aware without inventing new
planning logic.

## What makes this a fair test rather than a strawman

A learned policy beating a heuristic that was never told lead times exist
proves nothing. The study therefore requires, before any learned comparison:

1. A **lead-time-aware MDL-2** using the existing `reagent_purchase_lead_time`
   pathway, tuned on development data.
2. A measurement of that tuned heuristic's **own optimality gap** against a
   hindsight benchmark. This is the ceiling on any learned claim, and it is
   the study's first deliverable — if the gap stays near 1%, the regime change
   failed and the study stops there.

## The gating protocol still applies

Everything learned from the overtime study carries over unchanged: headroom,
then value of state-dependence against a tuned constant, then label stability
on fresh streams, then learnability from decision-time state — with a held-out
scenario and seed families reserved **before** exploration begins, enforced in
code.

- [Protocol](protocol.md)

## Sign-off record

| Role | Name | Decision | Date |
| --- | --- | --- | --- |
| Principal investigator | Howard Tseng | _pending_ | |
| Collaborator / reviewer | Zhaowei Li | _pending_ | |
