# Continuous Overtime Control (Step-0 Draft)

> **STATUS: SIGNED OFF 2026-08-29 — E1–E4 AUTHORIZED, NO TRAINING.** This is
> the step-0 change-control artifact for a NEW STUDY under the locked
> routing-primary execution plan
> (`docs/patient_indexed_specimen_routing_locked_execution_plan.md`). The
> routing-primary campaign remains closed at Stage E; nothing here amends it,
> reopens its formal holdout, or launches any training. Authorization covers
> flag-gated implementation (E1) and the three evaluation-only screens
> (E2–E4). Stage E5 (any training) requires its own spec and sign-off.

This specification designs the environment-first phase of the follow-up study
motivated by the online-RL attribution null (Stages F1/G0/G1/H0-H1). It adds one
primary continuous decision — per-facility **overtime capacity** — and one
optional secondary decision — a **production-rate throttle** — to the
patient-condition capacity-planning environment, together with the extended
heuristic field and the pre-training screening gates that must pass before any
learning algorithm is trained on the new channel.

Motivating analysis: `docs/online_rl_attribution_postmortem_and_followup_brief.md`.

- [Requirements](requirements.md) — decision semantics, cost model, invariants,
  prohibitions
- [Experiment plan](plan.md) — staged gates E0–E5 with prospective pass/fail
  criteria
- [Validation](validation.md) — regression, equivalence, and screening
  validation commands

## Sign-off record

| Role | Name | Decision | Date |
| --- | --- | --- | --- |
| Principal investigator | Howard Tseng | Approved | 2026-08-29 |
| Collaborator / reviewer | Zhaowei Li | Approved (via direct message to Howard, quoting the proposal summary) | 2026-08-29 |

Pre-merge self-review amendments (2026-08-29): borrowed-capacity accounting
with a fleet-conservation invariant (the original "never enters the shift
register" wording would have silently inflated the fleet on lot completion);
action slice `[4n:5n]` and `u_ot = (a+1)/2` mapping pinned;
`base_capacity = initial_idle_bioreactors` pinned; overtime cost charged on
committed surge; MDL-2-OT surge rule extended with a reagent-sufficiency
condition; dormant-throttle activation rejected by config validation.
