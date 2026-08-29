# Continuous Overtime Control (Step-0 Draft)

> **STATUS: DRAFT — NOT AUTHORIZED.** This is the step-0 change-control artifact
> for a NEW STUDY under the locked routing-primary execution plan
> (`docs/patient_indexed_specimen_routing_locked_execution_plan.md`). The
> routing-primary campaign is closed at Stage E; nothing here amends it, reopens
> its formal holdout, or launches any experiment. This spec becomes active only
> after (1) Howard and Zhaowei sign off, (2) a change-control entry is appended
> to the locked plan naming this spec, and (3) the signed spec is committed
> before any implementation or run.

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
| Principal investigator | Howard Tseng | _pending_ | |
| Collaborator / reviewer | Zhaowei Li | _pending_ | |
