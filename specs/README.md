# Project Specs — Research & Paper Plan

This directory is the **plan of record** for our paper on graph-aware deep
reinforcement learning for capacity planning in perishable, patient-specific
cell-therapy manufacturing networks. It follows a spec-driven development (SDD)
workflow: each roadmap phase is planned as a dated feature spec before
implementation.

## Why this structure

A research paper is part argument, part software, part experiment campaign, so
the plan is split so each concern can evolve independently:

| File | Purpose | Changes |
|------|---------|---------|
| [`mission.md`](mission.md) | The research argument: problem, contributions, differentiation wedge, scope, non-goals | Rarely |
| [`tech-stack.md`](tech-stack.md) | The system: codebase architecture, environment-extension strategy, algorithm roster, reproducibility rules | When a technical decision is made |
| [`roadmap.md`](roadmap.md) | The sequenced execution plan (phases, dependencies, pilot/validation gates) | As phases ship |
| [`research-context.md`](research-context.md) | Literature map, closest rivals, differentiation axes, citation candidates, reviewer-objection checklist | As the lit review firms up |

Each roadmap phase is scoped to later become its own `feature-spec` (a dated
`specs/YYYY-MM-DD-<phase>/` directory with `requirements.md`, `plan.md`,
`validation.md`) when we start it.

## Status

**Planning draft — awaiting review.** No code or manuscript has been changed to
produce this plan. Nothing here is committed as a decision until reviewed.

## Two standing constraints (see `tech-stack.md` for the full list)

- **English only** everywhere in this repo.
- **No fabricated results.** Every table/figure/claim must trace to a logged
  experiment output. Prose in `specs/` may describe *intended* experiments, but
  must never present projected numbers as if measured.

## A note on confidentiality

These specs encode pre-publication research strategy (our differentiation angle,
unpublished direction). Keep `specs/` in a **private** repository. The source
materials the plan draws on live in `docs/research_howard/`, which is
git-ignored on purpose — the plan references them by pointer and never
reproduces their contents.
