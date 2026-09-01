# Intertemporal Shared-Capacity Study

Status: **closed at J2; no policy training authorized**.

This study is separate from the closed continuous-overtime line in PR #9. It
does not reinterpret or replace that result. The earlier action acted in the
same epoch and was mostly captured by a tuned constant. This study asks a new
question: can online control add value when capacity must be reserved before a
scheduled referral wave, persists across epochs, and is allocated from a
shared network pool?

## Review state

- Zhaowei authorized mechanics development and cheap screening on 2026-09-01.
- Howard scientific review is pending.
- Formal confirmation, the Stage E routing holdout, and policy training are
  prohibited until both authors approve a later algorithm protocol.
- The prospective J2 fixed-state gate failed on 2026-09-01. This configuration
  cannot proceed to J3, J4, or DDPG training; see `outcome.md`.

## Dependency and isolation

- Development base: publication-audited PR #9 follow-up branch, commit
  `5fcba9d`.
- All behavior is behind default-off configuration fields.
- Existing scenarios, checkpoints, evidence, and result roots are immutable.
- New development seeds and output roots must be used.

## Decision rule

Mechanics are implemented first. The action channel is screened before any
agent is trained. It must pass all four gates in `protocol.md`: continuous
execution, state-dependent headroom beyond strong comparators, fresh-stream
label stability, and held-out-state ranking/value capture. A failure closes or
redesigns the channel; it never triggers algorithm tuning against a failed
environment.
