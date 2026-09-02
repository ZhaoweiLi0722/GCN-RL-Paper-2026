# Intertemporal Residual Headroom Screen

> **STATUS: IMPLEMENTED BUT NOT AUTHORIZED FOR SCIENTIFIC EXECUTION.** Zhaowei
> approved preparation on 2026-09-02. Howard's explicit review of this exact
> action library, seed reservation, and gate is still pending. The executable
> refuses to generate states or create a result directory until both approval
> records are true.

## Why this screen exists

The closed J2 screen compared ten global policy parameterizations. Its
strongest frozen comparator was graph forecast with full shared budget and
0.25 graph smoothing. The prospective improvement from choosing among those
ten policies was 0.0692%, and even the optimistic validation oracle reached
only 0.0936%, both far below the 0.5% gate.

That negative result is binding for global policy tuning, but it does not span
the per-facility continuous action directions available to a DDPG actor. This
screen tests the one remaining narrow question: while holding the total
overtime request fixed, can moving a small amount of capacity from one
facility to another create material, reproducible value around the strongest
graph-forecast allocation?

This is an action-support audit, not an RL experiment. Passing it would only
authorize a separate observable-state ranking screen. It does not authorize
DDPG training, a formal holdout, or a manuscript performance claim.

## Files

- `frozen_protocol.md`: scientific contract, candidates, seeds, gates, and
  decision consequences.
- `validation.md`: no-seed validation commands and execution-lock checks.
- `experiments/configs/intertemporal_residual_allocation_headroom.json`:
  machine-readable contract with Howard approval set to `false`.
- `evaluation/screen_intertemporal_residual_allocation_headroom.py`:
  deterministic candidate mechanics and prospective evaluator.

## Sign-off record

| Role | Name | Protocol review | Scientific execution |
| --- | --- | --- | --- |
| Collaborator | Zhaowei Li | Approved 2026-09-02 | Approved for this screen only |
| Principal investigator | Howard Tseng | Pending | **Not authorized** |

No result directory belongs in this branch before the pending row is resolved
in a new reviewed commit.
