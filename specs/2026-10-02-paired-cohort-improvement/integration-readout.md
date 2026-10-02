# Substantive next step after the completed null

2026-10-02. Preparation only, zero new scientific calls.

## Delivered

- `src/rl/paired_cohort_objective.py`: differentiable all-candidate expected
  cost loss, detached raw diagnostics and explicit paired seed declarations.
  Unlike sampled PPO credit, every candidate receives a supplied rollout cost.
- `src/rl/paired_cohort_plan.py`: exact remaining-horizon branch descriptors
  and non-refundable maximum accounting for the proposed complete comparison.
- `experiments/configs/paired_cohort_improvement_20261002.json` and protocol:
  one numerical package, including collection, conditional branches, actor/BC
  fits, controls, independent tests, readout, preservation and stop rules.
- 28 artificial forward/gradient/accounting tests passed, including a one-unit
  difference on a1e9cost reaching float32 actor gradients without label rounding.
  These tests use no optimizer steps or patient models; they are not RL results.
  Full repository compileall also passed.

## Decision-changing evidence reused

The completed recovery2 readout already establishes the exact observed greedy
null. Do not rerun that matrix or its archives. The saved continuation readout
at `reports/2026-10-01-s1-continuation-recovery/readout.md` already establishes
non-reference exploration and changed probabilities; it is an earlier attempt,
not a fresh inference over the new cohort-trained checkpoints. Together these
justify testing different action credit, not asserting a diagnosed causal fault.

This candidate deliberately avoids another reward-weight change. Full-horizon
cost accounting was already tested and did not change greedy behavior. Whether
individual decisions have useful conditional effects under that same objective
remains uncertain. Full-cohort branches differ from the older short-horizon
labels, but may still be noisy or offer no usable headroom. One direct comparison,
not an indefinitely escalating signal-screen ladder, will decide this mechanism.

## Remaining blockers and authority

The pure objective is usable but the scientific runner is NOT ready: a new
versioned snapshot/conditional-future branch collector, actor-only update wrapper,
serial entry and raw verifier still need integration with existing budget,
restore, recording, sealing and archive facilities. The old strict PPO collector
must not be silently repurposed. Source inspection confirms prefix snapshots
contain environment RNG and patient/queue state; exact branch integration still
needs zero-update mock tests and the approved counted preflight, not an uncounted
real environment probe.

Scope-specific execution approval and final committed source/runtime/input/seed
locks are also pending. The existing `gcn-rl` automation remains PAUSED; no
training or evaluation process is running. No full historical audit repeated.

One advancement agent completed the pure objective and22artificial tests, then
closed. Coordinator integrated the float64-to-float32 gradient regression and
branch-plan delivery. The one read-only efficiency agent advised reusing valid
checks and making one different finite comparison, not adding gates; it is closed.
No role definition is reported as a background-running agent.
