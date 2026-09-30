# Candidate PPO objective acceptance

Date: 2026-09-30. Base checkout:
`3fd16bfebdd00369ac4d973341e5540dd2279bdb`, persistent local
`codex/september-research-integration` worktree. This is synthetic-only
engineering preparation. The capped optimizer adapter, complete update-boundary
recovery and public-input collector audit remain pending. No scientific run or
positive RL result is claimed.

## Completed implementation

`src/rl/candidate_ppo_objective.py` adapts the existing
`src/baselines/ppo.py::_update_from_rollout` loss arithmetic:

- Clipped categorical policy surrogate, with positive and negative advantages.
- Unclipped value mean squared error, without an implicit one-half multiplier.
- Negative entropy bonus in the minimized combined loss.
- Optional population-standard-deviation advantage normalization over the full
  rollout, before minibatching. Constant and singleton cases are explicit.
- Detached old log probabilities, advantages and returns. Only current policy
  log probabilities, values and entropies retain gradients.
- Strict finite, shape, dtype and device checks. No implicit broadcasting or
  silent importance-ratio overflow repair. Diagnostics are detached.

Candidate logits and likelihoods still come from the sealed-support interface
in `candidate_rollout.py`. The objective alone cannot authenticate that an old
likelihood came from a real behavior policy. Its future caller must reproduce
the receipt and freeze support, observations, precision and old quantities.

This reuses PPO mathematics, not the historical agent's full implementation.
The candidate encoder is shared by policy and value heads, so both losses
contribute to its gradient; the historical agent has separate actor/critic
networks. There is no new dependency, optimizer step, value clipping, KL cutoff,
candidate search, collection loop or experimental hyperparameter selection.

## Verification

The 13 new tests cover hand-calculated clipping and value/entropy arithmetic,
gradient direction and clipped branches, a finite-difference check, detached
old quantities, normalization, malformed inputs and numeric overflow. Invented
graph/self-only/flat receipts integrate with the loss: unchanged-policy ratios
are one, gradients are finite and weights remain unchanged without an optimizer.

First focused run: 69 tests passed in 0.136 seconds. Final related run:
156 tests passed in 4.746 seconds using project Python 3.9.6 / PyTorch 2.8.0.
Full repository compileall passed. Both commands exited zero.

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests.test_candidate_ppo_objective tests.test_candidate_policy_rollout \
  tests.test_routing_candidate_contract tests.test_formal_replay_contract \
  tests.test_formal_graph_contract tests.test_validated_returns \
  tests.test_matched_inputs tests.test_prospective_adapter \
  tests.test_prospective_learner
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m compileall -q .
```

New-test inputs are invented tensors/receipts, with forward/backward operations
only. Existing related DDPG tests include bounded invented-tensor optimizer
steps. No new patient trajectory, scientific fitting, research campaign or
performance comparison ran. The read-only host process check found only its
own matching shell/filter, not a related research workload.

SHA256 at acceptance:

| File | SHA256 |
| --- | --- |
| `src/rl/candidate_ppo_objective.py` | `8439a415174087ea61c34753dae7027080791d0ddc4a0d3a7a217002ca809b24` |
| `tests/test_candidate_ppo_objective.py` | `fd46a558b37c8f22f6d76737bece9102d921cf1da2ccc0c1ce72f42cf1479145` |

All eight preservation fingerprints in
`docs/team_updates/2026-09-30-gcn-rl-research-direction.md` still match, including
the prior readouts, clean DDPG kernel, return and input contracts, and G0 report.

## Continuation and stop boundary

Zhaowei explicitly requested automatic continuation. A new active in-thread
heartbeat, `gcn-rl`, was created and independently checked in its configuration
on 2026-09-30, every 30 minutes. This is a newly scoped engineering task, not
revival of a closed experimental campaign or evidence of a running learner.
The previously paused `monitor-pc-conservative-td3` task was not modified.

The finite authorized chain is:

1. Implement and test capped categorical PPO updates, atomic failure rollback
   and full update-boundary model/optimizer/rollout/private-RNG restoration.
   Use invented fixtures only and reuse the repository's existing core.
2. Source-audit and prepare the public-input collector without stepping the
   patient environment. Check information parity, request precision, terminal
   liabilities, support coverage and recovery using mocks/invented fixtures.
3. Draft one bounded scientific pilot with named arms, initialization, same-start
   frozen/no-RL comparison, graph attribution limits, clean DDPG comparability,
   outcomes, fresh streams and per-arm query/update/time caps. Request explicit
   execution approval; do not launch it automatically.

Each continuation must inspect actual HEAD, edits, outputs and processes, run
relevant tests plus compileall for code edits, update the Live checkpoint and
preserve historical evidence. Local commits are permitted. The task must delete
itself when this finite chain is complete or only new-scope approval remains.

No reward/scenario/model search, new scientific fitting, real patient episodes,
formal holdout, remote Git, messages or cloud action is authorized by this
automation. R6 test evidence is not untouched confirmation; Stage E stays
closed; operational calibration is missing; Howard approval is not inferred.
