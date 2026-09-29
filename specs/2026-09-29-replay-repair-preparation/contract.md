# Prospective replay-repair preparation, not an experiment amendment

September 29, after Zhaowei renewed the request for continued local progress.
This packet implements an opt-in numeric contract and synthetic tests. It does
not modify the old replay buffer, agent, collector, cache, execution config or
results. No agent imports this module. No scientific experiment or formal
holdout use is authorized by this document.

## Why this comes before a new scenario

The [historical replay audit](../2026-09-29-formal-replay-contract/readout.md)
found discontinuous counterfactual records assembled into multi-step windows,
absolute versus anchor-relative reward mixing, and differing calibration/update
bootstrap conventions. A changed environment cannot by itself resolve these
target-consistency problems. Repairing them may or may not improve performance;
the old null result remains a result of the executed implementation.

## Prototype interface

Implementation: `src/rl/validated_returns.py`.
Tests: `tests/test_validated_returns.py`.

`ReplaySemantics` declares reward kind and definition ID, reward scale, gamma,
state/action schemas and dimensions, and explicit bootstrap-on-truncation policy.
No scientific reward choice, scale or discount is supplied as a module default.

`OneStepRecord` carries immutable numeric state/action/reward/next-state values,
termination and truncation separately, collector execution/source ID, origin,
trajectory ID/step where applicable, and full-state lineage tokens. A batch can
contain distinct sources/episodes with the same semantics; a multi-step window
cannot join their trajectories.

| Input condition | Required behavior |
| --- | --- |
| Independent one-step counterfactual | Admit as one step; never concatenate with another branch or trajectory |
| Actual continuous trajectory | Require same collector/source and trajectory, consecutive step indices, matching full-state tokens and exact next/current observation values |
| Terminal/truncation before window end | Reject rather than crossing into reset data |
| Short tail | Use the actual number of available steps; scheduling/flush ownership remains with a future collector adapter |
| Different reward definition, scale, gamma, state/action schema or truncation policy | Reject within the declared critic contract, even when numeric reward values happen to match |
| Legacy arrays without lineage/semantics | Reject automatic upgrading; missing metadata cannot be invented from matching observations |
| Nonfinite values, invalid dimensions/flags, target shape mismatch | Reject, including implicit broadcasting of a one-dimensional Q vector |

Lineage tokens should eventually identify the collector's full simulator and
RNG state under a documented hashing scheme; they are not hashes of the neural
observation alone. This prototype checks declared relationships, **not their
authenticity**. Fabricated matching tokens can pass. A collector provenance
audit and actual clone/trajectory tests remain necessary. No current teacher
cache has been relabeled or certified as a valid trajectory by this work.

## One target definition

For a valid window of actual length `n`, with unscaled rewards `r_i`:

```text
R = reward_scale * sum(gamma**i * r_i for i = 0, ..., n-1)
d = 0                                if terminated
d = 0                                if truncated and truncation bootstrap is disabled
d = gamma**n                         otherwise
y = R + d * Q_target(next_state, target_action)
```

The prototype stores a separately scaled first-step reward and returns the
**complete** bootstrap discount. A future adapter must not multiply by another
gamma. This differs from the existing buffer convention, where
`discount_multiplier = gamma**(n-1)` is multiplied by gamma in the main update.
The older calibration helper omits that multiplier. Do not pass the new field
straight into the old API and assume it is compatible.

`bellman_targets` is a NumPy reference for both prospective calibration and
ordinary critic-update implementations. It has not replaced either production
path, does not calculate anchor-relative rewards from raw environment outcomes,
and does not establish that a declared reward matches the scientific objective.
Absolute and relative conventions are supported for explicit tests but cannot
silently share one critic. The choice for a new study remains a protocol decision.

Truncation is not automatically termination: an artificial data-collection cut
and the modeled end of a finite-horizon task may require different targets.
The producer/protocol must state which occurred. When both flags are true,
termination masks bootstrap. A truncated record always ends the collection
segment even if its final state permits bootstrapping.

## Required future integration checks

1. A collector/loader adapter records true reward/action provenance and window
   lineage. Independent teacher counterfactuals remain one-step unless an actual
   branch rollout with verified continuation is collected.
2. Calibration and online updates consume the same target definition. Test their
   numeric targets directly, not just matching config strings or finite losses.
3. Counterfactual label horizon/replications/continuation and action support must
   have separate metadata. This target module does not repair teacher-ranking
   labels, choose a loss weight, filter support, or regenerate the old cache.
4. Check source/mode/schema rejection at resume and sampling, queue tail flushing,
   true termination versus truncation, and no accidental double reward scaling.
5. Freeze graph/flat feature, proposal-conditioned gate and action-head treatment
   independently. Both critics must receive the declared comparable information.
6. Only an approved, committed development protocol may enable a revised agent or
   launch training. Never use the closed formal outcomes to tune these repairs.

## Reproduction and result boundary

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_validated_returns tests.test_formal_replay_contract \
  tests.test_offline_replay_training
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Tests use hand-authored numeric trajectories and independent backward return
recursion, with no environment, model optimizer, rollout or outcome selection.
They establish the prototype's software behavior, not that current training is
fixed or that online DDPG will improve. The old source and evidence remain intact.

September 29 verification: all 19 new tests and the combined 109-test suite
passed, together with full Python compilation and `git diff --check`. The
combined suite includes offline replay, historical method/replay/graph audits,
attribution recovery, crossed evidence verification, and service-effort/queue
mechanics, observations and comparators. Existing `src/models`, `src/env`,
`src/baselines`, `src/rl/replay_buffer.py`, `experiments/configs`,
`experiments/evidence` and `reports` have no tracked differences from `04d8c67`.
A source-reference scan finds no agent integration of `validated_returns`.
