# Capped categorical PPO kernel acceptance

Date: 2026-09-30. Base checkout
`34ffdc17c97d0b1b635159cbd124db92dfde9571`, persistent local
`codex/september-research-integration` worktree. This is the next engineering
packet under Zhaowei's explicit continuation request. No scientific collection,
patient simulation, reward change or performance comparison ran.

## Implementation

`src/rl/candidate_ppo_kernel.py` owns an explicitly enabled, cloned candidate
policy, Adam optimizer, bounded pending closed segments, consumed trajectory
identities and separate CPU sampling/shuffle generators. It reuses the accepted
PPO objective, sealed candidate likelihoods, closed-segment GAE, typed replay
and actor-state decoder. Historical agents and the clean DDPG kernel are intact.

Settings explicitly declare learning rate, loss coefficients, clipping, GAE,
normalization, epochs, batch size, maximum pending steps, total rollout updates
and total optimizer steps. The fixture settings are not selected scientific
hyperparameters. An update must fit its entire epoch/minibatch count in the
remaining budget; the adapter never starts a partly affordable update.

Before admission/update, it recomputes GAE from raw typed rewards and verifies
old policy evaluations, including a declared truncation bootstrap, against the
current behavior snapshot. Duplicate and previously consumed trajectory step
identities are rejected. It does not accept old replay by relabelling it as PPO
data. Advantages are normalized once across the admitted rollout, not separately
in each minibatch or after each epoch. Every epoch visits every row exactly once,
including a short final minibatch, using the private shuffle RNG.

Current likelihoods use the recorded candidate support and original public
inputs; old likelihoods, values/returns and advantages remain fixed. All losses,
gradients, parameters and Adam moments must be finite. The gradient norm is
clipped before each optimizer step. A single shared optimizer updates the
candidate encoder, scorer and value head; this is not the historical separate
actor/critic PPO implementation or an isolated algorithm-only intervention.

`mode="frozen"` uses the same initial numerical policy and sampling behavior
but has no optimizer and rejects training buffers/updates. `mode="online"`
only enables parameter updates in this adapter. That name does not establish
scientific deployment-time adaptation or a positive online-RL contribution.

## Atomicity and recovery

A complete rollout update runs on a private copy. The live learner is replaced
only after all minibatches and the resulting full state validate. Exceptions,
including failure after a successful first optimizer step, leave live model,
optimizer, pending rollout, counters and private RNGs unchanged. There is no
automatic retry. A scientific runner must preserve/report a failure before
deciding whether any new attempt is authorized.

Checkpoints preserve model definition and initial-weight binding, input/reward
contract, settings, mode, Adam configuration/moments/step counts, pending sealed
segments, completed rollout sizes, consumed identities and both RNG states.
Load checks exact model layout/precision, optimizer mapping, finite moments,
nonnegative second moments, consistent budgets/history and recomputed pending
receipts on a private copy. Invalid loads cannot partially replace live state.

Saved envelopes contain a SHA256 payload checksum and use weights-only loading.
Publication uses a same-directory temporary file and a no-replacement hard link.
Existing files are not overwritten, and failed publication removes its temporary
file. Checksums detect mismatches, not fabricated provenance or maliciously
resealed data. This is not a signed audit record or a power-loss durability claim.

Recovery is at a **closed-rollout/update boundary** on this CPU runtime. It does
not include an environment, an unfinished collection segment or a mid-minibatch
cursor. Exact CPU continuation is tested, not bitwise GPU/MPS recovery. The
scientific runner still needs an independent wall-clock/query cap and collector
checkpoint contract; optimizer iteration caps alone do not provide those limits.

## Verification

26 new tests and 156 prior related tests pass: **182 tests in 5.516 seconds**.
Full repository compileall exits zero. Python 3.9.6, PyTorch 2.8.0; CPU float32
and float64 fixtures, no external dependency added.

The new tests cover graph, self-only and flat updates, explicit settings/caps,
frozen parity, receipt tampering, duplicate consumption, once-per-rollout
normalization, gradient clipping, single-class support, and an independent
first-Adam-step calculation from the clipped objective. They verify exact
continuation of the next update and next eight sampled decisions after a saved
pending-rollout checkpoint. Corrupted models, optimizer state, history, RNG,
candidate support and envelope checksums are rejected without live mutation.

Injected second-minibatch and nonfinite-gradient/moment failures verify full
atomicity. The following successful update matches an untouched reference
learner exactly. Python, NumPy and global CPU Torch RNG states are unchanged.
These are bounded updates on invented tensors/receipts, not scientific fits.

Development failures retained here: the initial 24-test run had one assertion
failure because the duplicate fixture also exceeded the rollout cap, which was
checked first. The fixture was separated from the cap case. The subsequent
26-test run had one test-mock error because its replacement recursively called
the patched candidate helper; retaining the original helper fixed the fixture.
Neither was a recorded research run or a scientific retry.

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests.test_candidate_ppo_kernel tests.test_candidate_ppo_objective \
  tests.test_candidate_policy_rollout tests.test_routing_candidate_contract \
  tests.test_formal_replay_contract tests.test_formal_graph_contract \
  tests.test_validated_returns tests.test_matched_inputs \
  tests.test_prospective_adapter tests.test_prospective_learner
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m compileall -q .
```

| File | SHA256 |
| --- | --- |
| `src/rl/candidate_ppo_kernel.py` | `c9bca67cb808423e72e17b344d3d6eac40558f6ab53cb20eb462171166b4e8f8` |
| `tests/test_candidate_ppo_kernel.py` | `4c83990822368e4575221dde24a6fafd8e8fa48a8e97b8b68327d9fc5c12c0d3` |

All eight preservation fingerprints from the research-direction memo match.
The default sandbox process query was denied; an approved read-only host query
found only the query's own shell/filter, no related workload. All test/compile
sessions ended. The `gcn-rl` heartbeat remains ACTIVE every 30 minutes, verified
from its saved configuration; no automation change was made this turn.

## Next decision

The next automatic step is a source-level public-input collector audit and
mocked preparation, with no patient environment steps. Verify observation
availability, original requests and inference precision, terminal liabilities,
candidate support, same-information baselines and collector recovery. Request
classes are still not a physical-feasibility mask, and graph/flat parameter
matching remains unresolved.

Then prepare one bounded scientific pilot decision packet and request specific
execution approval. No reward/scenario/model search, new research training,
formal holdout, remote Git, messaging or Howard sign-off is implied. Stage E
stays closed; R6 test labels are not untouched validation. The finite heartbeat
must end when only new-scope execution approval remains.
