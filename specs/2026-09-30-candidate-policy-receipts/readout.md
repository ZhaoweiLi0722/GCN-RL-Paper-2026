# Candidate scoring and on-policy receipt acceptance

Date: 2026-09-30. Base checkout:
`c090af7486414aab60d1ba276565a27cf99d5709`, persistent local
`codex/september-research-integration` worktree. This implements the next
engineering packet authorized by Zhaowei's continuation request. It is not a
scientific execution protocol or evidence of a positive RL effect.

## Completed implementation

- `src/models/candidate_policy.py`: opt-in CPU graph/flat categorical candidate
  scorer and state-value baseline, reusing `GraphConvolution` and matched-input
  validation. No environment, optimizer, algorithm registration or checkpoint
  migration. Widths, architecture, message mode and initialization seed are
  explicit constructor settings, not proposed experimental hyperparameters.
- `src/rl/candidate_rollout.py`: immutable behavior evaluations, candidate
  decisions, explicit-generator sampling, old-snapshot reproduction, new-policy
  likelihood evaluation and closed-segment return/advantage preparation.
- `src/rl/routing_candidate_contract.py`: factors the existing request precision
  check into a callable pre-collection guard. Sampling failure, including a
  precision failure, restores the caller-supplied RNG before raising. No action
  is returned for execution, and no synthetic transition is manufactured.
- `tests/test_candidate_policy_rollout.py`: 32 invented-input tests. Historical
  DDPG/PPO agents, environment dynamics, rewards, configs and results are intact.

## Information and attribution boundaries

Every scorer receives the same raw node features, globals, physical-link matrix,
specimen-anchor request and reference request. Per-class features contain the
canonical integer-request features and reference/anchor membership flags. The
same scorer weights are applied to every class; there is no candidate-position
embedding. Token strings, hashes, realized outcomes and alias multiplicity are
not neural features. Option permutation and repeated aliases leave scores and
probabilities unchanged on the fixed synthetic fixture.

The graph encoder uses the existing normalized physical message operator.
Self-only replaces that operator with identity while preserving physical-link
metadata and the same initial weights. Flat uses a dense encoder over the same
raw node entries. All use the same scoring/value head layouts. This is common
information, not equal statistical efficiency or parameter matching:

| Invented two-facility fixture | Total parameters |
| --- | ---: |
| Graph, physical messages | 348 |
| Graph, self-only messages | 348 |
| Flat dense encoder | 360 |

These are fixture counts, not manuscript-model sizes. Graph/self-only can
isolate a message-operator intervention at this engineering boundary; graph/flat
still differs in encoder structure and parameter count. Node ordering is fixed
by the declared schema, not permutation invariance or unseen-network-size
generalization. No representation was selected based on performance.

Candidates remain **request classes**, not certified feasible patient flows.
The specimen-anchor option retains the reference's other three action groups;
it is not automatically a complete MDL-2 policy comparator. A collector must
verify the upstream public-information producer and candidate coverage. The
previous contract's hidden-state, integer-rounding and fallback limitations
remain in force.

## Behavior probability and return contract

`PolicyEvaluation` binds the exact input/reward/discount contract, canonical
candidate support, observed input vector, normalized class log-probabilities,
value estimate, inference precision, policy-definition hash and weight-snapshot
hash. `CandidateDecision` adds the selected class and its original request; the
old log-probability is indexed from that immutable evaluation.

Re-evaluating the behavior snapshot must reproduce the stored evaluation.
For a subsequent PPO likelihood, weights may change but the input schema,
architecture, operator, precision, observation and collected support may not.
The unchanged-policy likelihood ratio is one in tests, and a selected-class
log-probability gradient agrees with an independent finite difference. This
avoids pretending that a continuous Gaussian density is a probability for an
integer candidate class.

Sampling uses an explicit CPU `torch.Generator`, not global RNG or a greedy
training choice. Restoring its state reproduces the next 12 synthetic decisions
exactly. The selected original request is checked before return for precision
conversion that would change its integer specimen request. Failures restore
sampling RNG; this is not permission to retry a scientific attempt.

The segment adapter requires contiguous actual-trajectory records from one
behavior snapshot with matching reward, input, action and precision contracts.
It rejects counterfactual rows, missing receipts, cross-episode sequences,
state/action mismatches and segments exceeding the explicit step cap.

Rewards are scaled once through the declared reward contract. Values are
declared in those same units. A true terminal state never bootstraps, including
when both terminal and truncated flags are set. A truncated segment only
bootstraps if declared and supplied with a sealed next-state evaluation from
the same policy and exact next observation/token. GAE never connects it to a
subsequent episode. The adapter reuses the repository's `_compute_gae`, whose
advantage core is float32 even for float64 policy inference; this precision
choice is explicit, checked for overflow, and not a new float64 GAE claim.

Hashes detect mismatch, not dishonest collector metadata. A caller can still
fabricate a structurally valid record. Reproduction checks and a source-audited
collector are required before scientific use. These helpers do not authenticate
actual simulation continuity or prove that a submitted request was executed.

## Verification

Runtime: project Python 3.9.6, PyTorch 2.8.0. Tests exercise CPU float32/float64
only. No GPU training, environment episodes or scientific optimizer fit ran.
The newly added tests use invented observations, weights and trajectory receipts;
they calculate gradients without optimizer steps. The existing related DDPG
unit suite also performs bounded invented-tensor optimizer steps.

All 32 new tests and 111 related existing tests pass, 143 total in 4.687 seconds.
Full repository compileall passes. A final read-only host scan found only its
own matching shell/filter commands, with no related workload. Coverage includes common information, self-only isolation,
ordering/alias invariance, probability normalization, explicit RNG and sampling
recovery, snapshot tamper detection, differentiable likelihoods, integer precision
guards, terminal/truncation precedence, reward/discount contract changes,
segment lineage and hand-calculated GAE cases. Eight prior source/evidence
fingerprints from the research-direction memo were rechecked unchanged.

Development failures are retained in this account: the first import failed
because Python 3.9 evaluated a union annotation without future annotations;
that compatibility error was fixed. A later test tried to relabel float32
log-probabilities as float64 without renormalization and correctly failed during
receipt validation, before the intended bootstrap check. The test now supplies
a valid double-precision normalized fixture for that check. Neither was a
failed or repeated scientific experiment.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests.test_candidate_policy_rollout tests.test_routing_candidate_contract \
  tests.test_formal_replay_contract tests.test_formal_graph_contract \
  tests.test_validated_returns tests.test_matched_inputs \
  tests.test_prospective_adapter tests.test_prospective_learner
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m compileall -q .
```

## Remaining integration and preservation requirements

This is not a complete PPO learner or a resumable experiment. Stable Baselines3
was not installed in the inspected runtime; no dependency was installed. The
next finite engineering packet is a capped PPO update adapter using an audited
existing core, tested only on invented data. It must preserve old probabilities,
support multiple PPO passes over one sealed rollout without mixing behavior
versions, and test clipping/value/entropy calculations and failed-update rollback.
Do not copy the old continuous-action agent and silently call it categorical.

Before scientific rollout collection, finish the source-audited public-input
environment adapter and prospective baseline/initialization choices. A new
categorical head is not tensor-identical to the historical continuous actor;
its frozen no-RL comparator must be constructed prospectively. Do not claim a
same-start training ablation merely because both use a GCN.

Full recovery will require policy/value weights, model/input/reward definitions,
optimizer moments, schedule/update counters, sampler and shuffle RNG, all pending
rollout receipts and their source hashes, and environment/collector state at a
declared save boundary. Model `state_dict` and sampler restoration tests alone
do not establish this. Persist source/config commits, runtime versions and raw
records, and use versioned archive/Dropbox byte verification for a later
authorized run. No cloud copy, sync or collaborator-access claim is made here.

New scientific fitting/trajectories still require a committed bounded protocol
and explicit execution approval. Prior R6 test labels cannot serve as untouched
confirmation. Keep the first method comparison on the existing scenario and
fixed reward; do not launch a broad search, reopen Stage E, or claim Howard
approval. No push, PR, merge, message or automatic campaign launch occurred.
