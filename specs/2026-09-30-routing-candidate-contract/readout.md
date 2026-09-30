# Routing request-class engineering acceptance

Date: 2026-09-30. Prepared from checkout
`4e2d9cd5217503e271f6457fc8d7eba4bc85fe52` on the persistent local
`codex/september-research-integration` branch. This is software preparation
under Zhaowei's continuation request, not a scientific execution protocol.

## Decision

Keep one clean DDPG reference path and develop a categorical graph-policy
interface for specimen-routing requests. The first implemented boundary is
**request-class identity**, not a physical-feasibility oracle. No new patient
episode, scientific fit, model selection, reward change or performance result
was produced. Stage E and every previous experimental packet remain closed.

The observation currently available to the policy is insufficient to certify
which individual patients will move for every proposed request. The environment
receives in-transit arrivals before routing, then considers individual patient
eligibility, transfer limits and edge priorities. A mask inferred only from
aggregate waiting counts would not be an exact execution mask. Do not add hidden
patient state or outcome labels to make such a mask appear exact.

## Implemented contract

New opt-in module: `src/rl/routing_candidate_contract.py`.
New tests: `tests/test_routing_candidate_contract.py`.
Existing agents, environment dynamics, experiment configs and evidence are
unchanged. The new module is not registered in a production training launcher.

| Boundary | Behavior |
| --- | --- |
| Explicit schema | Requires action-schema ID, facility count, specimen scale and candidate cap; accepts only finite normalized 4N requests without silent clipping |
| Original support | Retains a reference request, a specimen-anchor request and explicitly supplied options; all other request groups must match the reference |
| Equivalence | Groups requests only when the existing float64 decoder receives identical rounded specimen lot requests and unchanged other groups |
| Canonical order | Sorts unique classes independently of option enumeration; representative priority is reference, then anchor, then lexicographically smallest original request |
| Probability | One categorical logit per unique request class, using PyTorch Categorical; repeated aliases cannot receive extra probability merely by appearing more often |
| Submission | Returns an original request, not its class number, normalized integer features or realized flow |
| Provenance | Immutable candidate copies and SHA256 bind the schema, state token, original requests and distinguished reference/anchor roles |
| Replay | Requires a typed actual-trajectory receipt with the same state token, original action and declared reward/action semantics |
| Precision | Rejects replay float32/float64 conversion when it changes the first-slice integer request; never silently changes the recorded action |
| Information boundary | Rejects extra environment, patient-registry, outcome or external-mask payload fields; upstream decision-time provenance still requires a collector audit |

The specimen anchor preserves the reference's non-specimen groups. It is not
automatically the full MDL-2 policy action. If reference and anchor are aliases,
both input records remain but share one selectable class; the reference is the
submitted representative. A later scientific protocol must separately retain
its intended full-policy baselines and cannot call this one-slice comparison
an evaluation of the entire MDL-2 controller.

Different integer requests that happen to execute no transfers on an empty
queue remain separate. Their blocked-request accounting can differ, and a
sampled identical flow is not proof of general equivalence. Conversely, the
classes certify equality of decoder inputs, not equal future controller states,
return distributions, or global physical feasibility. Unbalanced or blocked
requests are not silently removed. Candidate coverage and public feasibility
remain open collector-design questions.

The state token and candidate digest are audit metadata, not policy features.
A strict key whitelist does not prove that an upstream producer used only
public information. Likewise, selecting an action does not prove it was actually
executed; the collector must supply a valid observed transition. Precision
validation covers specimen quantization, not equality of every numerical effect
in the other three request groups after conversion.

## Clean DDPG integration boundary

The synthetic end-to-end contract test passes the selected original request
through `OneStepRecord`, `ReplayInputContract` and `prepare_replay_batch`.
Physical-message and self-only graph views preserve the same request and
terminal reward target. Category IDs, execution outcomes and counterfactual
labels cannot silently substitute for the submitted behavior action.

`prospective_ddpg_kernel.py` remains unchanged. It is a CPU-float32 engineering
kernel, not an already repaired scientific campaign. The new precision check
must be invoked explicitly by a future collector; existing training code has
not been rewired. A request rejected at a float32 rounding boundary requires a
prospectively specified representation/precision policy, not an ad hoc shift
of the action or bypass of the check. DDPG's continuous actor and candidate
selection are different controller interfaces; this receipt test does not
replace the DDPG actor with a categorical selector.

## Existing PPO is not the proposed categorical learner

Source inspection found `src/models/gcn_ppo.py` and `src/baselines/ppo.py` already
implement continuous tanh-Gaussian PPO. They reuse `PPOTransition` and
`_compute_gae`, and do not carry sealed candidate supports or categorical
choices. Their existing receipt uses a single done flag, and the save method
does not provide full optimizer/rollout/RNG recovery. These are compatibility
gaps for this new interface, not a measured explanation of historical results.

Do not relabel the existing `gcn_ppo` config as a categorical experiment or
silently modify those historical agents. Reuse applicable graph components
and a verified PPO implementation after auditing the interface. The next finite
engineering packet is a matched graph/flat candidate-scoring and on-policy
receipt contract: explicit policy version, candidate seal, selected class,
original request, old log-probability, actual reward, terminal/truncation
semantics and observation provenance. Test with invented tensors only first.
Saved off-policy R6 rows are not new on-policy PPO rollouts.

## Validation and limits

The focused suite contains 24 new tests plus 87 existing replay, graph, matched
input, adapter and clean-DDPG kernel tests. All 111 pass in 4.643 seconds. Full
repository `compileall -q .` passes. Local commit follows Git whitespace review.
Tests use invented numbers, two-facility patient queues and existing synthetic
learner fixtures. They call the deterministic routing helper for mechanics
checks, not `env.step` or an experimental trajectory. Existing learner tests
include bounded invented-tensor optimizer updates, not scientific fitting.

Coverage includes all 24 permutations of the option fixture, immutable copies,
reference/anchor retention, single-class support, invalid numeric inputs,
undeclared fields, precision boundary crossings, actual-trajectory receipts,
decoder consistency on invented queues, stable categorical log-probabilities,
alias-independent probability and the typed replay adapter.

The initial 22-test development run had two incorrect fixture expectations:
one ULP below a half-integer boundary was rounded back upward by the existing
float64 `+0.5` calculation. The fixture was corrected to a representable value
farther below the boundary, and the actual one-ULP decoder behavior received a
separate regression test. Historical rounding was not changed. These were
unit-test development failures, not failed or retried scientific attempts.

Reproduce the bounded tests and compilation from this worktree:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests.test_routing_candidate_contract tests.test_formal_replay_contract \
  tests.test_formal_graph_contract tests.test_validated_returns \
  tests.test_matched_inputs tests.test_prospective_adapter \
  tests.test_prospective_learner
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m compileall -q .
```

No experiment config, new model architecture or new training algorithm has
been scientifically approved or selected by these tests. There is no evidence
yet that categorical PPO or repaired DDPG improves patient-network performance.
The eight prior fingerprints in
`docs/team_updates/2026-09-30-gcn-rl-research-direction.md` were rechecked and
match. They are not a claim to rehash every historical archive. At
2026-09-30T14:00:47Z an approved read-only host process scan found only its own
matching shell/filter commands, no related research workload. All test and
compile sessions ended with exit code 0. No remote Git or cloud action occurred.

## Next decision boundary

Continue the synthetic-only interface work without waiting for routine-step
approval. Before collecting new scientific trajectories or fitting a policy,
commit one bounded protocol with named comparator arms, common public inputs,
candidate/continuous support, objective, fresh independent streams, episode
and terminal accounting, metrics, query/update/time caps and a single-attempt
rule, then obtain specific execution approval. Keep the first comparison on
the existing scenario and fixed reward. Do not guarantee positive RL returns,
infer Howard's sign-off, reopen old studies, or start a broad algorithm search.
