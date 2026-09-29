# Common-information preparation, not a new controller

September 29 local queue packet N2. This is an isolated tensor-interface
prototype with synthetic tests, not a change to existing agents or an experiment
amendment. No environment, checkpoint, teacher cache or optimizer is used.

Implementation: `src/models/matched_inputs.py`.
Tests: `tests/test_matched_inputs.py`.
Prerequisites: [historical graph/flat audit](../2026-09-29-formal-graph-contract/readout.md)
and [N1 target contract](../2026-09-29-replay-repair-preparation/contract.md).

## Why this is necessary

The historical graph and flat controllers do not differ only in message passing.
Their state transforms, action heads and proposal-conditioned gate inputs differ.
The graph representation also aggregates some pipeline detail that the raw flat
input retains. Copying a config flag or matching total parameters does not remove
these differences. Historical measured package-level effects remain unchanged;
this prototype neither isolates their cause nor changes their interpretation.

## Canonical input layout

`InputSchema` names an ordered node list, ordered per-node feature list, ordered
global feature list and ordered action list. `definition_id` must identify a
prospectively specified producer contract, including units, scales, anchor,
action coordinates, physical-link meaning and time-availability rules. A mismatch
in any ID or ordering fails even when shapes happen to match. No config flag
silently enables a private transform in one arm.

For batch size `B`, nodes `N`, node features `F`, globals `G` and actions `A`:

| Block | Required shape | Ordering / meaning |
| --- | --- | --- |
| nodes | `(B, N, F)` | All declared slots, node-major and then feature-major; no reduction or aggregation |
| globals | `(B, G)` | Declared order; `G=0` is allowed, batch size must still be explicit |
| physical_links | `(B, N, N)` | Current declared symmetric nonnegative weighted links, zero diagonal, in the same node order |
| anchor_actions | `(B, A)` | Declared action order, available to actor, critic and gate in both representations |
| candidate/proposed action | `(B, A)` | Same action coordinates as the anchor, supplied only for critic/gate |

Both representations receive physical-link metadata. The graph cannot obtain
extra time-varying relationship information that is omitted from the flat
representation. This prototype supports **one symmetric relation matrix only**;
directed links, separate relation types and edge attributes are not implemented.
Their inclusion would require an explicit shared schema, not silent truncation.
Physical feasibility can also depend on constraints outside this matrix; the
module does not implement, certify or change those constraints.

For each role, graph inputs are `nodes` plus `context`; flat inputs are exactly
`concat(nodes.reshape(B, N*F), context)`. Context always starts with
`concat(globals, physical_links.reshape(B, N*N), anchor_actions)`.

| Role | Context suffix | Contract |
| --- | --- | --- |
| Actor | None | Reject its own proposal as input; no circular dependence |
| Critic | Candidate action | Preserve action gradients and provide the anchor explicitly |
| Gate | Proposed action minus anchor | Both views get the same proposal; detach policy must be explicitly true or false |

Gate proposals are intended to be the request **after the common upstream
action transform, before the learned gate**, not a mixture of raw residuals and
executed integer actions. The interface does not clip, rescale, mask or project
actions. Those operations and their bounds must be declared and tested in a
future producer/adapter, identically for both arms. Proposal detachment affects
only the delta block, not the separate anchor context block.

Shape broadcasting, automatic dtype/device conversion, nonfinite values,
unknown roles and ambiguous detach settings fail. Only float32/float64 tensors
are accepted. Returned values are copied away from caller storage while keeping
autograd paths when required. Returned tensors are still mutable; callers must
not alter one view and then claim the pair remains matched.

## Neural message ablation

`message_mode=physical` adds self-loops to the physical-link weights and uses
symmetric degree normalization, matching the existing GCN helper on each batch
item. `self_only` uses an identity neural operator. In both modes, node values,
context, flat representation, anchor and proposal remain identical. No physical
environment config, route, transfer feasibility or action dimension changes.

This contrast tests the message operator, not removal of all topology
information: physical links intentionally remain visible in context. It is
different from deleting physical routes or withholding information from a
baseline. A topology-permutation or information-removal contrast would need its
own prespecified treatment. No such experiment was run here.

## Heads and parameter accounting

The prototype supplies inputs, not trained networks or architecture settings.
A synthetic forward test reuses the existing `GraphConvolution` with identity
weights and the **same** dense head for both views. Under self-only messages,
outputs agree; changing the message operator changes encoded values without
changing physical inputs. This tests plumbing only, not a trained-policy effect.

A future encoder-only contrast must also use comparable readout/action heads,
gate output definitions, action transforms and deployment rules. Shared edge
heads versus a global dense head are a separate treatment; retaining the old
head difference and calling it a pure GCN ablation would remain invalid.
Fixed-order flattening makes no permutation-equivariance or variable-network-
size claim. The actual layer widths, parameter budget, head choice and temporal
information contract remain scientific protocol decisions.

`parameter_inventory(actor=..., critic=..., gate=...)` reports:

- parameters in each explicitly supplied component;
- unique Parameter objects across those components, including frozen ones;
- currently trainable unique parameters;
- duplicated component counts due to shared Parameter objects.

Buffers and optimizer state are excluded. Target/reference copies are not
arguments and must not be supplied as deployment components. Their compute and
memory cost are not asserted to be zero. The old parameter counter is unchanged;
this prospective helper handles cross-component sharing more explicitly. No
parameter matching, hyperparameter search or latency equivalence is claimed.

## Producer and integration limits

Tensor equality is necessary, not sufficient, for a valid scientific comparison.
This module cannot determine whether the producer mislabeled a column, used a
latent capacity value, leaked future outcomes, omitted an observable pipeline
slot or computed the anchor with different information. Matching schema labels
are assertions, not independent provenance proof. The real producer is **not**
implemented here. Existing graph/flat feature helpers are not silently reused
because the historical audit showed that their transforms differ.

Next packet N3 will join N1/N2 in a synthetic, no-training acceptance harness
and specify remaining integration/approval checks. A future study still needs
collector audits, time-valid observation construction, actual head parity,
checkpoint/optimizer/normalization handling, conventional adaptive baselines,
and a locked development protocol before any training. Equal inputs alone do
not guarantee equal effective information use or positive online RL gains.

## Reproduction

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_matched_inputs tests.test_validated_returns \
  tests.test_formal_graph_contract tests.test_gcn_heads
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Only hand-authored tensors and untrained component forwards are used. There is
no scientific seed selection, simulator rollout, policy evaluation or training.

September 29 validation: 21 new contract tests and the combined 140-test suite
passed. The combined suite covers N1, historical method/replay/graph contracts,
offline replay, attribution recovery, crossed evidence checks, service-effort
and queue contracts, existing GCN heads and patient graph features. Full Python
compilation and `git diff --check` passed. The source-reference scan finds no
existing agent import. Existing tracked source, execution configs, evidence and
reports are unchanged from `e17474a`; only this new opt-in source and tests were
added alongside documentation. This is software verification, not a performance
result or certification of the historical graph/flat comparison as input-matched.
