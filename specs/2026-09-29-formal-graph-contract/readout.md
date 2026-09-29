# Historical graph/flat contract audit

## Decision

The formal result remains a measured advantage of the **complete graph-aware
controller over the implemented flat controller**. It is not an isolated causal
effect of GCN message passing. This distinction matters independently of the
null final-versus-frozen online-update result.

This is September queue packet G1, not the earlier Stage G1 legal-action-ranker
experiment. It is a read-only source/interface audit, not a new scientific run.
No historical code, cache, checkpoint, raw row, summary or frozen publication
map was changed. No research training/evaluation campaign, historical-checkpoint
inference or remote action was performed. The interface audit itself takes no
environment transition; the adjacent regression tests exercise synthetic fixtures.

## Evidence and provenance

- Training source: `9c0718b0da49d34f7f878ed2f036e96bf7bb3693`.
- Historical training manifest SHA256 verified:
  `0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2`.
- All ten effective configs and summaries were read; current hashes are recorded
  in `reports/2026-09-29-formal-graph-contract/audit.json`. Manifest identities
  and parameter counts agree. This is not independent historical hash proof for
  every config or summary; that provenance limit is retained from M1.
- Models, graph specification and numeric helpers were reconstructed from
  pinned source in isolated in-memory namespaces. No current environment code
  or full agent/replay buffer was imported. CPU forward checks use untrained
  synthetic tensors and test interfaces, not performance.
- The frozen publication evidence map was reviewed without alteration. Its
  graph-attribution entry supports a controller contrast; it supplies no
  feature/head/gate-matched message-edge ablation or held-out-topology result.
  This is a scoped evidence gap, not a claim about every file ever produced.

## Claim-to-code matrix

Source paths below refer to the pinned training commit. Exact source excerpts
and line bounds are embedded in the audit JSON.

| Question | Executed contract / source | Claim boundary |
| --- | --- | --- |
| Same physical action feasibility? | `src/rl/experiment.py:apply_graph_ablation` returns the same physical config for `flat_state_no_graph`. Pairwise configs differ in graph metadata but not other environment fields. | The flat policy retains the same specimen/resource/capacity feasibility. It is not an isolated-facility baseline. |
| Same raw observation? | 561 values: 20 facilities times 14 base and 14 patient-summary values, plus time. `src/models/graph_features.py:build_graph_spec`. | Same raw interface does not imply identical neural inputs. |
| GCN neural features? | Actor/critic: 21 nodes times 36 values. Includes MDL-2 requests, three derived demand features, broadcast time and hub indicator. Direct capacity columns are idle and total, not every production slot. `flat_state_to_node_features`. | Engineered inputs change along with the encoder. Derived values use current state/config, not privileged future observations. |
| Flat neural features? | Actor/gate: raw 561 plus 80 MDL-2 requests. Critic: raw 561 plus candidate action 80. Non-temporal `_actor_input_tensor` in `src/baselines/flat_ddpg.py` does not append adaptive-demand features despite the shared config flag. | Flat retains detailed raw pipeline slots, but lacks the explicit derived-demand block and critic anchor block. Neither representation simply contains everything in the other. |
| Actor readout? | GCN `network_residual`: shared edge heads and replenishment head, not fixed-order global actor flattening. Flat: MLP `[292,212,128]` to 80 outputs. `src/models/gcn.py:GCNActor`; flat constructor. | This changes the decision-head inductive bias, not just the encoder. |
| Gate parity? | GCN gets four per-node proposed clipped action deltas, for 40 node features. Flat gate sees 641 raw-plus-anchor values; its pinned implementation does not consume `include_proposed_residual_features`. | Same config text does not establish equivalent gate conditioning. Effect size attributable to this difference is unknown. |
| Critic / gate readout? | Separate graph encoders; concatenate 20 facility embeddings and mean over all 21 nodes: 672 values. `GCNCritic`, `GCNCorrectionGate`, `graph_readout`. | No complete-policy permutation-equivariance or variable-size generalization claim. Actor sharing does not remove the ordered gate/critic. |
| Graph type? | One deduplicated, travel-time-weighted adjacency: 36 clinic links plus 20 hub links, with self-loops in normalization. Separate actor edge metadata has distance/time/resource-lead/cost features. | Not a relation-specific multiplex GNN. Static configured links, changing node observations. |
| Edge ablation evidence? | Frozen map has package, backbone and transport-timing contrasts, not isolated neural-topology contrasts. Physical `no_interfacility_edges` also removes feasible transfers. | Removing physical links is a different treatment from changing only neural message edges. Transport-lead sensitivity does not isolate graph mechanism. |
| Online attribution? | Existing final-versus-frozen rows are unchanged. | This audit neither creates positive online evidence nor proves the input differences caused the null online increment. |

The direct graph-capacity layout was tested by redistributing two production
pipeline slots while preserving their total. Its raw node block is unchanged,
whereas the flat raw state changes. The anchor-derived block is intentionally
excluded from this synthetic probe: it does **not** prove the entire GCN policy
cannot distinguish those states through its anchor. A second pinned-function
probe changes a proposed residual and observes the corresponding GCN gate-input
change. Flat input reconstruction preserves its 561 raw values and appends only
the 80 anchor values. These are software interface tests, not outcome ablations.

## Parameter accounting

The exact historical counter counts actor, critic and correction-gate parameters,
deduplicating repeated module objects. It excludes target/reference copies,
buffers and optimizer state. All ten runs reproduce their archived totals.

| Component | GCN | Flat |
| --- | ---: | ---: |
| Actor | 324,964 | 287,164 |
| Critic | 238,433 | 276,973 |
| Correction gate | 49,889 | 43,201 |
| Total | 613,286 | 607,338 |

Gap relative to flat: **0.97936%**, within the declared 1% budget. This is total
parameter matching, not per-component, compute, latency, feature or effective
specimen-channel-capacity matching. Formal residual scales disable corrections
in the other three output groups; their network weights still enter this count.

## Manuscript correction and remaining work

The manuscript now describes the actual actor/gate/critic interfaces, merged
adjacency, parameter-count scope, and package-level interpretation. Formal
estimates and confidence intervals are unchanged. The M1 wording that called
the entire primary readout fixed-order was too broad: it applies to critic and
gate, not the network-residual actor. Its readout note has been corrected.

Before claiming an isolated GCN effect in a new study, a prospective design
would need identical information transforms and proposal-conditioned gates;
separate comparisons for message passing and shared edge heads; and physical
links held fixed during neural topology ablations. This is a methodological
requirement, **not authorization to launch that study**. The M2 reward/replay
issues must also be addressed in any newly authorized online-learning protocol.
The old holdout cannot be used to choose the corrections or architectures.

Routine local work can now move to E1's operational decision/measurement
contract, then C1's completion-only comparator. Do not train DDPG on the solved
synthetic queue fixture or weaken its comparator to seek an RL win.

## Reproduction

From the persistent integration worktree:

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
ROOT='/Users/lizhaowei/gcnrl-patient-routing-persistent/results/patient_indexed_specimen_routing_mac_mps_primary/confirmation/ddpg_routing_primary_100/patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100'
"$PY" -m evaluation.audit_formal_graph_contract --training-root "$ROOT" --output /private/tmp/formal-graph-audit-new.json
"$PY" -m unittest tests.test_formal_graph_contract -v
```

Use a fresh output filename; the audit refuses to overwrite evidence. Full
Python compilation and the focused regression suite are required before the
local checkpoint. TeX rendering is unavailable on this host, so source checks
do not establish PDF layout correctness.

Validation completed: eight new interface-audit tests and 42 adjacent audit,
recovery and queue tests passed (50 total); full Python compilation and
`git diff --check` passed. A second audit reproduced the saved JSON byte for
byte. This is a local-only checkpoint; historical results remain unchanged.
