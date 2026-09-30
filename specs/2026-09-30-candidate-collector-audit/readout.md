# Candidate public collector audit and boundary preparation

Date: 2026-09-30. Base checkout
`b960e4462eba86a8d179b95d278c2bb9bb250679`, persistent local
`codex/september-research-integration` worktree. Source inspection and invented
fixtures only. No patient environment construction/reset/step, new trajectory,
scientific fitting or performance comparison was performed in this packet.

## Source findings

| Boundary | Source evidence | Consequence |
| --- | --- | --- |
| Public state | `src/env/patient_capacity_planning.py:166`, `src/env/capacity_planning.py:412`, `src/rl/patient_replay_collector.py:38` | Raw float32 state has base facility blocks, patient summaries and optional time; the producer losslessly interleaves them into node features. Hidden registry/RNG/snapshots are not accepted by `observe` |
| Anchor information | `src/baselines/heuristics.py:715`, `src/rl/patient_replay_collector.py:121` | MDL-2 uses public state plus declared configuration/nominal demand rates. This is common configured information, not proof that an unknown changing regime is observed or learned |
| Graph | `src/rl/patient_replay_collector.py:29` | Producer rejects heterogeneous relation graphs and unsupported overtime/procurement/hub layouts. Graph/self-only retains identical physical metadata; this cannot establish heterogeneous-network or changing-capacity results |
| Action timing | `src/env/patient_capacity_planning.py:283` | Pending transfers arrive before specimen routing; production follows routing. Start-of-step eligible-count summaries are not a complete execution feasibility oracle |
| Integer requests | `src/env/specimen_routing.py:65` | Half-away-from-zero rounding determines requested lots. Executed patients additionally depend on queues, patient eligibility/transfer history, edges and tie-breaking. Request classes are not equal-outcome classes |
| Blocked flows | `src/env/specimen_routing.py:252` | Headline blocked count is the maximum of unfilled inbound and outbound requests, not their sum. Record both sides and conserve executed net flow |
| Old collector | `src/rl/patient_replay_collector.py:188` | Expects proposal/gate/request, casts to float32, and reconstructs post-gate actions. Categorical requests must not be forced through this interface by inventing gate metadata |
| Full neural information | `src/models/candidate_policy.py:98`, `src/models/prospective_forward_agent.py:31` | Candidate scoring also consumes a reference request, candidate features and role flags. Equality of the raw actor-state vector is not equality of every neural input with clean DDPG; a frozen-reference model carries its own training budget/provenance |
| Reward | `src/env/patient_capacity_planning.py:506`, `src/env/capacity_planning.py:1205` | Reward is the negative sum of operating plus patient-loss/expiry/urgency costs. Preserve raw reward, scale once downstream and reconcile aliases without adding transfer fees twice |
| Horizon | `src/env/capacity_planning.py:1440`, `src/env/patient_capacity_planning.py:566` | `done` means configured time horizon reached, not every patient resolved. The final step can enroll arrivals and leave active identities; no extra terminal liability or salvage is automatically applied |
| Recovery | `src/rl/prospective_patient_session.py:122`, `src/env/patient_capacity_planning.py:1339` | The old bounded DDPG session combines environment, OU, replay and learner recovery. It assumes DDPG windows/updates and is not a categorical PPO session. PPO kernel recovery alone does not restore collection |

The horizon finding is a documented objective boundary, not proof that past
negative RL findings were caused by a reward bug. A finite-window observed-cost
objective can be valid when declared, but it is not a completed-treatment
lifecycle objective. Unfinished patients are not automatically deaths, and an
arbitrary terminal penalty would be a new scientific choice, not a code repair.

## Implemented pure boundary

New module: `src/rl/candidate_collection_boundary.py`. It does not own, query or
step an environment; no automatic run or historical-agent change is introduced.

- `public_candidate_context` reuses the existing raw producer and full MDL-2
  anchor. Reference and every option must share MDL-2's non-specimen groups.
  It rejects a mixed anchor instead of quietly changing their common raw-state
  and anchor block. Routing must be enabled. Additional reference/candidate
  features still differ from clean DDPG: these controllers are not an isolated
  optimizer comparison, and raw-input parity is not complete feature parity.
- Candidate generation and reference initialization remain external declared
  choices. This helper does not select epsilons, rank options or use hidden
  patient-state masks. It supplies no empirical coverage estimate.
- `candidate_submission` reproduces the sealed behavior evaluation, checks
  current-state identity and inference precision, then preserves the original
  normalized request in float64, matching the environment's conversion. It
  does not substitute a class center or a float32 re-encoding. The returned
  array is a read-only copy.
- `audit_candidate_step` reconciles reported cost components/aliases, exact
  negative raw reward, decoded requests, conserved executed flows and blocked
  inbound/outbound counts. It constructs a typed record retaining the original
  submitted request and once-unscaled reward. Unsupported new cost channels
  require a declared extension rather than silently disappearing.
- `CollectionBoundary` checks explicit horizon and collection limits. It maps
  true finite-window termination separately from collection truncation. A
  continuing bootstrap is a prospective objective choice, not the default cure
  for censored outcomes.
- Post-step audit data retains losses, completions, active identities and
  unresolved identities at the boundary. Remaining identities outside waiting,
  manufacturing and specimen transit are retained as `other_active_patients`,
  not guessed to be losses or assigned a monetary value. Added terminal cost
  is exactly zero; reward weights and environment behavior are unchanged.

The audit record reports request-class count and whether the selected class
differs from the reference. A different request can still execute zero routes;
this is explicitly tested. Neither nonzero routing nor candidate diversity is
evidence of a useful learned policy or a clinically safe candidate.

## Verification

17 new tests pass on their first recorded run in0.055s. The full related suite
passes199 tests in5.570s; after adding an explicit missing-component assertion,
the final run passes **199 tests in5.595s**. Full compileall exits zero. Python3.9.6,
PyTorch2.8.0. Eight previous preservation fingerprints match.

New tests patch patient constructor, reset and step to raise immediately. A
metadata-only shell uses `object.__new__` with dataclass configuration and
invented raw arrays; it has no patient registry, simulator RNG or real episode.
This checks actual producer layout code without fresh simulator data. Optional
observation blocks, whole-anchor parity across graph/self-only/flat views,
unsupported layouts and hidden-input rejection are covered.

Other tests cover exact requests rather than class centers, stale receipts,
precision crossings, component/alias reconciliation, raw reward scaling,
integer flow accounting, finite/missing-invalid fields, explicit closure,
unresolved patients, immutable numeric audit copies and terminal-record GAE
integration. No new test trains a model. The existing PPO/DDPG related suite
contains bounded optimizer updates on invented tensors, not research training.

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests.test_candidate_collection_boundary tests.test_candidate_ppo_kernel \
  tests.test_candidate_ppo_objective tests.test_candidate_policy_rollout \
  tests.test_routing_candidate_contract tests.test_formal_replay_contract \
  tests.test_formal_graph_contract tests.test_validated_returns \
  tests.test_matched_inputs tests.test_prospective_adapter tests.test_prospective_learner
env PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m compileall -q .
```

| File | SHA256 |
| --- | --- |
| New boundary module | `fe77b982cc93230b847d0d40cd8120bfc44a07a973bb653fba31b942e6d3d3b5` |
| New boundary tests | `243fb6b60797aab491b7d9a437f7c7a6b70720675812bc32beb87bff22676e95` |
| Existing `patient_replay_collector.py` | `1f62db840b7176e2b92c6848330e30ad1feb32afd978a707cb76580dd9cbd58a` |
| Existing `prospective_patient_session.py` | `6fc909ac9eefef325bc2bb4f01ac0ae48ad2998d55642d7fc3ff9a069b705a6e` |
| Existing `patient_capacity_planning.py` | `6d04e1351549b488e56d8b9868afcb23d22239d7005b8ba65ef9211653acd2d4` |
| Existing `capacity_planning.py` | `529215bc180463fb1965534f994539616198e95371776acd814fd8bf05a1e2f1` |
| Existing `heuristics.py` | `3ed3f944d47b52f76f8d7581ff706c6903e238c43ed077301ece1ddc0db0de35` |
| Existing `specimen_routing.py` | `5d9de6539c212362f55d943b975c810e0f21aa11bea7872f28615c550157f2ec` |

Read-only host scan found only its own shell/filter, no related research job.
Test and compile sessions ended. The existing gcn-rl heartbeat was rechecked
ACTIVE every30min; it was not modified. Prior sources/configs/results were not edited.
No new archive, cloud-sync or collaborator-access claim is made.

## Remaining execution gates

This completes source audit and pure boundary preparation, not an end-to-end
patient collector acceptance. Before a scientific run, the committed protocol
must name and budget a real preflight that checks:

1. Candidate option family and initialization, same-start frozen comparator,
   full MDL-2 comparator, graph attribution and clean DDPG comparability. Beating
   a random untrained policy alone is not an additional-online-value result.
2. Exact raw/public producer and reference-model input parity on the selected
   scenario; unsupported topology/layout and architecture parameter mismatch
   cannot be overlooked. Missing operational calibration stays missing.
3. Whole environment, collection cursor, unfinished segment, decision receipts,
   sampling RNG and PPO kernel recovery in one coherent transaction. No such
   categorical environment driver was implemented or exercised here. Its
   source/tests must pass and be frozen before the separately approved run.
4. Query/wall-clock limits counting clone checks and all evaluation work. A
   clone step is another simulator query, not a free diagnostic.
5. Declared finite-window or lifecycle objective, patient denominators and
   unresolved-at-boundary reporting. No results-driven reward adjustment or
   implicit drain/terminal-penalty extension.
6. Fresh development streams and independently sealed evaluation with complete
   raw costs, clinical components, request/route coverage and negative outcomes.
   R6 inspected test labels are not a fresh confirmation set.

Next is one bounded scientific pilot decision packet, not an automatic launch.
No new collection/training, reward/scenario/model search, formal holdout, remote
Git, messaging or Howard sign-off is authorized. Stage E remains closed. The
finite heartbeat must be removed once the packet is ready and only a new-scope
execution decision remains.
