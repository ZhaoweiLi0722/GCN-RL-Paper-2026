# Phase 6 — Graph Method Family (validation)

## Automated

Run before each group's commit and once at the end:

```
conda run -n gcn-rl python -m compileall src evaluation tests
conda run -n gcn-rl python -m unittest discover -s tests -v
```

Required assertions (new tests):

- **Plumbing (group 1):** existing GCN-DDPG tests stay green after the
  extraction — pure move, zero behaviour change.
- **Patient node features (group 2):**
  - `build_graph_spec` on a patient config yields `expected_state_dim` equal to
    the patient env `observation_size`, and `node_feature_dim` equal to the
    per-node width of `graph_observation()["node_features"]`.
  - `flat_state_to_node_features(env.observation())` is **elementwise equal** to
    `env.graph_observation()["node_features"]` (select-path ≡ replay-path).
  - Base (non-patient) env path is byte-for-byte unchanged (regression).
- **Graph heads (group 3):** shapes correct; squashed-Gaussian actions ∈ (−1,1);
  log-prob shape `(batch,1)`; `deterministic` ≠ a stochastic `sample` under a
  fixed seed; encoder adjacency symmetric with self-loops; `backward()` populates
  gradients on encoder parameters (gradient flow through the graph).
- **Each new agent (groups 4–5):** builds on the patient env; `select_action`
  returns shape `(action_size,)` with all entries in `[-1, 1]`; one
  `observe`+`update` cycle returns finite loss metrics.
- **Patient-env sanity (group 6):** for each of `gcn_td3`, `gcn_sac`, `gcn_ppo`,
  `patient_env_sanity(...)` returns `beats_random == True` and
  `learned_cost < random_cost` by a non-flaky margin at the short CI budget.
- **Registry (group 6):** `available_algorithms()` contains the three new names;
  `get_agent_class` returns each class without importing the DDPG residual
  machinery.

All torch-dependent tests must **skip cleanly** (not error) when torch is
unavailable, matching the existing suite's guard pattern.

## Manual

- Read one new agent's `update()` side-by-side with its flat sibling and confirm
  the only differences are (a) node-feature conversion and (b) graph nets — the
  RL math (target computation, clipping, entropy, GAE) is identical.
- Confirm `GCNSquashedGaussianActor.sample` reproduces flat SAC's tanh log-prob
  correction term exactly (no dropped Jacobian).
- Confirm no new agent imports `heuristics` / residual / imitation code — they
  learn from scratch.
- Confirm `graph_observation()` and the reconstructed replay node features agree
  on a config **with** the central capacity hub (hub row zero-padded on both
  paths).

## Tone / docs check

- English only across all new files and comments.
- Provenance table updated; roadmap Phase 6 ticked; any hyperparameter deviation
  noted in `tech-stack.md`'s decision log.

## Definition of done

- GNN-TD3, GNN-SAC, GNN-PPO implemented, registered, and running on the patient
  env; GNN-DDPG retained as ablation and still green.
- Shared graph plumbing extracted and patient-aware; select-path ≡ replay-path
  proven by test.
- Full test suite green; `compileall` clean.
- Provenance extended; roadmap updated; changelog written; branch merged to
  `main` and pushed to `mine`.
- **Explicitly out of scope and not done here:** LQR-graph retrofit, V4
  feasibility ablation, multi-seed IQM curves, temporal encoder — all deferred to
  the pilot (Phase 7) / Phase 8 per `roadmap.md`.
