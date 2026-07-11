# Phase 6 — Graph Method Family (plan)

Task groups are ordered so each is independently reviewable and committed on its
own. Groups 1–2 are pure plumbing (no behaviour change to existing agents);
groups 3–5 add the new agents; group 6 verifies and wires; group 7 documents.

## Group 1 — Extract shared graph plumbing (no behaviour change)

1. Create `src/models/graph_features.py`; move `GraphStateSpec`,
   `build_graph_spec`, `flat_state_to_node_features`, `_normalize_node_features`,
   `_infer_num_facilities`, `_configured_edges`, `_dedupe_edges` from
   `gcn_ddpg.py` into it.
2. In `gcn_ddpg.py`, import and **re-export** those names (`from
   src.models.graph_features import ...`) so existing imports/tests are untouched.
3. Run the existing GCN-DDPG tests + `compileall` — must stay green (pure move).

## Group 2 — Patient-aware graph spec + node features

1. Add `patient_summary_width: int = 0` to `GraphStateSpec`.
2. In `build_graph_spec`: detect the patient env
   (`env_config.get("env_type") == "patient_condition"`), derive `summary_width`
   from `survival_bucket_edges` (`3 + len(edges) + 1`, default edges
   `(0.85, 0.90, 0.97)` → width 7), grow the expected state-dim check by
   `n·summary_width`, grow `node_feature_dim` by `summary_width`, and record
   `patient_summary_width`.
3. In `flat_state_to_node_features`: when `patient_summary_width > 0`, split the
   flat state into base (`n·features_per_facility`) + summary (`n·summary_width`)
   blocks, parse the base with the existing path, reshape the summary to
   `(batch, n, summary_width)`, hstack onto the clinic nodes, zero-pad the hub
   row — matching `PatientConditionCapacityEnv.graph_observation()`.
4. Tests (`tests/test_graph_features_patient.py`): (a) spec state-dim + node dim
   match the patient env's `observation_size`/`graph_observation()` widths; (b)
   `flat_state_to_node_features(env.observation())` equals
   `graph_observation()["node_features"]` elementwise (select-path ≡ replay-path);
   (c) base-env path unchanged (regression).

## Group 3 — Graph network heads (`src/models/gcn.py`)

1. Factor the encoder→readout step used by `GCNActor`/`GCNCritic` into a small
   reusable `graph_readout(encoded, num_facilities, include_global_context)`
   helper (flatten facility rows + optional mean global context) so all heads
   share one readout definition. Keep `GCNActor`/`GCNCritic` behaviour identical.
2. Add `GCNSquashedGaussianActor` (SAC): encoder + readout + mean/log-std heads;
   `forward`, `sample`, `deterministic` copy flat SAC's tanh-squash + log-prob
   correction and `LOG_STD_MIN/MAX` clamps verbatim.
3. Add `GCNGaussianActor` (PPO): encoder + readout + mean head + state-independent
   `log_std` parameter; `sample`/`log_prob`/`entropy`/`evaluate_actions` match
   flat PPO.
4. Add `GCNValue` (PPO): encoder + readout + scalar head (no action input).
5. Provide torch-null `# pragma: no cover` stubs for each, matching the existing
   fallback pattern at the bottom of `gcn.py`.
6. Component tests (`tests/test_gcn_heads.py`): output shapes; action bounds
   (squashed actor ∈ (−1,1)); log-prob shape `(batch,1)`; deterministic ≠ sample
   under seed; adjacency symmetry + self-loops (regression on encoder);
   `loss.backward()` populates encoder grads (gradient flow through the graph).

## Group 4 — GNN-TD3 (`src/models/gcn_td3.py`)

1. `GCNTD3Agent(algorithm="gcn_td3")` mirroring `TD3Agent`: build `graph_spec`
   via `build_graph_spec`; actor/target = `GCNActor`, twin critics/targets =
   `GCNCritic`; OU/Gaussian exploration as in flat TD3; `select_action` converts
   state→node features then projects; `observe` → replay; `update` reproduces
   TD3's clipped double-Q, target smoothing, delayed policy update — on node
   features. No residual/imitation. `save`/`load_actor` like GNN-DDPG.
2. Test (`tests/test_gcn_td3.py`): builds on the patient env, emits valid action
   `(action_size,)` in `[-1,1]`, one `observe`+`update` cycle runs and returns
   finite losses; skips cleanly if torch is unavailable.

## Group 5 — GNN-SAC and GNN-PPO

1. `src/models/gcn_sac.py`: `GCNSACAgent(algorithm="gcn_sac")` mirroring
   `SACAgent` with `GCNSquashedGaussianActor` + twin `GCNCritic`; auto entropy
   tuning carried over; node-feature conversion on select + update paths.
2. `src/models/gcn_ppo.py`: `GCNPPOAgent(algorithm="gcn_ppo")` mirroring
   `PPOAgent` (on-policy `observe` buffering transitions, `update` doing GAE +
   clipped surrogate) with `GCNGaussianActor` + `GCNValue`.
3. Tests (`tests/test_gcn_sac.py`, `tests/test_gcn_ppo.py`): same
   builds/valid-action/one-cycle checks as group 4, on the patient env.

## Group 6 — Registry, wiring, and patient-env sanity

1. `src/rl/agents.py`: add `gcn_td3`/`gcn_sac`/`gcn_ppo` to `available_algorithms`
   and `get_agent_class` (lazy imports).
2. Confirm `src/rl/experiment.py` `build_env` already routes the patient env for
   these agents (env_type branch is agent-agnostic — verify, don't duplicate).
3. Extend `evaluation/verify_algorithms.py` `patient_env_sanity` to accept the
   three graph algorithms (config plumbing only; the harness is agent-agnostic).
4. Sanity test (`tests/test_gcn_patient_sanity.py`): each of `gcn_td3`,
   `gcn_sac`, `gcn_ppo` beats random on the 2-clinic patient dev config at a
   short train budget (`beats_random` True; learned cost < random cost with a
   non-flaky margin). torch-guarded skip. Keep budgets small (CI-friendly).

## Group 7 — Provenance, changelog, merge

1. Extend `specs/2026-07-11-benchmark-algorithms/provenance.md` with the three
   graph agents (source, hyperparameter source, verification status = encoder
   tests + patient sanity).
2. Tick Phase 6 in `roadmap.md`; add a decision-log note to `tech-stack.md` if
   any hyperparameter deviates from the flat sibling.
3. Run the changelog skill; merge `phase-6-graph-method-family` → `main`; push to
   the private remote (`mine`).

## Validation checkpoints (per group)

`python -m compileall src evaluation tests` clean, and the group's own tests +
the full suite green, before each commit. See `validation.md`.
