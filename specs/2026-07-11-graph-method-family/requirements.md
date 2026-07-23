# Phase 6 — Graph Method Family (requirements)

Feature branch: `phase-6-graph-method-family`
Roadmap phase: **Phase 6 — Graph method family** (verification gate).
Depends on: Phase 4 (patient-condition env), Phase 5 (verified backbones + harness).

## Goal

Deliver the **graph actor-critic method family** — GNN-TD3, GNN-SAC, GNN-PPO —
built as *verified GNN encoder ∘ verified RL backbone*, with the existing
**GNN-DDPG** kept as the dominated **ablation**. Each new agent must run on the
**patient-condition env** and pass the phase's verification gate before it can
produce any paper number.

This closes RQ1 (does a graph representation beat flat control?) and RQ2 (which
backbone, and why not DDPG alone?) on the method side; the pilot (Phase 7) then
picks the flagship.

## Scope

### In scope

| Item | Detail |
|------|--------|
| **GNN-TD3** | Twin graph Q-critics + delayed deterministic graph actor + target-policy smoothing. Mirrors `src/baselines/td3.py` update logic; swaps MLP nets for GCN nets. |
| **GNN-SAC** | Twin graph Q-critics + squashed-Gaussian graph actor + entropy temperature (auto-tuned as in flat SAC). Mirrors `src/baselines/sac.py`. |
| **GNN-PPO** | Graph Gaussian actor + graph value net; clipped surrogate + GAE. Mirrors `src/baselines/ppo.py` (on-policy `observe`/`update` contract). |
| **GNN-DDPG** | Existing `src/models/gcn_ddpg.py` — kept as ablation. Only touched to share the graph plumbing extracted below; its residual/imitation features stay intact and default-off. |
| **Patient-aware graph plumbing** | `build_graph_spec` + `flat_state_to_node_features` extended so graph agents run on the patient env's larger observation (base block **+** per-clinic patient-summary block). |
| **Registry + config wiring** | `get_agent_class` learns `gcn_td3` / `gcn_sac` / `gcn_ppo`; `available_algorithms` lists them. Per-algorithm sane defaults. |
| **Verification** | Encoder component tests + patient-env sanity (each new agent beats random). Provenance table extended. |

### Explicitly NOT in scope (deferred, as decided)

- **No warm-start on the new agents.** GNN-TD3/SAC/PPO **learn from scratch** —
  no residual-action anchor, no imitation pretrain. (User decision: keep the
  flat-vs-graph and backbone comparisons honest; residual stays a GNN-DDPG-only
  variant.) The new agents therefore do **not** implement `pretrain_with_heuristic`
  or `fit_action_batch`.
- **No LQR retrofit for graphs.** The LQR gate (V2/V3) already certified the flat
  TD3/SAC/PPO learning machinery in Phase 5; we do not give the LQR a synthetic
  graph. Graph agents are certified by *encoder component tests* (the new part) +
  *patient-env sanity* (the composed agent learns). (User decision.)
- **No V4 feasibility-projection ablation this phase.** Carried to the pilot
  (Phase 7), per the roadmap's "check feasibility-projection load" item.
- **No temporal encoder** (Phase 8, pilot-gated). Static graph encoder only.
- **No new third-party dependency.** No PyTorch Geometric — the existing dense
  `GCNEncoder` (≤21 nodes) is sufficient and already verified in GNN-DDPG.

## Decisions

### D1 — Full family, learn-from-scratch
Build all three new backbones (TD3/SAC/PPO) plus the DDPG ablation. Every new
agent learns from scratch. Rationale: the paper's RQ2 ("why not DDPG alone?")
needs the whole family measured under identical, warm-start-free conditions;
Phase 5's LQR finding already predicts DDPG will be the unstable one.

### D2 — Composition, not duplication of the graph encoder
The GNN encoder and the flat→node feature mapping are **shared**; only the
backbone-specific heads and update rules differ per agent.

- **Extract** `GraphStateSpec`, `build_graph_spec`, `flat_state_to_node_features`
  out of `src/models/gcn_ddpg.py` into a new `src/models/graph_features.py`, and
  re-export them from `gcn_ddpg.py` for backward compatibility (existing imports
  and tests keep working). This lets the new agents depend on the plumbing
  **without** importing the DDPG agent's heuristic/residual machinery.
- **Add graph network heads** to `src/models/gcn.py`, each = `GCNEncoder` +
  readout + a backbone-appropriate head, reusing the encoder/readout the existing
  `GCNActor`/`GCNCritic` already use:
  - `GCNActor` (deterministic tanh) + `GCNCritic` (Q) — **reused** for TD3/DDPG.
  - `GCNSquashedGaussianActor` — SAC. Mean/log-std heads on the graph readout;
    `sample()` / `deterministic()` reproduce flat SAC's tanh-squash + log-prob
    correction and `LOG_STD_MIN/MAX` clamps **exactly** (only the backbone
    changes).
  - `GCNGaussianActor` + `GCNValue` — PPO. State-independent log-std parameter,
    `log_prob`/`entropy`/`evaluate` matching flat PPO; value net = encoder +
    scalar readout, no action input.
- **One file per new agent** (`src/models/gcn_td3.py`, `gcn_sac.py`,
  `gcn_ppo.py`), each mirroring its flat sibling's `__init__`/`select_action`/
  `observe`/`update` so it is independently readable and independently
  verifiable — matching the repo's existing "one self-contained file per agent"
  convention.

### D3 — Patient-aware node features (the crux)
The patient env observation is `[ base (n·features_per_facility) | summary
(n·summary_width) ]`; its `graph_observation()` already hstacks the per-clinic
summary columns onto each clinic node (hub row zero-padded). Two current blockers:

1. `build_graph_spec` **hard-raises** when `state_dim != n·features_per_facility`
   — the appended summary block trips this.
2. `flat_state_to_node_features` only knows the base per-facility layout, so the
   replay path (which reconstructs node features from flat states) would drop the
   patient signal even if (1) were bypassed.

**Fix:** make both patient-aware. `build_graph_spec` detects the patient env
(via `env.env_type == "patient_condition"` in config), computes `summary_width`
from `survival_bucket_edges`, adds `patient_summary_width` to `GraphStateSpec`,
grows `expected_state_dim` by `n·summary_width`, and grows `node_feature_dim` by
`summary_width`. `flat_state_to_node_features` splits the flat state into base +
summary blocks, parses the base block with the existing logic, reshapes the
summary block to `(n, summary_width)`, hstacks it onto the clinic nodes, and
zero-pads the hub row — reproducing `graph_observation()` exactly so the
select-action path and the replay path agree.

### D4 — Config-appropriate hyperparameters
Each agent takes its backbone's published defaults (SAC/PPO actor-lr 3e-4;
TD3/DDPG actor 1e-4, critic 1e-3), same as the flat siblings and the Phase 5
harness fairness fix. GCN hidden sizes default to `[64, 64]` as in GNN-DDPG.
Deviations recorded in the provenance table.

## Context

- **Verification standard (this phase):** encoder component tests are the *new*
  surface and must be tested directly; the composed agents are certified by
  patient-env sanity (`beats_random`, wide margin), reusing Phase 5's
  `patient_env_sanity` harness — extended to accept `gcn_td3/gcn_sac/gcn_ppo`.
  Backbone learning machinery is already LQR-certified (Phase 5) and unchanged.
- **Env constraints inherited from Phase 4:** patient env is `action_mode =
  facility_net`, `transfer_lead_time = 0`, specimen-transfer action ignored
  (autologous identity). Graph agents emit the full facility-net action; the
  env's projection/handling is unchanged.
- **Reproducibility rules (tech-stack.md):** English only; params in config;
  multi-seed for anything reported; `python -m compileall .` + tests before done;
  no fabricated numbers. `docs/research_howard/` stays private.
- **Provenance:** extend the table in
  `specs/2026-07-11-benchmark-algorithms/provenance.md` with the three new graph
  agents (source = "verified GNN encoder ∘ verified backbone"; verification =
  encoder tests + patient sanity).
- **MVP framing:** July target. This phase is deliberately the *composition +
  sanity* cut; deeper certification (V4 ablation, multi-seed IQM curves) lands at
  the pilot where it informs the flagship decision.
