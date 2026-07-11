# Tech Stack & System Design

## System design

The system is a research pipeline in four layers, all already present in the
repo and extended by this plan:

```
Environment (simulation)  →  Graph construction  →  Models (policies)  →  Training/Eval
   src/env/                     src/graph/            src/models/,           training/,
                                                      src/baselines/         evaluation/
```

- **Environment** — a NumPy simulation of the distributed PRM manufacturing
  network. Facilities ("clinics") are nodes; state per facility includes demand,
  waiting specimens, reagent inventory, and a bioreactor production pipeline.
  Actions are normalized continuous controls for reagent replenishment and
  signed transfers of specimens / capacity / reagents over typed edges. Reward =
  −cost (purchase, holding, shortage, transfer). Supports 20 clinics
  (`make_20_clinic_config()`), Poisson demand with shocks, Bernoulli supplier
  disruption, and production/transfer lead times.
- **Graph** — typed edge sets (information / specimen / resource / capacity),
  with ablation switches to remove edge types.
- **Models** — GCN encoder feeding actor-critic heads; flat MLP baselines;
  heuristic policies.
- **Training/Eval** — thin config-driven entry points over a shared experiment
  runner; multi-seed training, Monte Carlo evaluation, a benchmark manifest,
  aggregation and plotting.

**What the current environment does NOT model** (and this plan adds): per-patient
health-condition states, product/material shelf-life or expiry, patient queues
as first-class entities, and any temporal/recurrent learnable layers.

## Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11 (conda env `gcn-rl`) | Matches repo; env already set up and smoke-tested |
| Deep learning | PyTorch | Existing models (GCN-DDPG, TD3, SAC, PPO) |
| Numerics | NumPy | Environment is pure NumPy |
| Plotting | matplotlib | Existing figure pipeline |
| Heuristic baselines | Gurobi (MILP) | MYO/MDL use MILP subproblems in Tseng et al. code; free academic license via Georgia Tech — see decision log |
| Tests | unittest (pytest optional) | 57 existing tests pass under unittest |
| Config | JSON (env scenarios) + YAML (algorithms) | Existing convention |

New dependencies require approval and a decision-log entry below.

## Environment-extension strategy (decided)

Add patient conditions and expiry as a **new environment layer/module**, leaving
`src/env/capacity_planning.py` and its passing tests **intact**. Rationale:
adding per-patient sub-state and age-bucketed inventory is invasive (it changes
state dimensionality and touches every flat baseline that assumes a fixed
`state_dim`); isolating it protects the working manuscript baseline and keeps
the two comparable.

Reusable patterns to build on:

- The existing **bioreactor production pipeline** (age-advancing array) is the
  template for **age-bucketed, expiry-aware** specimen/material inventory.
- The SimPAC patient model (private source) provides the **patient-condition
  dynamics** to port: a continuous survival state that decays over time with a
  stochastic deterioration shock and an eligibility threshold (an expiry-like
  deadline). Port the *mechanism*, re-parameterized for the network setting.

## Algorithm roster (decided)

| Tier | Algorithm | Status in repo | Role |
|------|-----------|----------------|------|
| Heuristic baseline | MYO, ISO, MDL-1, MDL-2 | Implemented (`src/baselines/heuristics.py`); MILP variants in previous code | Strong/standard baselines to beat |
| Flat DRL baseline | Flat DDPG | Implemented (`src/baselines/flat_ddpg.py`) | Isolates value of the graph |
| Method family | GNN-DDPG | Implemented (`src/models/gcn_ddpg.py`) | **Ablation** — show it is dominated |
| Method family | GNN-TD3, GNN-SAC | Flat versions exist; graph variants to build | **Flagship candidates** |
| Method family | GNN-PPO | Flat PPO exists; graph variant to build | On-policy comparator (matches closest rival) |
| Contribution (pilot-gated) | Condition/expiry-aware temporal encoder | Not present (no temporal machinery yet) | Build only if pilot supports |
| Future work | GNN-MARL | Not planned | Out of scope this paper |

Building the graph TD3/SAC/PPO variants is largely swapping the GCN encoder onto
existing actor-critic backbones — incremental, not from scratch.

## Algorithm implementation: sourcing & verification

RL results are highly implementation-sensitive; a subtly wrong backbone
under-performs silently and invalidates comparisons. Every algorithm is sourced
from the highest available rung and **must pass verification before it produces
any number that appears in the paper.**

**Sourcing priority ladder** (drop a rung only when the one above doesn't fit):

1. The repo's existing implementations (audit before trusting) — DDPG, TD3, SAC,
   PPO, GCN-DDPG, heuristics all already present.
2. Our prior peer-reviewed code — ISO/MYO/MDL MILP heuristics and Fujimoto-style
   TD3/DDPG in `docs/research_howard/code-of-my-previous-drl-paper` (private).
3. Original author reference implementations — Fujimoto TD3/DDPG (`sfujim/TD3`),
   SAC authors' release.
4. Established benchmarked libraries — **CleanRL** (single-file, benchmarked
   curves — best cross-check), OpenAI **SpinningUp**, **Stable-Baselines3**;
   **PyTorch Geometric** for GNN layers (GCN/GAT/GATv2/SAGE).
5. Last resort — implement from the paper plus a vetted checklist (e.g. the PPO
   implementation-details checklist).

**Composition rule (graph variants):** GNN-TD3/SAC/PPO exist nowhere
off-the-shelf. Build each as a *verified GNN encoder ∘ verified RL backbone*,
then verify the composition.

**Verification gate** (each algorithm, before paper use):

- **V1 Component tests** — shapes, action bounds, feasibility projection,
  gradient flow (extend the existing test suite).
- **V2 Reference-task sanity** — solve a standard continuous-control task
  (e.g. `Pendulum-v1`) to the known reference return. Highest-value check; broken
  implementations fail here cheaply.
- **V3 Cross-library curve check** — compare the toy-task learning curve against
  CleanRL/SB3; divergence signals a bug.
- **V4 Feasibility-projection ablation** — quantify how much action-repair
  contributes (answers the "projection masks non-learning" objection).
- **V5 Multi-seed variance** — report IQM/CIs; never trust a single seed.

**Provenance table** (maintained as algorithms land; becomes the reproducibility
appendix): algorithm → source/reference → license → hyperparameter source →
verification status.

**Hyperparameters:** start from each reference's published defaults; document any
deviation and its rationale in the provenance table.

## Reproducibility & conduct rules

- Keep experiment parameters in config files, not hard-coded in models.
- Store large outputs in git-ignored folders (`results/`, `runs/`,
  `checkpoints/`, …) — already configured.
- Multi-seed everything reported; report uncertainty (CIs / IQM), not single
  runs. Prefer paired comparisons across algorithms on shared seeds/scenarios.
- **No fabricated results.** Numbers come only from logged runs.
- **English only** in all files, comments, and docs.
- `docs/research_howard/` is private and git-ignored: read for context, never
  copy its contents into tracked files, never commit it.
- Validation before finishing any coding task: `python -m compileall .` plus the
  relevant tests; a quick smoke run when training/eval code changes (see
  `AGENTS.md`).

## Project layout (intended additions)

```
src/env/            # existing PRM env (left intact) + new patient-condition/expiry env layer
src/graph/          # graph construction (+ any temporal-graph utilities, if pilot proceeds)
src/models/         # GCN encoder + actor-critic; new graph TD3/SAC/PPO variants
src/baselines/      # heuristics + flat DRL (+ ported MILP baselines from previous code)
experiments/configs # env-scenario JSON (+ new patient-condition / expiry scenarios)
configs/            # per-algorithm YAML (+ _20_clinic variants)
specs/              # this plan of record
```

## Decision log

### 2026-07-11

- **Verification harness built** (`src/verification/`, `evaluation/verify_algorithms.py`):
  an LQR task with an analytic (Riccati) optimum, scoring each real agent
  between random (0) and optimal (1). Implements V2/V3 of the verification gate.
- **Empirical finding — DDPG is unstable, TD3/SAC are not.** On the LQR task
  (20k steps), flat DDPG scored 0.996 / 0.401 / −70.4 across seeds 1/0/2
  (near-optimal to full divergence), while TD3 (0.83–0.96) and SAC (0.97–0.99)
  were stable across seeds. This is empirical backing for demoting DDPG (and
  GNN-DDPG) to an ablation and featuring TD3/SAC. *Consequence:* any
  DDPG-family result in the paper **must report all seeds + IQM**, never a single
  run — folded into the Phase 3 statistical protocol.
- **PPO passes — after a harness fairness fix.** PPO first scored 0.45–0.56 and
  did *not* improve with budget (4k→80k), which looked like a failure. Root
  cause was the harness: it reused the off-policy learning rate (1e-3) for PPO
  instead of PPO's tuned 3e-4. With an algorithm-appropriate config (3e-4 policy
  LR, entropy bonus, longer rollout) PPO passes consistently: 0.985 / 0.987 (80k
  seeds 0) and 0.885 (80k seed 1). *Lesson (now built into the harness):*
  verification must give each algorithm fair, appropriate hyperparameters, or it
  tests the config rather than the implementation.
- *Note:* the harness validates the learning machinery on an easy task; it does
  not yet exercise GNN encoders (Phase 6) or the capacity-planning action
  projection. It is the foundation of the gate, not the whole gate.

### 2026-07-10

- **Positioning:** claim problem speciality (perishable, identity-bound,
  patient-condition-driven manufacturing); treat algorithms as applied tools.
  *Why:* closest prior art (`research-context.md`, Paper 3) already does
  temporal + heterogeneous graph + PPO, so algorithm novelty is indefensible.
- **Environment:** implement patient conditions + expiry as a new module, not an
  in-place edit. *Why:* protect the working, tested manuscript baseline;
  contain invasive state-shape changes.
- **Backbones:** feature GNN-TD3 / GNN-SAC; keep GNN-DDPG as an ablation; keep
  GNN-PPO as on-policy comparator. *Why:* turns "why DDPG in 2026?" into a
  result; matches the closest rival's on-policy choice.
- **Temporal encoder:** pilot first, then decide. *Why:* motivated by
  deterioration/perishability, but must earn its place empirically.
- **MARL:** out of scope this paper. *Why:* large scope increase, overlaps prior
  art, dilutes the problem-first story.
- **Gurobi dependency (resolved):** use Gurobi for the MILP heuristics via a
  **free academic license** (Georgia Tech; the collaborator is an eligible
  student). Add `gurobipy` to requirements when Phase 5 begins. Keep the repo's
  non-MILP heuristics as a fallback if licensing slips.
- **Same paper, preserve the original:** this evolves the existing manuscript
  rather than starting a new one. Before substantial edits, freeze the current
  manuscript (git tag `manuscript-v0`, and/or a read-only copy) so the original
  is recoverable. *Why:* the user wants the starting point preserved.
- **Venue / timeline:** target EAAI, aim to submit July 2026. *Why it matters
  here:* the timeline forces an MVP-first technical scope (defer the temporal
  encoder; trim the experiment matrix) — see `roadmap.md`.
