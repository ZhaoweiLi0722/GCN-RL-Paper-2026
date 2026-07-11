# Mission

## Overview

We are writing a research paper on **graph-aware deep reinforcement learning
(DRL) for adaptive capacity planning in distributed, patient-specific
cell-therapy manufacturing networks**, where the manufactured product is
**perishable and identity-bound** (one patient ↔ one batch) and **demand is
driven by deteriorating patient health**.

The work builds on an existing codebase and manuscript (a GCN-DDPG policy for a
20-clinic personalized regenerative medicine network) but reframes the
contribution around the *problem*, not the algorithm.

What the paper delivers:

- A simulation of a multi-site autologous cell-therapy network that models
  **patient health conditions** and **product/material expiry** as first-class
  drivers of demand and feasibility.
- A **graph-aware actor-critic** controller for coordinated capacity, reagent,
  and specimen decisions across the network.
- A rigorous benchmark against strong operational heuristics and flat-state DRL,
  under supply disruption, forecast error, and a **new patient-condition /
  expiry stress regime**.

## Motivation

Autologous cell therapies (e.g. CAR-T) are made to order for a single patient
and **cannot be pre-produced or pooled**. Production can start only when a
patient specimen, an available bioreactor, and sufficient consumables coincide.
Turnaround is long, and while a patient waits, their condition **deteriorates**
— past a threshold they become ineligible, and the manufactured product itself
has a hard **shelf-life / vein-to-vein window**. Capacity decisions across a
network of clinics therefore interact with clinical urgency and perishability in
a way generic capacity planning does not capture.

Existing graph-neural-network + DRL work for capacity and resource allocation
(see `research-context.md`) treats demand as fungible, abstract requests and
does not model perishability, patient condition, or identity-bound production.
That is the gap we close.

## Target audience

- **Primary venue:** *Engineering Applications of Artificial Intelligence*
  (EAAI, Elsevier) — https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence.
- **Target submission:** July 2026. No hard external deadline, but this is an
  aggressive internal target — see the timeline note in `roadmap.md`; it implies
  an MVP-first scope.
- **Readers:** researchers and practitioners in intelligent decision support for
  healthcare manufacturing, supply-chain RL, and graph learning for operations.

## Core research questions

1. Does a **graph representation** of the clinic network improve coordinated
   capacity/inventory/specimen policies over flat-state control?
2. Which **actor-critic backbone** (DDPG / TD3 / SAC / PPO) is best suited to
   this constrained continuous-control problem, and why not DDPG alone?
3. Does modeling **patient condition and product expiry** change the optimal
   policy and expose failures of condition-blind planners?
4. (Pilot-gated) Does a **condition/expiry-aware temporal encoder** — memory of
   deterioration trajectories — beat a static graph encoder?
5. Do learned policies beat **strong lookahead heuristics** (not just weak ones)
   under disruption, forecast error, and patient-condition stress?

## Contributions (positioning)

**Primary — problem speciality (the defensible wedge):**

- A capacity-planning formulation for **perishable, identity-bound, patient-
  specific** manufacturing across a clinic network — with patient-condition
  dynamics and expiry windows that no prior GNN+DRL capacity work models.
- A simulation environment realizing this, reusable as a benchmark testbed.

**Secondary — a modest method contribution (candidate, pilot-gated):**

- A **condition/expiry-aware temporal graph encoder**, motivated by the
  deterioration and perishability dynamics of the problem (not proposed as
  generic temporal-GNN novelty). Included **only if** a pilot shows it helps;
  otherwise it becomes a discussed design choice / future work.

Method backbones (graph actor-critic across DDPG/TD3/SAC/PPO) are framed as
**established tools applied and compared**, not as claimed algorithmic novelty.

## Scope (this paper)

In scope:

- A new environment layer adding patient-condition states and product/material
  expiry, on top of the existing 20-clinic PRM network model.
- Graph actor-critic method family: GNN-DDPG (ablation), GNN-TD3, GNN-SAC,
  GNN-PPO; adopt the best-performing as the proposed policy.
- Baselines: MYO, ISO, MDL-1, MDL-2 heuristics and flat-state DDPG.
- Multi-seed training and Monte Carlo evaluation with proper statistics.
- A pilot phase that decides (a) the temporal-encoder contribution and (b) the
  flagship backbone before the full experiment campaign.

Non-goals (this paper):

- **Multi-agent RL (MARL).** The core stays a single-agent centralized graph
  controller; MARL is future work (adds scope and overlaps prior art).
- Claiming novelty of any RL or GNN algorithm in isolation.
- Real patient data or clinical validation; the patient-condition model is
  literature-parameterized (survival-curve based), not fit to proprietary data.

This is the **same paper** as the existing manuscript, evolved — not a separate
publication. We simply **preserve a frozen snapshot of the original manuscript**
before making substantial changes, so the starting point is recoverable (see
`roadmap.md` Phase 0).

## Success criteria

- Every research question above is answered with multi-seed results and
  reported uncertainty (CIs / IQM), traceable to logged outputs.
- The differentiation from the closest prior work (`research-context.md`) is
  demonstrated, not just asserted — the patient-condition / expiry regime
  produces policy differences a condition-blind planner cannot match.
- The manuscript's algorithm choices survive the standard reviewer objections
  ("why DDPG in 2026?", "RL only beats weak heuristics", "single seed").

## Open items to confirm

- **Collaborator alignment (Zhaowei):** the patient-condition direction was
  *proposed* in discussion, not finalized. Confirm scope before Phase 1 writing.
- **July 2026 timeline vs. scope:** the full plan is unlikely to fit a July
  submission. Agree an MVP-first cut (see `roadmap.md` timeline note) — most
  likely defer the temporal encoder and trim the experiment matrix.
