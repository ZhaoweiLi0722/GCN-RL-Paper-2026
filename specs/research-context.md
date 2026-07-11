# Research Context

Consolidated background for the Introduction / Literature Review rewrite and for
justifying method and experiment choices. Derived from analysis of the current
manuscript, four internal deep-research reports, three prior GNN+DRL papers, and
two of our own prior papers. Source files are private (git-ignored) — see
pointers at the end; this file summarizes analysis, it does not reproduce them.

## The closest prior work (rank by threat)

1. **GNN-ODOA — enterprise dynamic resource allocation (the head-to-head threat).**
   Heterogeneous graph attention + a temporal context module + multi-objective
   PPO, evaluated on manufacturing/IT/logistics graphs (1k–4.6k nodes) under
   stochastic demand and equipment failure, against LP, greedy, DQN/PPO-flat,
   LSTM-RL, GAT-PPO, GCN-PPO. **It already has temporal + heterogeneous graph +
   PPO for capacity/resource allocation.** We must not claim algorithmic
   novelty against it.
   *What it does NOT model:* perishable / expiring product, patient health or
   patient-indexed demand, identity-bound (1-to-1) production, cold-chain /
   vein-to-vein deadlines, clinical constraints.

2. **GNN + MARL survey for 6G wireless.** Establishes the GCN/GAT + DRL
   paradigm and vocabulary. As a survey it cannot be "the same paper," but it is
   cited to argue our method machinery is standard — which we concede by design.
   Wireless domain; no perishability, patients, or manufacturing capacity.

3. **Graph MARL for UAV search & tracking.** Structural lineage only
   (graph-of-entities + attention + CTDE actor-critic). Spatial robotics, not
   supply/capacity planning — weakest overlap; cite as paradigm lineage.

## Differentiation axes (ranked, use these to load the novelty)

1. **Perishable, identity-bound product** (shelf-life/expiry + vein-to-vein):
   none of the rivals model goods that expire or must return to the *same*
   patient. Strongest, cleanest wedge — it changes state, constraints, and
   reward.
2. **Patient health/condition states as demand drivers:** time-critical demand
   from deterioration-while-waiting is unmodeled anywhere in the prior art.
3. **Personalized 1-to-1 (autologous) manufacturing:** non-fungible
   per-patient production breaks the rivals' fungible-resource formulations.
4. **A concrete cell-therapy clinic-network testbed** with cold-chain / clinical
   constraints, vs. their UAV field / wireless cells / generic enterprise graph.
5. **(Do NOT lead with this)** temporal + heterogeneous graph + disruptions —
   Paper 3 already has all three; position as shared machinery only.

## Literature stack for the Intro / Lit Review

All four deep-research reports converge on a three-layer stack; build the review
as an argument that ends in design implications, not a citation list.

- **(a) Method foundations:** DDPG, TD3, SAC, PPO; GCN, GraphSAGE, GAT/GATv2,
  GIN, graph transformers (Graphormer / GraphGPS); over-squashing.
- **(b) Domain-adjacent decision learning:** supply-chain RL, constrained RL,
  MARL, graph+RL for inventory/coordination.
- **(c) Domain-specific:** PRM / cell-therapy operations, biomanufacturing
  process control, decentralized capacity planning.

Recommended restructure of the manuscript's opening: problem importance → gap →
explicit research questions → contributions, with the Literature Review placed
*before* the Problem Formulation, ending in ~four design implications
(continuous-control actor-critic; graph representation; TD3/SAC over DDPG;
evaluation beyond mean cost).

## Citation candidates (public — safe to cite)

Method foundations: Lillicrap et al. (DDPG, ICLR 2016); Fujimoto et al. (TD3,
ICML 2018); Haarnoja et al. (SAC, ICML 2018); Schulman et al. (PPO, 2017); Kipf
& Welling (GCN, ICLR 2017); Hamilton et al. (GraphSAGE, 2017); Veličković et al.
(GAT, 2018); Brody et al. (GATv2, 2022); Xu et al. (GIN, 2019); Rampášek et al.
(GraphGPS, 2022); Ying et al. (Graphormer, 2021); Topping et al. (over-squashing,
2022).

Evaluation rigor: Henderson et al. (Deep RL that Matters, AAAI 2018); Agarwal et
al. (Statistical Precipice / IQM, NeurIPS 2021).

Domain / adjacent: Zheng et al. (hybrid model-based RL for cell-therapy process
control, 2022); Kotecha & del Rio-Chanona (GNN + MARL inventory, 2024 — closest
inventory comparator); Ahn et al. (GNN supply/inventory prediction; generative
probabilistic planning, 2024); Bermúdez et al. (distributional constrained RL,
2023); Mousa et al. (MARL decentralized inventory, 2023); Eisenach et al.
(neural coordination & capacity control, 2024); Stranieri et al. (classical vs
DRL for pharma supply chains, 2025); Chang et al. (learning production functions
with GNNs, 2024). Benchmarks: OR-Gym; SafeOR-Gym; D4RL.

(Full bibliographic detail to be pulled during Phase 1 writing; the manuscript's
existing `references.bib` already contains many of these.)

## Reviewer-objection checklist (design the experiments to answer these)

- **"Why DDPG in 2026?"** → include GNN-TD3/SAC as flagship, GNN-DDPG as
  ablation; show DDPG is dominated.
- **"RL only beats weak heuristics."** → include the strong MDL / rolling-horizon
  lookahead baseline, not just MYO/ISO.
- **"Single seed / not significant."** → multi-seed (target 10–20 where
  feasible, ≥5 minimum), CIs, paired tests, IQM.
- **"Feasibility projection masks non-learning."** → report how much the action
  projection is doing; ablate it.
- **"Overlap with GNN-ODOA / Kotecha."** → explicit differentiation on the
  perishability + patient-condition + identity axes, demonstrated empirically.
- **"Placeholder / decorative citations"** (present in the current draft, e.g.
  `article1`, `article2`, a broken `Eq. (14)` reference, a malformed projection
  equation) → clean these up during the rewrite.

## Notes on the current manuscript (correcting an earlier assumption)

The manuscript's **methodology section actually matches its introduction**
(GCN-DDPG, 20-clinic graph, MYO/MDL/ISO heuristics) — it is *not* carried over
from a different paper, though it has drafting seams (marketing-tone bullets, a
couple of broken equation references). Treat it as this project's genuine method
when deciding what to reuse vs. extend.

## Source pointers (private, git-ignored — do not commit)

- Current manuscript: `docs/research_howard/current-manuscript/main.tex`
  (read Intro + Lit Review; treat methodology as this project's own).
- Deep-research reports (Intro/Lit-Review positioning + citations):
  `docs/research_howard/deep-research-based-on-current-manuscript/`.
- Prior GNN+DRL papers (differentiate from): the three PDFs in
  `docs/research_howard/`.
- Our prior DRL paper + code (heuristic/MILP + DDPG/TD3 baselines):
  `docs/research_howard/my-previous-drl-paper/`,
  `docs/research_howard/code-of-my-previous-drl-paper/`.
- Patient-condition model to port (survival decay + deterioration shock +
  eligibility window):
  `docs/research_howard/my-previous-patient-condition-simulation-paper/`.
