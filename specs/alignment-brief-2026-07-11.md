# Alignment Brief — Direction Check with Zhaowei

**Date:** 2026-07-11 · **Purpose:** agree the paper's direction and scope before
we start writing/building. This is a discussion doc, not a decision — everything
below is a proposal for us to confirm together. Full detail lives in `specs/`.

## TL;DR of the proposed direction

- Keep this as **one paper**, evolved from the current manuscript (we've frozen
  the original as git tag `manuscript-v0`, nothing lost).
- **Reframe the contribution around the problem, not the algorithm.** Recent
  papers already pair GNNs with deep RL for capacity/resource allocation, so an
  "algorithm novelty" claim won't survive review. Our defensible wedge: a
  **perishable, patient-specific** cell-therapy network — product expiry /
  vein-to-vein deadlines and **deteriorating patient conditions** driving demand.
- Expand the sim to the **20-clinic network** (your proposal) **and** add the
  patient-condition / expiry layer (my proposal) as the differentiator.

## Decisions I'd like your sign-off on

1. **Problem-first framing.** Lead novelty on perishability + patient condition +
   1-patient-1-batch identity; treat the algorithms as applied tools. Agree?
2. **Algorithm plan.** Feature **GNN-TD3 / GNN-SAC**, keep **GNN-DDPG as an
   ablation**, include **GNN-PPO** as an on-policy comparator. This directly
   answers the "why DDPG in 2026?" question you raised — and see the evidence
   below. A **temporal (condition-aware) encoder** is a *pilot-first* maybe, not
   a commitment. **MARL is future work**, not this paper.
3. **Scope of the patient-condition model.** I'd port the survival-decay +
   deterioration-shock + eligibility-window mechanism from my prior
   patient-condition work, re-parameterized for the network. Does that match what
   you had in mind, or do you want it lighter/heavier?
4. **July timeline reality.** Submitting to EAAI in July is very tight for the
   full plan. I propose an **MVP cut**: env + baselines + graph family on a
   *static* encoder + a trimmed experiment matrix; defer the temporal encoder and
   the full matrix to a revision. Is July firm (→ we commit to the minimal
   defensible result), or can it slip (→ we widen scope)?

## Evidence we already have (to make the discussion concrete)

- **The pipeline runs end-to-end** — I reproduced the full 9-algorithm × 3-
  scenario benchmark at smoke budget on my machine.
- **"Why not DDPG" now has a data-backed answer.** I built a small verification
  test (an LQR task with a known-optimal solution) and ran the repo's agents:
  **TD3 and SAC learn reliably; SAC is best. Plain DDPG is unstable** — across
  seeds it ranged from near-optimal to complete divergence. That's concrete
  support for demoting DDPG to an ablation and featuring TD3/SAC. PPO also works
  once properly tuned.

## Logistics to settle

- **Gurobi license** for the MILP heuristics (MYO/MDL) from my prior code — can
  you set up a Georgia Tech academic license, or should I?
- **Target venue confirmed?** EAAI (Elsevier). The current manuscript already
  uses the Elsevier class, so no template change.
- **Experiment data/scenarios** — patient-condition parameters will be
  literature-based (survival curves), not proprietary data. OK?

## Not changing (unless you disagree)

- Same paper, same core problem family (distributed PRM capacity planning).
- Single-agent centralized graph controller (MARL deferred).
- The reproducibility rules: multi-seed + reported uncertainty, no fabricated
  numbers, English-only repo.
