# Online RL Attribution Post-Mortem and Follow-Up Design Brief

Status date: 2026-08-29
Author: Howard Tseng (analysis assisted by Claude)
Evidence basis: Stages F1, G0, G1, H0/H1 (`docs/patient_indexed_specimen_routing_stage_{f1,g0,g1}_results.md`, `docs/patient_indexed_specimen_routing_ddpg_problem_redefinition_review.md`, `docs/patient_indexed_specimen_routing_stage_e_evidence_synthesis.md`), locked execution plan v1.5.
Shareable version: https://claude.ai/code/artifact/978d8f8d-89b2-4528-b2f4-94244ffc4f61

> **Scope note.** This document interprets completed, audited evidence and designs a
> *follow-up study*. It authorizes no experiment. The routing-primary campaign is
> closed at Stage E under the locked execution plan; anything proposed here is a new
> study subject to that plan's change-control requirements.

## 1. One-paragraph synthesis

Online DDPG failed because it was asked to extract a below-noise-floor improvement,
through a gradient path that integer-lot quantization zeroes out, toward a target
whose ground-truth ranking does not replicate, on top of an anchor that had already
banked the accessible gain offline. This is a compound, structural failure — not
undertraining and not hyperparameters. The fix is therefore not primarily
architectural or algorithmic: the environment must first expose a learnable online
signal, and the project already owns the audit tooling to verify that *before*
training anything.

## 2. The five-failure causal chain

Each failure alone would cripple DDPG's learning signal; all five are present.

| # | Failure | Evidence | Key numbers |
|---|---------|----------|-------------|
| 1 | Reachable headroom is discrete; the deterministic policy gradient dies at execution (projection + integer-lot quantization is piecewise-constant) | G0; H0/H1 | 25/27 actor movements annihilated before execution (~7.3% displacement survives; 2/27 executed actions changed). Continuous channels execute 100% but show 0/27 clinically-safe prospective wins at the 1M threshold in both H-screens. |
| 2 | The learning target itself is statistically unstable (the binding failure) | G1 | Discovery/validation best-legal-action agreement 85/156 = 54.5% (gate 70%; worst seed 48%). Pairwise sign agreement 82.5% — a weak ordering exists, but policy improvement needs the argmax, where noise concentrates. Per-action threshold 1M ≈ 0.04% of the 2,686M objective. |
| 3 | Horizon mismatch in the supervision | G0 | 4-step vs remaining-horizon optimal actions agree in 51.9% of states; short-horizon optima cluster on +0.10 (16/27). |
| 4 | Auxiliary paired-advantage signal swamped by Bellman loss | F1 | Paired term 0.245–0.343% of total critic loss. G1's normalization fixed this and still failed on #2. |
| 5 | Strong anchor + bounded residual leave a sliver for online updates | F1, Stage E | Offline AFD banks −0.658%; final-vs-frozen deltas ±0.001–0.004% against CI widths ±0.02–0.03% (131/150 exact ties). |

## 3. Three remedial levers, assessed

### Lever 1 — Graph NN architecture: lowest value

Representation is the component that already works: GCN beats parameter-matched flat
2× (−0.658% vs −0.338%; graph-vs-flat CI wholly favorable) and holds on the TD3
backbone. None of failures 1, 2, 3, 5 involves the encoder; G1 removed the actor
entirely and supervised a critic directly on enumerated actions — it still failed.
Keep architecture work (temporal encoder etc.) as a later item, after a viable
learning signal exists.

### Lever 2 — Algorithm: necessary but not sufficient

- Addresses failure 1: hybrid discrete–continuous action methods (P-DQN/MP-DQN,
  H-PPO, HyAR) that treat the specimen correction as a discrete head; or stochastic
  policy gradients (PPO/SAC with mixed heads), whose likelihood-ratio gradient needs
  no differentiability through execution.
- Wolpertinger caution: proto-action → k-NN candidates → critic ranks them. G0/G1
  measured exactly that ranking capability at 30% top-1 vs a 40% gate; Wolpertinger
  inherits the failed link. Only viable after a critic passes a ranking gate.
- Does NOT address failure 2: any algorithm still optimizes a rollout-estimated
  target; flipping labels reappear as gradient variance. The fixes are statistical:
  paired-CRN counterfactual evaluation inside the training loop, shorter effective
  credit horizons (anchor-relative shaping), many-rollout averaged targets.
  Paired-evaluation direct search (CMA-ES-style) over the bounded residual is a
  defensible comparator that consumes CRN pairing natively.

Verdict: hybrid/stochastic action handling + paired-CRN variance reduction +
horizon-aligned targets, together. Swapping DDPG→SAC alone reproduces the null.

### Lever 3 — Simulation environment: highest value (the audit-endorsed path)

Add genuinely continuous, persistent, high-leverage operational decisions. This
attacks failure 1 (real continuous control authority) and failure 5 (headroom MDL-2
does not already plan). Environment knobs that raise label stability directly:
larger/more persistent per-decision consequences (the regional-shift regime was
always strongest: network AFR −1.13%) and a weaker or absent anchor on the new
dimensions. Cost: new simulator assumptions, extended heuristic comparators, new
validation — a new study, exactly as the redefinition review classifies it.

## 4. Follow-up program: Environment ≻ Algorithm ≻ Architecture

For the current paper: change nothing; ship it. For the follow-up:

1. **Environment first** — add 1–2 continuous decisions: overtime capacity
   (primary) and a production-rate throttle (secondary, dormant). Design in §5;
   preregistration in `specs/2026-08-29-continuous-overtime-control/`.
2. **Cheap screens before any training**, reusing existing tooling:
   - Headroom screen (`evaluation/audit_ddpg_continuous_control_headroom.py`
     pattern): the new channel must show validated, clinically-noninferior
     opportunity in ≥30% of states, or the environment is redesigned, not the agent.
   - Label-stability gate (G1 machinery): discovery/validation best-action agreement
     ≥70% at an affordable replication budget, or leverage is increased / credit
     horizon shortened until it passes.
   - Critic/ranking gate: held-out legal/candidate-action ranking must pass before
     any actor trains against the critic.
3. **Then the algorithm**: hybrid-action stochastic policy (H-PPO or SAC with mixed
   heads, or the legal-action-aligned proto-action design retained in
   `docs/patient_indexed_specimen_routing_stage_g1_design_review.md`), with
   paired-CRN advantage estimation baked into training. DDPG retained as the
   ablation.
4. **Architecture last**: temporal/attention encoder only once online learning
   demonstrably moves the needle, so its contribution is attributable.

This inverts the usual order — verify the learning problem is well-posed before
optimizing the learner — and is itself the methodological contribution: a gating
protocol that predicts when online RL can contribute, plus an environment where it
does.

## 5. Environment extension design (step 1, draft for review)

### 5.1 The current decision surface, and why no existing channel can carry an online-RL claim

**State space** (routing-primary patient environment). Per-facility observation
row = base block + patient block, concatenated over all 20 clinics (the GCN
receives the identical columns as node features plus four edge sets and
geography), plus one global normalized-time scalar:

| Base block (`features_per_facility` = 3 + lead_time + flags) | Width |
|---|---|
| demand, waiting specimens, reagent inventory | 3 |
| bioreactor pipeline (idle + in-process stages, shift register) | `production_lead_time` (5) |
| supplier availability · demand forecast *(flags)* | 1 + 1 |
| pending transfer arrivals (specimen/reagent/capacity) *(flag)* | 3 |
| demand history (rolling mean, trend, forecast error) *(flag)* | 3 |
| demand sequence block *(flag)* | 3 × L |

| Patient block (`summary_width` = 6 + buckets + routing) | Width |
|---|---|
| legacy patient scalars | 6 |
| survival histogram (edges 0.85 / 0.90 / 0.97) | 4 |
| routing state *(flag)* | 4 |

**Action space** — `facility_net`, `4n` dims, each in [−1, 1]
(`src/env/capacity_planning.py`, `_step_facility_net`):

| Block | Slice | Scaling | Executed as |
|---|---|---|---|
| w — specimen net flow | `[:n]` | × `max_specimen_transfer` | **integer patient-lot routing** (projection + quantization) |
| e — reagent net transfer | `[n:2n]` | × `max_reagent_transfer` | continuous float |
| q — capacity net transfer | `[2n:3n]` | × `max_bioreactor_transfer` | continuous float |
| p — reagent replenishment | `[3n:4n]` | `(a+1)/2` × `max_reagent_replenishment` × supplier | continuous float |

**The crucial non-action**: production is automatic —
`production = min(specimens, idle_bioreactors, reagents)`, greedy maximum every
epoch. The policy chooses *where things are*; it never chooses *how much to
make* or *how much capacity exists*.

**Reward**: `r = −cost`, additive decomposition. Operating group (~46% of
MDL-2's 2,686M objective): reagent purchase/holding/shortage, bioreactor
holding/shortage, three transfer costs. Patient group: patient loss (weight
50k default, 500k in the calibrated configs; ~45% of total cost), expiry
(40k), urgency (5k).

**Channel-by-channel failure map.** A channel supports online RL only if it
passes all four columns; every current channel fails at least one:

| Channel | Continuous execution? | Safe headroom? | Not banked by MDL-2 anchor? | Verdict |
|---|:---:|:---:|:---:|---|
| w specimen routing | No — integer lots (G0: 25/27 actor moves annihilated) | Yes (55.6% of states) | Yes | gradient dead |
| e reagent transfer | Yes (100%) | No (H0/H1: 0/27 prospective) | No | nothing to gain |
| q capacity transfer | Yes (100%) | No (0/27) | No | nothing to gain |
| p replenishment | Yes | No (reagent group) | No | nothing to gain |
| production / capacity level | — | — | — | **not a decision at all** |

Continuous where there is nothing to gain; discrete where there is; anchored
everywhere. That is failures 1 and 5 of §2 in one table, and it is why no
amount of algorithm or architecture work on the existing action space can
change the attribution result. The current action space only lets a policy
*reallocate a fixed pie* between clinics — a near-zero-sum game with tiny
margins over a heuristic that already reallocates well. The new decisions let
the policy *grow the pie at a price*.

### 5.2 Decision A — overtime capacity `u_ot` (primary)

- **Action**: per-facility continuous `u_ot ∈ [0, 1]`, one extra block
  (`action_size` → `5n`), scaled to
  `surge = u_ot · max_overtime_fraction · base_capacity`.
- **Effect**: relaxes the binding capacity bound in the production expression —
  `production = min(specimens, idle_bioreactors + surge, reagents)`. Surge is
  transient: never enters the shift register, never transferable.
- **Cost**: convex `c_ot = α·surge + β·surge²` per facility-epoch, added to the
  operating decomposition as `overtime_cost`. Convexity gives a smooth,
  state-dependent *interior* optimum ("how much", not "whether") — the exact
  geometry DDPG-class methods need and the current simulator lacks.
- **Why it passes the four columns by construction**: (i) surge and its cost
  are float end-to-end — even when the integer production count is unchanged
  this epoch, the marginal overtime cost is nonzero and sign-correct, so the
  gradient never flatlines; (ii) expanding the binding bound during demand
  shocks directly averts patient loss, the dominant (~45%) cost component —
  verified prospectively by gate E2, not assumed; (iii) MDL-2 does not plan
  this dimension and the spec prohibits anchoring it; (iv) the interior
  optimum plus the optional fatigue-persistence state target the ≥70%
  label-stability gate (E3).
- **Optional persistence**: a fatigue state decaying over K epochs that scales
  marginal overtime cost; behind a flag, reserved as the prespecified E3
  remediation.

### 5.3 Decision B — production-rate throttle `u_prod` (secondary, default OFF)

- **Action**: per-facility continuous `u_prod ∈ [0, 1]`, scaling the production
  expression: `production = round(u_prod · min(...))` under the existing
  integer-lot mechanics. `u_prod = 1` reproduces current dynamics exactly.
- **Leverage mechanism**: starting a lot commits a bioreactor for
  `production_lead_time` epochs and consumes a reagent now; deferral preserves
  capacity for sicker imminent arrivals — a genuine intertemporal decision the
  greedy `min(...)` cannot express and MDL-2 does not plan.
- **Why dormant**: the output is rounded to integer lots, and at typical
  per-epoch production of a few lots per clinic `u_prod` has only a handful of
  attainable levels — the same piecewise-constant geometry that killed the
  specimen channel (failure 1). It may activate only after its own separate E2
  headroom screen shows sufficient lot-count granularity.

### 5.4 State, costs, comparators, invariants

- **State**: append `u_ot` history / fatigue level and (if B enabled) current
  throttle to per-facility features and graph node features; flat baseline gets the
  identical block (matched-representation discipline).
- **Objective**: overtime cost joins the base operating group; report it in the
  additive decomposition. Cost-weight sensitivity required before any economic
  claim (per Stage E synthesis rules).
- **Extended heuristic field** (or the comparison is a strawman): `MDL-2-OT`
  (MDL-2 + myopic overtime rule: surge when eligible-waiting > idle capacity and
  marginal patient-loss risk exceeds marginal overtime cost), an `uMYO`-style surge
  variant, and a static-overtime policy sweep as the "tuned scalar" reference.
- **Anchor choice for learning**: anchor remains MDL-2(-OT) for routing dims; the
  overtime dim is learned as a *direct* continuous action (weak/no anchor), because
  failure 5 showed a strong anchor leaves nothing to learn online.
- **Invariants**: `u_prod=1, u_ot=0` reproduces the current environment bit-for-bit
  under fixed seeds (regression test); patient identity, CRN stream discipline, and
  clinical noninferiority gates unchanged.

### 5.5 Execution order and gates (no training before gates pass)

| Step | Deliverable | Gate |
|------|-------------|------|
| 0 | Change-control entry / new-study spec under `specs/` | User + Zhaowei sign-off |
| 1 | Env extension behind config flags + regression tests (`compileall`, fixed-seed equivalence, smoke episode) | Bit-identical legacy behavior with flags off |
| 2 | Extended heuristics (MDL-2-OT, surge uMYO, static-OT sweep) | Sanity: OT heuristics beat plain MDL-2 somewhere, else no headroom exists to learn |
| 3 | Headroom screen on the overtime channel (H0-style, 27+ states, discovery+validation CRNs) | ≥30% of states with validated clinically-noninferior ≥1M opportunity |
| 4 | Label-stability screen (G1-style) on candidate overtime actions | ≥70% discovery/validation best-action agreement |
| 5 | Critic feasibility (leave-one-seed-out ranking) | Top-1 and pairwise gates set prospectively before the run |
| 6 | Only then: algorithm phase (§4.3) | Preregistered attribution protocol: final-vs-frozen AND vs-matched-control, paired CRN |

Steps 3–5 are cheap (no training). If step 3 fails, the environment is redesigned —
larger `max_overtime_fraction`, different cost curvature, more stressed scenarios —
and the screen repeats. The agent is never tuned against a channel that has not
passed its screens.

## 6. Key reference numbers

Formal routing-primary holdout (5 seeds × 4 scenarios × 100 paired reps, seed 91100000):

| Comparison | Δ cost | Relative | 95% CI |
|---|---:|---:|---|
| AFR-GCN-DDPG − MDL-2 | −17.69M | −0.658% | [−19.36, −15.73]M |
| AFR-Flat-DDPG − MDL-2 | −9.09M | −0.338% | [−10.59, −7.51]M |
| GCN − Flat | −8.60M | −0.321% | [−11.07, −6.43]M |

Attribution audits: F1 final−frozen −0.001% (CI ±0.02–0.03%, 131/150 ties); G0 critic
top-1 3/27 vs control 2/27; G1 label stability 54.5% (gate 70%), fitted critic 30.1%
top-1 / 57.7% pairwise (gates 40% / 65%); H0/H1 continuous-channel prospective wins
0/27 in both scenarios.

## 7. External references

- HyAR (arXiv:2109.05490); P-DQN (arXiv:1810.06394); H-PPO (arXiv:1903.01344)
- Pairing seeds / CRN variance reduction (arXiv:2512.24145); variance-reduced policy
  gradients (NeurIPS 2020); many-actions PG (arXiv:2210.13011)
- DRL vs inventory heuristics: IJPE 2023 (S0925527323003316, S0925527323003201)
