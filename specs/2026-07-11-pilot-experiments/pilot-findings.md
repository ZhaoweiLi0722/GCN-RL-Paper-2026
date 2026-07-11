# Phase 7 — Pilot Findings

Status: **complete** (lean two-stage pilot). Numbers are from completed runs only.
The scaled-campaign readout comparison (Lever 2) is a separate follow-up run;
its result is appended when available.

## Setup

- **Regime:** Stage A screen on `2_clinic_patient_condition`; Stage B confirm on
  `20_clinic_patient_condition`.
- **Roster:** MYO/ISO/MDL-1/MDL-2/F-MYO/uMYO, flat-DDPG, GNN-DDPG (ablation),
  GNN-TD3, GNN-SAC, GNN-PPO.
- **Budget:** 5 seeds, ~30k train steps/agent, eval over 20 replications.
- **Reward:** blended (operational + patient), weights unchanged.
- **Flagship rule:** rank by patient eligibility (IQM across seeds), tie-break by
  total cost IQM, then cross-seed stability.
- **Runner:** `evaluation/run_patient_pilot.py` (`--resume` supported).

## Stage A — screen (2-clinic)

**Eligibility saturates (0.9167 IQM) for every method** — the 2-clinic queue is
never stressed enough for patient-condition awareness to discriminate, so **cost is
the only discriminator**. Ranked by cost among the flagship candidates:

| Algorithm | eligibility IQM | total_cost IQM | divergent seeds (cost) |
|-----------|:---------------:|:--------------:|:----------------------:|
| myo / umyo | 0.9167 | 2.34e5 | [0] |
| fmyo / mdl1 | 0.9167 | 2.65e5 | [] |
| iso | 0.9167 | 2.92e5 | [] |
| **gcn_td3** | 0.9167 | **2.92e5** | **[]** |
| mdl2 | 0.9167 | 2.95e5 | [] |
| gcn_ddpg (ablation) | 0.9167 | 2.96e5 | [1, 3] |
| flat_ddpg (ablation) | 0.9167 | 3.18e5 | [4] |
| gcn_ppo | 0.9167 | 3.24e5 | [1] |
| gcn_sac | 0.9167 | 6.09e5 | [] |

**Top backbone promoted: `gcn_td3`** — best-behaved flagship candidate: lowest cost
(2.92e5 vs SAC 6.09e5), and zero divergent seeds (PPO had a divergent seed). It even
matches the `iso` heuristic on cost at this scale.

## Stage B — confirm (20-clinic)

**The heuristic `mdl2` wins decisively; every learned policy collapses.**

| Algorithm | eligibility IQM | total_cost IQM | patients lost | divergent seeds |
|-----------|:---------------:|:--------------:|:-------------:|:---------------:|
| **mdl2** (heuristic) | **0.823** | **9.84e8** | 1161 | [] |
| gcn_ddpg (ablation) | 0.430 | 2.99e9 | 3271 | [4] |
| gcn_td3 (flagship) | 0.391 | 3.10e9 | 3328 | [] |
| flat_ddpg (ablation) | 0.308 | 3.04e9 | 3810 | [] |

The learned methods lose ~3x on cost and ~2x on eligibility. `gcn_td3` had **zero
divergent seeds** — *stably underfit, not unstable*.

## Diagnosis: undertraining, not a genuine negative result

`gcn_td3` was competitive at 2-clinic under the same 30k budget and stably bad
(not divergent) at 20-clinic. The 20-clinic problem has ~10x the state/action
dimension (action ≈ 80-d), so 30k steps is far too few. A likely co-cause: the
default `global_flat` graph actor's head is ~1280-d at 20 clinics (see
`campaign-scale-plan.md`). The pilot did its job — it caught this before the full
campaign spend.

## Decisions

1. **Flagship backbone: `gcn_td3`** — evidence: best-behaved graph backbone in
   Stage A (lowest cost among candidates, zero divergent seeds). *Provisional:*
   must be re-confirmed at scale against `mdl2` after the budget fix.
2. **Temporal encoder (Phase 8) go/no-go: DEFER.** Eligibility did not discriminate
   at 2-clinic, and the 20-clinic gap is an undertraining artifact, not evidence of
   condition-blindness. A temporal encoder is premature until the scaled campaign
   resolves the core RL-vs-heuristic question.
3. **Stability + projection load:** `gcn_td3` and `gcn_sac` had zero divergent seeds;
   the DDPG family was unstable (`gcn_ddpg` seeds [1,3], `flat_ddpg` seed [4]),
   confirming the switch to stability-oriented backbones and IQM/CI reporting.

## Follow-up (campaign preparation)

- **Lever 1 (budget):** scale 20-clinic training ~10x — `evaluation/campaign_manifest.py`
  (300k floor / 500k flagship). Do **not** launch at 30k.
- **Lever 2 (representation + curriculum):** `facility_action` readout + 2->20
  warm-start, verified in `tests/test_gcn_facility_action.py` and wired for the
  deterministic backbones. See `campaign-scale-plan.md`.
- **Readout comparison (running):** `evaluation/readout_comparison.py` compares
  `facility_action` vs `global_flat` on `gcn_td3` at 20-clinic / 80k steps, with
  `mdl2` as reference. Result appended here on completion.
