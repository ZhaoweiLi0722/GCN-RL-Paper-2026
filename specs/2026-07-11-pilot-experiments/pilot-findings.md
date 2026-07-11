# Phase 7 — Pilot Findings

Status: **in progress** — harness built and smoke-validated; the lean two-stage
run is compute-bound and may span sessions. Numbers below are filled from
completed runs only; partial coverage is stated explicitly (no fabricated
results).

## Setup

- **Regime:** Stage A screen on `2_clinic_patient_condition`; Stage B confirm on
  `20_clinic_patient_condition`.
- **Roster:** MYO/ISO/MDL-1/MDL-2/F-MYO/uMYO, flat-DDPG, GNN-DDPG (ablation),
  GNN-TD3, GNN-SAC, GNN-PPO.
- **Budget:** 5 seeds, ~30k train steps/agent, eval over N replications.
- **Reward:** blended (operational + patient), weights unchanged.
- **Flagship rule:** rank by patient eligibility (IQM across seeds) on Stage B,
  tie-break by total cost IQM, then cross-seed stability.
- **Runner:** `evaluation/run_patient_pilot.py` (`--resume` supported).

## Stage A — screen (2-clinic)

_TBD — ranking by eligibility IQM (best-first), with cost IQM and divergent-seed
flags. Top flagship-candidate backbone promoted to Stage B._

| Algorithm | eligibility IQM | total_cost IQM | divergent seeds |
|-----------|-----------------|----------------|-----------------|
| _pending_ | | | |

Top backbone promoted: _TBD_

## Stage B — confirm (20-clinic)

_TBD — top backbone + GNN-DDPG (ablation) + flat-DDPG + MDL-2._

| Algorithm | eligibility IQM | total_cost IQM | divergent seeds |
|-----------|-----------------|----------------|-----------------|
| _pending_ | | | |

## Decisions (to record in tech-stack.md)

1. **Flagship backbone:** _TBD_ — evidence: _TBD_.
2. **Temporal encoder (Phase 8) go/no-go:** _TBD_. Watch the D6 question — do
   condition-aware / graph methods separate from condition-blind ones on
   eligibility? If not, that is the case for the temporal encoder (or a Phase 9
   weight revisit).
3. **Stability + projection load:** _TBD_ — per-seed spread / divergence (esp.
   DDPG-family) and mean projection repair magnitude per algorithm.
