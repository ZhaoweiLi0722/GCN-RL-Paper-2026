# 20-Clinic Campaign Results

Run: `evaluation/campaign_runner.py`, 20-clinic patient-condition env, **150k steps/seed,
5 seeds**, 20 eval replications. Deterministic graph backbones use the `facility_action`
readout. `results/campaign/` (`campaign_summary.csv`). IQM across seeds; 95% bootstrap CI
on eligibility.

Note: run at 150k (not the 300k manifest floor) for robustness — the environment repeatedly
sleep-killed longer runs, and seeds only bank on full completion. 150k is ~2x the 80k at
which `facility_action` already reached ~0.69.

## Ranking (by eligibility IQM)

| Rank | Algorithm | Eligibility IQM [95% CI] | Total cost IQM | Patients lost | Service |
|-----:|-----------|:------------------------:|:--------------:|:-------------:|:-------:|
| 1 | iso (heuristic) | 0.823 [0.823, 0.825] | 9.91e8 | 1158 | 0.680 |
| 2 | mdl2 (heuristic) | 0.823 [0.822, 0.824] | 9.84e8 | 1161 | 0.679 |
| 3 | myo (heuristic) | 0.805 [0.804, 0.807] | 1.01e9 | 1279 | 0.654 |
| 4 | mdl1 (heuristic) | 0.800 [0.799, 0.802] | 9.99e8 | 1293 | 0.652 |
| 5 | **gcn_ddpg** (facility_action) | 0.773 [0.742, 0.813] | 1.24e9 | 1628 | 0.592 |
| 6 | **gcn_td3** (facility_action) | 0.736 [0.639, 0.782] | 1.88e9 | 2423 | 0.410 |
| 7 | flat_ddpg (non-graph) | 0.282 [0.246, 0.337] | 2.86e9 | 3865 | 0.119 |

## Findings

**RQ1 — Graph vs flat: decisive graph win.** The graph backbones (0.74–0.77 eligibility)
crush non-graph `flat_ddpg` (0.28) — a ~2.6x eligibility gap and ~2x lower cost. The GCN
encoder is what makes learned control viable at 20-clinic scale; a flat state vector
collapses. This is the cleanest positive result of the study.

**RQ2 — Backbone: `gcn_ddpg` leads `gcn_td3` at scale (surprise).** The DDPG ablation
(0.773, cost 1.24e9, CI [0.742, 0.813]) *outperforms* the TD3 flagship (0.736, cost 1.88e9,
CI [0.639, 0.782]). `gcn_td3` also has much higher seed variance (seed0 = 0.60 drags it
down). The pilot's flagship choice was made at 2-clinic/30k; at 20-clinic/150k the ranking
flips. **Action: reconsider the flagship — report both, or add seeds / 300k to settle it.**

**RQ5 — Learned vs heuristics: heuristics still win.** All four heuristics (0.80–0.82)
beat all learned methods on both eligibility and cost. Best learned (`gcn_ddpg` 0.773)
trails the weakest heuristic (`mdl1` 0.800) and costs ~25% more. Learned control **narrowed
the gap dramatically** (from the 0.39 pilot collapse to 0.77) **but did not close it.**
Consistent with `Stranieri2025PharmaDRL` — RL does not uniformly dominate tuned heuristics
under perishability/non-stationarity.

## Implications for the paper

The honest, defensible narrative: **graph structure is the decisive factor for learned
capacity control here (graph >> flat), and the best graph-RL narrows but does not overtake
well-tuned look-ahead heuristics on the blended objective.** The contributions stand on the
problem formulation, the benchmark testbed, and the graph-vs-flat result — not on "RL beats
heuristics."

Open decisions:
1. **Flagship:** `gcn_ddpg` vs `gcn_td3` — settle with more seeds or the 300k budget
   (`gcn_td3`'s wide CI suggests it may improve; `gcn_ddpg` is currently the safer pick).
2. **Budget:** whether 300k closes the remaining gap to heuristics (eligibility trend is
   promising; the cost gap is less certain).
3. **SAC/PPO:** still excluded (global_flat). Add `facility_action` to their actors to
   complete the family comparison fairly.
