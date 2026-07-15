# Residual + Patient-Risk Mini Pilot Results

Date: 2026-07-15

## Scope

This note records the first end-to-end mini pilot after adding:

- geography-aware 20-clinic dynamics,
- patient risk types and waiting-time-dependent deterioration,
- residual MLP-DDPG baselines anchored on MDL-2 / ISO / MYO,
- residual GCN-DDPG benchmark arms,
- a patient-condition stress scenario in the residual benchmark plan.

The run is intentionally small and is not manuscript-final:

```bash
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase all \
  --force \
  --algorithms flat_residual_mdl2 gcn_residual_mdl2 gcn_td3 mdl2 iso myo umyo \
  --scenarios graph_dynamic_patient_forecast_geo patient_condition_stress
```

A follow-up run added the ISO/MYO residual anchors and then the patient-priority
MYO (`pmyo`) teacher:

```bash
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase train \
  --algorithms flat_residual_iso flat_residual_myo gcn_residual_iso gcn_residual_myo \
  --scenarios graph_dynamic_patient_forecast_geo patient_condition_stress

.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase evaluate \
  --algorithms flat_residual_iso flat_residual_myo gcn_residual_iso gcn_residual_myo \
  --scenarios graph_dynamic_patient_forecast_geo patient_condition_stress

.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase train \
  --algorithms flat_residual_pmyo gcn_residual_pmyo \
  --scenarios graph_dynamic_patient_forecast_geo patient_condition_stress

.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase evaluate \
  --algorithms flat_residual_pmyo gcn_residual_pmyo pmyo \
  --scenarios graph_dynamic_patient_forecast_geo patient_condition_stress

# After enabling residual_action.zero_init_actor, rerun the key patient-stress arms:
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase train \
  --force \
  --algorithms flat_residual_myo flat_residual_pmyo gcn_residual_myo gcn_residual_pmyo \
  --scenarios patient_condition_stress

.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget mini_pilot \
  --phase evaluate \
  --force \
  --algorithms flat_residual_myo flat_residual_pmyo gcn_residual_myo gcn_residual_pmyo \
  --scenarios patient_condition_stress
```

Budget:

- seeds: 0, 1
- learned training: 10 episodes, 12 steps per episode
- evaluation: 5 Monte Carlo replications per seed
- aggregate count per algorithm/scenario: 10

Primary output:

- `results/residual_policy_benchmark/mini_pilot/aggregate_summary.csv`
- `figures/residual_policy_benchmark/mini_pilot/`

## Results

### `graph_dynamic_patient_forecast_geo`

Lower total cost is better.

| Algorithm | Mean total cost | Gap vs best | Service level | Avg waiting time |
|---|---:|---:|---:|---:|
| `gcn_residual_mdl2` | 198.34M | 0.00% | 0.5395 | 3.0821 |
| `mdl2` | 198.36M | 0.01% | 0.5395 | 3.0824 |
| `flat_residual_mdl2` | 198.59M | 0.12% | 0.5394 | 3.0831 |
| `myo` | 212.24M | 7.01% | 0.5225 | 3.2170 |
| `pmyo` | 212.24M | 7.01% | 0.5225 | 3.2170 |
| `umyo` | 212.24M | 7.01% | 0.5225 | 3.2170 |
| `flat_residual_myo` | 212.27M | 7.02% | 0.5224 | 3.2154 |
| `flat_residual_pmyo` | 212.27M | 7.02% | 0.5224 | 3.2154 |
| `gcn_residual_myo` | 212.34M | 7.06% | 0.5218 | 3.2192 |
| `gcn_residual_pmyo` | 212.34M | 7.06% | 0.5218 | 3.2192 |
| `iso` | 245.45M | 23.75% | 0.4610 | 3.6271 |
| `gcn_residual_iso` | 245.48M | 23.77% | 0.4610 | 3.6273 |
| `flat_residual_iso` | 245.62M | 23.83% | 0.4610 | 3.6274 |
| `gcn_td3` | 523.62M | 164.00% | 0.4610 | 3.6271 |

Interpretation:

- The residual GCN arm is now competitive with the strongest tuned heuristic in the geography-aware scenario.
- The margin over MDL-2 is tiny under this small budget, so it should be treated as a promising smoke signal, not evidence of superiority.
- Flat residual is also close, which means part of the gain comes from residualizing around MDL-2 rather than from graph structure alone.
- MYO/ISO residual anchors do not help in the geography scenario; the learned policies largely inherit their weaker anchors.

### `patient_condition_stress`

| Algorithm | Mean total cost | Gap vs best | Service level | Avg waiting time |
|---|---:|---:|---:|---:|
| `myo` | 89.51M | 0.00% | 0.6518 | 2.0365 |
| `pmyo` | 89.66M | 0.17% | 0.6513 | 2.0375 |
| `iso` | 90.98M | 1.65% | 0.6585 | 2.0335 |
| `mdl2` | 91.02M | 1.69% | 0.6580 | 2.0351 |
| `umyo` | 113.56M | 26.87% | 0.6480 | 2.0375 |
| `gcn_residual_pmyo` | 119.63M | 33.66% | 0.5853 | 2.2148 |
| `flat_residual_myo` | 120.08M | 34.16% | 0.5833 | 2.2233 |
| `flat_residual_pmyo` | 120.12M | 34.20% | 0.5838 | 2.2220 |
| `gcn_residual_myo` | 120.24M | 34.34% | 0.5818 | 2.2286 |
| `flat_residual_mdl2` | 123.23M | 37.68% | 0.5706 | 2.2606 |
| `flat_residual_iso` | 123.32M | 37.77% | 0.5698 | 2.2626 |
| `gcn_residual_iso` | 123.69M | 38.20% | 0.5687 | 2.2722 |
| `gcn_residual_mdl2` | 123.89M | 38.42% | 0.5684 | 2.2761 |
| `gcn_td3` | 453.77M | 406.98% | 0.5972 | 2.1937 |

Interpretation:

- Patient-condition stress is still heuristic-favored.
- Switching the anchor from MDL-2 to MYO improves learned residual cost by about 3.4 percentage points of gap, but the learned residual policies still trail direct MYO/ISO/MDL-2.
- The new `pmyo` heuristic fixes most of the old `umyo` overreaction: it is within 0.17% of MYO on patient stress and identical to MYO on the non-patient geography scenario.
- With zero-initialized residual actors, `gcn_residual_pmyo` is the best learned patient-stress arm, but it still trails MYO by 33.66%.
- `pmyo` as a residual teacher improves the best GCN learned arm slightly, so the direction is useful, but the bottleneck is now residual drift / training control rather than only the patient-aware base heuristic.
- `umyo` underperforms plain MYO here, so its urgency logic is not tuned for this risk/cost regime.
- The current GCN-TD3 setting remains unstable at this tiny training budget.

## Decision

The next publishable path should not be "generic RL beats heuristics everywhere." The evidence currently supports a more precise story:

1. Residualizing learned policies around strong heuristics fixes most early DDPG instability.
2. Graph residual policies can match or slightly improve strong heuristics in geography-aware network scenarios.
3. Patient-risk scenarios need a stronger learned correction/training formulation, not just another patient-aware heuristic anchor.

## Anchor Fallback Update

After the mini pilot, the benchmark runner was extended with conservative anchor
fallback. For learned residual policies, evaluation now:

1. evaluates the learned residual checkpoint on a held-out validation seed stream;
2. evaluates its heuristic anchor on the same validation stream;
3. deploys the learned residual only if its validation cost is no worse than the anchor;
4. otherwise deploys the anchor while preserving the learned algorithm label and writing fallback metadata to the raw rows.

Smoke command:

```bash
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget smoke \
  --phase all \
  --force \
  --algorithms gcn_residual_pmyo pmyo \
  --scenarios patient_condition_stress
```

Smoke decision:

- `gcn_residual_pmyo` validation learned cost: 5.8641M
- `pmyo` anchor validation cost: 5.8640M
- selected policy: `anchor`

This confirms the safe-improvement behavior: if the learned residual does not
clear the anchor on validation, the evaluation falls back to the heuristic
instead of deploying a worse residual policy.

### Fair Evaluation Fix

During the first `targeted_100` evaluation pass, we found that learned policies
and heuristic baselines were not always using the same merged environment
definition. Learned configs merged benchmark-wide 20-clinic defaults with the
scenario file, while heuristic configs used only the scenario file. This could
make heuristic baselines easier than the learned-policy evaluation stream.

The runner now builds all evaluation environments through the same
scenario-merge helper. In `residual_policy_benchmark.json`, heuristic baselines
use `gcn_residual_mdl2` as the reference environment, so demand shocks,
geography, patient-risk settings, and graph defaults are shared.

Targeted 100-episode checkpoint status:

- completed: `gcn_residual_mdl2`, seeds `0, 1`
- scenarios: `graph_dynamic_patient_forecast_geo`, `patient_condition_stress`
- both scenarios selected the `mdl2` anchor during fallback validation

Fair re-evaluation summary for completed arms:

| Scenario | Algorithm | Mean total cost | Service level | Avg. wait | Patients lost |
| --- | --- | ---: | ---: | ---: | ---: |
| geography | `mdl2` | 3.474B | 0.5570 | 10.1349 | n/a |
| geography | `gcn_residual_mdl2` | 3.474B | 0.5570 | 10.1349 | n/a |
| geography | `iso` | 3.791B | 0.4896 | 10.5019 | n/a |
| geography | `pmyo` | 3.839B | 0.5348 | 11.4195 | n/a |
| patient stress | `mdl2` | 840.28M | 0.6025 | 2.8079 | 1884.99 |
| patient stress | `gcn_residual_mdl2` | 840.28M | 0.6025 | 2.8079 | 1884.99 |
| patient stress | `pmyo` | 844.58M | 0.5788 | 2.8144 | 2013.83 |
| patient stress | `iso` | 847.91M | 0.6029 | 2.8066 | 1883.98 |

Current interpretation: anchor fallback has achieved the conservative target
of making the learned residual arm no worse than its heuristic anchor. The
remaining research question is whether residual learning can beat the anchor
without relying on fallback; that requires the rest of the targeted 100-episode
matrix plus tighter residual/trust-region tuning.

## Next Experiment Changes

Highest-priority changes before any longer run:

1. Use the new `targeted_100` budget to run a decisive residual-vs-anchor pilot with anchor fallback enabled.
2. Keep `pmyo` as the fair condition-aware heuristic; de-emphasize the older `umyo` surge policy unless it is retuned.
3. Run a targeted 100-episode pilot on:
   - `graph_dynamic_patient_forecast_geo`
   - `patient_condition_stress`
   - `flat_residual_mdl2`
   - `flat_residual_myo`
   - `flat_residual_iso`
   - `flat_residual_pmyo`
   - `gcn_residual_mdl2`
   - `gcn_residual_myo`
   - `gcn_residual_iso`
   - `gcn_residual_pmyo`
   - `mdl2`
   - `myo`
   - `pmyo`
   - `iso`

SAC/PPO should stay outside the next decisive pilot until the residual/control formulation is stronger; otherwise they will mostly measure training instability rather than the value of graph-aware control.
