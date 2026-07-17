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

- completed learned arms: `gcn_residual_mdl2`, `gcn_residual_pmyo`,
  `flat_residual_mdl2`, `flat_residual_pmyo`
- completed scenarios: `graph_dynamic_patient_forecast_geo`,
  `patient_condition_stress`
- completed seeds: `0, 1`
- fallback decisions: all completed residual arms selected their heuristic
  anchor under the 0.5% validation-improvement rule

Fair re-evaluation summary for the completed targeted matrix:

| Scenario | Algorithm | Mean total cost | Service level | Avg. wait | Patients lost |
| --- | --- | ---: | ---: | ---: | ---: |
| geography | `mdl2` | 3.474B | 0.5570 | 10.1349 | n/a |
| geography | `flat_residual_mdl2` | 3.474B | 0.5570 | 10.1349 | n/a |
| geography | `gcn_residual_mdl2` | 3.474B | 0.5570 | 10.1349 | n/a |
| geography | `iso` | 3.791B | 0.4896 | 10.5019 | n/a |
| geography | `gcn_residual_pmyo` | 3.839B | 0.5348 | 11.4195 | n/a |
| geography | `pmyo` / `myo` | 3.839B | 0.5348 | 11.4195 | n/a |
| geography | `flat_residual_pmyo` | 3.839B | 0.5348 | 11.4195 | n/a |
| patient stress | `mdl2` | 840.28M | 0.6025 | 2.8079 | 1884.99 |
| patient stress | `flat_residual_mdl2` | 840.28M | 0.6025 | 2.8079 | 1884.99 |
| patient stress | `gcn_residual_mdl2` | 840.28M | 0.6025 | 2.8079 | 1884.99 |
| patient stress | `gcn_residual_pmyo` | 844.58M | 0.5788 | 2.8144 | 2013.83 |
| patient stress | `pmyo` | 844.58M | 0.5788 | 2.8144 | 2013.83 |
| patient stress | `flat_residual_pmyo` | 844.58M | 0.5788 | 2.8144 | 2013.83 |
| patient stress | `iso` | 847.91M | 0.6029 | 2.8066 | 1883.98 |
| patient stress | `myo` | 850.79M | 0.5784 | 2.8141 | 2016.87 |

Current interpretation: anchor fallback has achieved the conservative target
of making learned residual arms no worse than their heuristic anchors. However,
the 100-episode targeted matrix did not yet show a learned residual correction
that clears the 0.5% validation margin. Graph and flat residuals both reduce to
their anchors under conservative selection, so the current evidence supports
"safe heuristic-competitive learned control" rather than "learned control
outperforms tuned heuristics."

After evaluating `gcn_residual_pmyo`, we tightened the fallback decision rule:
pilot-scale budgets now require at least a 0.5% validation improvement before
deploying the learned residual. A seed with only a small validation edge did not
generalize on the 50-replication formal stream, so this margin is a practical
trust-region guardrail rather than a cosmetic threshold.

## Graph Residual Performance Diagnosis

The first targeted matrix did not show a graph advantage because the local
search teacher mostly perturbed replenishment decisions. Those corrections are
node-local and can be represented about as well by a flat residual model as by a
GCN. In other words, the learned policy was not being given many graph-structured
transfer corrections to imitate.

To give the graph policy a better target, the local-search candidate generator
now includes graph-aware reagent-transfer and capacity-transfer net-flow
patterns. The pressure signal combines demand, forecast, waiting workload,
reagent/capacity scarcity, and patient-risk counts when available. Mini-pilot
and targeted-pilot local search now keep only strictly improving demonstrations
(`min_improvement = 0.0`), while smoke remains permissive so the path can be
tested quickly.

Smoke validation:

```bash
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget smoke \
  --phase all \
  --force \
  --algorithms gcn_residual_mdl2 mdl2 \
  --scenarios patient_condition_stress
```

Result: the new graph-aware local search path completed end-to-end and found
2 improved demonstrations in the 3-step smoke setting. The next meaningful test
is a forced `targeted_100` rerun for `gcn_residual_mdl2` on
`patient_condition_stress`, followed by the same arm on the geography scenario.

Forced targeted rerun, `gcn_residual_mdl2`, `patient_condition_stress`, seed 0:

- local-search demonstrations increased from 170 to 236 improved steps
- mean step improvement increased from about 112.7k to 152.2k
- validation still selected the anchor:
  learned residual cost 910.74M vs. MDL-2 anchor cost 906.04M

This means the remaining performance bottleneck is not only finding better
one-step/short-horizon corrections; it is making the actor generalize those
corrections robustly across the validation stream.

The next model-side change adds the heuristic anchor action directly to each
facility node feature for residual GCN policies. Each node now receives its
anchor `(w,e,q,p)` action values, so the graph actor can learn "how to alter
the MDL-2 decision" rather than infer the anchor decision only from inventory,
demand, and patient-risk state. This is enabled only for residual graph policies
with `include_base_action_features=true`; pure GCN policies keep the original
feature layout.

Anchor-action feature smoke validation:

```bash
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget smoke \
  --phase all \
  --force \
  --algorithms gcn_residual_mdl2 mdl2 \
  --scenarios patient_condition_stress
```

Result: the path completed end-to-end. The learned residual was still rejected
by anchor fallback in smoke, but the feature-layout change is now verified
inside the actual benchmark runner.

Forced targeted rerun with anchor-action node features,
`gcn_residual_mdl2`, `patient_condition_stress`, seed 0:

- post-training local search again found 236 improved steps
- mean step improvement remained about 152.2k
- validation still selected the anchor:
  learned residual cost 912.34M vs. MDL-2 anchor cost 906.04M

Interpretation: the input representation alone did not solve the gap. The
teacher can find useful local corrections, but the student is still too likely
to over-apply those corrections outside the states where they are safe.

The next distillation change adds anchor-reference samples: local search still
keeps strictly improving actions, but it also samples a controlled fraction of
the heuristic anchor actions from the same visited state distribution. This
turns post-training into a conservative residual fit, teaching both "where to
correct" and "where to leave MDL-2 alone."

Anchor-reference targeted rerun, `gcn_residual_mdl2`,
`patient_condition_stress`, seed 0:

- local-search samples increased from 236 to 374
- anchor-reference steps: 138
- post-training fit loss improved from about 0.101 to 0.094
- validation still selected the anchor:
  learned residual cost 911.58M vs. MDL-2 anchor cost 906.04M

This helped fit quality slightly but did not close the validation gap. The next
structural change is pressure-projected residual actions: reagent transfer,
capacity transfer, and replenishment residuals are projected onto state-derived
shortage/surplus pressure patterns. This reduces the action head's freedom to
emit arbitrary per-clinic corrections and should make learned graph corrections
more consistent with the network-balancing teacher.

Pressure-projected targeted rerun, `gcn_residual_mdl2`,
`patient_condition_stress`, seed 0:

- elite-imitation losses dropped by roughly an order of magnitude during
  training, indicating that the projected correction class is much easier to fit
- validation improved modestly but still selected the anchor:
  learned residual cost 910.84M vs. MDL-2 anchor cost 906.04M

One remaining mismatch is correction magnitude. The local-search teacher tests
transfer perturbations as large as 0.05, but the learned transfer residual scale
was previously capped at 0.02. The next run aligns GCN reagent-transfer and
capacity-transfer residual scales with the teacher at 0.05, while keeping the
pressure projection active as a trust-region guardrail.

Scale-aligned targeted rerun, `gcn_residual_mdl2`,
`patient_condition_stress`, seed 0:

- post-training fit loss improved from about 0.094 to 0.046
- validation still selected the anchor:
  learned residual cost 911.21M vs. MDL-2 anchor cost 906.04M
- checkpoint diagnosis showed early stopping matters:
  episode 50 validation cost 909.23M, episode 100 validation cost 910.65M,
  post-local-search validation cost 911.21M

The next runner change enables validation-based learned-checkpoint selection
before anchor fallback. This prevents later training or local-search
distillation from overwriting the best learned residual checkpoint.

Checkpoint-selection rerun on the same trained artifacts:

- selected learned checkpoint: episode 50
- checkpoint-selection validation cost: 842.52M on the checkpoint-selection split
- anchor-fallback validation still selected MDL-2:
  selected learned checkpoint cost 909.23M vs. MDL-2 anchor cost 906.04M

This is the best patient-stress learned result so far for this seed. It reduces
the learned-vs-anchor validation gap to about 0.35%, but it does not yet justify
deploying the learned residual over MDL-2 in this scenario.

Geography scenario rerun, `gcn_residual_mdl2`,
`graph_dynamic_patient_forecast_geo`, seed 0:

- local search found 245 improved steps
- mean step improvement: 201.97k
- post-training fit loss: 0.062
- checkpoint selection chose the local-search checkpoint
- anchor-fallback validation: learned 3.578B vs. MDL-2 3.584B
- because the validation improvement was about 0.17%, below the 0.5% safety
  margin, fallback conservatively selected MDL-2

Learned-only formal evaluation on the 50-replication targeted stream:

- GCN residual learned-only total cost: 3.495B
- MDL-2 total cost on the same stream: 3.504B
- learned-only improvement: about 0.26%
- learned-only average waiting time: 10.10 vs. MDL-2 10.14

This is the first clear positive graph-residual signal: in the geography/network
scenario, the learned graph correction is slightly better than MDL-2 on both
validation and formal streams, but the improvement is still smaller than the
current conservative 0.5% deployment margin.

## Next Experiment Changes

Highest-priority changes before any longer run:

1. Run the same checkpoint-selection flow on seed 1 for both `patient_condition_stress` and `graph_dynamic_patient_forecast_geo` to determine whether the graph-residual advantage is stable.
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

## Seed 1 Follow-Up

Current targeted logs contain 100 training episodes per learned job. Across the
saved `targeted_100` matrix this is 16 learned jobs, or 1600 logged episodes.
Because several forced reruns overwrite the same CSVs, the actual compute spent
on `gcn_residual_mdl2` is higher than the logged artifact count.

Fresh seed 1 reruns with the current graph-residual formulation:

`patient_condition_stress`, seed 1:

- local search found 237 improved steps
- anchor-reference steps: 124
- mean step improvement: 168.47k
- post-training fit loss: 0.058
- checkpoint selection chose episode 100
- anchor fallback selected MDL-2:
  learned validation cost 834.56M vs. MDL-2 828.63M
- formal 50-rep stream therefore matches MDL-2 through fallback:
  828.48M total cost

`graph_dynamic_patient_forecast_geo`, seed 1:

- local search found 241 improved steps
- anchor-reference steps: 124
- mean step improvement: 192.78k
- post-training fit loss: 0.061
- checkpoint selection chose episode 50
- anchor fallback selected the learned policy:
  learned validation cost 3.385B vs. MDL-2 3.405B
- formal 50-rep stream:
  GCN residual 3.423B vs. MDL-2 3.444B, about 0.60% lower total cost

The geography/network result is now positive on both seed 0 and seed 1. Seed 0
was positive learned-only but below the conservative 0.5% deployment margin;
seed 1 clears that margin and deploys the learned residual. The patient-stress
scenario remains heuristic-dominant under the current reward/action design.
