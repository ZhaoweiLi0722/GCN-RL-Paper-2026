# RL Baselines

This project uses learning-based baselines to separate the value of graph-aware
state encoding from the value of actor-critic reinforcement learning itself.

## Implemented Baselines

### Heuristic Policies

The deterministic benchmarks are implemented under `src/baselines/heuristics.py`.
They share the same 20-clinic environment and facility-net action layout as the
learned agents.

- MYO: current-period myopic balancing with sharing enabled.
- ISO: local replenishment only, with sharing disabled.
- MDL-1: one-period mean-demand lookahead with sharing enabled.
- MDL-2: two-period mean-demand lookahead with sharing enabled.

Run formal Monte Carlo evaluation for a heuristic:

```bash
python -m evaluation.evaluate_formal \
  --algorithm myo \
  --env-config experiments/configs/20_clinic_disruption_0_3.json \
  --replications 500
```

### Flat-State DDPG / MLP-DDPG

Flat-state DDPG is the key baseline. It keeps the same environment and action
space as the graph-aware method, but removes the GCN encoder. Facility states
are flattened into one vector and passed to MLP actor and critic networks.

Use this baseline to test whether GCN-DDPG improves over DDPG because it uses a
graph representation, not merely because it uses actor-critic learning.

Run:

```bash
python -m training.train_flat_ddpg --config configs/flat_ddpg.yaml
```

For the manuscript-scale environment:

```bash
python -m training.train_flat_ddpg --config configs/flat_ddpg_20_clinic.yaml
```

### GCN-DDPG

GCN-DDPG uses the same continuous-control DDPG update as the flat baseline, but
rebuilds each replay-buffer state into graph node features before the actor and
critic. The default 20-clinic graph contains clinic nodes plus a central
capacity hub.

Run:

```bash
python -m training.train_gcn_ddpg --config configs/gcn_ddpg_20_clinic.yaml
```

### TD3

TD3 is the closest modern DDPG-style baseline. It uses two critics, clipped
target policy smoothing, delayed actor updates, replay buffer learning, and soft
target updates.

Run:

```bash
python -m training.train_td3 --config configs/td3.yaml
```

For the manuscript-scale environment:

```bash
python -m training.train_td3 --config configs/td3_20_clinic.yaml
```

### SAC

SAC is implemented as a flat-state, entropy-regularized off-policy baseline
with a tanh-squashed Gaussian actor, twin critics, target critics, and optional
automatic entropy tuning.

Run:

```bash
python -m training.train_sac --config configs/sac_20_clinic.yaml
```

### PPO

PPO is implemented as a flat-state on-policy baseline with a tanh-squashed
Gaussian actor, value network, generalized advantage estimation, and the clipped
surrogate objective.

Run:

```bash
python -m training.train_ppo --config configs/ppo_20_clinic.yaml
```

## Canonical Algorithm Code References

The baseline implementations in this repository are maintained in a unified
PyTorch codebase so that all methods share the same environment, action space,
logging, seeds, and evaluation runner. They should be described as our
implementations cross-checked against canonical sources, not as directly
vendored original code.

Last checked: 2026-07-11.

| Algorithm | Canonical/source implementation | How to use it in this project |
| --- | --- | --- |
| DDPG | No standalone DeepMind/Lillicrap author repository was identified during the source search. Use OpenAI Baselines DDPG (`https://github.com/openai/baselines/tree/master/baselines/ddpg`) and OpenAI Spinning Up DDPG (`https://spinningup.openai.com/en/latest/algorithms/ddpg.html`) as canonical implementation references. | Cross-check actor/critic targets, replay-buffer updates, target-network soft updates, and exploration noise in `src/baselines/flat_ddpg.py` and `src/models/gcn_ddpg.py`. |
| TD3 | Author implementation by Fujimoto: `https://github.com/sfujim/TD3`. This repository includes `TD3.py`, `DDPG.py`, and `OurDDPG.py`, matching the TD3 paper's DDPG-style comparisons. | Cross-check twin critics, delayed policy updates, clipped target noise, and the legacy files under `legacy/cp_decentralized/cp_RL/TD3/`. |
| SAC | Original author repository: `https://github.com/haarnoja/sac`. The authors point users to the newer Softlearning package at `https://github.com/rail-berkeley/softlearning`. | Cross-check tanh-squashed Gaussian policy, twin Q networks, entropy term, target Q update, and optional automatic entropy tuning in `src/baselines/sac.py`. |
| PPO | OpenAI Baselines PPO2: `https://github.com/openai/baselines/tree/master/baselines/ppo2`, with OpenAI's PPO release page at `https://openai.com/index/openai-baselines-ppo/`. | Cross-check clipped surrogate objective, value loss, entropy bonus, GAE, rollout batching, and update epochs in `src/baselines/ppo.py`. |

For the manuscript, a careful wording is:

> We implemented DDPG, TD3, SAC, and PPO in a unified PyTorch evaluation
> framework and cross-checked the update rules against canonical author or
> OpenAI implementations. This avoids codebase-level confounding while keeping
> all algorithms on the same PRM manufacturing environment, action semantics,
> training budgets, and Monte Carlo evaluation protocol.

Do not claim that all baselines use original authors' code. TD3, SAC, and PPO
have clear canonical repositories, while DDPG is best supported by OpenAI
Baselines/Spinning Up and by the DDPG variants included in the TD3 author's
comparison code.

## Planned Baselines

No additional baselines are required for the current manuscript-facing pipeline.
SAC and PPO should still be treated as second-phase benchmarks until multi-seed
and full-horizon stability checks are complete.

## Multi-Seed Runs

```bash
python -m evaluation.run_multi_seed --algorithm flat_ddpg --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm gcn_ddpg --config configs/gcn_ddpg_20_clinic.yaml --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm td3 --config configs/td3_20_clinic.yaml --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm sac --config configs/sac_20_clinic.yaml --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm ppo --config configs/ppo_20_clinic.yaml --seeds 0 1 2 3 4
```

For a tiny implementation smoke comparison only:

```bash
python -m evaluation.run_smoke_comparison --episodes 1 --steps 4 --batch-size 2
```

For a small multi-seed pilot across learned agents and heuristics:

```bash
python -m evaluation.run_small_pilot --seeds 0 1 --episodes 1 --steps 4 --batch-size 2
```

For the manuscript-facing benchmark matrix, inspect the plan first:

```bash
python -m evaluation.run_full_benchmark --phase dry-run --budget smoke
```

Run a smoke pass of the full train/evaluate/aggregate/plot pipeline:

```bash
python -m evaluation.run_full_benchmark --phase all --budget smoke --scenarios disruption_0_3
```

Check the resulting training logs before scaling up:

```bash
python -m evaluation.check_training_stability \
  --inputs results/full_benchmark/smoke/training/disruption_0_3/*.csv \
  --output results/full_benchmark/smoke/training_stability.csv
```

The benchmark manifest at `experiments/configs/full_benchmark.json` contains:

- disruption scenarios: `0.05`, `0.3`, and `0.6`
- learned baselines: GCN-DDPG, flat DDPG, TD3, SAC, PPO
- heuristic baselines: MYO, ISO, MDL-1, MDL-2
- budgets: `smoke`, `pilot`, and `full`
- full evaluation: five seeds and 500 Monte Carlo replications per algorithm,
  scenario, and seed

Use `--budget pilot` to confirm stability before launching `--budget full`.
The benchmark runner skips completed training and evaluation jobs by default,
which makes long full-horizon runs resumable after interruption. Add `--force`
only when outputs should be overwritten.
Use `--max-jobs` for controlled local execution, for example:

```bash
python -m evaluation.run_full_benchmark --phase train --budget full --max-jobs 1
```

The patient-forecast benchmark also includes the optional
`graph_dynamic_patient_forecast_geo` scenario. It uses the 20 treatment-center
locations from the B&B Paper 2026 Appendix A to construct geographic
k-nearest-neighbor information, specimen-sharing, and reagent-sharing edges.
See `docs/geographic_network.md` before using the geo scenario for manuscript
claims.

For formal Monte Carlo evaluation of a heuristic:

```bash
python -m evaluation.evaluate_formal \
  --algorithm myo \
  --env-config experiments/configs/20_clinic_disruption_0_3.json \
  --replications 500
```

Aggregate and plot raw evaluation CSVs:

```bash
python -m evaluation.aggregate_results --inputs results/formal_myo.csv --output results/aggregate_summary.csv
python -m evaluation.plot_results --summary results/aggregate_summary.csv --metric total_cost_mean --output figures/total_cost_summary.png
```

## Evaluation

After training a checkpoint:

```bash
python -m evaluation.evaluate_policy \
  --algorithm flat_ddpg \
  --config configs/flat_ddpg.yaml \
  --checkpoint checkpoints/flat_ddpg/flat_ddpg_seed0_episode3.pt \
  --output results/flat_ddpg_eval.csv
```

## Results

Training and evaluation write CSV files under `results/`, and checkpoints under
`checkpoints/`. Both folders are ignored by Git.

The two-clinic configs are development smoke tests only. Manuscript-facing
experiments should use the 20-clinic configs and 20-clinic environment smoke
test.

Currently available metrics include:

- algorithm
- seed
- scenario
- graph ablation setting
- episode
- total reward
- total cost
- service level
- average waiting time
- reagent shortage frequency
- bioreactor shortage frequency
- bioreactor utilization
- transshipment count
- transshipment cost
- runtime for training rows

## Interpretation

Current quantitative results should be considered preliminary until multi-seed
experiments, Monte Carlo replications, and graph ablations are completed. Do not
compare algorithms based on single-seed training curves only.
