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
