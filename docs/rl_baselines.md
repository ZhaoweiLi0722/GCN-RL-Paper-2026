# RL Baselines

This project uses learning-based baselines to separate the value of graph-aware
state encoding from the value of actor-critic reinforcement learning itself.

## Implemented Baselines

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

### TD3

TD3 is the closest modern DDPG-style baseline. It uses two critics, clipped
target policy smoothing, delayed actor updates, replay buffer learning, and soft
target updates.

Run:

```bash
python -m training.train_td3 --config configs/td3.yaml
```

## Planned Baselines

PPO and SAC have config and training placeholders, but they are not implemented
yet. Do not report PPO or SAC results until real implementations and smoke tests
are added.

## Multi-Seed Runs

```bash
python -m evaluation.run_multi_seed --algorithm flat_ddpg --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm td3 --seeds 0 1 2 3 4
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

Currently available metrics include:

- algorithm
- seed
- scenario
- graph ablation setting
- episode
- total reward
- total cost
- reagent shortage frequency
- bioreactor shortage frequency
- transshipment count
- transshipment cost
- runtime for training rows

Service level, average waiting time, and utilization are not yet exposed by the
environment. Do not invent those metrics; add them to the environment before
using them in tables.

## Interpretation

Current quantitative results should be considered preliminary until multi-seed
experiments, Monte Carlo replications, and graph ablations are completed. Do not
compare algorithms based on single-seed training curves only.
