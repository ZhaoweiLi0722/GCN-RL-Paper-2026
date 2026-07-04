# Graph-Aware Deep Reinforcement Learning for Adaptive Capacity Planning

Research codebase for graph-aware capacity planning in distributed personalized regenerative medicine (PRM) manufacturing networks.

The project is intended to support GCN-DDPG policies, flat-state DDPG / MLP-DDPG baselines, graph ablations, multiple random seeds, Monte Carlo replications, and organized experiment configuration and result logging.

## Current Status

This repository currently contains the manuscript package under `Paper/`. The manuscript source includes LaTeX files, BibTeX references, Elsevier style files, a compiled PDF, and manuscript figures. I did not find Python source code, notebooks, experiment configs, datasets, checkpoints, or raw experiment output folders during initial repository setup.

Any quantitative claims in the current manuscript should be treated as preliminary unless they can be traced to final, reproducible experiment outputs.

Legacy code from a previous capacity-planning project has been imported under `legacy/cp_decentralized/` for reference. The current implementation should migrate reusable pieces into `src/` rather than depending directly on legacy scripts.

## Existing Files

- `Paper/`: manuscript package.
- `Paper/.../main.tex`: primary LaTeX manuscript source.
- `Paper/.../title_page.tex`: LaTeX title page.
- `Paper/.../references.bib`: bibliography.
- `Paper/.../figures/`: manuscript figures.
- `Paper/.../main.pdf`: compiled manuscript PDF.
- `Paper/.../*.bst`, `Paper/.../elsarticle.cls`: journal style files.
- `Paper/.../*.zip`: generated manuscript archive, ignored by Git.
- `legacy/cp_decentralized/`: previous decentralized capacity-planning code imported for reference.
- `docs/previous_work/`: small PDF/Word reference artifacts from the previous project.

## Expected Project Structure

The repository skeleton reserves space for modular research code:

```text
src/
  env/          # PRM manufacturing simulation environments and MDP dynamics
  graph/        # graph construction, edge ablations, graph utilities
  models/       # GCN-DDPG actor/critic and shared model components
  baselines/    # flat-state DDPG / MLP-DDPG and non-graph baselines
  utils/        # logging, seeding, metrics, shared helpers
experiments/
  configs/      # experiment, seed, and Monte Carlo configuration files
  scripts/      # train/evaluate/sweep entry points
tests/          # unit and smoke tests
figures/        # generated publication figures that are not already in Paper/
docs/           # notes, design docs, and reproducibility documentation
```

Existing manuscript files were left in place to avoid breaking paths or LaTeX figure references.

## Installation

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Python runtime dependencies were detected during setup because no Python source files are present yet. Add dependencies to `requirements.txt` as implementation files are introduced.

## Smoke Test

For the current repository state, run:

```bash
python -m compileall .
```

On systems where `python` is not available, use `python3 -m compileall .`.

In restricted local sandboxes that cannot write to the default Python cache
directory, use:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
```

Run the current environment smoke test:

```bash
python3 experiments/scripts/run_env_smoke.py
```

Run the manuscript-aligned 20-clinic environment smoke test:

```bash
python3 experiments/scripts/run_env_smoke.py --config experiments/configs/20_clinic_capacity_planning.json
```

Run the smoke test with selected graph ablations:

```bash
python3 experiments/scripts/run_env_smoke.py --remove-capacity-edges --remove-resource-edges
```

When training or evaluation scripts are added, include a small smoke test that runs quickly on CPU with a tiny horizon, few facilities, and one seed.

## Planned Experiments

- GCN-DDPG for continuous capacity, inventory, and transshipment decisions.
- Flat-state DDPG / MLP-DDPG baseline to isolate the value of graph representation learning.
- Graph ablations that remove capacity-sharing edges.
- Graph ablations that remove resource-sharing edges.
- Multiple random seeds and Monte Carlo replications for uncertainty-aware evaluation.
- Structured result logging with run metadata, configuration snapshots, and aggregate metrics.

## RL Baselines

The repository includes modular PyTorch implementations for:

- flat-state DDPG / MLP-DDPG
- GCN-DDPG
- TD3
- SAC
- PPO

It also includes deterministic heuristic baselines:

- MYO
- ISO
- MDL-1
- MDL-2

SAC and PPO are implemented as flat-state second-phase baselines. Treat their
results as exploratory until smoke tests, multi-seed pilots, and full-horizon
evaluations are completed.

Run smoke-scale training after installing dependencies:

```bash
python -m training.train_flat_ddpg --config configs/flat_ddpg.yaml
python -m training.train_gcn_ddpg --config configs/gcn_ddpg_20_clinic.yaml
python -m training.train_td3 --config configs/td3.yaml
python -m training.train_td3 --config configs/td3_20_clinic.yaml
python -m training.train_sac --config configs/sac_20_clinic.yaml
python -m training.train_ppo --config configs/ppo_20_clinic.yaml
```

The `*_20_clinic.yaml` configs are aligned with the manuscript setting
(`N=20`, production lead time `T=3`, and a 52-epoch weekly horizon). The
two-clinic configs are retained only for fast development smoke tests. You can
also use `python -m training.train_off_policy --config <config>` for any
implemented off-policy algorithm.

Run multi-seed baseline experiments:

```bash
python -m evaluation.run_multi_seed --algorithm flat_ddpg --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm gcn_ddpg --config configs/gcn_ddpg_20_clinic.yaml --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm td3 --config configs/td3_20_clinic.yaml --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm sac --config configs/sac_20_clinic.yaml --seeds 0 1 2 3 4
python -m evaluation.run_multi_seed --algorithm ppo --config configs/ppo_20_clinic.yaml --seeds 0 1 2 3 4
```

Run a tiny pipeline smoke comparison between 20-clinic flat DDPG and GCN-DDPG:

```bash
python -m evaluation.run_smoke_comparison --episodes 1 --steps 4 --batch-size 2
```

Run a small pilot across learned agents and heuristic baselines:

```bash
python -m evaluation.run_small_pilot --seeds 0 1 --episodes 1 --steps 4 --batch-size 2
```

Run the manuscript-facing benchmark plan as a dry run before launching jobs:

```bash
python -m evaluation.run_full_benchmark --phase dry-run --budget smoke
```

Then validate the full pipeline on a single smoke budget:

```bash
python -m evaluation.run_full_benchmark --phase all --budget smoke --scenarios disruption_0_3
python -m evaluation.check_training_stability \
  --inputs results/full_benchmark/smoke/training/disruption_0_3/*.csv \
  --output results/full_benchmark/smoke/training_stability.csv
```

The formal benchmark manifest is
`experiments/configs/full_benchmark.json`. It defines the three disruption
scenarios (`0.05`, `0.3`, `0.6`), full 20-clinic horizon, five random seeds,
and 500 Monte Carlo replications per evaluation job. Use `--budget pilot` for
stability checks before launching `--budget full`.

Run formal Monte Carlo evaluation for a heuristic scenario:

```bash
python -m evaluation.evaluate_formal \
  --algorithm myo \
  --env-config experiments/configs/20_clinic_disruption_0_3.json \
  --replications 500
```

Aggregate and plot evaluation outputs:

```bash
python -m evaluation.aggregate_results --inputs results/formal_myo.csv --output results/aggregate_summary.csv
python -m evaluation.plot_results --summary results/aggregate_summary.csv --metric total_cost_mean --output figures/total_cost_summary.png
```

See [docs/rl_baselines.md](docs/rl_baselines.md) for details. Results remain
preliminary until multi-seed experiments and graph ablations are completed.

## Reproducibility Notes

- Keep experiment parameters in config files rather than hard-coded in model definitions.
- Store raw large outputs in ignored folders such as `results/`, `runs/`, `wandb/`, or `checkpoints/`.
- Commit small, stable example configs and tests.
- Do not fabricate or hand-enter experimental results; result tables and figures should be generated from logged experiment outputs.
