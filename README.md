# Graph-Aware Deep Reinforcement Learning for Adaptive Capacity Planning

Research codebase for graph-aware capacity planning in distributed personalized regenerative medicine (PRM) manufacturing networks.

The project is intended to support GCN-DDPG policies, flat-state DDPG / MLP-DDPG baselines, graph ablations, multiple random seeds, Monte Carlo replications, and organized experiment configuration and result logging.

## Current Status

This repository currently contains the manuscript package under `Paper/`. The manuscript source includes LaTeX files, BibTeX references, Elsevier style files, a compiled PDF, and manuscript figures. I did not find Python source code, notebooks, experiment configs, datasets, checkpoints, or raw experiment output folders during initial repository setup.

Any quantitative claims in the current manuscript should be treated as preliminary unless they can be traced to final, reproducible experiment outputs.

## Existing Files

- `Paper/`: manuscript package.
- `Paper/.../main.tex`: primary LaTeX manuscript source.
- `Paper/.../title_page.tex`: LaTeX title page.
- `Paper/.../references.bib`: bibliography.
- `Paper/.../figures/`: manuscript figures.
- `Paper/.../main.pdf`: compiled manuscript PDF.
- `Paper/.../*.bst`, `Paper/.../elsarticle.cls`: journal style files.
- `Paper/.../*.zip`: generated manuscript archive, ignored by Git.

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

When training or evaluation scripts are added, include a small smoke test that runs quickly on CPU with a tiny horizon, few facilities, and one seed.

## Planned Experiments

- GCN-DDPG for continuous capacity, inventory, and transshipment decisions.
- Flat-state DDPG / MLP-DDPG baseline to isolate the value of graph representation learning.
- Graph ablations that remove capacity-sharing edges.
- Graph ablations that remove resource-sharing edges.
- Multiple random seeds and Monte Carlo replications for uncertainty-aware evaluation.
- Structured result logging with run metadata, configuration snapshots, and aggregate metrics.

## Reproducibility Notes

- Keep experiment parameters in config files rather than hard-coded in model definitions.
- Store raw large outputs in ignored folders such as `results/`, `runs/`, `wandb/`, or `checkpoints/`.
- Commit small, stable example configs and tests.
- Do not fabricate or hand-enter experimental results; result tables and figures should be generated from logged experiment outputs.
