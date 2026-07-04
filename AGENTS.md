# AGENTS.md

This is a research codebase for graph-aware deep reinforcement learning in distributed personalized regenerative medicine manufacturing networks. The intended implementation includes GCN-DDPG, flat-state DDPG / MLP-DDPG baselines, graph ablations, multiple random seeds, and Monte Carlo evaluation.

## Coding Guidelines

- Keep environment dynamics, graph construction, model architectures, training loops, and evaluation scripts modular.
- Put PRM manufacturing simulation logic under `src/env/`.
- Put graph construction and edge-ablation logic under `src/graph/`.
- Put GCN-DDPG actor/critic models and shared neural network components under `src/models/`.
- Put flat-state DDPG / MLP-DDPG and other non-graph baselines under `src/baselines/`.
- Put reusable logging, seeding, metrics, and config helpers under `src/utils/`.
- Avoid hard-coding experiment parameters inside model files.
- Use config files under `experiments/configs/` whenever possible.
- Keep scripts under `experiments/scripts/` thin: load config, set seeds, call library code, and write logs.
- Do not fabricate experimental results. Tables, plots, and manuscript claims should be traceable to logged outputs.

## Validation

Before finishing any coding task, run at least:

```bash
python -m compileall .
```

If the local sandbox cannot write to the default Python cache directory, use:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
```

If tests exist, run the relevant tests. If training or evaluation scripts are modified, run a small smoke test when feasible, using a tiny horizon, few facilities, and a minimal seed count.
