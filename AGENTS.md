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

## Locked Routing-Primary Experiment Plan

Before changing, launching, extending, or interpreting any patient-indexed
specimen-routing experiment, read
`docs/patient_indexed_specimen_routing_locked_execution_plan.md`. That file is
the authoritative cross-session execution plan. Context compaction, a new
Codex task, convenience, an isolated seed result, or available compute is not
a reason to change the main sequence.

Any scientific amendment requires a completed-stage evidence review, explicit
user approval, an appended change-control entry, and a committed plan update
before the amended experiment is launched. Existing result roots, checkpoints,
teacher artifacts, CRN streams, and provenance remain immutable.

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
