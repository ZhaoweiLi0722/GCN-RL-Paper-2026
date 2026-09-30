# AGENTS.md

This is a research codebase for graph-aware deep reinforcement learning in distributed personalized regenerative medicine manufacturing networks. The intended implementation includes GCN-DDPG, flat-state DDPG / MLP-DDPG baselines, graph ablations, multiple random seeds, and Monte Carlo evaluation.

## Research Objective

Zhaowei clarified on 2026-09-30 that the objective is a rigorous, publishable
GCN-plus-reinforcement-learning study, targeting EAAI-level work, not preserving
DDPG or guaranteeing a positive result. DDPG remains a candidate to assess fairly;
other algorithms may be considered when they fit the actual decisions better.
Separate RL policy training in simulation from parameter adaptation during
deployment, and separate both from imitation/distillation. Do not rename an
imitation gain as an RL gain. Favor decisive, bounded comparisons over an
open-ended algorithm search. The locked-plan change control and local-only
execution boundaries still apply; this objective is not a new experiment permit.

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

## Research Artifact Preservation

- Run research jobs in a persistent project worktree, never a temporary
  `/tmp` checkout. Use a new run directory; preserve failed and old artifacts.
- Freeze source/config before execution. Save the execution commit, runtime
  versions, teacher/input hashes, effective configs, raw logs, summaries,
  checkpoints and full resume state. Record any RNG state absent from the
  existing checkpoint format separately; do not promise bitwise GPU recovery.
- Archive completed run boundaries with per-file and archive SHA256 checks.
  Use `src/utils/research_archive.py` or the campaign's established equivalent;
  verify all archive members and destination bytes, never overwrite a different
  backup. Keep originals after copying.
- Zhaowei requested Dropbox preservation on 2026-09-29. The current project
  destination is documented in the R4 protocol/config. Use versioned run
  subdirectories there for authorized research artifacts. Do not change sharing
  permissions, create public links, or send files to collaborators implicitly.
- Distinguish local archive verified, Dropbox local copy verified, cloud sync
  verified, and collaborator access verified. A sync-folder copy alone is not
  an independent off-device backup. Do not claim old campaigns were archived
  unless their payloads and checksums have actually been verified.

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
