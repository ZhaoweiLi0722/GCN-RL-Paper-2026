# Provenance and Artifact Isolation

## Git Identity

- Parent branch: `codex/rtx4090-matched-ablation`
- Exact parent commit: `ce9b6274419c8e0e7adf800f434e47d96c18c1dc`
- Routing branch: `codex/patient-indexed-specimen-routing`

The locked PC runner requires the exact reviewed routing commit as an argument,
verifies the branch and clean tracked worktree, and records source/config hashes.
It never uses `--force`, deletes outputs, kills processes, repairs runs, or
launches more than the explicitly selected stage.

## New Output Namespace

All generated artifacts live below:

```text
results/patient_indexed_specimen_routing/
```

Subtrees are reserved for mechanics gate, fresh teachers, routing/no-routing
training, final/frozen/sensitivity evaluation, analysis, logs, and provenance.
Every stage refuses to overwrite its expected outputs.

## Forbidden Legacy Inputs

The following prior artifacts remain immutable no-routing evidence and are
forbidden as inputs to a formal routing-enabled policy:

- every teacher cache outside the new routing namespace;
- every old pretrain or final policy checkpoint;
- every old normalization artifact;
- every old training-state checkpoint;
- every old log, result table, ZIP, SHA-256 sidecar, and Recovery record.

In particular, all `multiscenario_*`, `regional_*`, Recovery 1/2/3, and prior
attribution outputs remain untouched. They may be cited only as historical
no-routing baselines after an explicit comparability review; they are not
reinterpreted as routing results and are never copied into the new campaign.

The routing and matched no-routing control teachers are both regenerated. This
keeps their data-generation protocol matched and prevents the routing treatment
from inheriting a policy that never observed the new state/action dynamics.
