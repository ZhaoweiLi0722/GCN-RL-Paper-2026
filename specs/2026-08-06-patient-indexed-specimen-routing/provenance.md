# Provenance and Artifact Isolation

## Git Identity

- Parent branch: `codex/rtx4090-matched-ablation`
- Exact parent commit: `ce9b6274419c8e0e7adf800f434e47d96c18c1dc`
- Routing branch: `codex/patient-indexed-specimen-routing`

The locked PC runner requires the exact reviewed routing commit as an argument,
verifies the branch and clean tracked worktree, and records source/config hashes.
It never uses `--force`, deletes outputs, kills processes, repairs runs, or
launches more than the explicitly selected stage.

## Recovery 1 Output Namespace

The first formal teacher attempt at commit
`fd63bd4bf1ff2b191e9cccfd4ff67bde6a25e439` stopped before training because
the headroom state-probe configuration omitted its required top-level
`epsilons` field. That failed attempt, its transcript, and the partial original
namespace are immutable and invalid for paper use. No Smoke, Pilot, or Evaluate
stage began.

- Failed transcript: `results/patient_indexed_specimen_routing/logs/Teachers_20260806_233110.txt`
- Transcript SHA-256: `9fc9fb61298804c327b34311f6c0e64f1df2651331d6ac6e035f120794f9c09f`

All generated artifacts live below:

```text
results/patient_indexed_specimen_routing_recovery1/
```

Subtrees are reserved for the mechanics gate, a fresh routing teacher,
routing-enabled training, final/frozen/sensitivity evaluation, analysis, logs,
and provenance. Optional no-routing config paths remain reserved but are not
entered by the locked default runner. Every stage refuses to overwrite its
expected outputs. The runner records the superseded commit and root and never
reads from or writes to `results/patient_indexed_specimen_routing/`.

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

The primary campaign regenerates one routing-enabled teacher and shares its
reviewed generation protocol across GCN and matched flat. It does not regenerate
a no-routing teacher. This prevents the routing-enabled policies from inheriting
a teacher that never observed the new state/action dynamics while avoiding an
unnecessary second training campaign.
