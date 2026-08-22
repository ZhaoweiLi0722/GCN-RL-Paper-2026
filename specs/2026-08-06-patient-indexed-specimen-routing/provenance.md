# Provenance and Artifact Isolation

## Git Identity

- Parent branch: `codex/rtx4090-matched-ablation`
- Exact parent commit: `ce9b6274419c8e0e7adf800f434e47d96c18c1dc`
- Routing branch: `codex/patient-indexed-specimen-routing`

The locked PC runner requires the exact reviewed routing commit as an argument,
verifies the branch and clean tracked worktree, and records source/config hashes.
It never uses `--force`, deletes outputs, kills processes, repairs runs, or
launches more than the explicitly selected stage.

## Failed Attempt Evidence

The first formal teacher attempt at commit
`fd63bd4bf1ff2b191e9cccfd4ff67bde6a25e439` stopped before training because
the headroom state-probe configuration omitted its required top-level
`epsilons` field. That failed attempt, its transcript, and the partial original
namespace are immutable and invalid for paper use. No Smoke, Pilot, or Evaluate
stage began.

- Failed transcript:
  `results/patient_indexed_specimen_routing/logs/Teachers_20260806_233110.txt`
- Transcript SHA-256:
  `9fc9fb61298804c327b34311f6c0e64f1df2651331d6ac6e035f120794f9c09f`

Recovery 1 at commit `ecab3650780aafb746027fb26f2f02512b7b0495`
corrected that config contract and passed Validate. Its unique Teachers process
ran from 2026-08-07 13:02:39 through 13:25:33 EDT, then exited with signed code
120 after the foreground launcher output channel had timed out. It produced no
teacher cache, and Smoke, Pilot, and Evaluate never started. The transcript has
no Traceback, NaN/Inf, OOM, CPU fallback, native crash, driver reset, or config
mismatch. This is classified as a `launcher/output-channel failure`; that
classification is strongly supported by the closed output channel and exit
code, but is not a scientific Python exception diagnosis.

- Failed Recovery 1 transcript:
  `results/patient_indexed_specimen_routing_recovery1/logs/Teachers_20260807_130239.txt`
- Transcript SHA-256:
  `3307e86aafc80813b020a3102603f02001e7af50ddbea0b21020e9d81f255ee8`

Both failed roots and all of their partial contents are immutable, are invalid
for paper use, and may not be read as inputs to Recovery 2.

## Recovery 2 Output Namespace

All generated artifacts live below:

```text
results/patient_indexed_specimen_routing_recovery2/
```

Subtrees are reserved for the mechanics gate, a fresh routing teacher,
routing-enabled training, final/frozen/sensitivity evaluation, analysis, logs,
and provenance. Optional no-routing config paths remain reserved but are not
entered by the locked default runner. Every stage refuses to overwrite its
expected outputs. The runner records the superseded commit and root and never
reads from or writes to either failed namespace. Recovery 2 provenance records
the immediate superseded commit/root, the locked Recovery 1 transcript hash,
`superseded_outputs_reused=false`, and the failure classification.

Recovery 2 Validate passed at commit
`304dc83d6eb7447c6371150a551ea6d01999d5aa`. Its detached Teachers phase then
stopped after 325 state-probe states with exit code 1 and
`AttributeError: 'str' object has no attribute 'patient_id'` inside a copied
look-ahead environment. No teacher cache was generated and no later phase
started. The locked Recovery 2 transcript SHA-256 is
`d9a3e7f598b2991a90662ae3f1326be7cade6889888a3f5ff6f9d17f7b9bbfed`.
The Recovery 2 root is immutable and invalid for paper use.

## Recovery 3 Output Namespace and Teacher Transfer

All new campaign artifacts use:

```text
results/patient_indexed_specimen_routing_recovery3/
```

The formal routing teacher is generated from fresh Mac CPU shards at the exact
Recovery 3 commit. The canonical cache, state-probe rows, anchor rows, teacher
rows, and summary are packaged into a ZIP containing a provenance manifest.
The bundle records the commit, config SHA-256, source SHA-256 values, Python,
NumPy, and platform; a separate SHA-256 sidecar protects the ZIP itself.

On the PC, import is allowed only into a nonexistent Recovery 3 routing-teacher
directory. Import verifies the local commit/config/source identity, every
member hash and size, complete canonical rows, finite arrays and metrics, and
the positive headroom gate before an atomic rename. The extracted
`bundle_manifest.json` is retained as the transfer receipt. Recovery 1 and 2
outputs are never read, copied, or reused.

## Forbidden Legacy Inputs

The following prior artifacts remain immutable no-routing evidence and are
forbidden as inputs to a formal routing-enabled policy:

- every teacher cache outside the new routing namespace;
- every old pretrain or final policy checkpoint;
- every old normalization artifact;
- every old training-state checkpoint;
- every old log, result table, ZIP, SHA-256 sidecar, and Recovery record.

In particular, all `multiscenario_*`, `regional_*`, prior regional Recovery
1/2/3, both failed routing namespaces, and prior attribution outputs remain
untouched. They may be cited only as historical evidence after an explicit
comparability review; they are not reinterpreted as Recovery 3 routing results
and are never copied into the new campaign.

The primary campaign regenerates one routing-enabled teacher and shares its
reviewed generation protocol across GCN and matched flat. It does not regenerate
a no-routing teacher. This prevents the routing-enabled policies from inheriting
a teacher that never observed the new state/action dynamics while avoiding an
unnecessary second training campaign.
