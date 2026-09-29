# R4 result: three replacement baselines, verified local archives

Execution commit: `1845339fc0d0bf99eb91cb02f355e6f244d5e356`.
One production process (PID 35892), exit 0; completed in 398.235 seconds.
An after-completion PID/command check found no matching process. No retry,
online-attribution campaign, formal evaluation, reward tuning or remote Git
action occurred. The preflight field-name failure is retained in `preflight.md`.

## Actual outputs

GCN seeds 60, 61, 62 each completed 20 heuristic demonstration episodes /
1,040 simulator transitions, 20 imitation epochs, 300 cached-teacher
distillation epochs and 500 offline updates. Online episodes: **zero**.
Totals: 3,120 demonstration steps, 900 distillation epochs, 1,500 offline
updates. The separate smoke adds one engineering transition/epoch/update,
not a fourth baseline or scientific result.

All three policy checkpoints contain finite actor and gate tensors matching
their respective full saved states exactly. Each full state has optimizer,
replay, environment and the legacy RNG fields; the MPS RNG is retained in a
separate sidecar. All persisted numeric summaries and inspected numeric arrays
are finite. The 500-step critic warmup leaves offline actor-update indicators
zero; stdout `actor_loss=nan` is a missing-field placeholder, not a saved NaN.
Neither logs nor old model code were modified to hide that distinction.

Historical G1 actor-only digests match **0/3** of these rebuilt actors. They are
new baselines, not recovered F1 files or an exact historical replay. No policy
performance evaluation has been run, and this does not establish online gain.

## Archives and independent verification

The campaign payload contains 24 files at its final boundary: source/teacher
bundle, execution metadata, three logs, cumulative manifest, configs/summaries,
three policy checkpoints, three full states, three MPS RNG sidecars and three
per-seed receipts. The source bundle contains further nested tracked files.

| Boundary | Payload files | Compressed bytes | Archive SHA256 |
| --- | ---: | ---: | --- |
| Seed 60 | 10 | 14,535,466 | `0260439089afbfdf71d358ffdb26964bd9f6871b561d28f3ec9ebce2c517ed64` |
| Seed 61 | 17 | 27,192,983 | `509c16b9a39c6ff158a54af5f916612b52c0f4f5ebbe0b1076e6e38b4c9ce18a` |
| Seed 62 (all three) | 24 | 39,963,591 | `5959575f4b99ab98dcd12a9b893a942fc25f0872e783b496c628af4fbc7bf731` |

A separate read-only pass streamed every member in all three tar files,
recomputed hashes against inventories, checked all Dropbox destination bytes,
read all summaries/states and compared actor/gate tensors. Results:
`reports/2026-09-29-frozen-baseline-rebuild/verification.json`.
No archive extraction or new environment query was needed for this pass.

Local originals: `results/frozen_baseline_rebuild_20260929/`.
Dropbox destination is recorded in `protocol.md` and the execution config.
The final `through_seed62.tar.gz` is cumulative and contains all three seeds;
intermediate archives are retained as recovery boundaries.

**Dropbox local copies are verified; cloud synchronization and Howard's access
are not verified.** Reading the Dropbox client UI timed out. Do not treat two
directories on this Mac as confirmed independent off-device backups. No sharing
permissions or public links were changed. The searchable small verification
report, readout and manifests belong in Git; binary payloads remain archived.

## Next scientific boundary

The missing-file dependency now has an explicitly authorized replacement
option. R3 still names original F1 actors, so it must be amended to name these
new baselines before use. Then check strict policy/gate compatibility, implement
the fixed-window collector and lock fresh streams. Its capped value-label
pilot requires separate launch authorization. R4 does not justify a reward
change, reopening old failed gates or starting online training automatically.
