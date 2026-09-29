# Replacement GCN baseline archive, September 29, 2026

For Zhaowei and Howard: this package restores a usable **new** pretrained
baseline for further diagnostics after the historical F1 weights could not be
located. It does not replace old evidence or demonstrate an online RL gain.

Start with `through_seed62.tar.gz` and its `.manifest.json` sidecar. That final
archive includes all three GCN seeds (60, 61, 62); the seed60/61 archives are
earlier recovery boundaries, not additional independent models.

Final archive SHA256:
`5959575f4b99ab98dcd12a9b893a942fc25f0872e783b496c628af4fbc7bf731`.
Compressed size: 39,963,591 bytes. Verify the checksum before use.

## Contents

- `source.tar.gz`: frozen tracked code, configs, tests, protocol and teacher.
- `execution.json`: exact source commit, config, runtime versions and MPS setting.
- `training/frozen_baseline_rebuild_20260929/training_manifest.json`.
- Under `training/.../gcn_residual_mdl2_network_ddpg_afd/seed60` (also 61/62):
  effective config, summary, actor/gate checkpoint, complete legacy training
  state and supplemental MPS RNG state.
- `logs/seed60.log` (also 61/62) and per-seed hash receipts.

Each model used the fixed F1-style recipe: 20 heuristic demonstration episodes,
300 distillation epochs and 500 offline updates; zero online episodes. Source
commit: `1845339fc0d0bf99eb91cb02f355e6f244d5e356`. All three actor digests differ
from historical F1; do not silently reuse these with old result labels.
The initial critic warmup explains the logger's `actor_loss=nan` placeholders;
saved metrics/tensors are finite. See the readout and preflight record.

The nested source archive and effective configs record provenance, but legacy
paths are relative to the original project root. Do not blindly invoke an old
training launcher after extracting. Restore into a new directory, map paths
explicitly and verify inputs. The MPS RNG sidecar is not automatically loaded
by the legacy restore function. Never overwrite historical campaign roots.

## Backup status

The three archives and sidecars were byte-verified in both the persistent local
project and the Dropbox sync folder. **Cloud upload and Howard's permissions
have not been verified.** No public link or sharing-permission change was made.
`verification.json` records the independent checks; `launch_status.json` records
the completed job. The engineering smoke is separate and not for evaluation.

Next: name these replacement actors in a prospective protocol, then implement
and authorize a bounded frozen-policy value-label diagnostic. No performance
evaluation or online-learning conclusion is part of this package.
