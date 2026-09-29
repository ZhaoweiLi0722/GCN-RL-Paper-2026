# R4: authorized replacement pretraining and durable archives

## Authority and purpose

On September 29 Zhaowei explicitly authorized retraining if the old artifacts
could not be found, and requested Dropbox storage for future collaboration.
The scoped local/Dropbox search found no F1 seed60-62 artifacts. This authorizes
one bounded replacement-baseline rebuild, NOT another online-attribution
campaign. Howard approval is not asserted. Stage E stays closed.

Use a fresh persistent output root and label all outputs **replacement**, not
historically recovered F1 weights or an exact replication of G1. The current
source differs from August's environment; archive its exact committed version.
Old summaries, failed gates, configs, checkpoints and teacher remain immutable.
There is no reward correction or performance claim in this preparation stage.

## Fixed budget and procedure

- Original F1 control recipe and locked teacher; GCN seeds 60, 61, 62 only.
- Per seed: 20 MDL-2 initialization demonstration episodes, each at most 52
  steps; 20 imitation epochs; 300 cached-teacher distillation epochs; 500
  offline replay updates. Maximum total demonstration transitions: 3,120.
- Original hotspot seed assignment is preserved for initialization. This is
  not the prospective R3 nominal-history label-collection experiment.
- Set online episodes to zero and initial critic-calibration updates to zero,
  as in F1 pre-online state preparation. Save actor/gate and full episode-0
  state with optimizer/replay/environment/RNG. Do not import the old critic
  into a later corrected-value experiment.
- Preserve original architecture, costs, teacher labels, historical
  initialization seeds and hyperparameters. No best-seed selection, HPO,
  counterfactual label generation, performance evaluation or formal stream.
- MPS mandatory, `PYTORCH_ENABLE_MPS_FALLBACK=0`. Serial, one process,
  maximum one hour, one attempt per seed. Reject an existing output root.
  Stop on failure; preserve logs/partial artifacts, do not silently retry.
- A bounded engineering smoke may use one imitation step/epoch, one cached
  distillation epoch and one offline update, saved separately and explicitly
  excluded from the three production baselines. It is not a performance run.

Zero **online episodes** does not mean zero simulator steps or zero learning:
heuristic demonstrations and offline actor/critic updates are part of the
original pretraining recipe. This reconstructs a baseline, not online gain.

## Storage contract

Primary workspace: the existing persistent September integration worktree,
never a `/tmp` training directory. Campaign:
`results/frozen_baseline_rebuild_20260929`.

Each completed seed triggers a cumulative immutable `.tar.gz` snapshot and
sidecar JSON containing every relative path and SHA256, archive SHA256 and
size. Read every archived member back and verify against unchanged originals.
Keep source tarball, locked teacher, effective configs, summaries, per-seed
logs, pretrain checkpoint, full state, runtime versions and execution commit.
The legacy full-state format omits MPS RNG, so retain a separate MPS RNG
sidecar as well; legacy resume does not automatically consume this sidecar.
Neither archival completeness nor fixed seeds guarantees bitwise GPU replay.
Do not overwrite old archives or silently repair an inconsistent backup.

Copy the verified archive and sidecar to the existing project Dropbox folder:
`/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/GCN-DRL Paper 2026/Research Artifacts/frozen_baseline_rebuild_20260929`.
Verify the destination bytes before the next seed starts. Do not alter sharing
permissions, create a public link or send a message to Howard. Local Dropbox
copy verification is not proof of cloud sync, retention, or Howard access.
Record those three states separately. Local and synced-folder copies share
the same Mac until independent cloud confirmation is available.

Future research jobs should adopt the same snapshot-on-completion rule and
include full state at interruption-safe boundaries. This packet does not
retroactively back up all previous campaigns or claim that they are protected.

## Completion and next decision

Require three actor/gate checkpoints, three full states, configs/summaries,
finite saved tensors/metrics, zero online episodes, locked teacher, bounded
demonstration count, verified archives and local Dropbox copies. Record actual
counts and hashes, including any failure. Preserve the historical actor hashes
only as diagnostics; equality is neither promised nor required for replacements.

After completion, R3 still needs an explicit protocol amendment naming these
replacement baselines, compatibility tests, implementation and fresh CRNs
before a separately authorized value-label pilot. Do not start that pilot or
online DDPG automatically after pretraining.
