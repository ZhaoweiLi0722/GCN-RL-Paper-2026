# F1 artifact recovery checkpoint

Read-only inspection on 2026-09-29, starting from local integration HEAD
`17205200d55733ca4d9c6fb5f4faf080d3fb7d52`. No recovery, inference, simulator
step, fitting, remote access or scientific launch was performed.

## What is recoverable locally

The original temporary worktree no longer exists. Git still has the Stage F
branch at `9cc757831c27b79299d8a4cdcc43a835c671b98a` and the training execution
commit `48292683152e5a29107cd2d6ad9451612d8b22a3`. This preserves tracked source
and experiment configs, not necessarily run artifacts. Do not prune or recreate
the worktree as an attempted recovery of ignored files.

`git check-ignore -v` confirms the F1 checkpoint path is covered by the
`results/` rule. A reachable-all-refs Git log search for `.pt`, `.pth` and F1
manifest/inventory path names returned no entries. This does not inspect
unreachable objects, arbitrary renamed archives, cloud backups or other hosts;
it is not evidence of global data loss.

The archived Recovery 2 spec still hashes to
`eb87540dfb27a4a23f524d4f5f872e6e7fa561b001d4b9e7ef034a9b4bc001b2`.
The G1 summary still hashes to
`9e96deec850d6033fb5a2fb9e2f5090f8a040468df9faf8634015a14875d1c41`.
These recover the required identities below, not the missing tensor bytes.

## Expanded search and coverage limits

In addition to the locations in `readout.md`, this continuation checked:

| Scope | Observation |
| --- | --- |
| Primary `/Users/lizhaowei/GCN-RL Paper 2026` tree, including its worktrees | Targeted ignored-file-inclusive scan found no seed60-62 checkpoint or named F1 training manifest |
| `/Users/lizhaowei/GCN-RL-Paper-2026-worktrees` | Same filename scan found no match |
| Primary `output/frozen_experiment_archives` | One experiment archive, identified by its manifest as the August 11 formal confirmation campaign, not F1; not extracted or substituted |
| `training_data/regional_4090_training_data.zip` | ZIP member names inspected: 6 members, no F1 or seed60-62 name |
| `results/imported_4090_20260730/regional_cuda_results.zip` | ZIP member names inspected: 105 members, no F1 or seed60-62 name |
| Retained Stage F Git tree | Source/configs available; no matching tracked F1 binary/archive inventory found |

The primary filename scan reported permission-denied directories under the
July 30 imported CUDA extraction, stages 1-4, seeds 0-2. Those directories
remain uninspected; no permissions were changed. Reading the ZIP's directory
does not establish that it equals the extracted directories. The later shell
pipeline status must not be interpreted as a complete successful filesystem
scan. Cloud/shared folders, renamed packages and the RTX host remain unverified.

The August 29 collaborator drafts mention an RTX 4090 host. That is a search
lead, not a verified current location. Nothing was connected to or sent.

## Exact retrieval package

Preferred: a read-only copy of this **entire 151-file control-training tree**,
preserving relative paths and bytes:

```text
results/patient_indexed_specimen_routing_ddpg_online_paired_advantage_development/control/training/
```

Its locked tree digest is
`eb276c4402d766124080f1eddbfa1c365ff5a32d8baedb0204ff39e23c5041e8`.
The historical algorithm is SHA256 of the UTF-8 compact, key-sorted JSON
mapping `relative/path -> whole-file SHA256`, not a tar-file checksum. It is
defined in the Recovery 2 runner's `immutable_tree_snapshot` function. An
archive's own SHA256 is useful for transport but is not this tree digest.

If transferring the full tree is impractical, the minimum payload is ten files
under `patient_indexed_specimen_routing_ddpg_online_paired_advantage_control_100/`:

- `training_manifest.json`;
- for **each** seed 60, 61 and 62, under
  `gcn_residual_mdl2_network_ddpg_afd/seed<seed>/`: effective `config.json`,
  `summary.json`, and
  `checkpoints/gcn_residual_mdl2_network_ddpg_afd_seed<seed>_pretrain.pt`.

For partial transfer also supply the complete 151-entry relative-path/hash
mapping from the original tree. Reconcile that mapping with the locked digest
before using it to verify the ten files. Do not invent unavailable hashes.
The manifest alone is locked to
`0d0640f112ff55580d50f5435c8dc4ed4dd9e17a590d3b96c324fd6ccef452ed`.

Recovery 2's `launcher/artifact_sha256.json` inventories its evaluation-only
campaign root. It is **not** the control-training inventory and must not be
used as a substitute for training-file provenance.

The G1 actor-only tensor digests provide an additional check after trusted
file verification, but cannot certify gates/configs or whole checkpoints:

| Seed | Actor-only SHA256 |
| --- | --- |
| 60 | `91368bc77481ee048113589dbda7e4daefd645d1d7c68a25749004b50cf26a73` |
| 61 | `bce142e721446d9ae7a5d12dce38f8dca7097345a5d10fd5e2fe4cb18d52a3d6` |
| 62 | `49913cbd4bada31d01ef79a675343c0b9b145b45ee6664bfaa5df88ef5e518e2` |

## Source compatibility is not yet runtime parity

Compared byte-for-byte with training commit `4829268`:

| File | Current vs historical |
| --- | --- |
| `src/models/gcn_ddpg.py` | Identical; SHA256 `2eaea8a4a82559691b48c20bc62186ee926559f59459705fe71a80c85607b371` |
| `src/models/graph_features.py` | Identical; SHA256 `53f0c20fe2734ef8d313dbaf39189964c6a5582fddd0a579e860692d8affab9a` |
| F1 `online_paired_advantage_control_100.json` config | Identical; SHA256 `af46bf7e88f02b5fe444e0df09771030565d9b47cdfbeddd82bd59f9f806ade2` |
| `src/env/capacity_planning.py` | Different: historical `bfc561af473e94b412f103e5c9409ab1544a5587936db5a6af67757bca5d3886`, current `529215bc180463fb1965534f994539616198e95371776acd814fd8bf05a1e2f1` |

The environment has subsequently added optional procurement, overtime and
scheduled-referral functionality. A source difference does not prove behavior
changed with extensions off, nor do comments claiming flag-off equivalence
prove full parity. This four-file check is not a complete dependency audit.
Future acceptance must verify effective config, strict actor/gate loading,
feature/action processing and historical-versus-current behavior explicitly.
Do not silently overwrite current source with the old version.

## Stop and next action

No matching research Python process was found by the scoped PID/PPID/command
check. The sandbox initially denied `ps`; a separately approved read-only
check succeeded. No process was started, stopped or repaired.

Artifact location is still the immediate dependency; the prior question is
unanswered, so do not repeat it or pretend the pilot is running. Once a local
copy is available, verify the file/tree locks, then resolve runtime parity.
The collector implementation, fresh streams and capped scientific pilot
authorization remain separate. No reward tuning, replacement actor, recovery
training or new experiment is authorized by this document.
