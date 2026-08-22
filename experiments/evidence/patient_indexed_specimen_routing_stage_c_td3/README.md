# Stage C TD3 Screen — Final Results (Recovery Run Complete)

**Status: COMPLETE. All phases done, all audits passed, 245-file artifact inventory written. Headline: both learned policies beat the MDL-2 heuristic on cost with CIs entirely below zero — but the frozen-pretrain checkpoint matches the final checkpoint, so the gain comes from offline pretraining (BC + advantage distillation), not from the 100 online TD3 episodes.**

TL;DR（中文）：恢复运行全部完成（audit → final eval → pretrain eval → inventory，训练没有重跑）。结果：GCN-TD3 相对 MDL-2 heuristic 成本 **−0.735%**（CI 全部在零以下，528/600 配对获胜），Flat-TD3 −0.322%，临床指标全部 noninferior（GCN 甚至更好）。但 frozen-pretrain checkpoint 的成绩和 final checkpoint 几乎一样（GCN −0.7356% vs −0.7348%），说明提升来自 offline pretraining（BC + advantage distillation），100 个 online TD3 episodes 基本没有增益。这是 development evaluation（dev CRN streams），不是 formal holdout 证据。executor 的 csv bug 我在 `stage-c-td3-audit-recovery` 分支上打了最小 patch（附 diff），原始 crash 证据原封未动。

## Headline numbers (final checkpoint vs MDL-2 anchor, dev holdout, 600 paired reps)

| Arm | Cost Δ | 95% CI | Paired wins | Per-seed |
|---|---|---|---|---|
| **GCN-TD3** | **−0.735%** | [−0.861%, −0.625%] | 528/600 | −0.867% / −0.691% / −0.647% |
| Flat-TD3 | −0.322% | [−0.372%, −0.271%] | 445/600 | −0.354% / −0.301% / −0.311% |

Clinical guardrails (GCN): completion service level +0.0019 (CI [+0.0013, +0.0026], above zero); patients lost −26.9 (CI [−32.3, −22.3]); manufacturing-ineligibility rate lower. All noninferiority bounds pass for both arms.

## Frozen-pretrain comparison (the mechanism)

| Arm | Final checkpoint | Frozen-pretrain checkpoint |
|---|---|---|
| GCN | −0.7348% (528/600) | **−0.7356% (530/600)** |
| Flat | −0.3220% (445/600) | **−0.3282% (447/600)** |

The pretrain-only policy is statistically indistinguishable from (numerically a hair better than) the fully trained one. Interpretation: the beat-the-heuristic result is a **distillation/pretraining win**; online TD3 at this budget (100 episodes) neither helped nor collapsed. The graph architecture advantage is large and consistent (GCN ≈ 2.3× flat improvement in both variants).

## Caveats

- This is the **development evaluation** (dev CRN seeds 94000000/94100000; the eval config states "not formal holdout evidence"). A formal holdout pass (Stage D) is required before the paper can claim this.
- Anchor = pure MDL-2 teacher (deployment scale 0.0) under identical CRN streams; candidate = residual policy at scale 1.0 with checkpoint group thresholds.

## Execution provenance

- Training root: unchanged from the 2026-08-14 run (commit `8a7768e`, all six runs, audit passed: 100 episodes each, parameter gap 0.0038%, 20 checkpoints/run).
- Recovery run: branch `stage-c-td3-audit-recovery`, commit `8373f1b`, clean tracked worktree, all 15 locked hashes re-verified (runner hash updated to the patched file), MPS re-probed, 54 focused tests re-passed. Phases: focused_tests → training audit → evaluation_final → evaluation_pretrain → artifact inventory, all exit 0. Full trail in `launcher/recovery/` (claim.json, status.json, phase logs); the original crash evidence in `launcher/` is byte-identical to the failure report.
- The patch (attached diff): (1) `read_csv_rows` now raises `csv.field_size_limit(sys.maxsize)`, mirroring `audit_patient_indexed_specimen_routing_ddpg.py`; (2) a `--recover` flag that requires the existing training root, skips the training phase, and writes to `launcher/recovery/`. No scientific setting touched.

## For Zhaowei to decide

1. Review/merge the `stage-c-td3-audit-recovery` patch (or re-issue your own locked spec absorbing it).
2. Whether the pretrain≈final result changes the Stage D design — the interesting formal-holdout claim may be about the distilled/pretrained policy, with online TD3 as an ablation.
3. Formal holdout (Stage D) authorization to convert this from a development result into paper evidence.

## What this evidence directory contains

Committed here (curated from the gitignored `results/` root on Howard's Mac):
- `launcher/` — both complete audit trails: the original 2026-08-14 run (claim.json, crash status.json + traceback, all phase logs including training stdout/stderr) and `launcher/recovery/` (recovery claim/status/logs + `artifact_sha256.json`, the 245-file inventory of the full results root)
- `training/.../seed*/` — per-run `summary.json`, `config.json`, and the full episode-level `training.csv.gz` (100 rows each), plus `training_manifest.json`
- `evaluation/{final,pretrain}/` — top-level and per-run `summary.json`

Not committed (large; SHA256-inventoried in `artifact_sha256.json`, reproducible via locked CRN streams, available on request): the 132 checkpoint files (~1 GB) and the row-level evaluation CSVs (24 × ~30 MB).
