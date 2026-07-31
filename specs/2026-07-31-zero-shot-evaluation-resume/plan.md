# Resume the TD3 zero-shot evaluation

## Goal

Continue the pre-registered final-versus-frozen TD3 zero-shot evaluation without
retraining and without spending more time on root-cause investigation.

This task is evaluation only. It must use the existing episode-100 and
frozen-pretrain checkpoints produced by the completed RTX 4090 screen.

## Required source state

Work on `codex/rtx4090-matched-ablation` at the latest remote commit containing:

- `f9ba2f2` -- narrow recovery for the info-only patient risk count;
- `0183b94` -- persistence of risk-count recovery diagnostics in formal rows.

Fast-forward only. Do not merge, rebase, reset, stash, or overwrite local
results.

## Safety constraints

- Do not retrain any policy.
- Do not perform online RL updates.
- Do not use `--force`.
- Do not change checkpoints, scenarios, seeds, horizon, deployment scale,
  endpoint projection, action masks, clinical margins, or bootstrap settings.
- Do not modify the manuscript.
- Do not delete previous partial outputs, failed outputs, logs, or transcripts.
- Do not start more than one zero-shot runner.

## Preflight

1. Confirm no training, evaluation, or zero-shot runner is active.
2. Fast-forward to the latest remote branch commit.
3. Confirm the tracked worktree is clean.
4. Verify all 12 required checkpoints exist:
   - GCN and flat;
   - training seeds 0, 1, and 2;
   - episode-100 final and frozen-pretrain variants.
5. Confirm checkpoint hashes are unchanged from the completed training package.
6. Run:

```powershell
.venv\Scripts\python.exe -m unittest `
  tests.test_patient_capacity_planning `
  tests.test_patient_eval_metrics `
  tests.test_patient_condition `
  tests.test_patient_observation `
  tests.test_multiscenario_residual_evaluation

.venv\Scripts\python.exe -m compileall -q src evaluation tests
```

The focused suite should report 66 passing tests. Stop and report if it does
not pass.

## Preserve failed evidence

The previous run failed after validation during the final GCN seed-0 anchor
holdout. Preserve every existing partial/failed directory and log.

If either formal output root contains incomplete output, rename it atomically
to a unique path containing `failed_typeerror` and a timestamp:

- `results/regional_conservative_gcn_flat_td3_cuda_zero_shot_eval`
- `results/regional_conservative_gcn_flat_td3_cuda_zero_shot_pretrain_control_eval`

Never delete or overwrite these directories. The new formal run must start with
both official output roots absent.

## Formal execution

Launch the existing runner once as an independent background process:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File C:\gcnrl\scripts\run_regional_conservative_td3_zero_shot.ps1 `
  -ExportDirectory `
  "C:\Users\52623\OneDrive - Georgia Institute of Technology\GCN-RL-Paper-2026\RTX4090-Transfer"
```

Preserve the original protocol:

- final evaluation followed by frozen-pretrain evaluation;
- GCN and matched flat TD3;
- training seeds 0, 1, and 2;
- five pre-registered zero-shot scenarios;
- validation seed `40800000`;
- holdout seed `40900000`;
- 100 holdout replications per scenario and training seed;
- 52 epochs;
- fixed deployment scale `0.1`;
- original endpoint projection and clinical noninferiority margins.

The new source change affects only an info-accounting field. It must not alter
observations, actions, rewards, costs, patient dynamics, or policy decisions.

## Monitoring

After launch, report:

- source commit;
- focused test result;
- runner PID and evaluator PID;
- current final/pretrain, method, seed, and validation/holdout stage;
- CUDA device and absence of CPU fallback;
- measured ETA after the first complete policy run.

Use the existing heartbeat for read-only monitoring. Do not treat an idle Codex
task as evidence that the independent process stopped.

If `risk_type_count_recoveries` is nonzero, allow the evaluation to finish and
report its count, maximum, and affected runs/scenarios. Do not hide it.

Stop the affected runner, preserve evidence, and do not restart on:

- a new traceback;
- NaN or Inf;
- CUDA OOM;
- CPU fallback;
- checkpoint, config, or CRN mismatch;
- missing checkpoint;
- any retraining or online update.

## Final acceptance

Require:

- a final summary with six policy runs;
- a frozen-pretrain summary with six policy runs;
- five scenarios and 100 holdout replications for every run;
- identical final/pretrain CRN keys;
- identical paired anchor outcomes;
- finite costs and clinical metrics;
- zero retraining and zero online updates;
- unchanged checkpoint hashes;
- explicit risk-count recovery diagnostics;
- a ZIP and SHA256 sidecar;
- matching SHA256 after copying to OneDrive.

Report:

- GCN versus MDL-2, pooled and by scenario;
- flat versus MDL-2, pooled and by scenario;
- GCN versus matched flat, pooled and by scenario;
- final versus frozen-pretrain for both representations;
- GCN-minus-flat difference-in-differences;
- clinical noninferiority;
- residual usage and inference cost;
- online TD3 attribution as `Strong`, `Scenario-specific`, or
  `Not established`.

After delivery, stop. Do not start seeds 3-4, a 300-episode campaign, or any
other experiment until the zero-shot result is reviewed.
