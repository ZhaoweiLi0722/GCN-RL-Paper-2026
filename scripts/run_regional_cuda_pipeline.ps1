Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_cuda_4090.ps1 first."
}

$RequiredData = @(
    "results\decoupled_transfer_regional_teacher_h52_crn8_100traj_temporal\teacher_train.npz",
    "results\decoupled_transfer_regional_teacher_h52_crn8_100traj_temporal\teacher_validation.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_seed0_smoke\teacher_train_dagger.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_seed0_smoke_validation\teacher_dagger_only.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_iter2_seed0\teacher_train_dagger.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_iter2_seed0_validation\teacher_dagger_only.npz"
)
foreach ($Path in $RequiredData) {
    if (-not (Test-Path $Path)) {
        throw "Missing training bundle file: $Path"
    }
}

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".matplotlib-cache"
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null
New-Item -ItemType Directory -Force "results\regional_cuda_pipeline\logs" | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = "results\regional_cuda_pipeline\logs\pipeline_$Timestamp.txt"
Start-Transcript -Path $LogPath

try {
    & $Python scripts\verify_cuda.py
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA verification failed."
    }

    $TrainingConfigs = @(
        "experiments\configs\regional_cuda_stage1_base.json",
        "experiments\configs\regional_cuda_stage2_dagger1.json",
        "experiments\configs\regional_cuda_stage3_dagger2.json"
    )
    foreach ($Config in $TrainingConfigs) {
        Write-Host ""
        Write-Host "Running $Config"
        & $Python -m evaluation.train_multiscenario_network_residual --config $Config
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $Config"
        }
    }

    $EvaluationConfig = "experiments\configs\regional_cuda_stage4_fixed_eval.json"
    Write-Host ""
    Write-Host "Running $EvaluationConfig"
    & $Python -m evaluation.evaluate_multiscenario_network_residual --config $EvaluationConfig
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $EvaluationConfig"
    }

    Write-Host ""
    Write-Host "Regional CUDA pipeline completed."
    Write-Host "Results: results\regional_cuda_pipeline"
}
finally {
    Stop-Transcript
}
