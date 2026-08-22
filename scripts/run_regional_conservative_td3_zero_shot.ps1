param(
    [string]$ExportDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_cuda_4090.ps1 first."
}

$DirtyPaths = @(git status --porcelain --untracked-files=no)
if ($DirtyPaths.Count -gt 0) {
    throw "Commit or stash tracked source changes before a paper run."
}

$RunRoot = "results\regional_conservative_gcn_flat_td3_cuda_screen"
$TrainingManifest = Join-Path $RunRoot "training_manifest.json"
if (-not (Test-Path $TrainingManifest)) {
    throw "Missing completed TD3 training manifest: $TrainingManifest"
}

foreach ($Method in @(
    "gcn_residual_mdl2_network_td3_bc",
    "flat_residual_mdl2_network_td3_bc"
)) {
    foreach ($Seed in 0..2) {
        foreach ($Variant in @("episode100", "pretrain")) {
            $Checkpoint = (
                "$RunRoot\$Method\seed$Seed\checkpoints\" +
                "${Method}_seed${Seed}_${Variant}.pt"
            )
            if (-not (Test-Path $Checkpoint)) {
                throw "Missing required checkpoint: $Checkpoint"
            }
        }
    }
}

$FinalOutput = (
    "results\regional_conservative_gcn_flat_td3_cuda_zero_shot_eval"
)
$PretrainOutput = (
    "results\" +
    "regional_conservative_gcn_flat_td3_cuda_zero_shot_pretrain_control_eval"
)
foreach ($Path in @($FinalOutput, $PretrainOutput)) {
    if (Test-Path $Path) {
        throw "Refusing to overwrite existing evaluation output: $Path"
    }
}

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".matplotlib-cache"
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null

$LogRoot = "results\regional_conservative_td3_zero_shot_transfer"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogRoot "zero_shot_$Timestamp.txt"

$EvaluationConfigs = @(
    (
        "experiments\configs\" +
        "regional_conservative_gcn_flat_td3_cuda_zero_shot_eval.json"
    ),
    (
        "experiments\configs\" +
        "regional_conservative_gcn_flat_td3_cuda_zero_shot_pretrain_control_eval.json"
    )
)

Start-Transcript -Path $LogPath
try {
    & $Python scripts\verify_cuda.py
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA verification failed."
    }

    foreach ($Config in $EvaluationConfigs) {
        Write-Host ""
        Write-Host "Running evaluation only: $Config"
        & $Python -m evaluation.evaluate_multiscenario_network_residual `
            --config $Config
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $Config"
        }
    }
}
finally {
    Stop-Transcript
}

foreach ($Path in @(
    "$FinalOutput\summary.json",
    "$PretrainOutput\summary.json"
)) {
    if (-not (Test-Path $Path)) {
        throw "Evaluation did not create expected output: $Path"
    }
}

$ProvenancePath = Join-Path $LogRoot "provenance_$Timestamp.json"
$Commit = (git rev-parse HEAD).Trim()
$Branch = (git branch --show-current).Trim()
$Provenance = [ordered]@{
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    git_commit = $Commit
    git_branch = $Branch
    source_training_manifest = $TrainingManifest
    final_evaluation_config = $EvaluationConfigs[0]
    pretrain_control_config = $EvaluationConfigs[1]
    policy_runs_per_evaluation = 6
    training_seeds = @(0, 1, 2)
    scenarios = @(
        "patient_condition_geo_nominal_history",
        "patient_condition_geo_demand_drift_severe_history",
        "patient_condition_geo_regional_drift",
        "patient_condition_geo_abrupt_regime_shift",
        "patient_condition_geo_compound_regional_stress"
    )
    holdout_replications_per_scenario_seed = 100
    holdout_seed = 40900000
    retraining_performed = $false
}
$Provenance | ConvertTo-Json -Depth 5 | Set-Content `
    -Encoding UTF8 $ProvenancePath

$ZipPath = Join-Path $LogRoot (
    "regional_conservative_td3_zero_shot_$Timestamp.zip"
)
$ArchiveInputs = @(
    $FinalOutput,
    $PretrainOutput,
    $TrainingManifest,
    $ProvenancePath,
    $LogPath
)
Compress-Archive -Path $ArchiveInputs -DestinationPath $ZipPath
$Hash = Get-FileHash -Algorithm SHA256 $ZipPath
$HashPath = "$ZipPath.sha256"
(
    $Hash.Hash.ToLower() + "  " + (Split-Path -Leaf $ZipPath)
) | Set-Content -Encoding ASCII $HashPath

if ($ExportDirectory) {
    New-Item -ItemType Directory -Force $ExportDirectory | Out-Null
    Copy-Item $ZipPath $ExportDirectory
    Copy-Item $HashPath $ExportDirectory
}

Write-Host ""
Write-Host "Zero-shot TD3 evaluations completed without retraining."
Write-Host "ZIP: $ZipPath"
Write-Host "SHA256: $($Hash.Hash.ToLower())"
if ($ExportDirectory) {
    Write-Host "Copied to: $ExportDirectory"
}
