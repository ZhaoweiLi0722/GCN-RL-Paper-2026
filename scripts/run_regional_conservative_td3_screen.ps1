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

$RequiredAssets = @(
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_iter2_seed0\teacher_train_dagger.npz"
)
foreach ($Method in @(
    "gcn_residual_mdl2_network_ddpg_afd",
    "flat_residual_mdl2_network_ddpg_afd"
)) {
    foreach ($Seed in 0..2) {
        $RequiredAssets += (
            "results\regional_cuda_pipeline\regional_cuda_stage3_dagger2\" +
            "$Method\seed$Seed\checkpoints\" +
            "${Method}_seed${Seed}_pretrain.pt"
        )
    }
}
foreach ($Path in $RequiredAssets) {
    if (-not (Test-Path $Path)) {
        throw "Missing required Stage-3 asset: $Path"
    }
}

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".matplotlib-cache"
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null

$RunRoot = "results\regional_conservative_gcn_flat_td3_cuda_screen"
$LogRoot = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogRoot "screen_$Timestamp.txt"

Start-Transcript -Path $LogPath
try {
    & $Python scripts\verify_cuda.py
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA verification failed."
    }

    $TrainingConfig = (
        "experiments\configs\" +
        "regional_conservative_gcn_flat_td3_cuda_screen.json"
    )
    Write-Host ""
    Write-Host "Running matched conservative TD3 training"
    & $Python -m evaluation.train_multiscenario_network_residual `
        --config $TrainingConfig
    if ($LASTEXITCODE -ne 0) {
        throw "Conservative TD3 training failed."
    }

    $EvaluationConfigs = @(
        (
            "experiments\configs\" +
            "regional_conservative_gcn_flat_td3_cuda_screen_eval.json"
        ),
        (
            "experiments\configs\" +
            "regional_conservative_gcn_flat_td3_cuda_pretrain_control_eval.json"
        )
    )
    foreach ($Config in $EvaluationConfigs) {
        Write-Host ""
        Write-Host "Running $Config"
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

$ExpectedOutputs = @(
    "$RunRoot\training_manifest.json",
    "results\regional_conservative_gcn_flat_td3_cuda_screen_eval\summary.json",
    "results\regional_conservative_gcn_flat_td3_cuda_pretrain_control_eval\summary.json"
)
foreach ($Path in $ExpectedOutputs) {
    if (-not (Test-Path $Path)) {
        throw "Screen did not create expected output: $Path"
    }
}

$TransferRoot = "results\regional_conservative_td3_transfer"
New-Item -ItemType Directory -Force $TransferRoot | Out-Null
$ProvenancePath = Join-Path $TransferRoot "provenance_$Timestamp.json"
$Commit = (git rev-parse HEAD).Trim()
$Branch = (git branch --show-current).Trim()
$Provenance = [ordered]@{
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    git_commit = $Commit
    git_branch = $Branch
    training_config = (
        "experiments/configs/" +
        "regional_conservative_gcn_flat_td3_cuda_screen.json"
    )
    final_evaluation_config = (
        "experiments/configs/" +
        "regional_conservative_gcn_flat_td3_cuda_screen_eval.json"
    )
    pretrain_control_config = (
        "experiments/configs/" +
        "regional_conservative_gcn_flat_td3_cuda_pretrain_control_eval.json"
    )
    policy_runs = 6
    online_episodes_per_run = 100
    training_seeds = @(0, 1, 2)
    holdout_replications_per_seed = 100
    holdout_seed = 39900000
}
$Provenance | ConvertTo-Json -Depth 5 | Set-Content `
    -Encoding UTF8 $ProvenancePath

$ZipPath = Join-Path $TransferRoot (
    "regional_conservative_td3_screen_$Timestamp.zip"
)
$ArchiveInputs = @(
    $RunRoot,
    "results\regional_conservative_gcn_flat_td3_cuda_screen_eval",
    "results\regional_conservative_gcn_flat_td3_cuda_pretrain_control_eval",
    $ProvenancePath
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
Write-Host "Conservative TD3 screen completed."
Write-Host "ZIP: $ZipPath"
Write-Host "SHA256: $($Hash.Hash.ToLower())"
if ($ExportDirectory) {
    Write-Host "Copied to: $ExportDirectory"
}
