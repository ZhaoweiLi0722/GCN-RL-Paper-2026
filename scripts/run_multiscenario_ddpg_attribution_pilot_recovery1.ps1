param(
    [string]$ExportDirectory = "",
    [string]$TeacherBundle = "",
    [string]$ExpectedCommit = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_cuda_4090.ps1 first."
}

$Commit = (git rev-parse HEAD).Trim()
if ($ExpectedCommit -and $Commit -ne $ExpectedCommit) {
    throw "Expected commit $ExpectedCommit but found $Commit."
}
$DirtyPaths = @(git status --porcelain --untracked-files=no)
if ($DirtyPaths.Count -gt 0) {
    throw "Commit or stash tracked source changes before a paper run."
}

$TeacherAssets = [ordered]@{
    "results\multiscenario_teacher_screen\teacher_train.npz" = (
        "49b21ac92c743f1489b6e914cf17795c214ff3f4d061faec4561db5d91477953"
    )
    "results\multiscenario_teacher_screen\teacher_validation.npz" = (
        "6070d9a8b959252109e1a5ea597604ae804b873d6a1883686c6d3fd637fb6be3"
    )
    "results\multiscenario_teacher_screen\manifest.json" = (
        "7b2fb4ff6cd1cede6bf15d5e9570de533b3becac7f092ff7b15f55925f921160"
    )
}
$MissingTeacherAssets = @(
    $TeacherAssets.Keys | Where-Object { -not (Test-Path $_) }
)
if ($MissingTeacherAssets.Count -gt 0) {
    if (-not $TeacherBundle) {
        throw (
            "Missing teacher assets and no -TeacherBundle was supplied: " +
            ($MissingTeacherAssets -join ", ")
        )
    }
    if (-not (Test-Path $TeacherBundle)) {
        throw "Teacher bundle does not exist: $TeacherBundle"
    }
    $TeacherBundleHashPath = "$TeacherBundle.sha256"
    if (-not (Test-Path $TeacherBundleHashPath)) {
        throw "Missing teacher bundle SHA256 sidecar: $TeacherBundleHashPath"
    }
    $ExpectedBundleHash = (
        (Get-Content $TeacherBundleHashPath | Select-Object -First 1) `
            -split "\s+"
    )[0].ToLowerInvariant()
    $ActualBundleHash = (
        Get-FileHash -Algorithm SHA256 $TeacherBundle
    ).Hash.ToLowerInvariant()
    if ($ActualBundleHash -ne $ExpectedBundleHash) {
        throw "Teacher bundle SHA256 mismatch."
    }
    Expand-Archive -Path $TeacherBundle -DestinationPath $RepoRoot -Force
}
foreach ($Entry in $TeacherAssets.GetEnumerator()) {
    if (-not (Test-Path $Entry.Key)) {
        throw "Missing teacher asset after import: $($Entry.Key)"
    }
    $ActualHash = (
        Get-FileHash -Algorithm SHA256 $Entry.Key
    ).Hash.ToLowerInvariant()
    if ($ActualHash -ne $Entry.Value) {
        throw "Teacher asset SHA256 mismatch: $($Entry.Key)"
    }
}

$RunRoot = (
    "results\multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1"
)
$FinalOutput = (
    "results\multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1_eval"
)
$FrozenOutput = (
    "results\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1_pretrain_eval"
)
foreach ($Path in @($RunRoot, $FinalOutput, $FrozenOutput)) {
    if (Test-Path $Path) {
        throw "Refusing to overwrite existing campaign output: $Path"
    }
}

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONFAULTHANDLER = "1"
$env:TORCH_SHOW_CPP_STACKTRACES = "1"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".matplotlib-cache"
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null

$TransferRoot = "results\multiscenario_ddpg_attribution_transfer"
New-Item -ItemType Directory -Force $TransferRoot | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $TransferRoot "pilot_recovery1_$Timestamp.txt"
$ComparisonPath = Join-Path $TransferRoot (
    "checkpoint_attribution_recovery1_$Timestamp.json"
)

$TrainingConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1.json"
)
$FinalEvaluationConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1_eval.json"
)
$FrozenEvaluationConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1_pretrain_eval.json"
)

Start-Transcript -Path $LogPath
try {
    & $Python -m unittest discover -s tests `
        -p "test_compare_multiscenario_checkpoint_stages.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint-stage comparison tests failed."
    }

    & $Python scripts\verify_cuda.py
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA verification failed."
    }

    Write-Host ""
    Write-Host "Running matched multi-scenario AFR-GCN/Flat-DDPG training"
    & $Python -m evaluation.train_multiscenario_network_residual `
        --config $TrainingConfig
    $TrainingExitCodeSigned = [int32]$LASTEXITCODE
    $TrainingExitCodeUnsigned = [BitConverter]::ToUInt32(
        [BitConverter]::GetBytes($TrainingExitCodeSigned),
        0
    )
    $TrainingExitCodeHex = "0x{0:X8}" -f $TrainingExitCodeUnsigned
    Write-Host (
        "Training process exit code: signed=$TrainingExitCodeSigned; " +
        "unsigned=$TrainingExitCodeUnsigned; hex=$TrainingExitCodeHex"
    )
    if ($TrainingExitCodeSigned -ne 0) {
        throw (
            "Multi-scenario DDPG training failed with exit code " +
            "signed=$TrainingExitCodeSigned, " +
            "unsigned=$TrainingExitCodeUnsigned, " +
            "hex=$TrainingExitCodeHex."
        )
    }

    $TrainingManifestPath = Join-Path $RunRoot "training_manifest.json"
    $TrainingManifest = Get-Content $TrainingManifestPath -Raw |
        ConvertFrom-Json
    foreach ($Seed in 0..2) {
        $GraphRun = @(
            $TrainingManifest.runs | Where-Object {
                $_.algorithm -eq "gcn_residual_mdl2_network_ddpg_afd" `
                    -and [int]$_.seed -eq $Seed
            }
        )
        $FlatRun = @(
            $TrainingManifest.runs | Where-Object {
                $_.algorithm -eq "flat_residual_mdl2_network_ddpg_afd" `
                    -and [int]$_.seed -eq $Seed
            }
        )
        if ($GraphRun.Count -ne 1 -or $FlatRun.Count -ne 1) {
            throw "Missing matched graph/flat training run for seed $Seed."
        }
        $GraphParameters = [double]$GraphRun[0].parameter_count
        $FlatParameters = [double]$FlatRun[0].parameter_count
        $RelativeGap = [Math]::Abs(
            $GraphParameters - $FlatParameters
        ) / [Math]::Max($GraphParameters, $FlatParameters)
        if ($RelativeGap -gt 0.01) {
            throw (
                "Graph/flat parameter-count gap exceeds 1% for seed " +
                "${Seed}: $RelativeGap"
            )
        }
    }

    foreach ($Config in @(
        $FinalEvaluationConfig,
        $FrozenEvaluationConfig
    )) {
        Write-Host ""
        Write-Host "Running locked CRN evaluation: $Config"
        & $Python -m evaluation.evaluate_multiscenario_network_residual `
            --config $Config
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $Config"
        }
    }

    & $Python -m evaluation.compare_multiscenario_checkpoint_stages `
        --final-summary "$FinalOutput\summary.json" `
        --frozen-summary "$FrozenOutput\summary.json" `
        --output $ComparisonPath `
        --bootstrap-resamples 20000 `
        --bootstrap-seed 61000000
    if ($LASTEXITCODE -ne 0) {
        throw "Final-versus-frozen attribution failed."
    }
}
finally {
    Stop-Transcript
}

$ExpectedOutputs = @(
    "$RunRoot\training_manifest.json",
    "$FinalOutput\summary.json",
    "$FrozenOutput\summary.json",
    $ComparisonPath
)
foreach ($Path in $ExpectedOutputs) {
    if (-not (Test-Path $Path)) {
        throw "Campaign did not create expected output: $Path"
    }
}

$ProvenancePath = Join-Path $TransferRoot (
    "provenance_recovery1_$Timestamp.json"
)
$Branch = (git branch --show-current).Trim()
$Provenance = [ordered]@{
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    git_commit = $Commit
    git_branch = $Branch
    training_config = $TrainingConfig
    final_evaluation_config = $FinalEvaluationConfig
    frozen_evaluation_config = $FrozenEvaluationConfig
    comparison = $ComparisonPath
    teacher_asset_sha256 = $TeacherAssets
    algorithms = @(
        "gcn_residual_mdl2_network_ddpg_afd",
        "flat_residual_mdl2_network_ddpg_afd"
    )
    training_seeds = @(0, 1, 2)
    online_episodes_per_run = 100
    policy_runs = 6
    scenarios = @(
        "patient_condition_geo_nominal_history",
        "patient_condition_geo_abrupt_regime_shift",
        "patient_condition_geo_regional_drift",
        "patient_condition_geo_compound_regional_stress"
    )
    holdout_replications_per_scenario_seed = 100
    holdout_seed = 59900000
    deployment_was_pre_registered = $true
    retraining_from_holdout_feedback = $false
    recovery_attempt = 1
    recovery_reason = "Original process exited during AFD pretraining."
    scientific_parameters_changed = $false
}
$Provenance | ConvertTo-Json -Depth 8 | Set-Content `
    -Encoding UTF8 $ProvenancePath

$ZipPath = Join-Path $TransferRoot (
    "multiscenario_ddpg_attribution_pilot_recovery1_$Timestamp.zip"
)
$ArchiveInputs = @(
    $RunRoot,
    $FinalOutput,
    $FrozenOutput,
    $ComparisonPath,
    $ProvenancePath,
    $LogPath
)
Compress-Archive -Path $ArchiveInputs -DestinationPath $ZipPath
$Hash = Get-FileHash -Algorithm SHA256 $ZipPath
$HashPath = "$ZipPath.sha256"
(
    $Hash.Hash.ToLowerInvariant() + "  " + (Split-Path -Leaf $ZipPath)
) | Set-Content -Encoding ASCII $HashPath

if ($ExportDirectory) {
    New-Item -ItemType Directory -Force $ExportDirectory | Out-Null
    Copy-Item $ZipPath $ExportDirectory
    Copy-Item $HashPath $ExportDirectory
}

Write-Host ""
Write-Host "Multi-scenario DDPG attribution recovery1 completed."
Write-Host "ZIP: $ZipPath"
Write-Host "SHA256: $($Hash.Hash.ToLowerInvariant())"
if ($ExportDirectory) {
    Write-Host "Copied to: $ExportDirectory"
}
