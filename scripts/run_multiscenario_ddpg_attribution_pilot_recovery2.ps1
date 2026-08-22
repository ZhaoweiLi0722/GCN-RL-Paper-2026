param(
    [string]$ExportDirectory = "",
    [string]$TeacherBundle = "",
    [string]$ExpectedCommit = "",
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-IndependentRelatedProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$ProcessSnapshot,
        [Parameter(Mandatory = $true)]
        [int]$CurrentProcessId
    )

    $ProcessById = @{}
    foreach ($Process in $ProcessSnapshot) {
        $ProcessById[[int]$Process.ProcessId] = $Process
    }

    # Start-Process leaves a short-lived PowerShell ancestor whose command line
    # contains this runner path. Ancestors are launchers, not other campaigns.
    $ExcludedProcessIds = @($CurrentProcessId)
    $CursorId = $CurrentProcessId
    while ($ProcessById.ContainsKey($CursorId)) {
        $ParentId = [int]$ProcessById[$CursorId].ParentProcessId
        if (
            $ParentId -le 0 -or
            $ExcludedProcessIds -contains $ParentId
        ) {
            break
        }
        $ExcludedProcessIds += $ParentId
        $CursorId = $ParentId
    }

    return @(
        $ProcessSnapshot | Where-Object {
            -not ($ExcludedProcessIds -contains [int]$_.ProcessId) -and
            $_.CommandLine -and (
                $_.CommandLine -match "train_multiscenario_network_residual" -or
                $_.CommandLine -match "evaluate_multiscenario_network_residual" -or
                $_.CommandLine -match (
                    "(?:run|resume)_multiscenario_ddpg_attribution"
                )
            )
        }
    )
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = if ($PythonExecutable) {
    [string](Resolve-Path $PythonExecutable)
} else {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_cuda_4090.ps1 first."
}
$PythonVersion = ((& $Python --version 2>&1) | Out-String).Trim()
$PythonSha256 = (
    Get-FileHash -Algorithm SHA256 $Python
).Hash.ToLowerInvariant()

$Commit = (git rev-parse HEAD).Trim()
if ($ExpectedCommit -and $Commit -ne $ExpectedCommit) {
    throw "Expected commit $ExpectedCommit but found $Commit."
}
$DirtyPaths = @(git status --porcelain --untracked-files=no)
if ($DirtyPaths.Count -gt 0) {
    throw "Commit or stash tracked source changes before a paper run."
}
$ProcessSnapshot = @(Get-CimInstance Win32_Process)
$RelatedProcesses = @(
    Get-IndependentRelatedProcesses `
        -ProcessSnapshot $ProcessSnapshot `
        -CurrentProcessId ([int]$PID)
)
if ($RelatedProcesses.Count -gt 0) {
    throw "A related DDPG attribution process is already active."
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
    "results\multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2"
)
$FinalOutput = (
    "results\multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_eval"
)
$FrozenOutput = (
    "results\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_pretrain_eval"
)
$GateRoot = (
    "results\multiscenario_gcn_ddpg_attribution_recovery2_stability_gate"
)
foreach ($Path in @($GateRoot, $RunRoot, $FinalOutput, $FrozenOutput)) {
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
$LogPath = Join-Path $TransferRoot "pilot_recovery2_$Timestamp.txt"
$ComparisonPath = Join-Path $TransferRoot (
    "checkpoint_attribution_recovery2_$Timestamp.json"
)

$TrainingConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2.json"
)
$StabilityGateConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_ddpg_attribution_recovery2_stability_gate.json"
)
$FinalEvaluationConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_eval.json"
)
$FrozenEvaluationConfig = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_pretrain_eval.json"
)

function Invoke-CheckedPython {
    param(
        [string]$Stage,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "Starting isolated stage: $Stage"
    & $Python @Arguments
    $ExitCodeSigned = [int32]$LASTEXITCODE
    $ExitCodeUnsigned = [BitConverter]::ToUInt32(
        [BitConverter]::GetBytes($ExitCodeSigned),
        0
    )
    $ExitCodeHex = "0x{0:X8}" -f $ExitCodeUnsigned
    Write-Host (
        "Stage exit code: stage=$Stage; signed=$ExitCodeSigned; " +
        "unsigned=$ExitCodeUnsigned; hex=$ExitCodeHex"
    )
    if ($ExitCodeSigned -ne 0) {
        throw (
            "$Stage failed with exit code signed=$ExitCodeSigned, " +
            "unsigned=$ExitCodeUnsigned, hex=$ExitCodeHex."
        )
    }
}

Start-Transcript -Path $LogPath
try {
    $TestPatterns = @(
        "test_compare_multiscenario_checkpoint_stages.py",
        "test_off_policy_training_state.py",
        "test_multi_scenario_env.py",
        "test_train_multiscenario_network_residual.py"
    )
    foreach ($Pattern in $TestPatterns) {
        Invoke-CheckedPython `
            -Stage "focused test $Pattern" `
            -Arguments @(
                "-m", "unittest", "discover", "-s", "tests", "-p", $Pattern
            )
    }

    Invoke-CheckedPython `
        -Stage "CUDA verification" `
        -Arguments @("scripts\verify_cuda.py")

    $GatePretrainCheckpoint = (
        "results\" +
        "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1\" +
        "gcn_residual_mdl2_network_ddpg_afd\seed0\checkpoints\" +
        "gcn_residual_mdl2_network_ddpg_afd_seed0_pretrain.pt"
    )
    if (-not (Test-Path $GatePretrainCheckpoint)) {
        throw "Missing Recovery 1 pretrain checkpoint for stability gate."
    }
    $GatePretrainHashBefore = (
        Get-FileHash -Algorithm SHA256 $GatePretrainCheckpoint
    ).Hash.ToLowerInvariant()

    Invoke-CheckedPython `
        -Stage "non-paper GCN seed 0 online stability gate" `
        -Arguments @(
            "-m", "evaluation.train_multiscenario_network_residual",
            "--config", $StabilityGateConfig,
            "--algorithm", "gcn_residual_mdl2_network_ddpg_afd",
            "--seed", "0"
        )
    $GatePretrainHashAfter = (
        Get-FileHash -Algorithm SHA256 $GatePretrainCheckpoint
    ).Hash.ToLowerInvariant()
    if ($GatePretrainHashAfter -ne $GatePretrainHashBefore) {
        throw "Stability gate modified the Recovery 1 pretrain checkpoint."
    }
    $GateRunDirectory = Join-Path $GateRoot (
        "gcn_residual_mdl2_network_ddpg_afd\seed0"
    )
    $GateRows = @(Import-Csv (Join-Path $GateRunDirectory "training.csv"))
    if ($GateRows.Count -ne 20) {
        throw "Stability gate did not complete exactly 20 episodes."
    }
    $GateOnlineUpdates = 0
    foreach ($Row in $GateRows) {
        $GateOnlineUpdates += [int]$Row.online_rl_updates
    }
    if ($GateOnlineUpdates -le 0) {
        throw "Stability gate completed with zero online RL updates."
    }
    $GateExpectedOutputs = @(
        (Join-Path $GateRunDirectory "summary.json"),
        (Join-Path $GateRunDirectory (
            "checkpoints\" +
            "gcn_residual_mdl2_network_ddpg_afd_seed0_episode20.pt"
        )),
        (Join-Path $GateRunDirectory (
            "checkpoints\" +
            "gcn_residual_mdl2_network_ddpg_afd_seed0_training_state.pt"
        ))
    )
    foreach ($Path in $GateExpectedOutputs) {
        if (-not (Test-Path $Path)) {
            throw "Stability gate did not create expected output: $Path"
        }
    }

    $RunSpecs = @()
    foreach ($Algorithm in @(
        "gcn_residual_mdl2_network_ddpg_afd",
        "flat_residual_mdl2_network_ddpg_afd"
    )) {
        foreach ($Seed in 0..2) {
            $RunSpecs += [pscustomobject]@{
                Algorithm = $Algorithm
                Seed = $Seed
            }
        }
    }
    foreach ($RunSpec in $RunSpecs) {
        $Algorithm = [string]$RunSpec.Algorithm
        $Seed = [int]$RunSpec.Seed
        Invoke-CheckedPython `
            -Stage "official training $Algorithm seed $Seed" `
            -Arguments @(
                "-m", "evaluation.train_multiscenario_network_residual",
                "--config", $TrainingConfig,
                "--algorithm", $Algorithm,
                "--seed", [string]$Seed
            )
        $RunDirectory = Join-Path $RunRoot "$Algorithm\seed$Seed"
        $Rows = @(Import-Csv (Join-Path $RunDirectory "training.csv"))
        if ($Rows.Count -ne 100) {
            throw "$Algorithm seed $Seed did not complete 100 episodes."
        }
        $OnlineUpdates = 0
        foreach ($Row in $Rows) {
            $OnlineUpdates += [int]$Row.online_rl_updates
        }
        if ($OnlineUpdates -le 0) {
            throw "$Algorithm seed $Seed completed with zero online updates."
        }
        foreach ($Path in @(
            (Join-Path $RunDirectory "summary.json"),
            (Join-Path $RunDirectory (
                "checkpoints\${Algorithm}_seed${Seed}_pretrain.pt"
            )),
            (Join-Path $RunDirectory (
                "checkpoints\${Algorithm}_seed${Seed}_episode100.pt"
            )),
            (Join-Path $RunDirectory (
                "checkpoints\${Algorithm}_seed${Seed}_training_state.pt"
            ))
        )) {
            if (-not (Test-Path $Path)) {
                throw "$Algorithm seed $Seed missing output: $Path"
            }
        }
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
        Invoke-CheckedPython `
            -Stage "locked CRN evaluation $Config" `
            -Arguments @(
                "-m", "evaluation.evaluate_multiscenario_network_residual",
                "--config", $Config
            )
    }

    Invoke-CheckedPython `
        -Stage "final-versus-frozen attribution" `
        -Arguments @(
            "-m", "evaluation.compare_multiscenario_checkpoint_stages",
            "--final-summary", "$FinalOutput\summary.json",
            "--frozen-summary", "$FrozenOutput\summary.json",
            "--output", $ComparisonPath,
            "--bootstrap-resamples", "20000",
            "--bootstrap-seed", "61000000"
        )
}
finally {
    Stop-Transcript
}

$ExpectedOutputs = @(
    "$GateRoot\training_manifest.json",
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
    "provenance_recovery2_$Timestamp.json"
)
$Branch = (git branch --show-current).Trim()
$Provenance = [ordered]@{
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    git_commit = $Commit
    git_branch = $Branch
    python_executable = $Python
    python_version = $PythonVersion
    python_sha256 = $PythonSha256
    training_config = $TrainingConfig
    stability_gate_config = $StabilityGateConfig
    stability_gate_diagnostic_only = $true
    stability_gate_online_episodes = 20
    stability_gate_online_updates = $GateOnlineUpdates
    stability_gate_pretrain_checkpoint_sha256 = $GatePretrainHashBefore
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
    recovery_attempt = 2
    recovery_reason = (
        "Two Windows native access violations: Recovery 0 during AFD " +
        "pretraining and Recovery 1 after online episode 10."
    )
    isolated_process_per_policy_run = $true
    atomic_training_state_interval_episodes = 5
    exact_resume_includes = @(
        "actor_and_target",
        "critic_and_target",
        "optimizers",
        "replay_buffer",
        "imitation_regularization_cache",
        "exploration_rng",
        "global_rng",
        "episode_rows"
    )
    scientific_parameters_changed = $false
}
$Provenance | ConvertTo-Json -Depth 8 | Set-Content `
    -Encoding UTF8 $ProvenancePath

$ZipPath = Join-Path $TransferRoot (
    "multiscenario_ddpg_attribution_pilot_recovery2_$Timestamp.zip"
)
$ArchiveInputs = @(
    $GateRoot,
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
Write-Host "Multi-scenario DDPG attribution recovery2 completed."
Write-Host "ZIP: $ZipPath"
Write-Host "SHA256: $($Hash.Hash.ToLowerInvariant())"
if ($ExportDirectory) {
    Write-Host "Copied to: $ExportDirectory"
}
