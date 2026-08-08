param(
    [ValidateSet("Preflight", "ImportTeacher", "Smoke", "Pilot", "Evaluate")]
    [string]$Phase = "Preflight",
    [Parameter(Mandatory = $true)]
    [string]$ExpectedCommit,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9]+(?:,[0-9]+)*$")]
    [string]$ControlProcessIds,
    [switch]$ApprovePilot,
    [string]$TeacherBundle = "",
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LockedBranch = "codex/patient-indexed-specimen-routing"
$LockedParent = "ce9b6274419c8e0e7adf800f434e47d96c18c1dc"
$ResultRoot = "results\patient_indexed_specimen_routing_recovery5"
$RecoveryName = "Recovery 5"
$SupersededCommit = "0628af8bbfa584b344d145eaff1234e1e49b122a"
$SupersededResultRoot = "results\patient_indexed_specimen_routing_recovery4"
$SupersededFailureEvidenceSha256 = [ordered]@{
    validate_status = "5d0bf8d478096d7241d2aeb4bfd8ecd77cc15598169eca6532d945802a7f00a3"
    mechanics_report = "7f8b1f3774ab0e9e2ae5453615052e273a027c5c4f34a90a240159c8eb925e27"
    import_teacher_claim = "4b34fb0703b49ca0817ffefc30a31a05474df6b59dc58413a3031487b8094a72"
    import_teacher_stdout = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    import_teacher_stderr = "8e2d82058532bef73bbb3b59325c0162eaf9aaf76846b5387be919b36597b6f8"
    import_teacher_status = "4243c0944f0ec01be48a34a9c6886574bffd21750a77ff25e4b57c3b416585c4"
}
$FailureClassification = (
    "PC launcher process-gate false positive before frozen teacher import"
)
$PriorRecovery2Commit = "304dc83d6eb7447c6371150a551ea6d01999d5aa"
$PriorRecovery2ResultRoot = "results\patient_indexed_specimen_routing_recovery2"
$PriorRecovery2TranscriptSha256 = (
    "d9a3e7f598b2991a90662ae3f1326be7cade6889888a3f5ff6f9d17f7b9bbfed"
)
$GraphAlgorithm = "gcn_residual_mdl2_network_ddpg_afd"
$FlatAlgorithm = "flat_residual_mdl2_network_ddpg_afd"
$FrozenTeacherConfigName = (
    "patient_indexed_specimen_routing_teacher_routing.json"
)
$FrozenTeacherSourceRoot = (
    "results\patient_indexed_specimen_routing_recovery4\teachers\routing"
)
$MacValidationCommit = "4cc04d298417ae65d6a0c8f4c9ca6414a6da47ee"
$MacValidationEvidence = (
    "experiments\evidence\patient_indexed_specimen_routing_mac_validation.json"
)
$MacValidationEvidenceSha256 = (
    "4805af6790999a4403ebb35495179444f667da079a4cd5a08015371316d14953"
)
$MacMechanicsReportSha256 = (
    "1409c76ae01673b87108f311c91a941535796cfe0edd43f218ae8d344faa0260"
)

function Get-IndependentRelatedProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$ProcessSnapshot,
        [Parameter(Mandatory = $true)]
        [int]$CurrentProcessId,
        [int[]]$ControlProcessIds = @()
    )

    $ProcessById = @{}
    foreach ($Process in $ProcessSnapshot) {
        $ProcessById[[int]$Process.ProcessId] = $Process
    }
    $ExcludedProcessIds = @(
        @($CurrentProcessId) + @($ControlProcessIds) |
            Sort-Object -Unique
    )
    $CursorId = $CurrentProcessId
    while ($ProcessById.ContainsKey($CursorId)) {
        $ParentId = [int]$ProcessById[$CursorId].ParentProcessId
        if ($ParentId -le 0 -or $ExcludedProcessIds -contains $ParentId) {
            break
        }
        $ExcludedProcessIds += $ParentId
        $CursorId = $ParentId
    }
    return @(
        $ProcessSnapshot | Where-Object {
            $ProcessName = [string]$_.Name
            $CommandLine = [string]$_.CommandLine
            $IsPythonWorkload = (
                $ProcessName -match "(?i)^python(?:w)?\.exe$" -and
                $CommandLine -and (
                    $CommandLine -match "train_multiscenario_network_residual" -or
                    $CommandLine -match "evaluate_multiscenario_network_residual" -or
                    $CommandLine -match "network_residual_headroom"
                )
            )
            $IsDetachedRoutingLauncher = (
                $ProcessName -match "(?i)^(?:powershell|pwsh)\.exe$" -and
                $CommandLine -and
                $CommandLine -match (
                    "(?i)-File\s+.*(?:run_patient_indexed_specimen_routing|" +
                    "invoke_patient_indexed_specimen_routing_phase_detached)\.ps1"
                )
            )
            -not ($ExcludedProcessIds -contains [int]$_.ProcessId) -and
                ($IsPythonWorkload -or $IsDetachedRoutingLauncher)
        }
    )
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Stage,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "Starting locked stage: $Stage"
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

function Assert-FreshPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path $Path) {
        throw "Refusing to overwrite existing routing output: $Path"
    }
}

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -PathType Leaf $Path)) {
        throw "Missing required routing artifact: $Path"
    }
}

function Assert-FiniteNumber {
    param(
        [Parameter(Mandatory = $true)][object]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Number = 0.0
    $Style = [Globalization.NumberStyles]::Float
    $Culture = [Globalization.CultureInfo]::InvariantCulture
    if (-not [double]::TryParse([string]$Value, $Style, $Culture, [ref]$Number)) {
        throw "Non-numeric diagnostic $Label=$Value"
    }
    if ([double]::IsNaN($Number) -or [double]::IsInfinity($Number)) {
        throw "Non-finite diagnostic $Label=$Value"
    }
    return $Number
}

function Get-StringLeaves {
    param([object]$Value)
    if ($null -eq $Value) {
        return @()
    }
    if ($Value -is [string]) {
        return @([string]$Value)
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $Leaves = @()
        foreach ($Key in $Value.Keys) {
            $Leaves += Get-StringLeaves $Value[$Key]
        }
        return $Leaves
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $Leaves = @()
        foreach ($Item in $Value) {
            $Leaves += Get-StringLeaves $Item
        }
        return $Leaves
    }
    $Properties = @($Value.PSObject.Properties)
    if ($Properties.Count -gt 0) {
        $Leaves = @()
        foreach ($Property in $Properties) {
            $Leaves += Get-StringLeaves $Property.Value
        }
        return $Leaves
    }
    return @()
}

function Assert-NewArtifactNamespace {
    $ConfigFiles = @(
        Get-ChildItem "experiments\configs\patient_indexed_specimen_routing_*.json"
    )
    foreach ($ConfigFile in $ConfigFiles) {
        $Payload = Get-Content $ConfigFile.FullName -Raw | ConvertFrom-Json
        foreach ($Leaf in @(Get-StringLeaves $Payload)) {
            $Normalized = $Leaf.Replace("/", "\")
            $IsFrozenTeacherReference = (
                $ConfigFile.Name -eq $FrozenTeacherConfigName -and
                $Normalized -in @(
                    "$FrozenTeacherSourceRoot\teacher_cache.npz",
                    $FrozenTeacherSourceRoot
                )
            )
            if (
                $Normalized -match "^results\\" -and
                -not $Normalized.StartsWith("$ResultRoot\") -and
                -not $IsFrozenTeacherReference
            ) {
                throw (
                    "Routing config references a forbidden legacy artifact: " +
                    "$($ConfigFile.Name): $Leaf"
                )
            }
        }
    }
}

function Assert-CudaReady {
    Invoke-CheckedPython -Stage "CUDA verification" -Arguments @(
        "scripts\verify_cuda.py"
    )
}

function Assert-FrozenMechanicsReport {
    Assert-RequiredFile $MechanicsReport
    $ActualHash = (
        Get-FileHash -Algorithm SHA256 $MechanicsReport
    ).Hash.ToLowerInvariant()
    if ($ActualHash -ne $MacMechanicsReportSha256) {
        throw "Frozen Mac mechanics report hash mismatch: $ActualHash"
    }
    $Gate = Get-Content $MechanicsReport -Raw | ConvertFrom-Json
    if ($Gate.status -ne "PASS") {
        throw "Frozen Mac mechanics/headroom gate did not pass."
    }
}

function Assert-MacValidationEvidence {
    Assert-RequiredFile $MacValidationEvidence
    $ActualHash = (
        Get-FileHash -Algorithm SHA256 $MacValidationEvidence
    ).Hash.ToLowerInvariant()
    if ($ActualHash -ne $MacValidationEvidenceSha256) {
        throw "Frozen Mac validation evidence hash mismatch: $ActualHash"
    }
}

function Get-TrainingRunDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][string]$Algorithm,
        [Parameter(Mandatory = $true)][int]$Seed
    )
    return Join-Path $ResultRoot (
        "training\$RunName\$Algorithm\seed$Seed"
    )
}

function Assert-TrainingRun {
    param(
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][string]$Algorithm,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Episodes
    )

    $RunDirectory = Get-TrainingRunDirectory $RunName $Algorithm $Seed
    $CsvPath = Join-Path $RunDirectory "training.csv"
    $SummaryPath = Join-Path $RunDirectory "summary.json"
    Assert-RequiredFile $CsvPath
    Assert-RequiredFile $SummaryPath
    $Rows = @(Import-Csv $CsvPath)
    if ($Rows.Count -ne $Episodes) {
        throw "$RunName $Algorithm seed $Seed has $($Rows.Count), expected $Episodes rows."
    }
    $OnlineUpdates = 0
    $RouteCount = 0.0
    foreach ($Row in $Rows) {
        $OnlineUpdates += [int](
            Assert-FiniteNumber $Row.online_rl_updates "online_rl_updates"
        )
        $RouteCount += Assert-FiniteNumber `
            $Row.specimen_route_count `
            "specimen_route_count"
        foreach ($Property in @($Row.PSObject.Properties)) {
            if (
                $Property.Name -match "(?:cost|loss|drift)" -and
                $Property.Value -ne ""
            ) {
                Assert-FiniteNumber `
                    $Property.Value `
                    "$RunName/$Algorithm/seed$Seed/$($Property.Name)" | Out-Null
            }
        }
    }
    if ($OnlineUpdates -le 0) {
        throw "$RunName $Algorithm seed $Seed completed with zero online updates."
    }
    if ($RouteCount -le 0.0) {
        throw "$RunName $Algorithm seed $Seed produced no specimen routes."
    }

    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $Drift = Assert-FiniteNumber `
        $Summary.actor_drift_from_pretrain.rms `
        "actor_drift_from_pretrain.rms"
    if ($Drift -le 0.0) {
        throw "$RunName $Algorithm seed $Seed has zero actor drift after online training."
    }
    $CheckpointRoot = Join-Path $RunDirectory "checkpoints"
    foreach ($Path in @(
        (Join-Path $CheckpointRoot "${Algorithm}_seed${Seed}_pretrain.pt"),
        (Join-Path $CheckpointRoot "${Algorithm}_seed${Seed}_episode${Episodes}.pt"),
        (Join-Path $CheckpointRoot "${Algorithm}_seed${Seed}_training_state.pt")
    )) {
        Assert-RequiredFile $Path
    }
}

function Assert-ParameterMatch {
    param([Parameter(Mandatory = $true)][string]$ManifestPath)
    Assert-RequiredFile $ManifestPath
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    foreach ($Seed in 0..2) {
        $Graph = @(
            $Manifest.runs | Where-Object {
                $_.algorithm -eq $GraphAlgorithm -and [int]$_.seed -eq $Seed
            }
        )
        $Flat = @(
            $Manifest.runs | Where-Object {
                $_.algorithm -eq $FlatAlgorithm -and [int]$_.seed -eq $Seed
            }
        )
        if ($Graph.Count -ne 1 -or $Flat.Count -ne 1) {
            throw "Missing matched GCN/flat manifest rows for seed $Seed."
        }
        $GraphCount = [double]$Graph[0].parameter_count
        $FlatCount = [double]$Flat[0].parameter_count
        $Gap = [Math]::Abs($GraphCount - $FlatCount) /
            [Math]::Max($GraphCount, $FlatCount)
        if ($Gap -gt 0.01) {
            throw "GCN/flat parameter gap exceeds 1% for seed ${Seed}: $Gap"
        }
    }
}

function Write-PhaseProvenance {
    param(
        [Parameter(Mandatory = $true)][string]$CompletedPhase,
        [Parameter(Mandatory = $true)][string]$Timestamp
    )
    $ProvenanceRoot = Join-Path $ResultRoot "provenance"
    New-Item -ItemType Directory -Force $ProvenanceRoot | Out-Null
    $ConfigHashes = [ordered]@{}
    foreach ($Config in @(
        Get-ChildItem "experiments\configs\patient_indexed_specimen_routing_*.json"
    )) {
        $ConfigHashes[$Config.FullName.Substring($RepoRoot.Path.Length + 1)] = (
            Get-FileHash -Algorithm SHA256 $Config.FullName
        ).Hash.ToLowerInvariant()
    }
    $TeacherManifest = $null
    if (Test-Path -PathType Leaf $RoutingTeacherManifest) {
        $TeacherManifest = Get-Content $RoutingTeacherManifest -Raw |
            ConvertFrom-Json
    }
    $Payload = [ordered]@{
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        phase = $CompletedPhase
        git_branch = $Branch
        git_commit = $Commit
        parent_commit = $LockedParent
        python_executable = $Python
        python_version = $PythonVersion
        python_sha256 = $PythonSha256
        config_sha256 = $ConfigHashes
        result_root = $ResultRoot
        recovery = $RecoveryName
        supersedes_failed_commit = $SupersededCommit
        superseded_result_root = $SupersededResultRoot
        superseded_outputs_reused = $false
        superseded_failure_evidence_sha256 = $SupersededFailureEvidenceSha256
        failure_classification = $FailureClassification
        launcher_control_process_ids = $ExplicitControlProcessIds
        frozen_teacher_config_source_root = $FrozenTeacherSourceRoot
        validation_origin = "frozen Mac evidence"
        mac_validation_commit = $MacValidationCommit
        mac_validation_evidence = $MacValidationEvidence
        mac_validation_evidence_sha256 = $MacValidationEvidenceSha256
        mac_mechanics_report_sha256 = $MacMechanicsReportSha256
        prior_recovery2_commit = $PriorRecovery2Commit
        prior_recovery2_result_root = $PriorRecovery2ResultRoot
        prior_recovery2_transcript_sha256 = $PriorRecovery2TranscriptSha256
        teacher_origin = if ($null -ne $TeacherManifest) {
            "frozen Mac CPU shard bundle"
        } else {
            $null
        }
        teacher_bundle_sha256 = if ($null -ne $TeacherManifest) {
            $TeacherManifest.bundle_sha256
        } else {
            $null
        }
        teacher_source_commit = if ($null -ne $TeacherManifest) {
            $TeacherManifest.provenance.git_commit
        } else {
            $null
        }
        routing_name = (
            "patient-indexed, identity-preserving, pre-manufacturing specimen routing"
        )
        formal_training_started = (
            $CompletedPhase -eq "Pilot" -or $CompletedPhase -eq "Evaluate"
        )
        formal_evaluation_started = ($CompletedPhase -eq "Evaluate")
        formal_no_routing_training = $false
        manuscript_protocol_wording_updated = $true
        manuscript_results_inserted = $false
    }
    $Path = Join-Path $ProvenanceRoot (
        "${CompletedPhase}_$Timestamp.json"
    )
    $Payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $Path
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$ExplicitControlProcessIds = @(
    $ControlProcessIds.Split(",") |
        ForEach-Object { [int]$_ } |
        Sort-Object -Unique
)
$Python = if ($PythonExecutable) {
    [string](Resolve-Path $PythonExecutable)
} else {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -PathType Leaf $Python)) {
    throw "Missing CUDA virtual environment Python: $Python"
}
if ($Phase -eq "ImportTeacher" -and -not $TeacherBundle) {
    throw "ImportTeacher requires -TeacherBundle."
}
if ($Phase -ne "ImportTeacher" -and $TeacherBundle) {
    throw "-TeacherBundle is only valid for ImportTeacher."
}
$PythonVersion = ((& $Python --version 2>&1) | Out-String).Trim()
$PythonSha256 = (
    Get-FileHash -Algorithm SHA256 $Python
).Hash.ToLowerInvariant()

$Branch = (git branch --show-current).Trim()
$Commit = (git rev-parse HEAD).Trim()
if ($Branch -ne $LockedBranch) {
    throw "Expected branch $LockedBranch but found $Branch."
}
if ($Commit -ne $ExpectedCommit) {
    throw "Expected commit $ExpectedCommit but found $Commit."
}
git merge-base --is-ancestor $LockedParent $Commit
if ([int32]$LASTEXITCODE -ne 0) {
    throw "Locked parent $LockedParent is not an ancestor of $Commit."
}
$DirtyPaths = @(git status --porcelain --untracked-files=no)
if ($DirtyPaths.Count -gt 0) {
    throw "Commit or stash tracked source changes before a locked routing stage."
}
Assert-NewArtifactNamespace

$ProcessSnapshot = @(Get-CimInstance Win32_Process)
$RelatedProcesses = @(
    Get-IndependentRelatedProcesses `
        -ProcessSnapshot $ProcessSnapshot `
        -CurrentProcessId ([int]$PID) `
        -ControlProcessIds $ExplicitControlProcessIds
)
if ($RelatedProcesses.Count -gt 0) {
    $Details = ($RelatedProcesses | ForEach-Object {
        "PID=$($_.ProcessId) PPID=$($_.ParentProcessId) $($_.CommandLine)"
    }) -join "`n"
    throw "Another related routing process is active:`n$Details"
}

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONFAULTHANDLER = "1"
$env:TORCH_SHOW_CPP_STACKTRACES = "1"
$env:PYTHONPYCACHEPREFIX = Join-Path $ResultRoot "python-cache"
$env:MPLCONFIGDIR = Join-Path $ResultRoot "matplotlib-cache"
New-Item -ItemType Directory -Force $env:PYTHONPYCACHEPREFIX | Out-Null
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null
$LogRoot = Join-Path $ResultRoot "logs"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogRoot "${Phase}_$Timestamp.txt"

$MechanicsReport = Join-Path $ResultRoot "mechanics_gate\report.json"
$RoutingTeacher = Join-Path $ResultRoot "teachers\routing\teacher_cache.npz"
$RoutingTeacherManifest = Join-Path $ResultRoot (
    "teachers\routing\bundle_manifest.json"
)
$SmokeGate = Join-Path $ResultRoot "gates\smoke_gate.json"
$PilotGate = Join-Path $ResultRoot "gates\pilot_gate.json"

$RoutingSmokeName = "patient_indexed_specimen_routing_smoke_routing"
$RoutingPilotName = "patient_indexed_specimen_routing_pilot_routing"

Start-Transcript -Path $LogPath
try {
    switch ($Phase) {
        "Preflight" {
            Assert-FreshPath $MechanicsReport
            Assert-MacValidationEvidence
            Invoke-CheckedPython -Stage "verify frozen Mac validation evidence" -Arguments @(
                "-m", "evaluation.verify_patient_indexed_specimen_routing_validation",
                "--evidence", $MacValidationEvidence,
                "--output", $MechanicsReport,
                "--expected-commit", $Commit
            )
            Assert-FrozenMechanicsReport
            Assert-CudaReady
        }
        "ImportTeacher" {
            Assert-FrozenMechanicsReport
            if (-not $TeacherBundle) {
                throw "ImportTeacher requires -TeacherBundle."
            }
            $ResolvedTeacherBundle = [string](Resolve-Path $TeacherBundle)
            Assert-RequiredFile $ResolvedTeacherBundle
            Assert-FreshPath (Split-Path $RoutingTeacher -Parent)
            Invoke-CheckedPython -Stage "verify and import frozen Mac teacher" -Arguments @(
                "-m", "evaluation.headroom_teacher_bundle", "extract",
                "--bundle", $ResolvedTeacherBundle,
                "--destination", (Split-Path $RoutingTeacher -Parent),
                "--config", (
                    "experiments\configs\" +
                    "patient_indexed_specimen_routing_teacher_routing.json"
                )
            )
            Assert-RequiredFile $RoutingTeacher
            Assert-RequiredFile $RoutingTeacherManifest
        }
        "Smoke" {
            Assert-FrozenMechanicsReport
            Assert-RequiredFile $RoutingTeacher
            Assert-FreshPath (Join-Path $ResultRoot "training\$RoutingSmokeName")
            Assert-FreshPath $SmokeGate
            Assert-CudaReady
            $Jobs = @(
                @("patient_indexed_specimen_routing_smoke_routing.json", $GraphAlgorithm),
                @("patient_indexed_specimen_routing_smoke_routing.json", $FlatAlgorithm)
            )
            foreach ($Job in $Jobs) {
                $Config = [string]$Job[0]
                $Algorithm = [string]$Job[1]
                Invoke-CheckedPython -Stage "smoke $Algorithm $Config" -Arguments @(
                    "-m", "evaluation.train_multiscenario_network_residual",
                    "--config", (Join-Path "experiments\configs" $Config),
                    "--algorithm", $Algorithm,
                    "--seed", "0"
                )
                Assert-TrainingRun $RoutingSmokeName $Algorithm 0 5
            }
            $RoutingManifest = Join-Path $ResultRoot (
                "training\$RoutingSmokeName\training_manifest.json"
            )
            Assert-ParameterMatch $RoutingManifest
            New-Item -ItemType Directory -Force (Split-Path $SmokeGate -Parent) |
                Out-Null
            [ordered]@{
                status = "PASS"
                completed_at = (Get-Date).ToUniversalTime().ToString("o")
                git_commit = $Commit
                learned_runs = 2
                episodes_per_run = 5
                tuning_from_smoke_permitted = $false
            } | ConvertTo-Json | Set-Content -Encoding UTF8 $SmokeGate
        }
        "Pilot" {
            if (-not $ApprovePilot) {
                throw "Pilot requires explicit -ApprovePilot."
            }
            Assert-RequiredFile $SmokeGate
            $Smoke = Get-Content $SmokeGate -Raw | ConvertFrom-Json
            if ($Smoke.status -ne "PASS" -or $Smoke.git_commit -ne $Commit) {
                throw "Smoke gate is absent, failed, or from another commit."
            }
            Assert-FreshPath (Join-Path $ResultRoot "training\$RoutingPilotName")
            Assert-FreshPath $PilotGate
            Assert-CudaReady
            $PilotSpecs = @()
            foreach ($Algorithm in @($GraphAlgorithm, $FlatAlgorithm)) {
                foreach ($Seed in 0..2) {
                    $PilotSpecs += ,@(
                        "patient_indexed_specimen_routing_pilot_routing.json",
                        $RoutingPilotName,
                        $Algorithm,
                        $Seed
                    )
                }
            }
            foreach ($Spec in $PilotSpecs) {
                $Config = [string]$Spec[0]
                $RunName = [string]$Spec[1]
                $Algorithm = [string]$Spec[2]
                $Seed = [int]$Spec[3]
                Invoke-CheckedPython -Stage "pilot $RunName $Algorithm seed $Seed" -Arguments @(
                    "-m", "evaluation.train_multiscenario_network_residual",
                    "--config", (Join-Path "experiments\configs" $Config),
                    "--algorithm", $Algorithm,
                    "--seed", [string]$Seed
                )
                Assert-TrainingRun $RunName $Algorithm $Seed 100
            }
            Assert-ParameterMatch (Join-Path $ResultRoot (
                "training\$RoutingPilotName\training_manifest.json"
            ))
            New-Item -ItemType Directory -Force (Split-Path $PilotGate -Parent) |
                Out-Null
            [ordered]@{
                status = "PASS"
                completed_at = (Get-Date).ToUniversalTime().ToString("o")
                git_commit = $Commit
                learned_runs = 6
                episodes_per_run = 100
                checkpoint_interval = 5
            } | ConvertTo-Json | Set-Content -Encoding UTF8 $PilotGate
        }
        "Evaluate" {
            Assert-RequiredFile $PilotGate
            $Pilot = Get-Content $PilotGate -Raw | ConvertFrom-Json
            if ($Pilot.status -ne "PASS" -or $Pilot.git_commit -ne $Commit) {
                throw "Pilot gate is absent, failed, or from another commit."
            }
            $EvaluationRoots = @(
                "evaluation\routing_final",
                "evaluation\routing_pretrain",
                "evaluation\routing_lead0_sensitivity",
                "evaluation\routing_return1_sensitivity",
                "analysis\routing_attribution.json"
            )
            foreach ($RelativePath in $EvaluationRoots) {
                Assert-FreshPath (Join-Path $ResultRoot $RelativePath)
            }
            Assert-CudaReady
            foreach ($Config in @(
                "patient_indexed_specimen_routing_pilot_routing_eval.json",
                "patient_indexed_specimen_routing_pilot_routing_pretrain_eval.json",
                "patient_indexed_specimen_routing_lead0_sensitivity_eval.json",
                "patient_indexed_specimen_routing_return1_sensitivity_eval.json"
            )) {
                Invoke-CheckedPython -Stage "locked CRN evaluation $Config" -Arguments @(
                    "-m", "evaluation.evaluate_multiscenario_network_residual",
                    "--config", (Join-Path "experiments\configs" $Config)
                )
            }
            Invoke-CheckedPython -Stage "paired routing attribution" -Arguments @(
                "-m", "evaluation.compare_patient_indexed_specimen_routing",
                "--config",
                "experiments\configs\patient_indexed_specimen_routing_attribution.json"
            )
        }
    }
}
finally {
    Stop-Transcript
}

Write-PhaseProvenance -CompletedPhase $Phase -Timestamp $Timestamp
Write-Host ""
Write-Host "Patient-indexed specimen-routing phase completed: $Phase"
Write-Host "Commit: $Commit"
Write-Host "Transcript: $LogPath"
