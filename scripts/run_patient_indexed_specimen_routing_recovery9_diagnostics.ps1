param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit,
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9]+(?:,[0-9]+)*$")]
    [string]$ControlProcessIds,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LockedBranch = "codex/patient-indexed-specimen-routing"
$Recovery7Commit = "30fe642f641e0592abc59009ae3f154019a14145"
$Recovery8Commit = "d37870f040117eba3c76cfe4d9e128006cd4cee8"
$Recovery7Root = "results\patient_indexed_specimen_routing_recovery7"
$Recovery8Root = "results\patient_indexed_specimen_routing_recovery8"
$Recovery9Root = "results\patient_indexed_specimen_routing_recovery9"
$Algorithm = "gcn_residual_mdl2_network_ddpg_afd"
$Seed = 0
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$SmokeConfig = "experiments\configs\patient_indexed_specimen_routing_smoke_routing.json"
$BenchmarkConfig = "experiments\configs\patient_indexed_specimen_routing_benchmark.json"
$RunRoot = Join-Path $Recovery7Root (
    "training\patient_indexed_specimen_routing_smoke_routing\" +
    "$Algorithm\seed0"
)
$CheckpointRoot = Join-Path $RunRoot "checkpoints"
$TrainingState = Join-Path $CheckpointRoot (
    "${Algorithm}_seed0_training_state.pt"
)
$EpisodeOneCheckpoint = Join-Path $CheckpointRoot (
    "${Algorithm}_seed0_episode1.pt"
)
$PretrainCheckpoint = Join-Path $CheckpointRoot (
    "${Algorithm}_seed0_pretrain.pt"
)
$TrainingCsv = Join-Path $RunRoot "training.csv"
$SmokeGate = Join-Path $Recovery7Root "gates\smoke_gate.json"
$FlatRoot = Join-Path $Recovery7Root (
    "training\patient_indexed_specimen_routing_smoke_routing\" +
    "flat_residual_mdl2_network_ddpg_afd"
)
$DiagnosticRoot = Join-Path $Recovery9Root "diagnostics\windows_native_crash"
$EvidenceRoot = Join-Path $DiagnosticRoot "evidence"
$DiagnosticGate = Join-Path $Recovery9Root (
    "gates\windows_native_crash_diagnostic_gate.json"
)

$AllowedChangedPaths = @(
    "evaluation/diagnose_patient_indexed_specimen_routing_recovery8.py",
    "evaluation/verify_patient_indexed_specimen_routing_validation.py",
    "scripts/invoke_patient_indexed_specimen_routing_recovery8_diagnostics_detached.ps1",
    "scripts/invoke_patient_indexed_specimen_routing_recovery9_diagnostics_detached.ps1",
    "scripts/run_patient_indexed_specimen_routing.ps1",
    "scripts/run_patient_indexed_specimen_routing_recovery8_diagnostics.ps1",
    "scripts/run_patient_indexed_specimen_routing_recovery9_diagnostics.ps1",
    "scripts/start_patient_indexed_specimen_routing_phase.ps1",
    "scripts/start_patient_indexed_specimen_routing_recovery8_diagnostics.ps1",
    "scripts/start_patient_indexed_specimen_routing_recovery9_diagnostics.ps1",
    "scripts/invoke_patient_indexed_specimen_routing_phase_detached.ps1",
    "tests/test_patient_indexed_specimen_routing_recovery8_diagnostics.py"
)

$ScientificSourcePaths = @(
    $SmokeConfig,
    $BenchmarkConfig,
    "configs\gcn_residual_20_clinic.yaml",
    "evaluation\train_multiscenario_network_residual.py",
    "evaluation\train_network_residual_history_screen.py",
    "src\baselines\heuristics.py",
    "src\graph\edges.py",
    "src\models\gcn_ddpg.py",
    "src\models\graph_features.py",
    "src\rl\experiment.py",
    "src\rl\replay_buffer.py",
    "src\rl\training_state.py"
)

$LayerIterations = [ordered]@{
    complete_undirected_edges = 250000
    facility_net_action_from_state = 25000
    batched_base_action = 1000
    tensor_cpu_numpy = 50000
    cpu_actor_critic_update = 128
    cuda_actor_critic_update = 256
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][object]$Value,
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Depth = 12
    )
    if (Test-Path $Path) {
        throw "Refusing to overwrite Recovery 9 evidence: $Path"
    }
    $Parent = Split-Path $Path -Parent
    if (-not (Test-Path -PathType Container $Parent)) {
        New-Item -ItemType Directory -Force $Parent | Out-Null
    }
    $Temporary = "$Path.tmp.$PID"
    $Value | ConvertTo-Json -Depth $Depth |
        Set-Content -Encoding UTF8 $Temporary
    Move-Item $Temporary $Path
}

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -PathType Leaf $Path)) {
        throw "Missing frozen Recovery 7 input: $Path"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLowerInvariant()
}

function Get-ExitDetails {
    param([Parameter(Mandatory = $true)][int32]$Signed)
    $Unsigned = [BitConverter]::ToUInt32(
        [BitConverter]::GetBytes($Signed),
        0
    )
    return [ordered]@{
        signed = $Signed
        unsigned = [uint32]$Unsigned
        hex = ("0x{0:X8}" -f $Unsigned)
    }
}

function Get-IndependentRelatedProcesses {
    param(
        [Parameter(Mandatory = $true)][object[]]$ProcessSnapshot,
        [Parameter(Mandatory = $true)][int]$CurrentProcessId,
        [int[]]$ExcludedControlProcessIds = @()
    )
    $ProcessById = @{}
    foreach ($Process in $ProcessSnapshot) {
        $ProcessById[[int]$Process.ProcessId] = $Process
    }
    $Excluded = @(
        @($CurrentProcessId) + @($ExcludedControlProcessIds) |
            Sort-Object -Unique
    )
    $Cursor = $CurrentProcessId
    while ($ProcessById.ContainsKey($Cursor)) {
        $Parent = [int]$ProcessById[$Cursor].ParentProcessId
        if ($Parent -le 0 -or $Excluded -contains $Parent) {
            break
        }
        $Excluded += $Parent
        $Cursor = $Parent
    }
    return @(
        $ProcessSnapshot | Where-Object {
            $Name = [string]$_.Name
            $CommandLine = [string]$_.CommandLine
            $IsRoutingPython = (
                $Name -match "(?i)^python(?:w)?\.exe$" -and
                $CommandLine -and (
                    $CommandLine -match "patient_indexed_specimen_routing" -or
                    $CommandLine -match "train_multiscenario_network_residual"
                )
            )
            $IsRoutingPowerShell = (
                $Name -match "(?i)^(?:powershell|pwsh)\.exe$" -and
                $CommandLine -and
                $CommandLine -match "patient_indexed_specimen_routing"
            )
            -not ($Excluded -contains [int]$_.ProcessId) -and
                ($IsRoutingPython -or $IsRoutingPowerShell)
        }
    )
}

function Get-FrozenInputManifest {
    $Files = @(
        Get-ChildItem -Path $Recovery7Root -File -Recurse -ErrorAction Stop
    )
    $Files += @(
        Get-ChildItem -Path $Recovery8Root -File -Recurse -ErrorAction Stop
    )
    foreach ($RelativePath in $ScientificSourcePaths) {
        Assert-RequiredFile $RelativePath
        $Files += Get-Item $RelativePath
    }
    $Entries = @()
    foreach ($File in @($Files | Sort-Object FullName -Unique)) {
        $RootPrefix = ([IO.Path]::GetFullPath([string]$RepoRoot)).TrimEnd("\") + "\"
        $FullPath = [IO.Path]::GetFullPath([string]$File.FullName)
        if (-not $FullPath.StartsWith(
            $RootPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Frozen input is outside the repository: $FullPath"
        }
        $Relative = $FullPath.Substring($RootPrefix.Length).Replace("\", "/")
        $Entries += [ordered]@{
            path = $Relative
            length = [int64]$File.Length
            sha256 = Get-Sha256 $File.FullName
        }
    }
    return @($Entries)
}

function Get-ManifestDigest {
    param([Parameter(Mandatory = $true)][object[]]$Manifest)
    $Json = $Manifest | ConvertTo-Json -Depth 5 -Compress
    $Bytes = [Text.Encoding]::UTF8.GetBytes($Json)
    $Stream = [IO.MemoryStream]::new($Bytes)
    try {
        return (Get-FileHash -Algorithm SHA256 -InputStream $Stream).Hash.ToLowerInvariant()
    } finally {
        $Stream.Dispose()
    }
}

function Assert-FrozenInputManifest {
    param(
        [Parameter(Mandatory = $true)][object[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Stage
    )
    $Actual = @(Get-FrozenInputManifest)
    $ExpectedDigest = Get-ManifestDigest $Expected
    $ActualDigest = Get-ManifestDigest $Actual
    if ($ActualDigest -ne $ExpectedDigest) {
        throw (
            "Frozen Recovery 7/8/source input hash mismatch after $Stage; " +
            "expected=$ExpectedDigest actual=$ActualDigest"
        )
    }
    return $ActualDigest
}

function Get-NewWerEvents {
    param([Parameter(Mandatory = $true)][datetime]$StartedAt)
    try {
        return @(
            Get-WinEvent -FilterHashtable @{
                LogName = "Application"
                StartTime = $StartedAt
            } -ErrorAction Stop |
                Where-Object {
                    $_.ProviderName -match (
                        "(?i)Windows Error Reporting|Application Error"
                    ) -and
                    $_.Message -match "(?i)python|BEX64|0xc0000005"
                } |
                Sort-Object RecordId |
                ForEach-Object {
                    [ordered]@{
                        record_id = [int64]$_.RecordId
                        time_created = $_.TimeCreated.ToUniversalTime().ToString("o")
                        provider = [string]$_.ProviderName
                        event_id = [int]$_.Id
                        message = [string]$_.Message
                    }
                }
        )
    } catch {
        return @(
            [ordered]@{
                collection_error = $_.Exception.Message
            }
        )
    }
}

function Get-NewCrashArtifacts {
    param([Parameter(Mandatory = $true)][datetime]$StartedAt)
    $Roots = @(
        (Join-Path $env:LOCALAPPDATA "CrashDumps"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\Windows\WER"),
        (Join-Path $env:ProgramData "Microsoft\Windows\WER"),
        "C:\Windows\Minidump"
    )
    $Artifacts = @()
    foreach ($Root in $Roots | Select-Object -Unique) {
        if (-not (Test-Path -PathType Container $Root)) {
            continue
        }
        try {
            foreach ($File in @(
                Get-ChildItem $Root -File -Recurse -ErrorAction Stop |
                    Where-Object {
                        $_.LastWriteTimeUtc -ge $StartedAt.ToUniversalTime() -and
                        $_.Extension -match "(?i)^\.(?:dmp|wer|xml|txt)$"
                    }
            )) {
                $Artifacts += [ordered]@{
                    path = [string]$File.FullName
                    length = [int64]$File.Length
                    last_write_time_utc = $File.LastWriteTimeUtc.ToString("o")
                    sha256 = Get-Sha256 $File.FullName
                }
            }
        } catch {
            $Artifacts += [ordered]@{
                root = [string]$Root
                collection_error = $_.Exception.Message
            }
        }
    }
    return @($Artifacts)
}

function Invoke-DiagnosticLayer {
    param(
        [Parameter(Mandatory = $true)][string]$Layer,
        [Parameter(Mandatory = $true)][int]$Iterations,
        [Parameter(Mandatory = $true)][object[]]$FrozenManifest,
        [Parameter(Mandatory = $true)][int]$Ordinal
    )
    $LayerRoot = Join-Path $DiagnosticRoot ("{0:D2}_{1}" -f $Ordinal, $Layer)
    if (Test-Path $LayerRoot) {
        throw "Refusing to reuse Recovery 9 diagnostic layer: $LayerRoot"
    }
    New-Item -ItemType Directory $LayerRoot | Out-Null
    $Stdout = Join-Path $LayerRoot "stdout.log"
    $Stderr = Join-Path $LayerRoot "stderr.log"
    $Result = Join-Path $LayerRoot "result.json"
    $Status = Join-Path $LayerRoot "status.json"
    $Arguments = @(
        "-X", "faulthandler",
        "-m", "evaluation.diagnose_patient_indexed_specimen_routing_recovery8",
        "--layer", $Layer,
        "--training-state", $TrainingState,
        "--smoke-config", $SmokeConfig,
        "--algorithm", $Algorithm,
        "--seed", [string]$Seed,
        "--recovery-number", "9",
        "--iterations", [string]$Iterations,
        "--output", $Result
    )
    $StartedAt = Get-Date
    $Process = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru
    $Win32Process = Get-CimInstance Win32_Process `
        -Filter "ProcessId = $($Process.Id)" `
        -ErrorAction SilentlyContinue
    $Process.WaitForExit()
    $CompletedAt = Get-Date
    $Exit = Get-ExitDetails ([int32]$Process.ExitCode)
    $WerEvents = @(Get-NewWerEvents $StartedAt)
    $CrashArtifacts = @(Get-NewCrashArtifacts $StartedAt)
    $FrozenDigest = Assert-FrozenInputManifest `
        -Expected $FrozenManifest `
        -Stage $Layer
    $ResultStatus = "MISSING"
    $ResultSha256 = ""
    if (Test-Path -PathType Leaf $Result) {
        $ResultSha256 = Get-Sha256 $Result
        try {
            $ResultStatus = [string](
                (Get-Content $Result -Raw | ConvertFrom-Json).status
            )
        } catch {
            $ResultStatus = "INVALID"
        }
    }
    $StatusPayload = [ordered]@{
        schema_version = 1
        layer = $Layer
        ordinal = $Ordinal
        iterations = $Iterations
        state = $(
            if ($Exit.signed -eq 0 -and $ResultStatus -eq "PASS") {
                "completed"
            } else {
                "failed"
            }
        )
        started_at = $StartedAt.ToUniversalTime().ToString("o")
        completed_at = $CompletedAt.ToUniversalTime().ToString("o")
        parent_pid = [int]$PID
        child_pid = [int]$Process.Id
        observed_ppid = $(
            if ($null -eq $Win32Process) { $null }
            else { [int]$Win32Process.ParentProcessId }
        )
        command_line = $(
            if ($null -eq $Win32Process) { "" }
            else { [string]$Win32Process.CommandLine }
        )
        exit = $Exit
        stdout = [string]$Stdout
        stdout_sha256 = Get-Sha256 $Stdout
        stderr = [string]$Stderr
        stderr_sha256 = Get-Sha256 $Stderr
        result = [string]$Result
        result_status = $ResultStatus
        result_sha256 = $ResultSha256
        frozen_input_manifest_sha256 = $FrozenDigest
        frozen_inputs_unchanged = $true
        wer_events = $WerEvents
        crash_artifacts = $CrashArtifacts
    }
    Write-AtomicJson -Value $StatusPayload -Path $Status
    if ($Exit.signed -ne 0 -or $ResultStatus -ne "PASS") {
        throw (
            "Recovery 9 layer $Layer failed: signed=$($Exit.signed), " +
            "unsigned=$($Exit.unsigned), hex=$($Exit.hex), " +
            "result_status=$ResultStatus"
        )
    }
    return $StatusPayload
}

if ((git branch --show-current).Trim() -ne $LockedBranch) {
    throw "Recovery 9 diagnostics require branch $LockedBranch"
}
if ((git rev-parse HEAD).Trim() -ne $ExpectedCommit) {
    throw "Recovery 9 diagnostic commit mismatch"
}
git merge-base --is-ancestor $Recovery8Commit $ExpectedCommit
if ($LASTEXITCODE -ne 0) {
    throw "Recovery 9 diagnostic commit is not descended from Recovery 8"
}
$ChangedPaths = @(
    git diff --name-only $Recovery7Commit $ExpectedCommit |
        ForEach-Object { ([string]$_).Replace("\", "/") } |
        Where-Object { $_ }
)
$UnexpectedPaths = @(
    $ChangedPaths | Where-Object { $AllowedChangedPaths -notcontains $_ }
)
if ($UnexpectedPaths.Count -gt 0) {
    throw (
        "Recovery 9 changed scientific or unauthorized paths: " +
        ($UnexpectedPaths -join ", ")
    )
}
if (@(git status --porcelain --untracked-files=no).Count -ne 0) {
    throw "Recovery 9 requires a clean tracked worktree"
}
if (-not (Test-Path -PathType Leaf $PythonExecutable)) {
    throw "Missing locked Python executable: $PythonExecutable"
}
foreach ($Path in @(
    $TrainingState,
    $EpisodeOneCheckpoint,
    $PretrainCheckpoint,
    $TrainingCsv,
    $SmokeConfig,
    $BenchmarkConfig
)) {
    Assert-RequiredFile $Path
}
if (Test-Path $SmokeGate) {
    throw "Recovery 7 unexpectedly contains a PASS Smoke gate"
}
if (Test-Path $FlatRoot) {
    throw "Recovery 7 unexpectedly started matched-flat Smoke"
}
if ((Test-Path $DiagnosticRoot) -or (Test-Path $DiagnosticGate)) {
    throw "Refusing to reuse existing Recovery 9 diagnostic outputs"
}
if (-not (Test-Path -PathType Container $Recovery8Root)) {
    throw "Missing immutable Recovery 8 failure evidence"
}
$Recovery8Evidence = @(
    Get-ChildItem $Recovery8Root -File -Recurse -ErrorAction Stop
)
if ($Recovery8Evidence.Count -lt 4) {
    throw "Recovery 8 failure evidence is incomplete"
}
if (Test-Path -PathType Container $Recovery9Root) {
    $UnexpectedRecovery9Entries = @(
        Get-ChildItem $Recovery9Root -Force -ErrorAction Stop |
            Where-Object { $_.Name -ne "launcher-logs" }
    )
    if ($UnexpectedRecovery9Entries.Count -ne 0) {
        throw "Recovery 9 root contains output outside the current launcher claim"
    }
} elseif (-not $PreflightOnly) {
    throw "Recovery 9 launcher claim root is missing"
}
$Rows = @(Import-Csv $TrainingCsv)
if ($Rows.Count -ne 1) {
    throw "Recovery 7 must contain exactly one completed GCN episode"
}
$Row = $Rows[0]
if ([int]$Row.episode -ne 0 -or [int]$Row.online_rl_updates -ne 52) {
    throw "Recovery 7 boundary mismatch: expected episode 1 and 52 updates"
}
foreach ($Metric in @("actor_loss", "critic_loss", "imitation_loss")) {
    $Name = "online_rl_${Metric}_final"
    if (-not ($Row.PSObject.Properties.Name -contains $Name)) {
        throw "Recovery 7 CSV is missing $Name"
    }
    $Value = [double]$Row.$Name
    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
        throw "Recovery 7 CSV contains non-finite $Name"
    }
}
$ControlIds = @(
    $ControlProcessIds.Split(",") | ForEach-Object { [int]$_ }
)
$Related = @(
    Get-IndependentRelatedProcesses `
        -ProcessSnapshot @(Get-CimInstance Win32_Process) `
        -CurrentProcessId $PID `
        -ExcludedControlProcessIds $ControlIds
)
if ($Related.Count -ne 0) {
    throw "Related routing/training processes already exist"
}
if ($PreflightOnly) {
    Write-Host (
        "Recovery 9 pre-claim binding preflight PASS; " +
        "no output root or launcher claim was created"
    )
    return
}

New-Item -ItemType Directory -Force $EvidenceRoot | Out-Null
$FrozenManifest = @(Get-FrozenInputManifest)
$FrozenDigest = Get-ManifestDigest $FrozenManifest
Write-AtomicJson `
    -Value ([ordered]@{
        schema_version = 1
        captured_at = (Get-Date).ToUniversalTime().ToString("o")
        recovery7_commit = $Recovery7Commit
        recovery7_root = $Recovery7Root
        superseded_recovery8_commit = $Recovery8Commit
        superseded_recovery8_root = $Recovery8Root
        manifest_sha256 = $FrozenDigest
        files = $FrozenManifest
    }) `
    -Path (Join-Path $EvidenceRoot "frozen_inputs.before.json")

$Incident = [ordered]@{
    schema_version = 1
    classification = "windows_native_access_violation"
    classification_source = "locked Recovery 7 PC audit"
    recovery7_commit = $Recovery7Commit
    recovery7_root = $Recovery7Root
    recovery7_outputs_read_only = $true
    recovery7_outputs_reused_for_training = $false
    superseded_recovery8_commit = $Recovery8Commit
    superseded_recovery8_root = $Recovery8Root
    superseded_recovery8_outputs_reused = $false
    superseded_recovery8_failure_classification = (
        "PowerShell parameter-binding failure before Python diagnostics"
    )
    completed_gcn_episodes = 1
    planned_gcn_episodes = 5
    online_rl_updates = 52
    native_exit_signed = -1073741819
    native_exit_unsigned = 3221225477
    native_exit_hex = "0xC0000005"
    wer_bucket = "BEX64"
    cuda_device = "NVIDIA GeForce RTX 4090"
    no_oom = $true
    no_driver_reset = $true
    no_cpu_fallback = $true
    no_hash_mismatch = $true
    recovery7_retry_or_resume_permitted = $false
    formal_training_started_by_diagnostics = $false
}
Write-AtomicJson `
    -Value $Incident `
    -Path (Join-Path $EvidenceRoot "recovery7_incident.json")

$Statuses = @()
$Ordinal = 0
foreach ($Entry in $LayerIterations.GetEnumerator()) {
    $Ordinal += 1
    $Statuses += Invoke-DiagnosticLayer `
        -Layer ([string]$Entry.Key) `
        -Iterations ([int]$Entry.Value) `
        -FrozenManifest $FrozenManifest `
        -Ordinal $Ordinal
}
$FinalManifest = @(Get-FrozenInputManifest)
$FinalDigest = Assert-FrozenInputManifest `
    -Expected $FrozenManifest `
    -Stage "all diagnostic layers"
Write-AtomicJson `
    -Value ([ordered]@{
        schema_version = 1
        captured_at = (Get-Date).ToUniversalTime().ToString("o")
        recovery7_commit = $Recovery7Commit
        recovery7_root = $Recovery7Root
        superseded_recovery8_commit = $Recovery8Commit
        superseded_recovery8_root = $Recovery8Root
        manifest_sha256 = $FinalDigest
        files = $FinalManifest
    }) `
    -Path (Join-Path $EvidenceRoot "frozen_inputs.after.json")

$GatePayload = [ordered]@{
    schema_version = 1
    status = "PASS"
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    diagnostic_commit = $ExpectedCommit
    recovery7_commit = $Recovery7Commit
    recovery7_root = $Recovery7Root
    superseded_recovery8_commit = $Recovery8Commit
    superseded_recovery8_root = $Recovery8Root
    superseded_recovery8_outputs_reused = $false
    recovery7_outputs_read_only = $true
    recovery7_outputs_reused_for_training = $false
    frozen_input_manifest_sha256 = $FrozenDigest
    frozen_inputs_unchanged = $true
    layers = @($Statuses | ForEach-Object { $_.layer })
    layer_count = $Statuses.Count
    formal_training_permitted = $false
    smoke_requires_separate_locked_commit = $true
    pilot_permitted = $false
}
Write-AtomicJson -Value $GatePayload -Path $DiagnosticGate
Write-Host (
    "Recovery 9 native-crash diagnostics PASS; " +
    "formal_training_permitted=false; gate=$DiagnosticGate"
)
