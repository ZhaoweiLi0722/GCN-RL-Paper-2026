param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Validate", "ImportTeacher", "Smoke", "Pilot", "Evaluate")]
    [string]$Phase,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit,
    [switch]$ApprovePilot,
    [string]$TeacherBundle = "",
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LockedBranch = "codex/patient-indexed-specimen-routing"
$ResultRootName = "results\patient_indexed_specimen_routing_recovery5"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Get-ControlProcessIds {
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
    $ProcessIds = @($CurrentProcessId)
    $CursorId = $CurrentProcessId
    while ($ProcessById.ContainsKey($CursorId)) {
        $ParentId = [int]$ProcessById[$CursorId].ParentProcessId
        if ($ParentId -le 0 -or $ProcessIds -contains $ParentId) {
            break
        }
        $ProcessIds += $ParentId
        $CursorId = $ParentId
    }
    return @($ProcessIds | Sort-Object -Unique)
}

$Branch = (git branch --show-current).Trim()
$Commit = (git rev-parse HEAD).Trim()
if ($Branch -ne $LockedBranch) {
    throw "Expected branch $LockedBranch but found $Branch."
}
if ($Commit -ne $ExpectedCommit) {
    throw "Expected commit $ExpectedCommit but found $Commit."
}
$DirtyPaths = @(git status --porcelain --untracked-files=no)
if ($DirtyPaths.Count -gt 0) {
    throw "Tracked worktree must be clean before a detached phase launch."
}
if ($PythonExecutable -and -not (Test-Path -PathType Leaf $PythonExecutable)) {
    throw "Missing requested Python executable: $PythonExecutable"
}
if ($Phase -eq "ImportTeacher" -and -not $TeacherBundle) {
    throw "ImportTeacher requires -TeacherBundle."
}
if ($Phase -ne "ImportTeacher" -and $TeacherBundle) {
    throw "-TeacherBundle is only valid for ImportTeacher."
}
if ($TeacherBundle -and -not (Test-Path -PathType Leaf $TeacherBundle)) {
    throw "Missing frozen teacher bundle: $TeacherBundle"
}

$ResultRoot = Join-Path $RepoRoot $ResultRootName
if ($Phase -eq "Validate" -and (Test-Path $ResultRoot)) {
    throw "Recovery 5 result root already exists; refusing to launch Validate."
}
if ($Phase -ne "Validate" -and -not (Test-Path -PathType Container $ResultRoot)) {
    throw "Recovery 5 result root does not exist for phase $Phase."
}

$LauncherRoot = Join-Path $ResultRoot "launcher-logs"
$ClaimPath = Join-Path $LauncherRoot "$Phase.claim.json"
if (Test-Path $ClaimPath) {
    throw "Phase $Phase already has a Recovery 5 launch claim."
}
New-Item -ItemType Directory -Force $LauncherRoot | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StandardOutputPath = Join-Path $LauncherRoot "${Phase}_${Timestamp}.stdout.log"
$StandardErrorPath = Join-Path $LauncherRoot "${Phase}_${Timestamp}.stderr.log"
$StatusPath = Join-Path $LauncherRoot "${Phase}_${Timestamp}.status.json"
foreach ($Path in @($StandardOutputPath, $StandardErrorPath, $StatusPath)) {
    if (Test-Path $Path) {
        throw "Detached launcher output already exists: $Path"
    }
}

$DetachedWrapper = Join-Path $PSScriptRoot (
    "invoke_patient_indexed_specimen_routing_phase_detached.ps1"
)
$ControlProcessIds = @(
    Get-ControlProcessIds `
        -ProcessSnapshot @(Get-CimInstance Win32_Process) `
        -CurrentProcessId ([int]$PID)
)
$SerializedControlProcessIds = $ControlProcessIds -join ","
$Arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $DetachedWrapper,
    "-Phase", $Phase,
    "-ExpectedCommit", $ExpectedCommit,
    "-StatusPath", $StatusPath,
    "-ControlProcessIds", $SerializedControlProcessIds
)
if ($PythonExecutable) {
    $Arguments += @("-PythonExecutable", $PythonExecutable)
}
if ($TeacherBundle) {
    $Arguments += @("-TeacherBundle", $TeacherBundle)
}
if ($ApprovePilot) {
    $Arguments += "-ApprovePilot"
}

$Claim = [ordered]@{
    phase = $Phase
    state = "claimed"
    expected_commit = $ExpectedCommit
    requested_at = (Get-Date).ToUniversalTime().ToString("o")
    stdout_path = $StandardOutputPath
    stderr_path = $StandardErrorPath
    status_path = $StatusPath
    launcher_pid = $null
    control_process_ids = $ControlProcessIds
    teacher_bundle = $TeacherBundle
}
$Claim | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $ClaimPath

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StandardOutputPath `
    -RedirectStandardError $StandardErrorPath `
    -WindowStyle Hidden `
    -PassThru

$Claim.state = "launched"
$Claim.launcher_pid = [int]$Process.Id
$Claim | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $ClaimPath

Write-Host "DETACHED_PHASE_STARTED=$Phase"
Write-Host "PID=$($Process.Id)"
Write-Host "STDOUT=$StandardOutputPath"
Write-Host "STDERR=$StandardErrorPath"
Write-Host "STATUS=$StatusPath"
Write-Host "CLAIM=$ClaimPath"
