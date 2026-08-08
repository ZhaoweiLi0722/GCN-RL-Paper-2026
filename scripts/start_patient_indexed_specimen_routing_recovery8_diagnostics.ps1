param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit,
    [string]$PythonExecutable = "C:\gcnrl\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LockedBranch = "codex/patient-indexed-specimen-routing"
$Recovery7Commit = "30fe642f641e0592abc59009ae3f154019a14145"
$ExpectedPythonSha256 = (
    "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"
)
$ResultRootName = "results\patient_indexed_specimen_routing_recovery8"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Get-ControlProcessIds {
    param(
        [Parameter(Mandatory = $true)][object[]]$ProcessSnapshot,
        [Parameter(Mandatory = $true)][int]$CurrentProcessId
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

if ((git branch --show-current).Trim() -ne $LockedBranch) {
    throw "Recovery 8 diagnostics require branch $LockedBranch"
}
if ((git rev-parse HEAD).Trim() -ne $ExpectedCommit) {
    throw "Recovery 8 diagnostic commit mismatch"
}
git merge-base --is-ancestor $Recovery7Commit $ExpectedCommit
if ($LASTEXITCODE -ne 0) {
    throw "Recovery 8 diagnostic commit is not descended from Recovery 7"
}
if (@(git status --porcelain --untracked-files=no).Count -ne 0) {
    throw "Recovery 8 diagnostics require a clean tracked worktree"
}
if (-not (Test-Path -PathType Leaf $PythonExecutable)) {
    throw "Missing locked Python executable: $PythonExecutable"
}
$PythonSha256 = (
    Get-FileHash -Algorithm SHA256 $PythonExecutable
).Hash.ToLowerInvariant()
if ($PythonSha256 -ne $ExpectedPythonSha256) {
    throw (
        "Locked Python executable hash mismatch: " +
        "expected=$ExpectedPythonSha256 actual=$PythonSha256"
    )
}

$ResultRoot = Join-Path $RepoRoot $ResultRootName
if (Test-Path $ResultRoot) {
    throw "Refusing to reuse existing Recovery 8 root: $ResultRoot"
}
$LauncherLogs = Join-Path $ResultRoot "launcher-logs"
New-Item -ItemType Directory -Force $LauncherLogs | Out-Null
$ClaimPath = Join-Path $LauncherLogs (
    "WindowsNativeCrashDiagnostics.claim.json"
)
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutPath = Join-Path $LauncherLogs (
    "WindowsNativeCrashDiagnostics_${Timestamp}.stdout.log"
)
$StderrPath = Join-Path $LauncherLogs (
    "WindowsNativeCrashDiagnostics_${Timestamp}.stderr.log"
)
$StatusPath = Join-Path $LauncherLogs (
    "WindowsNativeCrashDiagnostics_${Timestamp}.status.json"
)

$ProcessSnapshot = @(Get-CimInstance Win32_Process)
$ControlProcessIds = @(
    Get-ControlProcessIds `
        -ProcessSnapshot $ProcessSnapshot `
        -CurrentProcessId $PID
)
$SerializedControlProcessIds = ($ControlProcessIds -join ",")
$PythonExecutableBase64 = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($PythonExecutable)
)
$Wrapper = Join-Path $PSScriptRoot (
    "invoke_patient_indexed_specimen_routing_recovery8_diagnostics_detached.ps1"
)
$Arguments = @(
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", $Wrapper,
    "-ExpectedCommit", $ExpectedCommit,
    "-StatusPath", $StatusPath,
    "-ControlProcessIds", $SerializedControlProcessIds,
    "-PythonExecutableBase64", $PythonExecutableBase64
)
$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru

$Claim = [ordered]@{
    schema_version = 1
    phase = "WindowsNativeCrashDiagnostics"
    launched_at = (Get-Date).ToUniversalTime().ToString("o")
    branch = $LockedBranch
    expected_commit = $ExpectedCommit
    recovery7_commit = $Recovery7Commit
    recovery7_retry_or_resume_permitted = $false
    recovery8_root = $ResultRootName
    formal_training_permitted = $false
    pilot_permitted = $false
    launcher_pid = [int]$PID
    detached_pid = [int]$Process.Id
    control_process_ids = $ControlProcessIds
    python_executable = $PythonExecutable
    python_sha256 = $PythonSha256
    stdout = [string]$StdoutPath
    stderr = [string]$StderrPath
    status = [string]$StatusPath
}
$TemporaryClaim = "$ClaimPath.tmp.$PID"
$Claim | ConvertTo-Json -Depth 6 |
    Set-Content -Encoding UTF8 $TemporaryClaim
Move-Item $TemporaryClaim $ClaimPath

Write-Host (
    "Recovery 8 diagnostics launched: PID=$($Process.Id); " +
    "STATUS=$StatusPath; CLAIM=$ClaimPath"
)
