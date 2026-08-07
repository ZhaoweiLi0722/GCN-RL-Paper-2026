param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Validate", "Teachers", "Smoke", "Pilot", "Evaluate")]
    [string]$Phase,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit,
    [switch]$ApprovePilot,
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LockedBranch = "codex/patient-indexed-specimen-routing"
$ResultRootName = "results\patient_indexed_specimen_routing_recovery2"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

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

$ResultRoot = Join-Path $RepoRoot $ResultRootName
if ($Phase -eq "Validate" -and (Test-Path $ResultRoot)) {
    throw "Recovery 2 result root already exists; refusing to launch Validate."
}
if ($Phase -ne "Validate" -and -not (Test-Path -PathType Container $ResultRoot)) {
    throw "Recovery 2 result root does not exist for phase $Phase."
}

$LauncherRoot = Join-Path $ResultRoot "launcher-logs"
$ClaimPath = Join-Path $LauncherRoot "$Phase.claim.json"
if (Test-Path $ClaimPath) {
    throw "Phase $Phase already has a Recovery 2 launch claim."
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
$Arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $DetachedWrapper,
    "-Phase", $Phase,
    "-ExpectedCommit", $ExpectedCommit,
    "-StatusPath", $StatusPath
)
if ($PythonExecutable) {
    $Arguments += @("-PythonExecutable", $PythonExecutable)
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
