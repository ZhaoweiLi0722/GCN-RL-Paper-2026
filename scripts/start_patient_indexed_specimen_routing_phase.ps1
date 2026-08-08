param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Preflight", "ImportTeacher", "Smoke", "Pilot", "Evaluate")]
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

function Resolve-RoutingPython {
    param(
        [string]$RequestedPython,
        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot
    )

    $Candidates = @()
    if ($RequestedPython) {
        $Candidates += $RequestedPython
    }
    $Candidates += @(
        (Join-Path $RepositoryRoot ".venv\Scripts\python.exe"),
        "C:\gcnrl\.venv\Scripts\python.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe")
    )

    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        try {
            $LauncherOutput = @(
                & $PyLauncher.Source `
                    -3.11 `
                    -c "import sys; print(sys.executable)" `
                    2>$null
            )
            if ([int32]$LASTEXITCODE -eq 0 -and $LauncherOutput.Count -gt 0) {
                $Candidates += [string]$LauncherOutput[-1]
            }
        } catch {
            # The regular candidate scan below remains authoritative.
        }
    }
    foreach ($Command in @(Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
        if ($Command.Source) {
            $Candidates += [string]$Command.Source
        }
    }

    $ProbeScript = @'
import json
import sys

import numpy
import torch
import evaluation.train_multiscenario_network_residual

cuda_available = bool(torch.cuda.is_available())
payload = {
    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    "numpy_version": numpy.__version__,
    "torch_version": torch.__version__,
    "cuda_available": cuda_available,
    "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "",
}
print("ROUTING_PYTHON_PROBE=" + json.dumps(payload, sort_keys=True))
'@

    $Failures = @()
    $Seen = @{}
    foreach ($CandidateValue in $Candidates) {
        if (-not $CandidateValue) {
            continue
        }
        $Candidate = [Environment]::ExpandEnvironmentVariables(
            [string]$CandidateValue
        )
        if (-not (Test-Path -PathType Leaf $Candidate)) {
            $Failures += "missing: $Candidate"
            continue
        }
        $ResolvedCandidate = [string](Resolve-Path $Candidate)
        $CandidateKey = $ResolvedCandidate.ToLowerInvariant()
        if ($Seen.ContainsKey($CandidateKey)) {
            continue
        }
        $Seen[$CandidateKey] = $true

        $PreviousPythonPath = $env:PYTHONPATH
        $PreviousCudaDevices = $env:CUDA_VISIBLE_DEVICES
        try {
            $env:PYTHONPATH = $RepositoryRoot
            $env:CUDA_VISIBLE_DEVICES = "0"
            $ProbeOutput = @(& $ResolvedCandidate -c $ProbeScript 2>&1)
            $ProbeExitCode = [int32]$LASTEXITCODE
        } catch {
            $ProbeOutput = @($_.Exception.Message)
            $ProbeExitCode = 1
        } finally {
            $env:PYTHONPATH = $PreviousPythonPath
            $env:CUDA_VISIBLE_DEVICES = $PreviousCudaDevices
        }
        if ($ProbeExitCode -ne 0) {
            $Failures += (
                "probe failed: $ResolvedCandidate :: " +
                (($ProbeOutput | ForEach-Object { [string]$_ }) -join " | ")
            )
            continue
        }
        $ProbeLine = @(
            $ProbeOutput | Where-Object {
                ([string]$_).StartsWith("ROUTING_PYTHON_PROBE=")
            }
        ) | Select-Object -Last 1
        if (-not $ProbeLine) {
            $Failures += "probe payload missing: $ResolvedCandidate"
            continue
        }
        try {
            $Probe = ([string]$ProbeLine).Substring(
                "ROUTING_PYTHON_PROBE=".Length
            ) | ConvertFrom-Json
        } catch {
            $Failures += "probe payload invalid: $ResolvedCandidate"
            continue
        }
        if ([string]$Probe.python_version -ne "3.11.9") {
            $Failures += (
                "Python version mismatch: $ResolvedCandidate :: " +
                [string]$Probe.python_version
            )
            continue
        }
        if ([string]$Probe.numpy_version -ne "2.0.2") {
            $Failures += (
                "NumPy version mismatch: $ResolvedCandidate :: " +
                [string]$Probe.numpy_version
            )
            continue
        }
        if (
            -not [bool]$Probe.cuda_available -or
            [int]$Probe.cuda_device_count -lt 1 -or
            [string]$Probe.cuda_device_name -notmatch "(?i)RTX\s*4090"
        ) {
            $Failures += (
                "RTX 4090 CUDA mismatch: $ResolvedCandidate :: " +
                [string]$Probe.cuda_device_name
            )
            continue
        }
        return [pscustomobject]@{
            path = $ResolvedCandidate
            sha256 = (
                Get-FileHash -Algorithm SHA256 $ResolvedCandidate
            ).Hash.ToLowerInvariant()
            python_version = [string]$Probe.python_version
            numpy_version = [string]$Probe.numpy_version
            torch_version = [string]$Probe.torch_version
            cuda_device_count = [int]$Probe.cuda_device_count
            cuda_device_name = [string]$Probe.cuda_device_name
        }
    }

    throw (
        "No existing Python satisfies the locked Python 3.11.9, NumPy 2.0.2, " +
        "project-import, and RTX 4090 CUDA gates. No environment was changed. " +
        ($Failures -join "`n")
    )
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
if ($Phase -eq "ImportTeacher" -and -not $TeacherBundle) {
    throw "ImportTeacher requires -TeacherBundle."
}
if ($Phase -ne "ImportTeacher" -and $TeacherBundle) {
    throw "-TeacherBundle is only valid for ImportTeacher."
}
if ($TeacherBundle -and -not (Test-Path -PathType Leaf $TeacherBundle)) {
    throw "Missing frozen teacher bundle: $TeacherBundle"
}

$PythonProbe = Resolve-RoutingPython `
    -RequestedPython $PythonExecutable `
    -RepositoryRoot $RepoRoot
$ResolvedPythonExecutable = [string]$PythonProbe.path

$ResultRoot = Join-Path $RepoRoot $ResultRootName
if ($Phase -eq "Preflight" -and (Test-Path $ResultRoot)) {
    throw "Recovery 5 result root already exists; refusing to launch Preflight."
}
if ($Phase -ne "Preflight" -and -not (Test-Path -PathType Container $ResultRoot)) {
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
    "-ControlProcessIds", $SerializedControlProcessIds,
    "-PythonExecutable", $ResolvedPythonExecutable
)
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
    requested_python = $PythonExecutable
    python_executable = $ResolvedPythonExecutable
    python_sha256 = [string]$PythonProbe.sha256
    python_probe = $PythonProbe
}
$Claim | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $ClaimPath

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
$Claim | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $ClaimPath

Write-Host "DETACHED_PHASE_STARTED=$Phase"
Write-Host "PID=$($Process.Id)"
Write-Host "STDOUT=$StandardOutputPath"
Write-Host "STDERR=$StandardErrorPath"
Write-Host "STATUS=$StatusPath"
Write-Host "CLAIM=$ClaimPath"
