param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Validate", "ImportTeacher", "Smoke", "Pilot", "Evaluate")]
    [string]$Phase,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit,
    [Parameter(Mandatory = $true)]
    [string]$StatusPath,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9]+(?:,[0-9]+)*$")]
    [string]$ControlProcessIds,
    [switch]$ApprovePilot,
    [string]$TeacherBundle = "",
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$StartedAt = (Get-Date).ToUniversalTime().ToString("o")
$ExitCode = 1
$State = "failed"
$FailureMessage = ""

try {
    $Runner = Join-Path $PSScriptRoot "run_patient_indexed_specimen_routing.ps1"
    $RunnerArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Runner,
        "-Phase", $Phase,
        "-ExpectedCommit", $ExpectedCommit,
        "-ControlProcessIds", $ControlProcessIds
    )
    if ($PythonExecutable) {
        $RunnerArguments += @("-PythonExecutable", $PythonExecutable)
    }
    if ($TeacherBundle) {
        $RunnerArguments += @("-TeacherBundle", $TeacherBundle)
    }
    if ($ApprovePilot) {
        $RunnerArguments += "-ApprovePilot"
    }

    & powershell.exe @RunnerArguments
    $ExitCode = [int32]$LASTEXITCODE
    if ($ExitCode -eq 0) {
        $State = "completed"
    } else {
        $FailureMessage = "Locked phase runner exited nonzero."
    }
} catch {
    $FailureMessage = $_.Exception.Message
} finally {
    $StatusDirectory = Split-Path $StatusPath -Parent
    if (-not (Test-Path -PathType Container $StatusDirectory)) {
        New-Item -ItemType Directory -Force $StatusDirectory | Out-Null
    }
    $Payload = [ordered]@{
        phase = $Phase
        state = $State
        expected_commit = $ExpectedCommit
        wrapper_pid = [int]$PID
        control_process_ids = $ControlProcessIds
        started_at = $StartedAt
        completed_at = (Get-Date).ToUniversalTime().ToString("o")
        exit_code = $ExitCode
        failure_message = $FailureMessage
        teacher_bundle = $TeacherBundle
    }
    $TemporaryStatusPath = "$StatusPath.tmp.$PID"
    $Payload | ConvertTo-Json -Depth 4 |
        Set-Content -Encoding UTF8 $TemporaryStatusPath
    Move-Item $TemporaryStatusPath $StatusPath
}

exit $ExitCode
