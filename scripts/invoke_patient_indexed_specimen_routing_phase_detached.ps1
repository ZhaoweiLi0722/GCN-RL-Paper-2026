param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Preflight", "ImportTeacher", "Smoke", "Pilot", "Evaluate")]
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
    [string]$TeacherBundleBase64 = "",
    [ValidatePattern("^$|^[0-9a-fA-F]{64}$")]
    [string]$ExpectedTeacherBundleSha256 = "",
    [string]$PythonExecutableBase64 = "",
    [switch]$TransportProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$StartedAt = (Get-Date).ToUniversalTime().ToString("o")
$ExitCode = 1
$State = "failed"
$FailureMessage = ""
$FailureDetails = ""
$TeacherBundle = ""
$PythonExecutable = ""

function ConvertFrom-Utf8Base64 {
    param([Parameter(Mandatory = $true)][string]$Value)
    if (-not $Value) {
        return ""
    }
    return [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($Value)
    )
}

try {
    if ($TeacherBundleBase64) {
        $TeacherBundle = ConvertFrom-Utf8Base64 $TeacherBundleBase64
    }
    if ($PythonExecutableBase64) {
        $PythonExecutable = ConvertFrom-Utf8Base64 $PythonExecutableBase64
    }
    $Runner = Join-Path $PSScriptRoot "run_patient_indexed_specimen_routing.ps1"
    $RunnerParameters = @{
        Phase = $Phase
        ExpectedCommit = $ExpectedCommit
        ControlProcessIds = $ControlProcessIds
    }
    if ($PythonExecutable) {
        $RunnerParameters["PythonExecutable"] = $PythonExecutable
    }
    if ($TeacherBundle) {
        $RunnerParameters["TeacherBundle"] = $TeacherBundle
        $RunnerParameters["ExpectedTeacherBundleSha256"] = (
            $ExpectedTeacherBundleSha256
        )
    }
    if ($ApprovePilot) {
        $RunnerParameters["ApprovePilot"] = $true
    }

    if (-not $TransportProbe) {
        & $Runner @RunnerParameters
    }
    $ExitCode = 0
    $State = "completed"
} catch {
    $FailureMessage = $_.Exception.Message
    $FailureDetails = ($_ | Out-String).Trim()
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
        failure_details = $FailureDetails
        teacher_bundle = $TeacherBundle
        expected_teacher_bundle_sha256 = $ExpectedTeacherBundleSha256
        argument_transport = "utf8_base64"
        python_executable = $PythonExecutable
        transport_probe = [bool]$TransportProbe
    }
    $TemporaryStatusPath = "$StatusPath.tmp.$PID"
    $Payload | ConvertTo-Json -Depth 4 |
        Set-Content -Encoding UTF8 $TemporaryStatusPath
    Move-Item $TemporaryStatusPath $StatusPath
}

exit $ExitCode
