param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit,
    [Parameter(Mandatory = $true)]
    [string]$StatusPath,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9]+(?:,[0-9]+)*$")]
    [string]$ControlProcessIds,
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutableBase64
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$StartedAt = (Get-Date).ToUniversalTime().ToString("o")
$ExitCode = 1
$State = "failed"
$FailureMessage = ""
$FailureDetails = ""
$PythonExecutable = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String($PythonExecutableBase64)
)

try {
    $Runner = Join-Path $PSScriptRoot (
        "run_patient_indexed_specimen_routing_recovery8_diagnostics.ps1"
    )
    & $Runner `
        -ExpectedCommit $ExpectedCommit `
        -PythonExecutable $PythonExecutable `
        -ControlProcessIds $ControlProcessIds
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
        phase = "WindowsNativeCrashDiagnostics"
        state = $State
        expected_commit = $ExpectedCommit
        wrapper_pid = [int]$PID
        control_process_ids = $ControlProcessIds
        started_at = $StartedAt
        completed_at = (Get-Date).ToUniversalTime().ToString("o")
        exit_code = $ExitCode
        failure_message = $FailureMessage
        failure_details = $FailureDetails
        python_executable = $PythonExecutable
        argument_transport = "utf8_base64"
        recovery7_retry_or_resume_permitted = $false
        formal_training_permitted = $false
        pilot_permitted = $false
    }
    $TemporaryStatusPath = "$StatusPath.tmp.$PID"
    $Payload | ConvertTo-Json -Depth 8 |
        Set-Content -Encoding UTF8 $TemporaryStatusPath
    Move-Item $TemporaryStatusPath $StatusPath
}

exit $ExitCode
