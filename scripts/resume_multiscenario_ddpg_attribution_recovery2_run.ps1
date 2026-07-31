param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "gcn_residual_mdl2_network_ddpg_afd",
        "flat_residual_mdl2_network_ddpg_afd"
    )]
    [string]$Algorithm,
    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 2)]
    [int]$Seed,
    [string]$ExpectedCommit = "",
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$Python = if ($PythonExecutable) {
    [string](Resolve-Path $PythonExecutable)
} else {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python executable does not exist: $Python"
}

$Commit = (git rev-parse HEAD).Trim()
if ($ExpectedCommit -and $Commit -ne $ExpectedCommit) {
    throw "Expected commit $ExpectedCommit but found $Commit."
}
$DirtyPaths = @(git status --porcelain --untracked-files=no)
if ($DirtyPaths.Count -gt 0) {
    throw "Tracked worktree must be clean before an exact resume."
}

$Config = (
    "experiments\configs\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2.json"
)
$RunDirectory = (
    "results\" +
    "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2\" +
    "$Algorithm\seed$Seed"
)
$TrainingState = Join-Path $RunDirectory (
    "checkpoints\${Algorithm}_seed${Seed}_training_state.pt"
)
$Summary = Join-Path $RunDirectory "summary.json"
if (-not (Test-Path $TrainingState)) {
    throw "No atomic training state exists for this run: $TrainingState"
}
if (Test-Path $Summary) {
    throw "Run already has a completed summary; refusing to resume it."
}

$RelatedProcesses = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -match "train_multiscenario_network_residual" -or
            $_.CommandLine -match "run_multiscenario_ddpg_attribution"
        )
    }
)
if ($RelatedProcesses.Count -gt 0) {
    throw "A related training or runner process is already active."
}

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONFAULTHANDLER = "1"
$env:TORCH_SHOW_CPP_STACKTRACES = "1"

& $Python -m evaluation.train_multiscenario_network_residual `
    --config $Config `
    --algorithm $Algorithm `
    --seed $Seed `
    --resume-training-state $TrainingState
$ExitCodeSigned = [int32]$LASTEXITCODE
$ExitCodeUnsigned = [BitConverter]::ToUInt32(
    [BitConverter]::GetBytes($ExitCodeSigned),
    0
)
$ExitCodeHex = "0x{0:X8}" -f $ExitCodeUnsigned
Write-Host (
    "Resume exit code: signed=$ExitCodeSigned; " +
    "unsigned=$ExitCodeUnsigned; hex=$ExitCodeHex"
)
if ($ExitCodeSigned -ne 0) {
    throw (
        "Exact run resume failed with exit code signed=$ExitCodeSigned, " +
        "unsigned=$ExitCodeUnsigned, hex=$ExitCodeHex."
    )
}
if (-not (Test-Path $Summary)) {
    throw "Resumed run exited without a completed summary."
}

Write-Host "Recovered run completed: $Algorithm seed $Seed"
