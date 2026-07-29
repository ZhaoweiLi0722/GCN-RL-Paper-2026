Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_cuda_4090.ps1 first."
}

& (Join-Path $PSScriptRoot "prepare_regional_training_data.ps1")

$env:PYTHONPATH = $RepoRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".matplotlib-cache"
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null

& $Python scripts\verify_cuda.py
if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed."
}

& $Python -m evaluation.train_multiscenario_network_residual `
    --config experiments\configs\regional_cuda_smoke.json
if ($LASTEXITCODE -ne 0) {
    throw "Matched GCN/flat CUDA smoke failed."
}

$ExpectedSummaries = @(
    "results\regional_cuda_smoke\regional_cuda_smoke\gcn_residual_mdl2_network_ddpg_afd\seed0\summary.json",
    "results\regional_cuda_smoke\regional_cuda_smoke\flat_residual_mdl2_network_ddpg_afd\seed0\summary.json"
)
foreach ($Path in $ExpectedSummaries) {
    if (-not (Test-Path $Path)) {
        throw "CUDA smoke did not create expected summary: $Path"
    }
}

Write-Host ""
Write-Host "Matched GCN/flat CUDA smoke completed successfully."
Write-Host "The branch is ready for a fast-forward merge into main."
