Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "Checking NVIDIA driver..."
& nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed. Install or update the NVIDIA driver first."
}

if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    }
    else {
        throw "Python was not found. Install 64-bit Python 3.11."
    }
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "The virtual environment does not contain python.exe."
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

& $Python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) {
    throw "CUDA-enabled PyTorch installation failed."
}

& $Python -m pip install numpy==2.0.2 matplotlib==3.9.4
if ($LASTEXITCODE -ne 0) {
    throw "Project dependency installation failed."
}

$env:PYTHONPATH = $RepoRoot
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".matplotlib-cache"
New-Item -ItemType Directory -Force $env:MPLCONFIGDIR | Out-Null

& (Join-Path $PSScriptRoot "prepare_regional_training_data.ps1")

& $Python scripts\verify_cuda.py
if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed."
}

& $Python -m unittest tests.test_train_multiscenario_network_residual
if ($LASTEXITCODE -ne 0) {
    throw "CUDA pipeline unit test failed."
}

Write-Host ""
Write-Host "CUDA environment is ready."
Write-Host "Python: $Python"
