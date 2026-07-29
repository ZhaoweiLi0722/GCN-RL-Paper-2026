Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$BundlePath = Join-Path `
    $RepoRoot "training_data\regional_4090_training_data.zip"
$ExpectedHash = `
    "f7792fa97a889f463f22ebae2d2846cc475225412141a1f377b4a84d55efec17"
$RequiredData = @(
    "results\decoupled_transfer_regional_teacher_h52_crn8_100traj_temporal\teacher_train.npz",
    "results\decoupled_transfer_regional_teacher_h52_crn8_100traj_temporal\teacher_validation.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_seed0_smoke\teacher_train_dagger.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_seed0_smoke_validation\teacher_dagger_only.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_iter2_seed0\teacher_train_dagger.npz",
    "results\regional_specialist_temporal_gcn_ddpg_onpolicy_dagger_iter2_seed0_validation\teacher_dagger_only.npz"
)

$MissingData = @(
    $RequiredData | Where-Object { -not (Test-Path $_) }
)
if ($MissingData.Count -eq 0) {
    Write-Host "All regional teacher-cache files are present."
    return
}

if (-not (Test-Path $BundlePath)) {
    throw "Missing repository training bundle: $BundlePath"
}

$ActualHash = (Get-FileHash $BundlePath -Algorithm SHA256).Hash.ToLower()
if ($ActualHash -ne $ExpectedHash) {
    throw "Training bundle SHA-256 mismatch: $ActualHash"
}

Write-Host "Extracting verified regional teacher-cache bundle..."
Expand-Archive $BundlePath $RepoRoot -Force

foreach ($Path in $RequiredData) {
    if (-not (Test-Path $Path)) {
        throw "Bundle extraction did not create required file: $Path"
    }
}

Write-Host "Regional teacher-cache files are ready."
