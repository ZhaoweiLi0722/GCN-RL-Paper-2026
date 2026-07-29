# RTX 4090 Training Setup

This workflow runs the matched three-seed regional-drift experiment on an
NVIDIA RTX 4090. It trains GCN and flat residual DDPG with the same data,
training stages, seeds, action projection, and evaluation CRNs.

## 1. Prerequisites

- Windows 11 or a recent Windows 10 release
- current NVIDIA GeForce driver
- Git
- 64-bit Python 3.11
- Codex
- at least 20 GB of free disk space

Use a short repository path such as `C:\gcnrl` to avoid Windows path-length
issues.

The PyTorch wheel contains the CUDA runtime needed by this project. A separate
CUDA Toolkit installation is not normally required because the repository does
not compile custom CUDA extensions.

## 2. Clone the repository

```powershell
cd C:\
git clone --branch codex/rtx4090-matched-ablation --single-branch `
  https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026.git gcnrl
cd C:\gcnrl
git status --short --branch
```

The checked-out branch must be `codex/rtx4090-matched-ablation`.

## 3. Extract the teacher-cache bundle

Copy `regional_4090_training_data.zip` from the Mac to the PC, then extract it
at the repository root:

```powershell
Get-FileHash C:\path\regional_4090_training_data.zip -Algorithm SHA256
Expand-Archive C:\path\regional_4090_training_data.zip C:\gcnrl -Force
```

The expected SHA-256 is:

```text
f7792fa97a889f463f22ebae2d2846cc475225412141a1f377b4a84d55efec17
```

The archive restores six required `.npz` files under `results\`. These files
are intentionally excluded from Git.

## 4. Install and verify CUDA PyTorch

Run PowerShell from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cuda_4090.ps1
```

The script:

- checks `nvidia-smi`;
- creates `.venv`;
- installs PyTorch 2.8.0 with the CUDA 12.6 runtime;
- installs the matching NumPy and Matplotlib versions;
- runs a CUDA matrix multiplication;
- runs the warm-start template unit test.

Successful output must include:

```text
cuda_available=True
device_name=NVIDIA GeForce RTX 4090
```

## 5. Run the experiment

Prevent Windows from sleeping while the experiment is running, then execute:

```powershell
.\scripts\run_regional_cuda_pipeline.ps1
```

The script runs:

1. base teacher distillation for GCN and flat seeds 0, 1, and 2;
2. DAgger iteration-1 fine-tuning for all six policies;
3. DAgger iteration-2 fine-tuning for all six policies;
4. fixed top-1 endpoint-projection evaluation with 100 validation and 300
   holdout replications.

Completed policy runs are skipped when the script is restarted. The active
policy run itself restarts from its stage boundary if interrupted before its
final checkpoint is written.

Monitor GPU activity in a second PowerShell window:

```powershell
nvidia-smi -l 2
```

Logs and outputs are written to:

```text
results\regional_cuda_pipeline
```

## 6. Return results to the Mac

After completion:

```powershell
Compress-Archive results\regional_cuda_pipeline regional_cuda_results.zip -Force
```

Transfer `regional_cuda_results.zip` back to the Mac. The final analysis should
pair GCN and flat rows by training seed, evaluation seed, and replication
before computing confidence intervals.

The complete task prompt for Codex on the PC is in
`docs/CODEX_4090_PROMPT.md`.
