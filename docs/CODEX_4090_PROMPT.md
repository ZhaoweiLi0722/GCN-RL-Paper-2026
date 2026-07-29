# Codex Prompt: Configure and Run the RTX 4090 Experiment

Paste everything inside the following block into Codex on the Windows PC.

```text
You are configuring and running a reproducible CUDA experiment on my Windows
PC with an NVIDIA RTX 4090.

Repository:
https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026

Required branch:
codex/rtx4090-matched-ablation

Primary objective:
Run the matched three-seed GCN-versus-flat residual DDPG pipeline exactly as
specified in the repository. Both methods must use the same teacher caches,
three-stage optimization history, training seeds, endpoint projection, and
evaluation common random numbers. This is a controlled graph-representation
ablation, not a hyperparameter search.

Operating rules:
1. Read AGENTS.md and docs/CUDA_4090_SETUP.md before doing anything.
2. Use C:\gcnrl as the repository path to avoid Windows path-length problems.
3. Do not merge before the matched CUDA smoke succeeds. After it succeeds,
   use only the documented fast-forward merge. Never rewrite Git history or
   commit generated results.
4. Do not change algorithms, hyperparameters, seeds, training epochs,
   projection thresholds, evaluation replications, or CRNs.
5. Do not regenerate teacher caches. The required archive is versioned at
   training_data\regional_4090_training_data.zip.
6. CUDA is mandatory. Abort and diagnose the environment if PyTorch cannot
   see the RTX 4090; never silently fall back to CPU.
7. You may fix genuine Windows portability or dependency errors, but keep
   fixes minimal. Show me the proposed code change before altering experiment
   behavior.
8. Keep the PC awake while the run is active. Do not terminate a healthy
   long-running training process merely because it is quiet.
9. Give concise progress updates after setup, after each training stage, and
   after evaluation.

Step 1: Locate or clone the exact branch.

If C:\gcnrl does not exist, run:

cd C:\
git clone https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026.git gcnrl
cd C:\gcnrl
git switch --track origin/codex/rtx4090-matched-ablation

If it already exists, inspect its status first. Preserve unrelated user
changes. Fetch origin and switch to codex/rtx4090-matched-ablation only when
that is safe.

Confirm:

git status --short --branch
git log --oneline -3

The active branch must be codex/rtx4090-matched-ablation.

Step 2: Verify prerequisites.

Check:

nvidia-smi
git --version
py -3.11 --version

The GPU must be an NVIDIA GeForce RTX 4090. If Git or 64-bit Python 3.11 is
missing, install it with winget when available:

winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e

After installation, open a fresh PowerShell session if PATH has not refreshed.
Do not install a separate CUDA Toolkit unless a real dependency requires it;
this project uses the CUDA runtime distributed in the PyTorch wheel.

Step 3: Validate the repository teacher-cache bundle.

The archive must exist at:

C:\gcnrl\training_data\regional_4090_training_data.zip

Check its SHA-256:

Get-FileHash C:\gcnrl\training_data\regional_4090_training_data.zip `
  -Algorithm SHA256

Expected value:
f7792fa97a889f463f22ebae2d2846cc475225412141a1f377b4a84d55efec17

If the archive is missing or the hash differs, stop and report it. Do not
regenerate data. The setup script will verify and extract the archive without
creating an extra directory.

Step 4: Install the environment and verify real CUDA execution.

From C:\gcnrl:

Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cuda_4090.ps1

The required output includes:

cuda_available=True
device_name=NVIDIA GeForce RTX 4090

Also record:
- torch version;
- PyTorch CUDA runtime version;
- total GPU memory;
- verification matmul time.

If setup fails, diagnose driver, Python architecture/version, virtual
environment, and PyTorch wheel selection. Keep torch==2.8.0 with the cu126
wheel unless there is a documented incompatibility.

Step 5: Run the matched CUDA smoke and merge to main.

Run:

.\scripts\run_regional_cuda_smoke.ps1

This must complete one distillation epoch for both
gcn_residual_mdl2_network_ddpg_afd and
flat_residual_mdl2_network_ddpg_afd on seed 0, with real CUDA execution.
Confirm that both summaries and checkpoints exist under:

C:\gcnrl\results\regional_cuda_smoke

Only after this smoke succeeds, fast-forward the verified code into main:

git fetch origin
git switch main
git pull --ff-only origin main
git merge --ff-only codex/rtx4090-matched-ablation
git push origin main

Do not force push and do not use a non-fast-forward merge. If authentication
is unavailable or `--ff-only` fails, stop and report the exact condition; the
verified branch remains safe. Do not wait for the full experiment before
merging. After a successful push, remain on main for the full run.

Step 6: Run the verified pipeline.

Explain the planned workload before starting:
- stage 1: base distillation, 300 epochs;
- stage 2: DAgger iteration 1, 100 epochs;
- stage 3: DAgger iteration 2, 100 epochs;
- methods: GCN residual DDPG and matched flat residual DDPG;
- seeds: 0, 1, and 2;
- evaluation: fixed top-1 endpoint projection, 100 validation replications,
  and 300 independent holdout replications.

Then run:

.\scripts\run_regional_cuda_pipeline.ps1

This script is restartable at completed policy-run boundaries. Do not add
--force when resuming. Logs are under:

C:\gcnrl\results\regional_cuda_pipeline\logs

In another PowerShell window, monitor:

nvidia-smi -l 2

After the first policy has made measurable progress, estimate the remaining
runtime from observed throughput. Distinguish GPU training time from
CPU-bound environment/evaluation time.

Step 7: Validate completion.

Confirm that all six GCN/flat training-seed runs completed at each of the three
stages and that stage 4 produced validation and holdout summaries. Check logs
for exceptions, NaNs, silent CPU fallback, missing checkpoints, or incomplete
seed coverage.

Summarize, for every method and training seed:
- selected checkpoint;
- total-cost delta versus MDL-2;
- paired 95% confidence interval;
- win rate;
- completion-service delta;
- patients-lost delta;
- residual deployment/usage rate;
- clinical noninferiority result.

Also report the paired GCN-versus-flat comparison. Do not claim a graph
advantage unless the matched multi-seed holdout supports it.

Step 8: Package the outputs.

Run:

Compress-Archive C:\gcnrl\results\regional_cuda_pipeline `
  C:\gcnrl\regional_cuda_results.zip -Force

Report the final ZIP path and SHA-256:

Get-FileHash C:\gcnrl\regional_cuda_results.zip -Algorithm SHA256

Do not push the results ZIP or generated result directories to GitHub. I will
transfer regional_cuda_results.zip back to the Mac for final analysis.
```
