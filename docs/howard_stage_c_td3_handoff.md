# Howard Handoff: Stage C Matched TD3 Screen

Status: ready for Mac MPS execution

Branch: `patient-indexed-specimen-routing`

Publication plan:
`docs/patient_indexed_specimen_routing_locked_execution_plan.md`

## What Howard Runs

Run the bounded routing-primary TD3 development screen on a Mac with Apple
MPS:

- AFR-GCN-TD3: seeds 20, 21, 22;
- parameter-matched AFR-Flat-TD3: seeds 20, 21, 22;
- 100 online episodes per run;
- six runs in strict serial order;
- full checkpoint and training state every five episodes;
- final and frozen-pretrain development evaluation after training.

This is a development screen, not the final confirmation experiment. Its
purpose is to test whether TD3 produces stronger online-learning attribution
than the completed DDPG candidate while preserving the GCN-versus-flat matched
comparison.

## Checkout

Howard should use only:

```text
origin/patient-indexed-specimen-routing
```

Fetch the branch and check out the exact commit supplied in the delegation
message:

```bash
git fetch origin patient-indexed-specimen-routing
git switch --detach <STAGE_C_COMMIT>
git status --short
```

The tracked worktree must be clean. Local untracked environment files are
acceptable if they do not replace a locked source, config, teacher, or result.

## Scientific Lock

The experiment assets are:

- training config:
  `experiments/configs/patient_indexed_specimen_routing_stage_c_td3_100.json`;
- execution spec:
  `experiments/configs/patient_indexed_specimen_routing_stage_c_td3_execution.json`;
- frozen routing teacher:
  `experiments/evidence/patient_indexed_specimen_routing_stage_c/teacher_train_dagger.npz`;
- optional detached launcher:
  `evaluation/launch_patient_indexed_specimen_routing_stage_c_td3.py`.

The execution spec contains the locked file hashes, phase order, fresh output
roots, development CRNs, tests, and audit requirements. The runner verifies
them before creating training output.

Do not change:

- routing mechanics, patient identity, teacher, scenarios, action bounds, or
  residual-policy contract;
- seeds, episode budget, TD3 settings, parameter matching, or CRN streams;
- GCN/flat budgets or run order;
- checkpoint selection or deployment threshold;
- MPS to CPU fallback.

Do not add HPO, extend DDPG, launch Stage D, or substitute another algorithm.

## Mac Requirement

Use a project Python environment with a PyTorch build for which
`torch.backends.mps.is_available()` is true. The executor performs a real MPS
tensor operation before claiming the run and refuses CPU fallback.

Howard may use his own reliable background-process method. The repository also
provides this optional launcher:

```bash
PYTHONPATH=. .venv/bin/python \
  -m evaluation.launch_patient_indexed_specimen_routing_stage_c_td3 \
  --expected-commit "$(git rev-parse HEAD)"
```

It prints the detached PID. The scientific executor then runs focused tests,
training, final development evaluation, frozen-pretrain development
evaluation, audits, and the SHA256 inventory in order.

## Expected Evidence

Return:

- exact commit and Mac/Python/PyTorch/MPS fingerprint;
- `claim.json`, `status.json`, phase stdout/stderr, and process information;
- six `training.csv` files and six run summaries;
- 20 checkpoints per run plus atomic training state;
- final and frozen-pretrain evaluation rows and summaries;
- the final artifact SHA256 inventory.

The audit requires nonzero specimen routing, nonzero online updates after
warmup, nonzero learned residual use, finite persisted metrics, complete CRN
rows, and a GCN/flat parameter gap below 1%.

If a real failure occurs, preserve the evidence and report it before changing
anything. Do not overwrite a partial Stage C root.

## What Happens Next

After Howard returns the evidence, the project team compares:

1. final GCN-TD3 versus final matched Flat-TD3;
2. final GCN-TD3 versus routing MDL-2;
3. final versus frozen pretrain for both architectures;
4. TD3 versus the completed DDPG candidate on development CRNs.

At most one candidate can advance to Stage D with fresh training seeds and a
never-used formal holdout stream. If TD3 does not pass, the paper retains the
completed DDPG result and reports online attribution conservatively.
