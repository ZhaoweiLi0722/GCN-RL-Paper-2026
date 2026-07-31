# Recovery 2: Staged and Resumable DDPG Attribution Pilot

## Trigger

The original pilot and Recovery 1 both ended with Windows native access
violations (`0xC0000005`). The original attempt failed during GCN seed 0 AFD
pretraining. Recovery 1 completed AFD pretraining, entered online training, and
failed after the episode 10 boundary in the DDPG update path. Neither attempt
produced a valid six-run attribution result.

## Scientific contract

Recovery 2 preserves the locked scientific protocol:

- parameter-matched AFR-GCN-DDPG and AFR-Flat-DDPG;
- seeds 0, 1, and 2;
- 300 AFD pretraining epochs and 100 online episodes per official run;
- the same four scenarios and episode round-robin schedule;
- identical model, optimizer, reward, exploration, action, and safety settings;
- final and frozen-pretrain evaluations with 100 CRN holdout replications per
  scenario and seed, using holdout seed 59900000;
- the same paired bootstrap attribution analysis.

The 20-episode stability gate is diagnostic only. It loads the immutable
Recovery 1 GCN seed 0 pretrain checkpoint, rebuilds the retained teacher cache
without additional fitting, and exercises the online actor-critic update path.
Its output is never pooled with or cited as a paper result.

## Execution changes

Execution reliability changes do not alter learning behavior:

- the stability gate must pass before official training starts;
- each method/seed policy run executes in a fresh Python process;
- actor checkpoints and full training state are written every five completed
  episodes;
- the full state contains actor/critic and target networks, optimizer states,
  replay buffer, retained imitation tensors, exploration state, random-number
  generator states, update counters, completed rows, and the next episode;
- full-state files are replaced atomically;
- incremental manifests merge completed isolated runs and validate their common
  experiment contract;
- every subprocess records signed, unsigned, and hexadecimal exit codes.

## Locked order

1. Validate the exact commit, clean tracked worktree, teacher assets, focused
   tests, CUDA, and absence of all Recovery 2 roots.
2. Run the non-paper GCN seed 0 stability gate for 20 episodes and require
   nonzero online updates.
3. Run official GCN seeds 0, 1, and 2 as separate processes.
4. Run official matched-flat seeds 0, 1, and 2 as separate processes.
5. Verify six complete runs and the graph/flat parameter gap.
6. Run final-checkpoint and frozen-pretrain CRN evaluations.
7. Compute final-versus-frozen and graph-minus-flat attribution.
8. Write provenance, archive, SHA256 sidecar, and the OneDrive copy.

Any native crash, Python traceback, non-finite value, OOM, CPU fallback,
checkpoint/config mismatch, zero online updates, or unexpected exit stops the
runner. It must not auto-relaunch. A later audited resume may invoke
`resume_multiscenario_ddpg_attribution_recovery2_run.ps1` once for the failed
method/seed, using the last atomic state. Completed runs and prior failure
evidence remain untouched.
