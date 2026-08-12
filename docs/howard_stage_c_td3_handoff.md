# Howard Handoff: Stage C Matched TD3 Development Screen

Document status: orientation and responsibility handoff only

Branch: `patient-indexed-specimen-routing`

Publication plan:
`docs/patient_indexed_specimen_routing_locked_execution_plan.md`

## Required branch and checkout

The only branch Howard should pull or fetch is:

```text
origin/patient-indexed-specimen-routing
```

Do not use `main`, `codex/patient-indexed-specimen-routing`,
`codex/rtx4090-matched-ablation`, or
`howard-patient-condition-robustness` for this Stage C task.

Howard may read the latest handoff now by checking out the current remote
branch tip in detached-HEAD mode:

```bash
git fetch origin patient-indexed-specimen-routing
git switch --detach origin/patient-indexed-specimen-routing
git rev-parse HEAD
```

The current branch tip is for orientation only and does not authorize
training. The project team will later push the Stage C execution-lock commit to
the same `patient-indexed-specimen-routing` branch. Howard must then fetch that
branch again and check out the newly supplied exact commit in detached-HEAD
mode before running the one-time launcher.

## 1. Short version

Howard will independently execute one locked Stage C experiment after the
project team provides a separate execution-lock commit and one-time launch
instruction.

The experiment is a matched routing-primary TD3 screen:

- AFR-GCN-TD3, training seeds 20, 21, and 22;
- parameter-matched AFR-Flat-TD3, training seeds 20, 21, and 22;
- 100 online episodes per run;
- six total runs and 600 total online episodes;
- one machine, strict serial order, checkpoint every five episodes;
- the same routing teacher, routing mechanics, scenario distribution, action
  bounds, residual-policy contract, and training budget for graph and flat.

Howard is the execution owner, not the experiment designer. He must not tune,
repair, retry, substitute parameters, or start a formal confirmation.

## 2. Why this experiment is next

The routing-primary DDPG campaign is complete for GCN and matched flat across
training seeds 10-14 and 100 online episodes per run. Formal paired evaluation
established two useful results:

1. AFR-GCN-DDPG beat routing MDL-2 by about 17.690 million cost units
   (0.658%), with the 95% interval excluding zero.
2. AFR-GCN-DDPG beat matched AFR-Flat-DDPG by about 8.604 million cost units
   (0.321%), with the 95% interval excluding zero.

The unresolved result is online-learning attribution. Final GCN-DDPG did not
beat its frozen-pretraining checkpoint. The completed Stage B checkpoint curve
was classified `plateaued_or_inconclusive`: episode 75 to 100 did not show a
statistically resolved cost improvement. The locked plan therefore rejects a
blind 200- or 300-episode DDPG extension and moves to a bounded matched TD3
screen.

This Stage C experiment asks whether TD3's twin critics, delayed policy update,
and target smoothing can produce a clearer online increment without changing
the graph representation or routing problem.

## 3. Current project position

| Stage | Status | Main outcome |
|---|---|---|
| Routing mechanics and teacher | Complete | Identity-preserving routing and the frozen teacher passed validation |
| DDPG training | Complete | Five GCN and five matched-flat 100-episode runs |
| Formal DDPG evaluation | Complete | GCN beat MDL-2 and matched flat; online increment not established |
| Stage A timing sensitivity | Complete | Lead-0 and return-1 frozen-policy evaluations audited |
| Stage B DDPG checkpoint attribution | Complete | DDPG curve plateaued or remained inconclusive |
| Stage C matched TD3 screen | Preparing | Howard will execute after the execution lock is published |
| Stage D independent confirmation | Not authorized | Runs only if Stage C promotes one candidate |
| Manuscript and reproducibility freeze | Pending | Begins after the final algorithm decision |

## 4. Howard's exact responsibility

After receiving the Stage C execution-lock commit and delegation prompt,
Howard will:

1. Check out the exact commit on a clean tracked worktree.
2. Verify the source, config, teacher, launcher, and stage-spec SHA256 values.
3. Record Python, PyTorch, accelerator, driver, operating-system, and hardware
   versions.
4. Confirm that no related training process or Stage C output root exists.
5. Run the provided preflight and focused tests.
6. Launch exactly one detached Stage C runner.
7. Let the runner execute all six jobs in the locked serial order.
8. Preserve logs, manifests, checkpoints, training states, summaries, process
   evidence, and final artifact hashes.
9. Return the complete evidence report without starting Stage D.

The preferred platform is a stable Linux CUDA environment. A different
platform is acceptable only if the execution-lock commit explicitly supports
it and records the environment. Howard must use the supplied launcher rather
than translate it into an ad hoc shell or PowerShell workflow.

## 5. Locked scientific boundaries

The execution-lock commit will freeze the final values. Until that commit is
provided, this document is not launch authorization.

The following boundaries are already fixed:

- Routing is enabled and patient identity is preserved.
- GCN and flat receive the same teacher data and scientific budget.
- Training seeds are 20, 21, and 22 for both architectures.
- Each run has 100 online episodes.
- Full actor, critic, optimizer, replay, RNG, and training state is saved every
  five completed episodes.
- Development CRNs are fresh and distinct from formal holdout seed `91100000`
  and Stage B development stream `93100000`.
- No broad HPO, deployment threshold selection, checkpoint cherry-picking,
  DDPG extension, SAC/PPO screen, or no-routing experiment is allowed.
- A failed run is evidence. It is not overwritten or silently retried.

The following launch values remain intentionally unspecified here and must be
read from the later execution-lock commit:

- exact Stage C commit;
- exact training and evaluation config paths and hashes;
- exact launcher path and hash;
- exact fresh development CRN seeds;
- exact output root and phase order;
- exact preflight command and one-time detached launch command.

## 6. Required output package

Howard's returned evidence must contain:

- exact commit, branch, clean-worktree status, and allowed diff;
- environment and accelerator fingerprint;
- teacher, source, config, stage-spec, and launcher hashes;
- one launcher claim and exact PID/PPID/command-line process chain;
- per-run `training.csv`, `summary.json`, full checkpoints, and atomic training
  states;
- checkpoint counts at episodes 5, 10, ..., 100;
- nonzero routing, nonzero online updates after warmup, actor drift, residual
  use, and finite persisted losses;
- final and frozen-pretraining development evaluation rows on the locked CRNs;
- anomaly scan for Traceback, native crash, OOM, accelerator fallback,
  NaN/Inf, duplicate process, hash mismatch, missing checkpoint, and stale
  progress;
- a final manifest plus SHA256 inventory for all evidence.

## 7. Stop conditions

Howard must stop and report, without repair or retry, if any of the following
occurs:

- source, config, teacher, checkpoint, or CRN mismatch;
- duplicate or unauthorized process;
- CPU fallback when the locked run requires CUDA or MPS;
- Traceback, native crash, OOM, NaN/Inf, or missing asset;
- zero specimen routing;
- zero online updates after warmup;
- missing five-episode checkpoint or invalid resume state;
- unexpected process exit or stale output beyond the plausible phase time.

## 8. What happens after Howard finishes

Howard stops after returning the Stage C evidence. The core project team then:

1. Audits all hashes, row counts, checkpoints, routing, losses, and clinical
   guardrails.
2. Evaluates final versus frozen pretraining for GCN-TD3 and flat-TD3.
3. Compares final GCN-TD3 with matched flat-TD3 and routing MDL-2.
4. Compares TD3 with the completed DDPG candidate on a matched development
   stream.
5. Applies the preregistered Stage C promotion gate.

If one TD3 candidate passes, the team locks Stage D: at least five fresh
training seeds and a never-used formal holdout stream. If TD3 does not pass,
the team retains the completed DDPG result and narrows the paper claim to
graph-aware advantage-filtered residual control. There will be no open-ended
algorithm search.

## 9. Immediate instruction to Howard

Read this document and the locked publication plan, but do not start an
experiment yet. Reply only that the task boundary is understood and report the
available operating system, GPU, CUDA/driver version, Python version, free disk
space, and expected uninterrupted compute window. The project team will then
publish the exact Stage C execution lock and one-time launch prompt.
