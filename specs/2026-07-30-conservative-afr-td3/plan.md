# Conservative AFR-GCN-TD3 Completion Plan

## Objective

Convert the verified graph-policy distillation advantage into a genuine,
stable reinforcement-learning result while preserving clinical
noninferiority. The primary method is:

`MDL-2 anchor + graph residual actor + conservative TD3 fine-tuning`

The matched flat residual policy must receive the same observations, teacher
data, online trajectories, update budget, action constraints, and evaluation
seeds.

## Publication Levels

### Level 1: Stage result

Available now. Publish the verified 4090 audit as graph-policy distillation
evidence. Do not describe it as final RL evidence.

### Level 2: Candidate RL result

Publish after a three-seed targeted screen passes all advance gates. This can
support internal updates and a draft results section, but remains provisional.

### Level 3: Final paper result

Publish after five training seeds, 500 paired Monte Carlo replications, all
required scenarios, matched ablations, and robustness checks are complete.

## Phase A: Lock Evidence and Reproducibility

1. Record the archive SHA256, commit, scenario, seeds, holdout seed, and
   deployment candidate.
2. Keep raw checkpoints and CSVs in private OneDrive storage.
3. Commit only compact statistics and provenance to GitHub.
4. Preserve the current Stage-3 distilled checkpoints as immutable starting
   points for all actor-critic variants.

Exit gate:

- independent aggregation exactly matches the pipeline summary;
- all expected files and replications are present;
- no numerical or device anomalies are found.

Status: passed.

## Phase B: Conservative Actor-Critic Implementation

Implement or verify the following for both graph and matched-flat agents:

1. TD3 clipped double-Q targets.
2. Target-policy smoothing and delayed policy updates.
3. Critic-only warm-up before the first actor update.
4. Teacher/behavior-cloning regularization retained during online updates.
   In the implemented screen this is a frozen Stage-3 policy-output trust
   region, avoiding an extra supervised epoch before online training.
5. Residual L2 penalty and the existing endpoint projection.
6. Fixed deployment scale and threshold selected without holdout leakage.
7. Actor drift diagnostics relative to the Stage-3 checkpoint.
8. Q-value diagnostics for actor, anchor, and held-out teacher actions.
9. Immediate failure on NaN, OOM, device fallback, or missing RL updates.

Initial conservative settings:

- actor learning rate: `1e-5`;
- critic learning rate: `3e-4`;
- critic warm-up: at least 500 updates;
- policy delay: 2;
- exploration standard deviation: `0.005`;
- target-policy noise: `0.01`;
- target-noise clip: `0.02`;
- frozen reference-policy regularization weight: `25.0`;
- actor advantage baseline: the frozen Stage-3 policy, not raw MDL-2;
- conservative reference comparison: lower twin-Q for the candidate versus
  upper twin-Q for the frozen reference;
- residual deployment scale: fixed at `0.1`;
- specimen transfer residual: always zero.

## Phase C: Smoke and Stability Diagnostics

Run one matched GCN/flat seed for 3-10 episodes on CPU, CUDA, or MPS.

Required checks:

- online RL updates are strictly positive;
- the actor remains frozen during critic warm-up;
- both critics and target values remain finite;
- critic disagreement does not grow monotonically;
- actor residual norm and Stage-3 actor drift remain bounded;
- checkpoint loading preserves the Stage-3 policy before updates.

Stop and revise if:

- any metric is non-finite;
- actor updates occur before warm-up;
- candidate cost degrades by more than 0.25% in the smoke holdout;
- clinical noninferiority fails badly;
- the deployment scale collapses to zero.

Status: passed for both graph and matched-flat policies. The smoke run
produced real critic updates and delayed actor updates after warm-up, with
finite twin-Q diagnostics and bounded Stage-3 actor drift.

## Phase D: Three-Seed Targeted Screen

Training:

- GCN conservative TD3: 100 online episodes x seeds 0, 1, 2.
- Matched-flat conservative TD3: 100 online episodes x seeds 0, 1, 2.
- Start each run from its corresponding validated Stage-3 checkpoint.
- Use identical episode seeds and scenario schedules.

Evaluation:

- 100 paired holdout replications per seed;
- common random numbers;
- `patient_condition_geo_regional_drift`;
- no holdout-based checkpoint or scale selection.

Advance gates:

1. GCN pooled cost difference versus MDL-2 is below zero.
2. GCN is clinically noninferior on completion, patients lost, and
   manufacturing ineligibility.
3. GCN pooled cost difference versus matched flat is below zero.
4. At least two of three GCN seeds deploy a nonzero residual.
5. At least two of three GCN seeds do not degrade relative to their Stage-3
   distilled initialization.
6. The bootstrap interval is either below zero or narrow enough to justify the
   final campaign.

## Phase E: Multi-Scenario Confirmation

Required scenarios:

1. Nominal integrated patient-condition + geography.
2. Heterogeneous regional drift.
3. Abrupt mid-horizon regime shift.
4. Compound regional disruption + patient deterioration.

Methods required in the main comparison:

- MDL-2 fixed-prior;
- rolling-estimate or forecast-aware MDL-2;
- robust/tuned MDL-2;
- oracle MDL-2 as a nondeployable upper bound;
- ISO, MYO, and pMYO;
- pure GCN-DDPG or GCN-TD3;
- AFR-GCN-DDPG;
- AFR-GCN-TD3;
- matched flat DDPG;
- matched flat TD3.

SAC and PPO are secondary family benchmarks. Run them after the graph residual
TD3 line is stable; they are not allowed to delay the core matched experiment.

## Phase F: Final Benchmark

For methods that pass Phase D:

- 300-500 online episodes;
- five training seeds;
- 500 paired Monte Carlo replications per scenario and seed;
- frozen evaluation configurations;
- hierarchical bootstrap over training seeds and replications;
- cost, completion service, patients lost, manufacturing ineligibility,
  turnaround time, transfer activity, residual usage, and inference time.

Final success criteria:

1. AFR-GCN-TD3 has a negative pooled cost difference versus MDL-2 in the
   adaptive scenarios, with a 95% interval below zero in the primary scenario.
2. Clinical noninferiority passes.
3. AFR-GCN-TD3 outperforms matched flat TD3.
4. The result is not driven by one seed or a near-zero deployment policy.
5. Nominal performance is not materially worse than MDL-2.

## Phase G: Manuscript and Release

Only after Phase F:

1. Freeze tables from logged outputs.
2. Update the method definition to distinguish AFD/teacher distillation from
   AFR actor-critic fine-tuning.
3. Report negative or unstable baselines honestly.
4. Commit the manuscript to the paper-only Overleaf repository.
5. Merge the tested code branch through a GitHub pull request.
6. Publish a compact reproducibility bundle on GitHub.
7. Archive checkpoints and raw replication rows in a versioned research-data
   release such as OneDrive for collaborators and Zenodo for publication.

## Branch Strategy

- Keep current development on `codex/rtx4090-matched-ablation`.
- Open a PR to `main` after the Stage-4 audit, tests, and smoke configuration
  are committed.
- Merge pipeline/reproducibility code independently of final manuscript
  claims.
- Add final results and manuscript in a later, evidence-locked commit.

## Mac/PC Transfer Protocol

1. GitHub carries source code, configs, tests, compact summaries, and
   provenance manifests.
2. OneDrive carries checkpoints, teacher caches, raw CSV rows, and ZIP
   archives. These files remain outside Git history.
3. Only one machine edits the shared branch at a time. The Mac commits and
   pushes, the PC pulls and runs, then the PC returns a ZIP plus SHA256.
4. The PC command is:

   `powershell -ExecutionPolicy Bypass -File scripts\run_regional_conservative_td3_screen.ps1`

5. An optional OneDrive destination can be supplied with
   `-ExportDirectory "C:\path\to\OneDrive\RTX4090-Transfer"`.
6. The Mac independently verifies the SHA256, raw row counts, common-random-
   number pairing, final-versus-pretrain deltas, graph-versus-flat deltas, and
   clinical noninferiority before any result commit.
