# RTX 4090 Regional Matched-Ablation Audit

## Scope

This note records the independently verified result of the regional CUDA
pipeline at repository commit `d6ac371`. The evaluated scenario is
`patient_condition_geo_regional_drift`, with three training seeds and 300
paired holdout replications per seed.

The pipeline trained the actor through advantage-filtered teacher
distillation:

- Stage 1: 300 supervised epochs.
- Stage 2: 100 DAgger-1 supervised epochs.
- Stage 3: 100 DAgger-2 supervised epochs.
- Stage 4: fixed-candidate validation and paired holdout evaluation.

Every training configuration used `num_episodes=0`. These results therefore
establish a graph-policy distillation advantage, not yet an actor-critic RL
advantage.

## Provenance

- Source archive: `regional_cuda_results.zip`
- Archive size: `37,937,682` bytes
- SHA256:
  `70478c382abe0837a9984d467489591db18619c86237a57562fa92a9d4868b28`
- Training runs: 18/18 complete.
- Stage-4 evaluations: 6/6 complete.
- CUDA configuration: 18/18 training snapshots specify `device="cuda"`.
- Errors: no Traceback, NaN, OOM, or CPU fallback found.
- Raw evaluation rows checked: 5,400; no non-finite values.
- Paired holdout seed: `39500000`.
- Independent aggregation check: maximum absolute difference from the
  pipeline summary was `0.0`.

The raw archive and checkpoints remain in private research storage. They
should not be committed to the public source repository.

## Pooled Results

The pooled intervals use the pipeline's hierarchical two-level bootstrap over
three training seeds and 900 paired holdout observations.

| Comparison | Cost difference | Relative gap | Paired 95% CI | Cost win rate |
|---|---:|---:|---:|---:|
| GCN residual vs MDL-2 | -$2,018,879.59 | -0.081919% | [-$3,041,769.21, -$839,151.74] | 60.11% |
| Flat residual vs MDL-2 | +$1,807,332.91 | +0.073335% | [+$1,269,358.03, +$2,423,579.72] | 28.56% |
| GCN residual vs flat residual | -$3,826,212.50 | -0.155141% | [-$5,180,060.53, -$2,474,554.13] | 68.33% |

For GCN residual versus MDL-2:

| Clinical metric | Mean difference | Paired 95% CI | Direction |
|---|---:|---:|---|
| Completion service level | +0.000576 | [+0.000295, +0.000840] | Favorable |
| Patients lost | -4.1289 | [-5.8111, -2.1422] | Favorable |
| Manufacturing ineligibility rate | -0.000243 | [-0.000344, -0.000139] | Favorable |

All three GCN seeds pass the preregistered clinical noninferiority checks. All
three flat seeds fail them.

## Per-Seed Cost Results

| Method | Seed | Cost difference vs MDL-2 | Relative gap | Paired 95% CI | Residual usage |
|---|---:|---:|---:|---:|---:|
| GCN residual | 0 | -$2.458M | -0.099754% | [-$3.526M, -$1.391M] | 68.35% |
| GCN residual | 1 | -$0.818M | -0.033175% | [-$1.644M, +$0.008M] | 46.12% |
| GCN residual | 2 | -$2.781M | -0.112829% | [-$3.716M, -$1.846M] | 63.10% |
| Flat residual | 0 | +$1.365M | +0.0554% | Entirely above zero | 19.79% |
| Flat residual | 1 | +$1.647M | +0.0668% | Entirely above zero | 15.82% |
| Flat residual | 2 | +$2.409M | +0.0978% | Entirely above zero | 25.15% |

## Interpretation

The evidence supports the following statement:

> Under patient-condition-aware geographic regional demand drift, the
> distilled graph residual policy outperforms the parameter-matched flat
> residual policy and yields a small but statistically significant improvement
> over MDL-2 without degrading the prespecified clinical outcomes.

It does not yet support:

> GCN-DDPG reinforcement learning outperforms MDL-2.

The latter claim requires stable, nonzero actor-critic updates starting from
the validated Stage-3 checkpoints. Earlier 100-episode DDPG fine-tuning was
unstable across seeds and is not sufficient evidence.

## Release Status

Safe to publish now:

- this audit note;
- the committed pipeline/configuration code;
- the archive hash and provenance metadata;
- compact aggregate and per-seed statistics.

Keep private for now:

- raw ZIP and model checkpoints;
- patient- or institution-restricted input data, if any;
- any result labelled as final RL superiority.

Release as the final paper artifact only after the conservative actor-critic
campaign passes the matched graph-versus-flat and multi-scenario gates defined
in `plan.md`.
