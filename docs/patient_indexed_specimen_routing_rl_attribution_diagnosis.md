# Routing-Primary DDPG Online-Attribution Diagnosis

Status: completed read-only development diagnosis, 2026-08-12

Branch: `rl-attribution-refinement`

Source state: `patient-indexed-specimen-routing` at
`8a7768e3495be6d63e2091445ac6a7aa28ec0558`

## 1. Scope

This report diagnoses why the completed routing-primary DDPG campaign did not
show a reliable final-versus-frozen-pretrain online-learning gain. It uses only
immutable Stage B development artifacts. It does not train a policy, run a new
simulator trajectory, select a checkpoint, reuse the formal holdout stream, or
authorize a new experiment.

The principal conclusion is:

> Online DDPG is active, but its critic becomes poorly calibrated and loses
> action-ranking quality on the actual online replay distribution after an
> early improvement. The evidence is most consistent with bootstrapped critic
> instability or overestimation, compounded by a teacher-to-policy support
> mismatch. It is not evidence that more DDPG episodes will solve the problem.

This is a mechanism diagnosis, not a causal proof. Critic diagnostics are
in-sample retrospective measurements, and checkpoint associations use only
five time points.

## 2. Locked inputs

- Stage B checkpoint-curve SHA256:
  `0016e3bb8fa779984de1a21015bcade6f377d72a231b7581adc14ffc5a652ecf`
- Training manifest SHA256:
  `0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2`
- Routing teacher SHA256:
  `9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92`
- Development evaluation seed: `93100000`
- Formal holdout seed `91100000`: not used
- Training seeds: 10-14
- Checkpoints: pretrain and episodes 25, 50, 75, and 100
- Fixed online replay audit: 26,000 pooled transitions per architecture

The diagnostic implementation is
`evaluation/diagnose_ddpg_online_attribution.py`. Its deterministic JSON output
for this audit had SHA256
`15e8f726f3266c9b5fbbb223e1dc261c0e09e082c6c1dd106912ff099b192420`.

## 3. Checkpoint evidence

Cost differences below are candidate minus frozen pretrain, so negative values
favor online learning. Replay ranking compares the critic's behavior-action
advantage over the MDL-2 anchor with the stored four-step anchor-relative
return on the fixed final replay set.

### GCN-DDPG

| Checkpoint | Cost difference | Replay Spearman | Pairwise order | Prediction SD | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrain | 0 | 0.191 | 0.564 | 0.00092 | 0.371 |
| Episode 25 | -162,215 | 0.381 | 0.634 | 0.00805 | 0.061 |
| Episode 50 | -216,664 | 0.312 | 0.603 | 0.00990 | 0.035 |
| Episode 75 | -63,175 | 0.207 | 0.568 | 0.00934 | 0.029 |
| Episode 100 | +5,438 | 0.159 | 0.553 | 0.00878 | 0.023 |

The point-estimate cost gain is largest at episode 50, while replay ranking
peaks earlier and then erodes. The replay target SD is only 0.00163, whereas
the post-online critic prediction SD is roughly 0.008-0.010. Calibration slope
falls from 0.371 to 0.023. The critic therefore expands small advantage
differences without preserving enough of their ordering. The same erosion is
visible when each checkpoint is restricted to replay transitions observed by
that time: GCN replay Spearman declines from 0.355 at episode 25 to 0.159 at
episode 100, so the fixed-final-replay result is not solely a future-state
distribution artifact.

Across these five checkpoints, the descriptive correlation between lower GCN
cost and better replay Spearman ranking is -0.882; the corresponding
correlation with pairwise order accuracy is -0.843. These values are
descriptive because five checkpoints are not an independent statistical
sample.

### Matched-flat DDPG

| Checkpoint | Cost difference | Replay Spearman | Pairwise order | Calibration slope |
| --- | ---: | ---: | ---: | ---: |
| Pretrain | 0 | 0.171 | 0.555 | 0.469 |
| Episode 25 | +165,915 | 0.155 | 0.552 | 0.149 |
| Episode 50 | +216,308 | 0.112 | 0.536 | 0.061 |
| Episode 75 | +191,519 | 0.040 | 0.513 | 0.029 |
| Episode 100 | +259,490 | 0.020 | 0.508 | 0.008 |

Flat DDPG shows the same mechanism more strongly: ranking approaches chance
and every online checkpoint is worse than frozen pretrain by point estimate.

## 4. Teacher-policy support mismatch

The routing teacher contains 314 rows and 13 candidate options. The globally
best option belongs to:

- specimen transfer for 154 rows;
- reagent transfer for 89 rows;
- combined routing for 42 rows; and
- the anchor for 29 rows.

The deployed residual policy can change specimen transfer only. Counting the
anchor as executable support, 183 of 314 teacher rows are policy-supported and
131 of 314, or 41.72%, have a winning action outside policy support.

The completed DDPG confirmation and the current Stage C TD3 config do not set
`critic_teacher_advantage_calibration.allowed_option_groups`. Their critic
ranking regularizer can therefore sample reagent and combined-routing winners
that the actor cannot reproduce. On policy-supported teacher rows, GCN critic
Spearman remains roughly stable from 0.315 at pretrain to 0.302 at episode 100,
while online-replay Spearman declines to 0.159. The support mismatch is thus a
plausible source of avoidable regularization noise, but it does not by itself
explain the online replay failure.

## 5. The actor is active but unreliable

The failure is not a dormant residual policy:

- all runs execute exactly 52 online updates per episode;
- actor updates occur after warmup;
- persisted losses are finite;
- GCN final versus pretrain changes routing in 98.1% of paired outcomes;
- flat final versus pretrain changes routing in 94.0% of paired outcomes.

For GCN final minus pretrain, mean total cost changes by +5,438. This combines
an increase of +85,888 in base operating cost with reductions of -72,500 in
patient-loss cost and -13,200 in expiry cost. Bioreactor shortage cost alone
increases by +77,311. Patients lost improve by 0.145 per replication, while
patients completed decrease by 0.046. Route count increases by 0.323 and route
distance by 21.8 miles. The policy is making material patient-level decisions,
but the economic and clinical effects do not align reliably.

Heterogeneity is substantial. GCN final-minus-pretrain cost changes range from
-512,241 to +602,970 across training seeds. It improves compound stress and
nominal history by point estimate, but worsens abrupt regime shift. Matched
flat worsens every scenario by point estimate.

## 6. What the evidence rules out

The retained artifacts do not support the following explanations:

- zero online updates;
- an inactive correction gate or zero residual use;
- a frozen actor;
- nonfinite persisted losses;
- MPS-to-CPU fallback;
- simply too few episodes, because performance and critic ranking erode after
  their early peaks;
- online imitation or pretrain-reference regularization being the sole brake,
  because the actor changes behavior materially and the weighted reference
  loss is small relative to the actor objective.

Bellman loss alone is not a sufficient health signal. GCN mean Bellman loss
falls from approximately 0.0099 in episodes 1-25 to 0.0079 in episodes 76-100
while replay ranking and development performance deteriorate.

## 7. Stage C interpretation

Howard's locked matched TD3 Stage C is the correct next experiment. Its
implementation adds twin critics, clipped double-Q targets, target-policy
smoothing, and delayed policy updates while keeping the teacher, residual
action contract, training budget, and GCN-versus-flat comparison matched.
Those changes directly test the leading critic-instability explanation.

Stage C does not add policy-support filtering to teacher calibration. This is
intentional for the current gate: changing both the backbone and teacher
support would confound the TD3 comparison.

## 8. Bounded next decision

1. Do not launch new DDPG training while Stage C is in progress.
2. If matched TD3 passes its locked development gate, promote TD3 according to
   the main execution plan and do not run an additional DDPG refinement.
3. If matched TD3 fails, prepare exactly one matched DDPG development
   refinement under the final material dual-confirm contract. Its single
   scientific change is support-aligned critic calibration with
   `allowed_option_groups: ["specimen_transfer"]` for both GCN and flat.
4. That fallback must use fresh development seeds and CRNs, a new output root,
   and the same episode and update budget for GCN and flat. It is not approved
   for launch by this report; a locked config and preflight review are required.
5. Do not reopen residual-scale, update-frequency, actor-frequency, actor-LR,
   exploration, imitation-weight, reference-weight, or broad HPO sweeps. Those
   dimensions have already received targeted development tests.

## 9. Reproduction

From a source checkout containing the diagnostic implementation, with the
immutable campaign artifacts available under the persistent artifact root:

```bash
PYTHONPATH=. python -m evaluation.diagnose_ddpg_online_attribution \
  --source-root /path/to/source \
  --artifact-root /path/to/persistent/campaign \
  --output /path/to/ddpg_online_attribution_diagnosis.json
```

The script verifies the three locked input hashes, rejects the formal holdout
seed, and fails on missing, nonfinite, mismatched, or duplicate evidence.
