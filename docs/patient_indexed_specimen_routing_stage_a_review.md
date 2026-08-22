# Routing-Primary Stage A Review: Transport Timing

Review date: 2026-08-11

Plan commit: `ad46ae48cbd7da9910bfe14c192b6fbf5c0e8188`

Training commit: `9c0718b0da49d34f7f878ed2f036e96bf7bb3693`

Evaluation asset commit: `df59a4448d29b5cac589ec90a261cd9ea6a90439`

## Protocol

Stage A evaluated the frozen final AFR-GCN-DDPG and matched AFR-Flat-DDPG
policies for training seeds 10-14. Each timing assumption used one routing
scenario and 100 paired CRN replications per algorithm and training seed.
Routing MDL-2 was the paired anchor. No training, checkpoint selection,
deployment retuning, or geography-speed sweep was performed.

The two assumptions were:

- Specimen availability lead 0 epochs instead of the primary lead of 1 epoch.
- Finished-product return lead 1 epoch instead of the primary immediate return.

The locked development-independent inputs were:

- Lead-0 config SHA256:
  `35769ea8ead9706f853d875a014bfd1568682bedf489f3954bc65b06bfef994b`.
- Return-1 config SHA256:
  `2fcb518bf6bb1f2f61ab64851cb5e1558ef3b7171aa472306360ff2ddb22120d`.
- Training manifest SHA256:
  `0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2`.
- Teacher SHA256:
  `9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92`.
- Launcher SHA256:
  `57f38ddcbeb0533ebfdb93dd831bce25371bf3caed531efacdc0c6b8fb0d94cd`.

## Terminal audit

Both phases completed with exit code 0. Each phase contained exactly 10 run
summaries, 1,000 learned-policy rows, and 1,000 anchor rows. All runs used the
expected scenario and 100 unique CRN keys. Persisted numeric fields were
finite, all stderr logs were empty, and no Traceback, native crash, OOM,
MPS/CPU fallback, NaN/Inf, duplicate process, or hash mismatch was found.

Routing and learned-policy use were material rather than nominal:

| Phase | Learned routes | Anchor routes | Corrected decisions | Applied residual L1 |
| --- | ---: | ---: | ---: | ---: |
| Lead 0 | 393,338 | 138,410 | 35,151 | 12,046.80 |
| Return 1 | 413,345 | 149,560 | 36,570 | 12,264.71 |

## Paired results

Differences are candidate minus comparator. Negative total-cost differences
favor the candidate. Intervals are two-level paired 95% bootstrap intervals
over five training seeds and 100 CRN replications per seed.

| Timing assumption | Comparison | Cost difference | Relative difference | 95% interval |
| --- | --- | ---: | ---: | ---: |
| Lead 0 | GCN minus MDL-2 | +16.742 million | +0.633% | [+14.294, +19.266] million |
| Lead 0 | GCN minus flat | +6.773 million | +0.255% | [+3.760, +10.327] million |
| Lead 0 | Flat minus MDL-2 | +9.969 million | +0.377% | [+7.977, +12.009] million |
| Return 1 | GCN minus MDL-2 | -29.181 million | -1.050% | [-31.332, -27.012] million |
| Return 1 | GCN minus flat | -8.369 million | -0.303% | [-10.646, -6.248] million |
| Return 1 | Flat minus MDL-2 | -20.811 million | -0.749% | [-22.503, -19.116] million |

Under lead 0, GCN reduced patients lost relative to MDL-2 by 14.204 on
average, but its completion service level was lower by 0.00131 and its total
cost was higher. Under return lead 1, GCN improved total cost, completion
service level, manufacturing ineligibility, and patients lost relative to both
MDL-2 and matched flat.

## Stage decision

The execution and data-integrity gate passed. Directional timing robustness did
not pass as a blanket claim: the primary GCN advantage strengthened under a
one-epoch finished-product return delay but reversed under immediate specimen
availability. The manuscript must report this asymmetry and must not claim
general transport-timing robustness.

Stage B proceeds unchanged because it diagnoses online DDPG learning on a new
development CRN stream under the primary four-scenario protocol; it does not
select a policy using Stage A or the formal holdout.
