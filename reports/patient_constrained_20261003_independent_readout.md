# Independent Patient-Constrained Raw Readout

Scope: 144 evaluation gzip files, 9,216 epochs, 36 matched worlds, four frozen evaluation arms. Costs were recomputed from all 14 components across epochs 0-63; losses/deliveries were counted from unique final patient IDs. All cohorts match within world; no unresolved patients remain. No project analysis/runner/learner/environment code was imported or executed.

## Paired Results

Savings = comparator cost minus constrained cost, in **million synthetic cost units**; negative is worse. Extra lost = constrained minus comparator. Each row averages 12 paired worlds (3 independently trained blocks, 4 worlds each). Extra delivered is exactly the negative of extra lost for every pair.

| Condition | Comparator | Savings, M | Extra lost | Positive savings |
|---|---|---:|---:|---:|
| No change | Fixed | -1.015 | 26.75 | 4/12 |
| No change | Cost-only | 0.884 | -4.17 | 7/12 |
| No change | ID-MPC | -2.717 | 24.17 | 2/12 |
| Persistent | Fixed | -2.830 | 40.50 | 2/12 |
| Persistent | Cost-only | 0.448 | -5.33 | 5/12 |
| Persistent | ID-MPC | -3.378 | 37.67 | 0/12 |
| Fast fluctuation | Fixed | -2.553 | 52.17 | 4/12 |
| Fast fluctuation | Cost-only | 0.555 | -5.25 | 9/12 |
| Fast fluctuation | ID-MPC | -1.925 | 34.92 | 3/12 |

Persistent arm means (cost M; lost; delivered): Constrained 42.181; 191.83; 97.17 | Fixed 39.352; 151.33; 137.67 | Cost-only 42.630; 197.17; 91.83 | ID-MPC 38.804; 154.17; 134.83.

Persistent block results, ordered blocks 0 / 1 / 2:

| Comparator | Savings, M by block | Extra lost by block |
|---|---|---|
| Fixed | -3.967 / -2.898 / -1.624 | 66.75 / 51.75 / 3.00 |
| Cost-only | -0.000 / -0.282 / 1.627 | 0.00 / -7.75 / -8.25 |
| ID-MPC | -3.883 / -3.211 / -3.039 | 62.75 / 41.00 / 9.25 |

**The prespecified patient-preserving training signal is false.** All three persistent block means are worse than fixed, and extra losses versus fixed are positive in every condition. The reported persistent-versus-fixed descriptive 95% savings interval is [-4.089, -1.313] M; the reported extra-loss interval is [3.915, 65.919]. Their signs and the main screen were reconciled; bootstrap generation was not duplicated. Constrained beats cost-only on pooled condition means, but persistent cost savings occur in only one of three blocks.

## Actions And Multipliers

Persistent control epochs 0-47 only; hours are total projected commitments across four sites per decision, not contemporaneously applied labor. Zero-site fractions describe the saved post-clipping requests.

| Block | Constrained hours | Cost-only hours | Constrained zero sites | Cost-only zero sites |
|---|---:|---:|---:|---:|
| 0 | 0.097 | 0.054 | 737/768 | 752/768 |
| 1 | 1.209 | 0.067 | 94/768 | 747/768 |
| 2 | 8.000 | 7.976 | 0/768 | 0/768 |

Constrained block2 hits the shared eight-hour limit at 184/192 persistent decisions, while neural site requests never reach four hours. All constrained-versus-fixed pairs differ at all 48 action boundaries, but block2's mean total L1 difference is only 0.0504 hours per episode; a changed-action count alone overstates its magnitude. Pre-clipping logits and gradients were not examined, so these action patterns do not establish a learning-failure mechanism.

All 36 multiplier receipts reconcile with clip(previous + 0.01 * paired training extra losses, 0, 20): 32 increases, 4 decreases, no bound contacts, after-values 0.80-3.68. Final vectors in condition order no-change / persistent / fast are block0 [1.94, 2.77, 3.68], block1 [1.10, 1.17, 1.12], block2 [2.23, 2.00, 1.95]. The cap did not bind; receipt arithmetic does not establish the cause of infeasibility. Training raw trajectories were not re-read.

## Components And Reconciliation

Persistent constrained-minus-fixed ledger means: patient-loss cost +2.025 M, expiry +1.390 M and bioreactor shortage +0.972 M, partly offset by reagent purchases -1.589 M and reagent shortage -0.173 M. Versus cost-only, patient-loss cost is -0.267 M and expiry -0.220 M, with reagent purchases +0.116 M. These are arithmetic contributions, not causal mechanisms or evidence that reward weights are wrong. All component means and condition/block differences are in the JSON.

The independent check agrees with comparison.json across 4,870 numeric fields, including per-world costs/components, paired means, block means, action differences and the main screen. Maximum absolute difference: 1.4901161193847656e-8; none exceeds 1e-6. Raw epoch cost-versus-component sums differ by at most 4.656612873077393e-10. Unique patient counts and new-loss counts agree exactly. Differences are numerical summation precision, not a substantive discrepancy.

Evidence: compressed-raw aggregate SHA256 `5c5affb357c069d2231c30148f9a23c7c27706b97c263ff158fd6e8807cd52f4`; comparison.json SHA256 `02c074e51c7bfd6fdd9e7e3598479fcd472ded78553d50687f56ddcad99f1b9b`. The JSON contains the aggregate formula, 144 file hashes, three update-file hashes, all 108 matched differences, 144 world totals, condition/block/component means and receipts.

Limitations: independent arithmetic is not independent scientific replication; only three training blocks were used. These are synthetic lost/delivered states, not clinical outcomes or validated safety evidence. This is simulation training followed by frozen evaluation, not deployment-online adaptation or isolated graph attribution. No processes, locks, archive or older history were checked. No source/results edits or scientific execution occurred.
