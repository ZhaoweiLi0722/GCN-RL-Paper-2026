# Reporting audit — 2026-09-15

Follow-up to blockers A and B of `submission_readiness_review_2026-09-15.md`.
Branch `e2b-run`. No training, evaluation, checkpoint selection, or artifact
replacement was performed. Existing evidence files are unchanged.

## 1. Demand-drift cost/weight contradiction: resolved by provenance

The manuscript's five-seed demand-prior-drift table (`tab:formal_drift_levels`)
reports an MDL-2 objective of 1.268257 billion with 3,665.59 patients lost,
while `tab:cost_coefficients` states a patient-loss weight of 500,000 "held
fixed across policies". Under that weight the patient-loss term alone would be
1.83 billion, exceeding the total.

Git history resolves it. The drift scenario config was created on 2026-07-23
(commit `1ce269f`, "Add robust residual RL and matched graph ablation") with

```
"weight_patient_lost": 50000,
"weight_expiry": 40000,
"weight_urgency": 5000,
```

Three days later, commit `1182003` (2026-07-26, "Add network AFR regional-shift
evaluation") raised the drift config to 500,000 / 100,000 / 25,000 and created
the four routing scenario configs already at those values. The demand-drift
five-seed study (`specs/2026-07-20-demand-drift-robustness/results.md`) and the
RL-family comparison that repeats its table (`specs/2026-07-23-rl-family-comparison`)
were therefore run under the 50,000 / 40,000 / 5,000 weights. The paired JSON
beside the drift note records `baseline_mean = 1,268,256,613`, matching the
table exactly.

Arithmetic check under the recovered weights: 3,665.59 × 50,000 = 183.3 million
and 5,206.41 × 5,000 = 26.0 million, both comfortably inside the 1.268 billion
total. The contradiction was a weight-regime mismatch between two studies, not
a units error or a different objective.

Consequences for the manuscript:

- Every table derived from the demand-drift study (`tab:demand_drift_results`,
  `tab:formal_drift_levels`, `tab:formal_drift_pairs`) must state that it used
  the earlier 50,000 / 40,000 / 5,000 penalty weights.
- `tab:cost_coefficients` must say the 500,000 / 100,000 / 25,000 weights apply
  to the regional-regime and routing-primary studies from 2026-07-26 onward,
  not to all calibrated experiments.
- Relative effects across the two weight regimes are not comparable; the
  drift-study percentages must not be placed beside routing-primary
  percentages as if measured on one objective.
- The routing-primary cost-component reconciliation (blocker D, 14/14 hashes)
  is unaffected. It was produced under 500,000 and reconciles under 500,000.

The `_severe` drift config carried the same 50,000 weight over the same window.

## 2. Formal row-level results: not on this machine

The compact publication evidence (`experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/`)
was frozen in commit `c3b8d8a` on 2026-08-17 by the collaborator on a machine
named `lis-MacBook-Pro.local`. All raw paths it records fall under the
gitignored root

```
results/patient_indexed_specimen_routing_mac_mps_primary/
```

which does not exist in this checkout. A search of the home directory,
mounted volumes, iCloud Drive, Spotlight, and Time Machine on this machine
found no copy. Recovery requires the collaborator to export the following.

Row-level formal holdout (relative to the root passed as
`--raw-formal-final-root`; layout `<root>/<algorithm>/seed<NN>/<file>`), with
the SHA-256 values the audit recorded:

| File | rows | SHA-256 |
| --- | ---: | --- |
| `gcn_residual_mdl2_network_ddpg_afd/seed10/holdout_rows.csv` | 400 | `019176b6…78e64` |
| `gcn_residual_mdl2_network_ddpg_afd/seed11/holdout_rows.csv` | 400 | `0f5b3c38…741a0a` |
| `gcn_residual_mdl2_network_ddpg_afd/seed12/holdout_rows.csv` | 400 | `c0f0c50e…df646f` |
| `gcn_residual_mdl2_network_ddpg_afd/seed13/holdout_rows.csv` | 400 | `3af8fb8a…fcfd109b` |
| `gcn_residual_mdl2_network_ddpg_afd/seed14/holdout_rows.csv` | 400 | `73e093af…3ccc72a9c` |
| `gcn_residual_mdl2_network_ddpg_afd/seed1{0..4}/holdout_anchor_rows.csv` | 400 each | see `formal/cost_component_summary.json` |

Full hashes are in `formal/cost_component_summary.json` under `source_files`.
Also required, because the compact summaries reference them and the graph-
versus-flat and final-versus-frozen contrasts need them:

- `flat_residual_mdl2_network_ddpg_afd/seed1{0..4}/holdout_rows.csv` and
  `holdout_anchor_rows.csv` (same root).
- The frozen-pretrain evaluation tree for both algorithms and all five seeds.
- `.../ddpg_routing_primary_100/patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100/training_manifest.json`
  and the ten `checkpoints/*_seed1{0..4}_episode100.pt` files.
- `.../development/ddpg_online_attribution_100/{pretrain,episode25,episode50,episode75,episode100}/summary.json`
  and their row files.
- The lead-0 and return-1 sensitivity evaluation trees.

The export should be a read-only archive with a manifest of SHA-256 hashes so
the ten recorded hashes can be verified before any recomputation.

## 3. Crossed-design bootstrap: method built, validated, partially run

### Why

The formal holdout is crossed: every training seed is evaluated on the same
CRN worlds, indexed by `(scenario, replication)`. The publication intervals
come from `paired_two_level_summary`, which resamples seeds and then resamples
rows independently inside each seed, treating shared worlds as independent per
seed and pooling scenarios in the inner draw.

### Tool

`evaluation/crossed_design_bootstrap_audit.py` recomputes each paired
contrast under four schemes on the same rows:

| Scheme | Resamples | Captures |
| --- | --- | --- |
| `nested_seed_then_row` | seeds, then rows within seed (the publication method, called directly) | reproduction baseline |
| `world_cluster` | worlds, stratified by scenario, same draw for every seed | world-level common uncertainty; fixed scenario mixture |
| `seed_cluster` | seeds only | policy-training variation |
| `two_way` | seeds and worlds jointly (Owen's pigeonhole bootstrap) | both; generally conservative |

It also reports the mean pairwise correlation across seeds of the world-level
differences, a two-way variance-component decomposition, and CRN checks that
the anchor cost is bit-identical across training seeds and across algorithms.

### Validation on local Stage C TD3 development rows

Root: `results/patient_indexed_specimen_routing_stage_c_td3_development/evaluation/{final,pretrain}`,
three seeds, four scenarios, 50 worlds per scenario. Output:
`reports/reporting_audit_2026-09-15/td3_dev_crossed_bootstrap.{json,md}`.

- CRN checks: anchor spread across seeds and across algorithms is exactly 0.
  World sharing is confirmed, not inferred.
- The nested scheme reproduces the stored `summary.json` point estimates to
  the last digit. Its interval differs from the stored one by about 1% of the
  width because the stored run seeded the generator per metric
  (`seed + index`) while the audit uses one seed; this is Monte Carlo
  resampling variation, not a method difference.
- World-level differences are strongly correlated across seeds (0.50–0.73
  for the favorable contrasts). The dependence the review flagged is real.

| Contrast (TD3 dev, total cost) | mean | nested | world_cluster | seed_cluster | two_way |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN − anchor | −19.65M | [−23.09, −16.78] | [−21.53, −17.79] | [−23.18, −17.31] | [−23.21, −16.34] |
| Flat − anchor | −8.61M | [−9.98, −7.28] | [−10.18, −6.98] | [−9.47, −8.04] | [−10.45, −6.75] |
| GCN − flat | −11.04M | [−13.81, −8.50] | [−12.93, −9.14] | [−13.72, −8.98] | [−14.13, −8.04] |
| GCN final − frozen | +0.02M | [−0.33, +0.35] | [−0.26, +0.31] | [−0.17, +0.14] | [−0.40, +0.44] |
| Flat final − frozen | +0.17M | [−0.23, +0.57] | [−0.17, +0.51] | [−0.07, +0.36] | [−0.33, +0.67] |

Intervals in millions of cost units. The two-way interval is 9–38% wider than
the nested one, so the publication method is mildly anticonservative on this
design, but no conclusion changes: the three favorable contrasts exclude zero
under every scheme and neither final-versus-frozen contrast excludes zero
under any scheme.

### What could be run on the formal evidence without rows

The compact `final_summary.json` and `pretrain_summary.json` contain per-seed,
per-scenario holdout cell means. The seed-level cluster bootstrap and a
5-seed t interval were computed from them. Output:
`reports/reporting_audit_2026-09-15/formal_compact_seed_level.{json,md}`.

| Contrast (formal DDPG, total cost) | mean | published nested | seed_cluster | seed t (df 4) |
| --- | ---: | ---: | ---: | ---: |
| GCN − MDL-2 | −17.69M | [−19.36, −15.73] | [−19.17, −15.85] | [−20.70, −14.68] |
| Flat − MDL-2 | −9.09M | [−10.59, −7.51] | [−10.44, −7.55] | [−11.62, −6.56] |
| GCN − flat | −8.60M | [−11.06, −6.43] | [−10.99, −6.64] | [−12.60, −4.60] |
| GCN final − frozen | +0.07M | not published | [−0.01, +0.14] | [−0.07, +0.21] |
| Flat final − frozen | −0.14M | not published | [−0.45, +0.17] | [−0.70, +0.42] |

Point estimates match the published values exactly. The anchor cell means are
identical across seeds, confirming world sharing in the formal run as well.
The seed-level t interval, the most conservative treatment of five training
seeds, still excludes zero for all three favorable contrasts. Neither
final-versus-frozen contrast excludes zero. This is the first directly paired
final-minus-frozen interval for the formal run; the review noted its absence.

### Still pending the rows

World-cluster and two-way intervals for the formal run cannot be computed from
cell means. Given the development-data pattern (two-way roughly 10–40% wider
than nested) and the formal margins above (GCN − MDL-2 upper bound is 15.7M
below zero on a 3.6M-wide interval), a sign reversal is implausible, but that
is an extrapolation and must not be written into the manuscript. When the
archive arrives:

```bash
python3 evaluation/crossed_design_bootstrap_audit.py \
  --root <formal_final_root> --frozen-root <formal_pretrain_root> \
  --algorithms gcn_residual_mdl2_network_ddpg_afd flat_residual_mdl2_network_ddpg_afd \
  --out-json reports/reporting_audit_2026-09-15/formal_crossed_bootstrap.json \
  --out-md   reports/reporting_audit_2026-09-15/formal_crossed_bootstrap.md
```

Reporting recommendation once complete: keep the published nested intervals
as the prespecified primary, add the two-way interval as the documented
sensitivity, and state the scheme in the methods text. Replacing the primary
interval would be a change-control amendment under the locked plan.
