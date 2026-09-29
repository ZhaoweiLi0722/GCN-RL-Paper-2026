# Formal cost inference: raw rows recovered and crossed audit completed

## Decision

The recovered raw rows support the existing graph/package cost conclusions
under crossed seed/world resampling. They still do not establish an incremental
online DDPG benefit. This strengthens reproducibility and uncertainty reporting;
it is not a new experiment or a newly positive RL result.

| Contrast (first minus second) | Mean, million cost units | Relative mean | Crossed 95% interval, million |
| --- | ---: | ---: | ---: |
| GCN final - MDL-2 | -17.689540 | -0.658474% | [-19.805786, -15.354809] |
| Flat final - MDL-2 | -9.085675 | -0.338205% | [-10.895859, -7.239471] |
| GCN final - flat final | -8.603865 | -0.321357% | [-11.284344, -6.087716] |
| GCN final - frozen pretrain | +0.072903 | +0.002732% | [-0.180322, +0.313172] |
| Flat final - frozen pretrain | -0.139378 | -0.005206% | [-0.546384, +0.278095] |

Negative differences favor the first policy. Percentages use each contrast's
comparator mean, not a newly selected avoidable-cost denominator. Cost units
are modeled objective units, not calibrated dollars. The online intervals
cross zero; this is not a proof of equivalence or of exactly zero online value.

## Data recovered

The old persistent source is
`/Users/lizhaowei/gcnrl-patient-routing-persistent/results/patient_indexed_specimen_routing_mac_mps_primary/confirmation/ddpg_routing_primary_100_evaluation`.
Both `formal_final` and `formal_pretrain` were read without modification.

- Forty CSVs, 400 rows each: 16,000 rows, comprising 8,000 learned-policy rows
  and 8,000 anchor rows across the two phases. Each individual policy/phase
  comparison has five training seeds crossed with four scenarios and 100 worlds.
- Both source summary files are byte-identical to the tracked historical
  final/pretrain summaries. The ten GCN-final CSVs have matching historical
  hashes in the old cost-component manifest.
- The remaining thirty CSVs have new hashes and all their scenario cost means
  reconcile to the corresponding historical summary. Historical byte-level
  provenance is not claimed for those thirty files.
- All 160 CSV/scenario means reconcile within 1e-5 absolute cost units. Row
  identities, horizon, graph labels and exact scenario/replication keys pass.
  Anchor costs agree exactly across phases, algorithms and all training seeds.
- The additive local projection contains nine identity/cost fields only. It
  excludes detailed patient trajectories and clinical metrics. Nothing in this
  audit re-estimates clinical noninferiority or justifies its margins.

The September 15 note that formal rows were unavailable on Howard's machine
was a machine-specific limitation, now resolved for this local cost audit.
It was not evidence that the underlying formal evaluation was missing.

## Analysis and independent checks

Recovery source/protocol freeze: `e1de6d3`. Howard's crossed-bootstrap program
was used unchanged, with 20,000 resamples, alpha 0.05 and bootstrap seed 0.
World resampling is stratified by the four original scenarios and shared
across selected training seeds. All five contrasts and all four resampling
methods are retained in `reports/2026-09-29-formal-crossed-audit/`.

An independent verifier reconstructed contrasts by explicit row keys and
computed the crossed bootstrap using seed/world multiplicity weights instead
of the original array-indexing implementation. All five means and crossed
interval endpoints agree within 1e-6 cost units; all forty projected CSV
hashes pass. See `verification.json`. This checks implementation, not exact
finite-sample coverage. Only five independently trained seeds are available.

Compared with the newly recomputed seed-0 nested intervals, crossed widths
increase by 21.50% (GCN-anchor), 17.54% (flat-anchor), 9.99% (GCN-flat), 26.68%
(GCN online increment) and 13.79% (flat online increment). These width ratios
refer to the audit's common settings, not the historical published intervals.
The historical aggregate uses holdout_seed + 500,000 and the graph-flat offset
+1; its Monte Carlo draws differ from this audit's seed 0. Its original values
remain untouched and remain the declared primary analysis.

The existing compact-summary audit has a separately identified t-quantile
lookup issue: its table is indexed inconsistently with `n-1`, giving 3.182
rather than 2.776 for five seeds at 95%. That code path is not used here.
Do not copy its seed-t interval into the manuscript as independently verified.
The row-level four-scheme audit above is unaffected.

## Reproduction

From the integration worktree, with the project virtual environment:

```bash
python -m evaluation.recover_formal_attribution_rows \
  --source /Users/lizhaowei/gcnrl-patient-routing-persistent/results/patient_indexed_specimen_routing_mac_mps_primary/confirmation/ddpg_routing_primary_100_evaluation \
  --output reports/2026-09-29-formal-crossed-audit/projection
python -m evaluation.crossed_design_bootstrap_audit \
  --root reports/2026-09-29-formal-crossed-audit/projection/formal_final \
  --frozen-root reports/2026-09-29-formal-crossed-audit/projection/formal_pretrain \
  --algorithms gcn_residual_mdl2_network_ddpg_afd flat_residual_mdl2_network_ddpg_afd \
  --metric total_cost --resamples 20000 --alpha 0.05 --seed 0 \
  --out-json reports/2026-09-29-formal-crossed-audit/formal_crossed_bootstrap.json \
  --out-md reports/2026-09-29-formal-crossed-audit/formal_crossed_bootstrap.md
python -m evaluation.verify_formal_crossed_audit \
  --root reports/2026-09-29-formal-crossed-audit \
  --output reports/2026-09-29-formal-crossed-audit/verification.json
```

These are the recorded commands, not instructions to overwrite the existing
packet. Recovery and verification refuse existing destinations. Reproductions
must use a separate root throughout. The validated projection also permits
reanalysis without access to the much larger original trajectory CSVs.

## Manuscript boundary

Add this as a labeled post-hoc cost sensitivity, preserving the original main
table and clinical analysis. Supported wording: the package/graph cost effects
remain favorable after accounting for shared evaluation worlds; a separate
increment from online DDPG updates remains unestablished. Do not claim all
scenarios, transport timings or clinical endpoints are robust from this test.
No fresh training/evaluation, checkpoint selection, remote push or main merge
occurred. Further paper needs include clinical-margin rationale, operational
calibration and a fair stronger-planner comparison, independently of online RL.

## Validation of this delivery

The combined focused suite passed 110 tests (recovery, statistics, independent
crossed verification, queue mechanics/diagnostics and preceding control
fixtures). Full `compileall` and `git diff --check` passed. All five rounded
intervals in `main.tex` were checked against the recorded JSON, and the new
bibliographic key and table label were checked. No LaTeX engine is installed
in this environment, so manuscript PDF compilation and visual layout remain
unverified; no newly rendered PDF is claimed.
