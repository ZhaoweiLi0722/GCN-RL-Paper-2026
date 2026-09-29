# Recovered formal rows: reporting sensitivity only

## Scope and timing

This is a post-hoc reporting audit of the already completed, closed formal
campaign. The original mean effects and primary intervals have already been
seen. It is NOT a prospective experiment, new confirmation, or untouched
holdout. No simulator, teacher, policy, checkpoint, training, deployment rule,
clinical margin or scenario selection will be changed. Source CSVs remain
read-only. The source/protocol commit precedes the recorded projection/audit.

The September 15 reporting audit could not locate these formal raw rows on
Howard's machine. A September 29 read-only search found final and pretraining
rows in Zhaowei's persistent patient-routing worktree. Ten GCN-final CSV hashes
and both summary hashes match the tracked historical evidence. The other
30 CSVs do not have historical byte-hash coverage in that compact manifest:
record current hashes and reconcile all scenario means to historical summaries.

## Fail-closed recovery

Use `evaluation.recover_formal_attribution_rows`, with the original
`ddpg_routing_primary_100_evaluation` root as input. Validate both phases,
GCN/flat algorithms, seeds 10-14, all four scenarios, replications 0-99, correct
scenario-specific CRN identifiers, 52 steps, finite cost, no duplicate/missing
keys, and graph labels. Verify all 160 per-file scenario means within absolute
1e-5 cost units. Require exact anchor cost equality across algorithms, seeds
and phases. Export only nine identity/cost fields; omit patient-route detail
and all clinical endpoints. Hash every input before/after and each projection.
Refuse existing output directories. Do not overwrite old evidence.

## Fixed analysis

Run Howard's existing `evaluation.crossed_design_bootstrap_audit` unchanged on
the validated projection: total_cost only, equal weights for the four original
scenarios, 20,000 resamples, alpha=0.05, bootstrap seed=0. Report all five
contrasts: GCN-anchor, flat-anchor, GCN-flat, GCN-final-frozen and flat-final-frozen.
For each retain the original nested scheme, world clustering, seed clustering,
and crossed seed/world resampling. No subgroup-based selection or adaptive
choice of resample count or interval method. No clinical noninferiority claim
is re-estimated from the cost-only projection.

Crossed resampling preserves common random worlds across training seeds.
It is a sensitivity analysis, not a guarantee of exact finite-sample coverage:
there are only five training seeds, four prespecified scenarios and 100 worlds
per scenario. The 2,000 rows per policy are not 2,000 independent training runs.
See Owen (2007), [The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111).
Keep the original declared primary analysis; do not silently replace it with
whichever interval looks more favorable. The old compact-summary t-interval
path is not used by this row-level audit.

## Deliverables

Separate dated report, compact CSV projection, provenance, all five contrasts,
and an explicit manuscript interpretation. Retain null online-attribution
findings. Local only: no upload, remote push, PR, or merge is authorized here.
