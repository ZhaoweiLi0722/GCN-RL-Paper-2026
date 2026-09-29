# R2: archived G1 decision-cost audit

Status: bounded retrospective arithmetic, not a new scientific experiment.
Zhaowei accepted continuing the fixed-weight legacy-simulator route. This
packet analyzes existing evidence only; it does not authorize fitting a critic,
changing rewards, collecting trajectories or reopening Stage E/G1.

## Question and provenance

Does historical top-1 disagreement represent approximately interchangeable
actions, or costly selection errors on the independently sampled validation
costs? Use all 156 existing seed/step states without selection. The published
G1 correction says these are **nominal-history** reconstructions. Persistent
hotspot strings in the CSV are recorded labels, not the actual scenario.
Freeze hashes of the two CSVs, NPZ, summary, old config, correction and original
metric implementation. Do not import an environment, learner or checkpoint.

The legal-action costs are already means of three discovery/five validation
replications. Individual replication rows and component/clinical outcomes are
absent here. The NPZ contains advantages, not additional independent evidence.

## Fixed calculation

- Audit all 1,560 cost and 1,560 prediction rows, both cost horizons, unique
  seed/step/action keys, labels, folds, horizons, CRN offsets and finite values.
  Reconcile every cost-derived advantage with CSV, NPZ and prediction labels.
  Verify discovery/validation copies of each critic prediction are identical.
- Reproduce old remaining-horizon aggregate top-1/pairwise/material metrics
  and discovery-validation best-set agreement/sign counts, including per seed.
  Do not import the historical implementation as the checker.
- Analyze only the originally primary `remaining` horizon for decision costs.
  Select action 0 (MDL-2), first-index argmax frozen critic, first-index argmax
  fitted leave-one-seed-out critic, and first-index argmin discovery cost.
  Validation never chooses these four selectors. Argmin of validation is only
  an explicitly optimistic within-sample reference, not a deployable policy.
- Record validation cost minus MDL-2 and validation cost minus validation-set
  minimum for every selector/state. Record fitted-minus-frozen differences.
  Report mean, median, min/max and better/tied/worse counts, pooled and per seed.
  Top-1 ties use original absolute tolerance 1e-9; best-set label replication
  uses original exact equality. All choices use deterministic first-index ties.
- Describe excess-cost bins: <=1e-9, (1e-9,250000], (250000,1000000], >1000000.
  The latter two boundaries reuse old reporting thresholds; they are not new
  clinically calibrated margins or replacement pass/fail gates. Also report
  bins among top-1 errors and among discovery/validation best-set disagreements.
- No p values, bootstrap intervals or new gate. States on a trajectory are
  dependent and their remaining-horizon costs overlap. Means per diagnostic
  state are not total episode savings. Three seeds do not establish stability.

## Execution and verification

Commit protocol/config and implementation before recording the JSON result in
`reports/2026-09-29-g1-decision-cost/audit.json`. Exclusive creation prevents
overwriting. One pass over the archived tables, no simulator/optimizer queries,
no matrix, no sampling, no tuning. Relevant synthetic tests and whole-repository
compileall must pass. Verify input hashes before/after and independently
recompute selected costs and paired differences using Decimal from the CSVs.
Preserve a failed recording; no automated repair/re-execution of a failed run.

## Interpretation boundary

The fitted model is an old offline ranker, not corrected online DDPG. Costs
describe one first-action intervention followed by the old continuation, not
repeated deployment of the selector. Even discovery-selected gains would not
establish online benefit, safety, a real optimum or response to capacity shifts.
Do not call validation-minimum differences true regret or an optimality bound.
This audit can motivate, not authorize, an independently designed fixed-weight
reward/replay/TD-consistency gate. Endpoint, data, practical effect margin,
budget and launch approval still need a separate decision.
