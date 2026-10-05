# Fixed-Parent MPC Policy-Tail: Terminal Readout

2026-10-05 UTC. Completed one approved attempt; **primary performance screen failed**.
User authorization remains the exact reply `启动下一轮` to the complete480-world/14h
package. No restart, reward change, extra update, patient world or model call.

## Execution And Preservation

- Entry HEAD2d90a3156d6230234a7d887623f990d2b14517c5; execution205225c7fd111142edc45b9ed2ab120bb3e8ae96.
- Scientific implementation88aa355; frozen implementationa9d6c8e.
- Packet583e63d8dc57fb01aa6a8c70f8d7f616dacd0123f1ff1d18480a0d724a396505.
- Root: results/capacity_policy_tail_20261005.
- Supervisor terminal: completed, scientific_completion_verified=true, child/parentexit0;
  elapsed27598.989s (7.666h), below50400s. PID12591 and12612 both absent from
  final full-command process check; no related runner. All failure/overrun markers absent;
  stdout/stderr/detachedlog0bytes. This is completed execution, not scheduler evidence.
- Reference120/120; each of three methods120fits and3840updates; total11520value,
  actor0. Historical ancestors carry7680old updates separately, not counted as new.
  All15new models and5ancestor bindings sealed before360/360frozen evaluations.
  Test optimizer debits0. Reuse earlier checked11520finite receipts/360after-fit states.
- Actual30720native steps/31680operations;36190forwards;9953280native planning
  prediction steps+4723200nested+5760prefix+46800pairedtail=14729040;
 12300nested decisions;720roots/1440clones;5988000total filter transitions.
  Global debit totals equal all25frozen counters exactly; phase counter bounds and
 855owner times pass. No pending chunk reservations or refunds.
- Phase seconds: admission5.288; reference/tails13291.513; fitting52.255;
  evaluation14170.993; analysis/archive78.534. All within original subcaps.
- Independently reconstructed480raw trajectories/30720rows and900paired-world
  comparisons (15pairs x3conditions x20worlds), including five-block means, harms,
  cost components, patient identity/terminal counts, requests/applied labor and latency.
 960raw/summary hashes match. Existing five-block bootstrap intervals are reused.
- Verified the existing2865-member archive and all current payload hashes.
  Archive445804056bytes, payload500371184bytes; no second archive created.
  Archive SHA256c710a6a6c777835ed39ea46231603dbc90d40c2d6eeb6d8e21256f3d20a19199.
  Comparison SHA25662136c0a0feac98830a8d2467a79b3452e8f432fba66ee983098a0fccb87aae2.
  Full ledger-file SHA256d112a702c03bbcf331879d36a8709c93e82ae80aa966e8de5513fa513899b892.
  The budget's ledger_sha256 is its final event-chain digest, not that file digest.
- Reused20necessary zero-update tests/compileall because scientific source/dependencies
  did not change. No scientific calls or model deserialization by terminal readers.
  No local TeX compiler (latexmk/pdflatex/tectonic) available; manuscript PDF compilation
  remains unverified. JSON, diff whitespace and added TeX environment/brace/label
  checks pass. No compiler installed, remote action or Dropbox export.

## Native Performance

Savings below are comparator minus policy-tail TD cost, in millions of synthetic
objective units; positive favors TD. Percentages are mean paired-world savings,
not percentage of pooled means. Extra losses are TD minus comparator; positive
is adverse. Each condition has20paired worlds in five reused historical blocks.
Intervals are the prespecified2000-resample two-level descriptive95% intervals,
not multiplicity-adjusted confirmation or clinical safety bounds.

| Comparator | Condition | Savings M [95%] | Savings % | Extra losses | Cost-positive blocks | Cost-harm worlds | Loss-harm worlds |
|---|---|---:|---:|---:|---:|---:|---:|
| Plain H8 MPC | No change | 0.424120 [-0.583349, 1.499259] | 0.6599 | -15.55 | 4/5 | 10/20 | 4/20 |
| Plain H8 MPC | Persistent | 0.044268 [-0.484021, 0.540357] | 0.1206 | -3.80 | 3/5 | 9/20 | 7/20 |
| Plain H8 MPC | Fast fluctuation | 1.814623 [0.564575, 3.292071] | 4.3494 | -29.85 | 5/5 | 3/20 | 1/20 |
| Existing frozen | No change | -0.347720 [-0.845603, 0.217029] | -0.9795 | -3.10 | 1/5 | 15/20 | 8/20 |
| Existing frozen | Persistent | -0.009655 [-0.656987, 0.600613] | -0.0083 | 1.45 | 2/5 | 11/20 | 14/20 |
| Existing frozen | Fast fluctuation | 0.420010 [-0.550352, 1.392735] | 0.9843 | -6.30 | 3/5 | 7/20 | 5/20 |
| Matched adaptive TD | No change | -0.025354 [-0.367128, 0.436687] | -0.0878 | -0.05 | 1/5 | 11/20 | 7/20 |
| Matched adaptive TD | Persistent | 0.006583 [-0.482301, 0.338406] | 0.0283 | -0.10 | 3/5 | 7/20 | 8/20 |
| Matched adaptive TD | Fast fluctuation | 0.375525 [-0.116305, 0.871031] | 0.7450 | -2.15 | 4/5 | 6/20 | 7/20 |
| Same-tail MC | No change | -0.031559 [-0.430288, 0.457679] | -0.0243 | -1.20 | 2/5 | 6/20 | 6/20 |
| Same-tail MC | Persistent | -0.110408 [-0.711381, 0.411014] | -0.2064 | 0.20 | 2/5 | 7/20 | 8/20 |
| Same-tail MC | Fast fluctuation | 0.425013 [-0.018339, 0.931606] | 1.1033 | -4.35 | 5/5 | 6/20 | 3/20 |
| Plain H16 MPC | No change | -0.338355 [-1.023093, 0.348066] | -0.8871 | -4.70 | 1/5 | 13/20 | 6/20 |
| Plain H16 MPC | Persistent | -0.189248 [-0.674345, 0.311362] | -0.4369 | 0.05 | 1/5 | 11/20 | 9/20 |
| Plain H16 MPC | Fast fluctuation | 0.641481 [-0.181591, 1.382893] | 1.5672 | -10.15 | 4/5 | 5/20 | 5/20 |

The primary conjunction fails for all three comparators: persistent-change cost
intervals cross zero and not all five blocks improve versus H8,existing_frozen or
matched adaptive TD. Against existing_frozen, persistent mean extra losses+1.45
also fails the patient criterion, with14/20worlds and4/5blocks showing extra losses.
Its descriptive loss interval[-0.95,3.70] does not establish population worsening.
The comparison does **not establish added benefit from changing the continuation
labels**, an increment over the existing frozen GCN, or TD superiority over
same-tail direct-return regression. It is not proof that every RL approach fails.

The favorable fast-fluctuation H8 contrast is real within this development sample:
4.3494% mean paired savings,1.814623M [0.564575,3.292071] absolute savings and29.85
fewer lost patients/world; all five cost means improve, but3/20worlds cost more and
1/20has extra loss. It does not reverse the failed persistent primary screen.
All condition-specific cost intervals against existing_frozen,matched adaptive TD,
same-tail MC and H16 include zero. Null intervals are not equivalence evidence.

Persistent TD/H8 actions differ at724/960boundaries and TD uses27.985more applied
flexible hours/world. Patient-loss/expiry savings190000/152000 are offset partly
by additional reagent purchases202435, reagent shortages73529, bioreactor
shortages37705 and flexible labor3170. More changed actions are not a gain.
Fast-fluctuation TD/H8 uses112.778additional hours/world. Full means, components,
all45pair-condition results and225block rows are in terminal-saved-data.json;
all900world pairs remain in the original comparison.json.

Evaluation median/p95 seconds per decision: H8 .6908/.8483; existing .6892/.8444;
adaptiveTD .6889/.8430; policyTD .6891/.8383; MC .6898/.8401; H16 1.3788/1.6135.
H16 spends twice the native prediction queries. TD's new training and nested
label-generation cost is additional, not hidden in deployment timing; same data
and update counts for TD/MC do not imply same forward computation.

## Mechanism And Claim Limits

On720saved training-tail pairs, frozen-parent MPC has lower public-model tail
cost than the adaptive rule547times, higher50times, equal123times. The mean
adaptive-minus-parent difference is2.941445M and positive in all five blocks.
This describes a different model-internal target, **not native counterfactual
performance** and not16-candidate ranking. Its large modeled saving did not
translate into a reliable incremental native evaluation gain.

The parent is fixed, not the changing student: this is not exact updated-policy
evaluation. Sampling is one preset candidate per root, unlike the preceding
96-tail study, so cross-study differences do not isolate continuation policy.
Five historical ancestors are not fresh independent training replications;
block0mixed-predictor provenance is retained. Reward/physical scenario/architecture/
candidate support remained frozen. Support labor is uncalibrated synthetic;
E1field data are missing. No isolatedGCN, model-freeDDPG, fullTD-MPC,
deployment-online-adaptation or clinical-benefit claim.

## Handoff And One Next Decision

Trigger: complete primary data fail the fixed-parent label amendment despite
substantial extra nested planning. Old action: finish this one comparison.
New action: close this package, preserve positive/null/adverse findings in the
manuscript, and pause the same gcn-rl monitor. No more epochs, label variants,
sample expansion or reward tuning under this authorization.

One next decision: whether to end public-model tail refinements and prepare a
separately scoped native-return-grounded value-learning comparison. That would
test dependence on forecast labels rather than assume a reward defect; no such
experiment, implementation or budget is approved or started here. Current reward
stays unchanged. A reward revision needs an accounting/terminal-liability defect
or an explicit objective change, not this negative result alone.

Saved-data reconciliation and manuscript handoff are complete. Ampere's finite
independent result interpretation agrees with the primary/secondary claim limits;
it is completed and closed, advisory rather than an additional acceptance gate.
Same automation PAUSED, retained visibly. No remaining execution blocker; only a
new scientific decision remains.
