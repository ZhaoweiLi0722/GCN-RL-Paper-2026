# R2 readout: action errors have unequal costs

Completed 2026-09-29. This is a retrospective analysis of immutable archived
G1 evidence, not new training or a recovered G1 result. Stage E/G1 remain closed.

## Finding

The existing archive does not justify either "there are no better actions" or
"a one-point top-1 improvement means the critic achieved nothing." Independent
validation costs show a more specific problem: candidate actions selected using
discovery costs have a favorable descriptive mean, but the learned critic
selectors fail to capture it reliably and sometimes make expensive mistakes.

All numbers below are **millions of original simulator cost units per diagnostic
state**, not currency, episode savings, patient benefits or a causal treatment
effect. Negative difference is better. Costs are remaining-window means; the
MDL-2 reference mean is 1,505.690 million. State horizons overlap and cannot be
summed into independent episode totals.

| Selector | Validation cost minus MDL-2, mean | Median difference | Better / tied / worse states |
| --- | ---: | ---: | ---: |
| MDL-2 reference | 0 | 0 | 0 / 156 / 0 |
| Frozen **critic argmax** | +4.488 | +0.141 | 70 / 3 / 83 |
| Historical fitted offline critic argmax | +3.181 | -0.078 | 80 / 3 / 73 |
| Discovery-cost-selected action | -0.932 | -0.357 | 85 / 43 / 28 |

**Frozen critic argmax is not the frozen deployed actor.** This table therefore
does not contradict the frozen pretrained policy's formal package benefit or
estimate an online-versus-frozen actor contrast. The fitted critic was trained
in the historical offline leave-one-seed-out audit; it was not trained by R2.
Discovery-cost selection uses simulator counterfactual labels, not information
freely available to a controller. Every cost describes one first-action change
followed by the archived continuation, not repeated use of that selector.

| Held-out seed | Frozen critic minus MDL-2 | Fitted critic minus MDL-2 | Fitted minus frozen | Discovery choice minus MDL-2 |
| --- | ---: | ---: | ---: | ---: |
| 60 | +10.911 | +8.905 | -2.006 | -1.290 |
| 61 | +2.555 | +0.266 | -2.288 | -0.373 |
| 62 | -0.002 | +0.371 | +0.373 | -1.135 |

The fitted-versus-frozen critic mean improves by 1.307 million, with 35 better,
97 tied and 24 worse state costs. It worsens in seed 62. Its slightly negative
median versus MDL-2 coexists with a positive mean and a worst recorded excess
over MDL-2 of 66.785 million. A win count alone would hide the cost asymmetry.

## Are ranking disagreements just near-ties?

No, not uniformly within this sample. Discovery/validation best sets disagree
in 71/156 states. For the discovery-selected action, excess over the validation
sample's minimum falls into these fixed legacy bins:

| Excess | Disagreement states |
| --- | ---: |
| Above tie tolerance through 250,000 | 9 |
| Above 250,000 through 1,000,000 | 22 |
| Above 1,000,000 | 40 |

The disagreement-state median excess is 1.179 million. Among the fitted
critic's 109 top-1 errors, 78 exceed 1 million. Thresholds are inherited from
G1, not calibrated clinical margins. The validation-set minimum is optimistically
selected on the same finite sample, **not a true optimum or regret bound**.
These differences cannot separate genuine return differences from Monte Carlo
noise without individual replication rows. They do show that automatically
dismissing all observed disagreements as harmless ties is unwarranted.

## What this changes in the next design

1. Preserve fixed business costs. Nothing here establishes that changing reward
   weights would improve the intended task. Fixing inconsistent reward/replay/TD
   semantics is distinct from making a more favorable optimization objective.
2. In a future separately approved protocol, report independently evaluated
   chosen-action cost, downside and clinical constraints alongside top-1 and
   pairwise ranking. Do not retrospectively replace G1's failed gates. Cost
   sensitivity matters for evaluation; this audit does not prove a particular
   loss, conservative selector or DDPG update will work.
3. Obtain replication-level cost and clinical components in any new data
   collection. The existing mean-only archive cannot estimate action-difference
   uncertainty or certify noninferiority. Do not fit uncertainty thresholds on
   these validation outcomes, silently add an anchor fallback or pick a best
   checkpoint after inspection.
4. Keep the matched-information online-versus-frozen actor comparison as the
   eventual question. A well-performing simulator lookahead selector would be
   a planning result, not evidence for online neural updates. Reward repair
   also does not remove the integer execution plateaus documented in G0.

Recommend the minimal-comparability endpoint for the next **protocol draft**:
the original 52-step finite-window incurred-cost objective with fixed weights,
explicit remaining time and coherent return semantics. Do not call it fully
settled welfare. Endpoint approval, fresh data/initialization provenance and a
bounded independent gate remain required before a scientific run. The existing
N1-N7 engineering need not be rebuilt. A corrected fixed-weight experiment
would test a repair package, not retrospectively prove the cause of old nulls.

## Limits retained

- G1's published correction makes these nominal-history diagnostics. Raw CSV
  hotspot labels are preserved for provenance, not accepted as scenario truth.
  This audit is not evidence of adaptation to a persistent capacity change.
- Only three held-out seeds with dependent within-trajectory states; each cost
  averages three discovery or five validation replications. No per-replication
  rows or clinical component outcomes are available in these locked inputs.
  No confidence interval, significance claim, calibrated safety claim or
  repeatedly deployed controller gain is made.
- The corrected H0/H1 screens independently failed to authorize escalation.
  Their failure and G1's `unstable_counterfactual_labels_close_extension`
  classification are unchanged. A favorable descriptive discovery-choice mean
  does not overrule them or reopen a neural campaign.

## Verification and reproduction

- Protocol/config commit: `5594989`.
- Recorded auditor commit: `1441c32e9a89f20640e91a524810ffa99b06b4f2`.
- Independent verifier commit: `dd62d6f`.
- Both processes exited 0. Seven locks checked before/after; 1,560 cost rows,
  1,560 prediction rows and NPZ advantages reconcile. Historical aggregate and
  per-seed top-1 results, label stability and pairwise counts reproduce.
- A separate stdlib Decimal implementation, with no import of the auditor or
  NumPy, verified all 156 state records / 624 choices and every pooled/per-seed
  cost summary to absolute tolerance 0.000002 original cost units.
- Fourteen new tests / 29 combined tests, whole-repository compileall and
  `git diff --check` pass. `pytest` is not installed; that attempted test command
  exited before tests. The repository's standard-library unittest runner passed.
- Fresh read-only process check found no related research Python process.
  Zero new environment queries, optimizer updates, neural runs or remote writes.

Artifacts (exclusive creation; existing output must not be overwritten):

- `reports/2026-09-29-g1-decision-cost/audit.json`:
  `a96fb9989da9b263020ade124a7c6ac2e60a4c92a04303eb391d0a7e6efda532`
- `reports/2026-09-29-g1-decision-cost/verification.json`:
  `1c033737491c18d33c2338279853361630ab7f5e8ab92dae83f85e4455e6effc`

```bash
python -m unittest tests.test_g1_decision_cost tests.test_reward_objective_bridge -v
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python -m compileall -q .
python -m evaluation.audit_g1_decision_cost --output <new-audit-path.json>
python -m evaluation.verify_g1_decision_cost --audit <new-audit-path.json> --output <new-verification-path.json>
```

Use the repository virtualenv. A later arithmetic recheck is not a new
experiment/replication. No simulator or critic is loaded by these commands.
