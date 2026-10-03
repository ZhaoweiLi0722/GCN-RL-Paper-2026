# Patient-Constrained Training: Completed, No Useful Gain

2026-10-03 UTC. The approved preparation-and-execution package completed in this
turn. Implementation0ca0502; executionc77bd4e8ce0339a330600370cf5f82f93c810d69;
packetdece9e38ffcbc8d6f8b4797854dc121c7454c8a6e1a335094cd678dda8e425b5.
User approval is recorded verbatim inapproval-intent.json. No separate launch
question was asked after readiness. This is one completed attempt, not a retry.

## Execution

Terminal exit0 after1916.307seconds (31.94minutes). Supervisor40789 andchild40801
exited; a read-only full process listing found no related runner. All36fixed
reference and72learned training worlds completed, followed by144evaluations.
The3initial and6final models were saved, and all final models were sealed before
test access. Real updates:3456actor,8448critic,36scalar multiplier updates.
Evaluation optimizer calls:0. Every global/phase counter exactly matches the
frozen contract, including16128native steps and38400neural forwards. All phase
and per-owner time limits were respected. Stderr and launcher logs are empty;
there are no terminal failure/overrun markers.

The 18 artificial zero-update tests and whole-repositorycompileall passed before
freeze. This preparation removed an actual entry-point blocker and proceeded
directly to training. No additional qualification fit or historical rerun occurred.

## Performance

The primary patient-preserving training signal is **false**. All three declared
criteria fail: positive persistent-condition savings in every block, a positive
descriptive cost interval, and no mean additional patient losses in any condition.

Below, positive cost change means the constrained policy is MORE expensive.
Percentages are means of matched-world percentage changes, not percentage change
of pooled means. Extra losses are mean unique lost-patient counts per simulated
world; these are simulator outcomes, not observed clinical results.

| Condition | Cost vs fixed | Extra lost vs fixed | Cost vs ID-MPC | Extra lost vs ID-MPC |
|---|---:|---:|---:|---:|
| No change | +2.966% | +26.750 | +7.409% | +24.167 |
| Persistent change (primary) | +7.729% | +40.500 | +9.581% | +37.667 |
| Fast fluctuation | +7.070% | +52.167 | +5.038% | +34.917 |

In the persistent condition, all three constrained-vs-fixed block means worsen:
cost+11.141%,+7.185%,+4.862%; additional losses66.75,51.75,3.00. The descriptive95%
paired percentage cost increase interval is[3.785%,11.164%]. Its positive sign
does not rescue the learner: it describes higher cost. Three blocks are a small
development pilot, not an independent confirmation study.

Compared with the matched cost-only learner, the constrained learner has lower
mean cost and fewer losses: savings1.846%,1.240%,1.311% and fewer losses4.167,
5.333,5.250 for the three conditions. However, cost intervals cross zero in every
condition and block directions differ. In the primary condition the descriptive
savings interval is[-0.717%,3.980%]. Modestly outperforming another inferior
learner does not demonstrate useful improvement over fixed allocation or ID-MPC.
The expected patient-loss constraint was not satisfied by the final greedy
policies relative to fixed. It was never a pathwise or clinical safety guarantee.

## Saved Behavior And Next Decision

Saved evaluation actions identify a concrete policy deterioration, not just
unchanged behavior. Fixed allocates8total flexible hours per decision. The three
constrained models average0.0977,1.2834,7.9996hours. In block0,95.833% of individual
site-hour actions are exactly zero; block1 has9.505% zeros. Cost-only block0/1
have97.917%/97.483% zeros. These numbers pool the48control decisions across12test
worlds per model; they do not include the forced zero-action settlement tail.

This supports describing severe under-allocation and seed dependence in the
learned controllers. It does NOT prove that critic extrapolation, reward scaling,
insufficient data or a particular gradient defect caused the failure. No new
model inference or diagnostic fit was run to make that claim. Neither does this
result prove that fixed allocation is near-optimal or that all RL is ineffective.

Decision: close this objective/learner attempt. Do not add epochs, increase lambda
or change weights in response to these test results. A better-motivated next
proposal would separate **where capacity is allocated** from **whether capacity
is purchased**: first investigate bounded redistribution around a competent
fixed/MPC controller, explicitly preserving a defined total resource budget.
That is a narrower/new action contract requiring its own numerical comparison,
not an automatic continuation and not a safety guarantee. Its strong frozen and
MPC controls must remain intact; new held-out worlds would be needed. No such
follow-on has been executed or authorized by this consumed package.

For the manuscript, retain the previously supported GCN/distillation evidence
and report this as negative development evidence. This experiment tests RL policy
training in simulation with frozen evaluation; it cannot establish deployment
online adaptation or isolated graph-representation contribution. Synthetic
support labor is not field-calibrated, and the missing E1 data remain missing.

## Evidence And Preservation

- Results: results/patient_constrained_capacity_20261003/payload/comparison.json.
- Raw: payload/raw,252gzip files/16128rows; independent reader reconstructs
  costs from components and outcomes from patient identity/status histories.
- Models/full state: payload/models andpayload/states; updates inpayload/updates.
- Execution/current locks: reports/patient_constrained_20261003_terminal_evidence.json;
  all525current source/input hashes and packet hash verified.
- Archive: results/patient_constrained_capacity_20261003/archives/payload.tar.gz;
  931members,269754159bytes; SHA256
  3da87299ce67eb36f743bda78ff33a512d4c05b59e3fbf23a0775952b44bfcbf.
  The runner read and verified every archive member; closure rechecked archive
  bytes and current source inventory against that receipt, without re-archiving.

The samegcn-rlmonitor isPAUSED after terminal completion. No training is running.
No Dropbox export, cloud-sync/access claim, remote push/PR/merge, external message,
holdout use, StageE reopening or Howard approval was performed or represented.
One finite independent saved-raw review is integrated below.

## Independent Reconciliation

Hypatia independently read144evaluation gzip files/9216rows without importing
project analysis, runner, learner or environment code. It reconstructed costs
from14components and losses/deliveries from unique terminal patient IDs. All4870
reconciled numerical fields agreed within1e-6 (maximum difference1.49e-8);
patient counts matched exactly. All36paired worlds had identical patient cohorts
across roles and no unresolved patients. Bootstrap generation was not duplicated;
reported interval signs and the main screen were checked against independent
means. This is independent arithmetic, not independent scientific replication.

The36multiplier receipts also match the declared rule:32increases,4decreases,
after-values0.80-3.68, no cap contacts. Increasing the cap is therefore not a
supported response to this run. Persistent constrained-vs-fixed extra modeled
patient-loss cost+2.025M,expiry+1.390M andreactor-shortage+0.972M are partly
offset by lower reagent purchases-1.589M andshortage-0.173M. These are ledger
contributions, not a causal diagnosis or proof that physical cost weights are wrong.

Reports: reports/patient_constrained_20261003_independent_raw_check.json and
reports/patient_constrained_20261003_independent_readout.md. They retain all
condition/block/component means,108paired differences and raw file hashes.
Comparison SHA25602c074e51c7bfd6fdd9e7e3598479fcd472ded78553d50687f56ddcad99f1b9b.
The finite assignment completed and closed; no additional fit or gate was added.
