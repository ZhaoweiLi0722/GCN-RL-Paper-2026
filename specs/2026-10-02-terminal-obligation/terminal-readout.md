# Cohort Comparison: Training Complete, Evaluation Deadline Failure

## Decision

The approved single attempt terminated during the first evaluation task. Do not
restart it or train new models. The changed-objective hypothesis is unresolved,
not rejected: no cohort-PPO, window-PPO or BC test comparison was completed.

The coordinator's90second per-controller/block evaluation allowance was too
tight for this observed execution. The watchdog enforced the frozen protocol;
the run did not exhaust its3hour global limit. No cost or patient outcome was
used to select this stopping point. The next useful action is evaluation-only
completion using the existing sealed models,subject to one new explicit permit.

## Evidence

- Execution commit:098b83f3e0457a3ec65568cbba90147670215bc6.
- Implementation commit:8a45c72269d23a99d36c2d6ca16c3eacb1b99e1a.
- Result root:`results/dynamic_candidate_cohort_objective_20261002`.
- `launcher/terminal.json`:failed,exit1,scientific_completion_verified=false.
- `launcher/supervisor.json`:wall_clock_deadline at
  `final_evaluation/block60/own_frozen`,child exit-15,no forced SIGKILL.
- Supervisor elapsed5214.478837seconds,about86.9minutes;global cap10800seconds.
- This section began at937084.053969 and deadline was937174.053969 in the
  declared shared monotonic clock:exactly90seconds.
- Original supervisor72066 and child72081 were both absent after exit.
- Stderr contains0bytes. Last running status is stale and must not override
  terminal.json. No automatic repair,retry,follow-on or training was started.

| Boundary | Completed |
| --- | ---: |
| Real preflight cohorts | 18/18 |
| Window PPO training | 96/96 |
| Cohort PPO training | 96/96 |
| BC continuation training | 96/96 |
| Training jobs | 9/9 |
| Actor/critic optimizer calls combined | 1,920/1,920 |
| Sealed frozen/trained model files | 12/12 |
| Full evaluation cohorts | 11/216 |

All12model byte lengths and SHA256s match `payload/model-seals.json`. Each of
the9training phase receipts has32complete raw indexes and8update boundaries.
The11complete test cohorts are only own-frozen/block60 worlds0-10. World11
contains17persisted prefix steps and is incomplete;never treat it as a full row.
Other controller test worlds have not been acquired. No paired effect size,
promotion-screen result or patient-performance claim is available.

The ledger charges20,288environment calls:1,434preflight including parity/clones,
18,144training (288x63), and710evaluation (11x63+17). The17partial calls remain
spent,not refunded. All1,920optimizer calls preceded evaluation. Training phases
finished under their nontransferable limits:windowPPO1789.342seconds,
cohortPPO1749.499seconds,BC1015.588seconds. The90second evaluation-owner cap,
not model fitting or the global allowance,was binding.

## Preservation and Handoff

Original source,configuration,raw records,partial episode,failed terminal record
and models remain untouched. Saved-only target reconciliation passed:all48PPO
rollouts,192PPO training cohorts and the full288three-arm acquisition inventory
match their raw/phase/kernel records. The reader checked1935file references and
confirmed immutable prefix rewards,exactly-once cohort-tail charges and saved
target histories. It only decoded checkpoint containers;no policy inference,
environment calls,gradients or optimizer reexecution occurred. Receipt:
`reports/2026-10-02-cohort-objective-closure/independent-training-targets.json`.
The separate metadata reader found no nonfinite numeric field in any of9training
phase records and confirmed source/input/model locks and the budget hash chain.

The non-overwriting failed-run archive completed with exit0 after reading every
member and confirming the entire original source tree unchanged:5297files,
2574714967archive bytes,SHA256
`d8b01bb8527532fbeb6a9770f7c291e3513bd7d745c430222679f0990d646705`.
Archive:`reports/2026-10-02-cohort-objective-closure/archives/failed-run.tar.gz`;
its adjacent `.manifest.json` lists every preserved file and hash. This is
post-termination preservation,not a resumed scientific archive phase or completed
performance comparison. The large archive is Git-ignored but retained locally;
the verified manifest and small closure receipts are included in the local commit.
No Dropbox export,cloud-sync or collaborator-access receipt is implied.

Same-thread automation `gcn-rl` is PAUSED,confirmed by the native tool,not deleted.
The two finite implementation agents already finished and are closed.
No remote push,PR,merge,Dropbox export,holdout use,StageE reopening or Howard
approval assertion occurred.

## One Proposed Recovery Decision

Keep all12sealed models,original36test worlds,six controllers,63step outcome,
costs,patient metrics and original comparisons unchanged. Reuse the11complete
evaluation cohorts and acquire205missing complete cohorts;the incomplete world
would restart from its original fixed seed into a new result root,not overwrite
or claim credit for the old17steps. No training,model selection or reward change.
These are remaining originally planned worlds,not a new independent test set.

Proposed caps:12,915new environment steps,0optimizer calls,240seconds per
controller/block owner,7,200seconds globally,one attempt,no automatic retry.
Runtime/model binding300seconds;18evaluation owner caps total4320seconds;
saved raw verification600seconds;archive900seconds;closure600seconds. Sum6720;
remaining global slack is not transferable between phase/owner caps. Up to12
saved policy artifact loads plus3unchanged R4 reference loads,3layout builds and
205episode builds;no initializer fitting or new qualification. No extra parity
trajectories or hidden numerical preflight. Source/inputs and the effective
recovery protocol must be frozen locally before any recovery call.

Approval for this numerical recovery was asked once. It has not been received
at this checkpoint. The old start authorization and unspent original allowances
do not authorize a new run. Current work stops at preservation and handoff.
