# Service-effort decision fixture readout

Completed locally on 2026-09-29. Execution source/protocol commit:
`8e102a0406c4300b582768f76c007103ef3f8450`.

Classification: deterministic comparator-software validation, not PRM
performance, clinical validation, online-RL evidence, or graph-benefit evidence.
There are zero independent statistical replications and no trained policies.

## Recorded results

All costs below are synthetic fixture units, lower is better. The changed
family averages two equiprobable, prescribed worlds. These are not estimated
population means or dollar costs.

| Controller | No change | Persistent response change |
| --- | ---: | ---: |
| Fixed balanced effort | 21.75 | 24.50 |
| Backlog feedback rule | 21.75 | 23.00 |
| Adaptive backlog rule | 21.75 | 23.00 |
| Fixed-model MPC | 19.50 | 23.25 |
| Identification plus MPC | 19.50 | 22.25 |

The finite-grid diagnostic knows the prescribed scenario distribution, whereas
the practical MPC baselines use a constant nominal or estimated response. It
is therefore model-privileged, not an information-matched practical competitor.

| Exact diagnostic on this grid and horizon | No change | Persistent change |
| --- | ---: | ---: |
| Best open-loop sequence | 19.50 | 22.25 |
| Best nonanticipative feedback policy | 19.50 | 22.25 |
| Clairvoyant policy | 19.50 | 21.25 |

Source: [recorded summary](../../reports/2026-09-29-service-effort-decisions/fixture/summary.json).
Full evidence: [inventory](../../reports/2026-09-29-service-effort-decisions/fixture/inventory.json),
[baseline decisions](../../reports/2026-09-29-service-effort-decisions/fixture/decisions.jsonl),
[episodes and settlement](../../reports/2026-09-29-service-effort-decisions/fixture/episodes.jsonl),
and [exact-tree transitions](../../reports/2026-09-29-service-effort-decisions/fixture/tree_transitions.jsonl).

## What this establishes

Identification lowers cost by 1.00 fixture unit relative to the fixed-model MPC
in the changed family. However, the best prescribed open-loop sequence also
costs 22.25: balanced effort for three decisions, followed by two idle
decisions. Feedback adds zero expected value beyond that sequence under the
known finite-world law, four-action grid, and declared settlement convention.

Consequently, an identification-versus-fixed-model improvement alone would
give a misleading reason to launch online RL here. It can reflect correcting
a misspecified comparator rather than an indispensable adaptation capability.
The selected open-loop sequence itself uses the diagnostic's scenario-law
knowledge; this does not imply that an uninformed deployed controller can
choose it without learning or prior data.

The remaining 1.00-unit clairvoyance gap is specific to this finite fixture.
Neither the gap nor the exact-grid optimum bounds a continuous-action patient
problem. This result does not establish that the prior PRM policy is globally
optimal, that capacity scenarios are exhausted, or that online learning can
never help. It does establish that this particular fixture is insufficient
evidence for a residual online-DDPG opportunity. No parameters were retuned
after seeing these results.

## Verification

- 15 baseline episodes, 75 decision rows, 15 settlement/drain rows, and 4,092
  exact-tree transition rows were recorded.
- The separate read-only auditor passed 31 file-hash checks, cost arithmetic,
  commitment timing, work/job conservation, receipt-based estimate timing,
  row identities and cardinalities, and terminal-obligation closure.
- The auditor independently enumerates open-loop and clairvoyant optima from
  recorded transitions. The nonanticipative solver is regression-tested, not
  independently reimplemented; no stronger audit claim is made.
- 72 relevant unit tests and full-repository `compileall` passed.
- Historical patient/environment sources and specified September 28 evidence
  were hash-checked against `1aae687c567eec4f8145b07eb95abd89f40aa32f`.

Recheck from the integration worktree with the project Python environment:

```bash
python -m evaluation.audit_service_effort_decisions \
  --output reports/2026-09-29-service-effort-decisions/fixture
```

## Decision and next work

Do not train DDPG on this fixture or add it to the manuscript as a performance
result. Retain it as reproducible validation of the comparator and attribution
checks. The current paper's established results remain unchanged.

The next useful step is an operational contract for a named, qualified
personnel-allocation task, with realistic intervention windows, commitment
delays, equipment constraints, measurement availability, and downstream queue
effects. The [engineering basis](engineering_basis.md) provides motivation,
not empirical calibration. More staff must not arbitrarily accelerate cell
growth, bypass assays, or create unconstrained processing equipment.

Before another patient-model experiment, specify and justify those assumptions
and a common observation/history interface. If the true decision is discrete
or nonpreemptible, represent it honestly instead of forcing a DDPG formulation.
Without operational measurements, explicitly label any follow-up as synthetic;
do not silently treat the current noiseless progress receipts as available data.

Only then screen sequential headroom and replicated action rankings beyond
strong feasible controls, including an adaptive rule and identification/MPC.
A frozen history-aware policy must receive the same information and offline
training opportunity as the online-updated policy. The eventual 2x2 belief-
update/policy-weight-update design separates system identification from online
DDPG. A feedback-only gain cannot establish the latter.

Adding noise, additional queues, or new scenarios solely until RL wins is not
the plan. Any added mechanism needs an operational reason, prospective scope,
and retained negative results. No patient campaign, formal confirmation,
remote push, main merge, or collaborator sign-off was performed here.
