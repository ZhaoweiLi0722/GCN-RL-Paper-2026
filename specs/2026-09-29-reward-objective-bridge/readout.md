# R1 readout: accounting passes; a scientific objective is still a decision

## Outcome

Ready within the receipt-accounting scope. This packet adds no evidence of
online-RL benefit, no new simulation and no reward intervention. It does not
reclassify the successful N7 engineering check as a failed scientific study.

All six archived N7 case records / 36 steps pass independent reconciliation:

- Raw reward equals negative incurred cost, with a single declared reward scale.
- Total cost equals base operating cost + patient loss + expiry + urgency.
  Specimen transfer is not counted twice as another top-level cost.
- Step indices, observed/full-state-token adjacency and source/episode lineage
  agree. Every start has its appropriate short tail, with no pending records.
- Six resumed copies equal the continuous records; these are duplicates for
  recovery verification, not six additional independent observations.
- All 9 input-file and 34 archived source hashes match before and after audit.

No new mixed-reward, numerical-cost or return-tail defect was found in these
receipts. This does not establish that every producer/model path is correct.
Learned Q targets were not independently recalculated: the receipts lack the
corresponding next-Q values. No binary checkpoint or model was loaded.

## Objective differences, not performance changes

Each case records the same cost sum of 7,019 hypothetical cost units. With the
engineering gamma of .9, its discounted segment cost is 5,790.540712; its sixth
step weight is .59049. The scaled undiscounted reward sum is -.07019. These
are different arithmetic summaries of the **same recorded path**, not a saving
from changing a policy or evidence that discounting caused a historical null.

N7 deliberately used gamma=.9 and three-step replay for mechanics coverage.
The N5 proposed complete-settlement design instead uses gamma=1 and one-step
replay. Four archived cases end in a bootstrapping truncation; two test a zero-
bootstrap terminal mask. They must not silently become a scientific objective
merely because their code and resume checks pass.

All six final receipts report zero active patients. That is a useful narrow
fact, but not proof of settled inventory, commitments, all future demand or
closure liabilities. The original N7 report explicitly makes no clinical
terminal-settlement claim. R1 preserves that distinction.

## The next research choice

The actionable design is in `next-study.md`. Separate an existing-simulator
consistency study from a new operational capacity/staffing scenario. Missing
E1 field inputs apply to the new task; they do not automatically prevent all
simulator-only investigation under fixed, explicitly hypothetical old weights.

The next feasibility proposal would fix one return definition across data,
calibration, replay and targets, then test executed-action ranking on independent
development evidence before actor training. It must preserve the existing G0/G1
negative findings, avoid old mixed replay/uncertified teacher labels, and retain
a competent frozen policy and adaptive controls. It is not another invitation
to train on the S4 screen or tune costs until RL wins.

Before that proposal can become executable, choose finite-window versus fully
settled cost, define common actor/gate/critic preparation and approve the fresh
data/budget/analysis protocol. A finite-window choice is a distinct estimand and
would need an explicit N5 amendment; this readout does not make it. Shaping is
a separate later question, not a substitute for these decisions.

## Reproduction and verification

Execution source: `541b73d`. Source/config were committed before the recorded
audit. Output:
`reports/2026-09-29-reward-objective-bridge/audit.json`.

SHA256: `677f3e13e665129de220ceb913bb4606b9021112e995570af247ad939f530f02`.

- 15 new receipt-audit tests, 48 combined relevant tests pass. Mutated copies
  detect bad rewards, component double counting, mixed semantics, disconnected
  records, missing tails and incorrect boundary flags. Existing files are not
  mutated by these tests.
- Full repository compileall and diff checks pass.
- A second stdout-only invocation reproduces the audit JSON byte for byte.
- A separate 50-digit Decimal calculation using backward recurrence verifies
  all six discounted segment costs and 36 window returns against the auditor's
  forward floating-point sums (relative tolerance 1e-14).
- Zero new environment queries, optimizer updates, cost changes or remote
  actions. No live evaluator was present at the initial read-only process check;
  both audit invocations returned exit 0. No background job was scheduled.

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_reward_objective_bridge tests.test_validated_returns tests.test_prospective_adapter
"$PY" -m evaluation.audit_reward_objective_bridge
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

The stdout invocation is read-only. `--output` refuses to overwrite any existing
destination. Later commits alter the execution-commit metadata, not the arithmetic;
byte comparison requires the same commit. Twelve E1 fields remain null. The N5
template has eight unfilled evidence fields, but this is not proof all evidence
is absent: N6/N7 provide bounded engineering acceptance outside that old template.
No clinical adequacy, fully settled target, launch authorization or performance
improvement is inferred from the audit's successful exit.
