# Service-effort mechanism: bounded engineering fixture

Status: local implementation and deterministic verification only, 2026-09-28.
Zhaowei said to continue the roadmap. This packet does not claim Howard's
approval, amend the formal campaign, or authorize neural training or a patient
simulation extension. The operational interpretation remains unvalidated.

## Code audit motivating this separation

- `PatientConditionCapacityEnv.step` floors the number of starts after limiting
  by waiting patients, idle/borrowed equipment and reagents. Its overtime path
  represents additional start capacity, not measured continuous labor service.
- `CapacityPlanningEnv._receive_transfer_arrivals` clips arriving idle equipment
  to the local storage ceiling. The earlier pilot retained a stock-loss fixture
  and explicitly used a different synthetic storage ceiling. This packet does
  not modify the historic implementation or assert its clinical results are
  unaffected; overflow semantics remain an integration prerequisite.
- The existing intertemporal overtime pipeline provides a useful delay pattern,
  but does not supply a measured work-progress stage. Reusing it as a labor
  productivity model without an explicit contract would change its meaning.

## Isolated model contract

The fixture has three sites with FIFO generic work orders, not patients or
therapies. Each order needs positive synthetic work units. A continuous effort
allocation advances remaining work. An order is counted complete only when its
whole requirement is fulfilled. Partial progress persists; unused service
capacity does not become stored labor. No arrivals, routing, reagent shortages,
QC constraints, patient deterioration, graph encoder or clinical reward exists
in this unit fixture.

Commitments are nonnegative hours, limited by site caps and one shared budget.
They take one interval to mature in the recorded fixture; the primitive also
tests a two-interval delay. Labor and switching charges are paid at commitment,
including idle or later unused effort. The response coefficient at service time
determines delivered work, capped by the remaining backlog.

All numbers are hand-checkable software-test inputs, not calibrated staffing
costs, clinical thresholds, representative regime ranges or research seeds.
The three eight-step cases use the same fixed allocation sequence:

1. Nominal response (1, 1, 1).
2. Response changes at service epoch 3 to (0.5, 1.25, 1).
3. Very small initial backlog, making throughput a censored response signal.

## Information contract

The observation reports epoch, aggregate remaining measured work, unfinished
and completed job counts, pending effort and previous commitment. A receipt
reports committed/applied hours, pre-service available work, delivered work,
whole-job completions and charges. These are proposed instrument readings;
their availability in PRM operations is NOT established.

The private response tape, change time and case name never enter the public
observation or receipt. The harness logs a case name for auditing, but passes
only receipts to the estimator. No controller receives the environment object.
This is API isolation, not a security boundary against malicious Python code.

When backlog is exhausted, delivered work divided by allocated hours is a
lower bound, not the response coefficient. The estimator skips those samples,
as well as zero-effort intervals. Equality at the backlog limit is censored.
There is no measurement noise; one uncensored measurement suffices when the
estimator's smoothing factor is 1. This is deliberately a simple identifier,
not evidence that RL is needed. More realistic measurement noise must be
justified operationally, not added solely to make this baseline fail.

## Checks and reporting limits

- Work, indivisible job count, and committed/applied/pending effort conserve.
- Small action changes can alter accumulated work without instantly completing
  an additional job. This does not imply a smooth clinical-cost gradient.
- Two latent profiles with equal observed histories remain indistinguishable
  until their executed responses differ. Detection precedes no observation.
- Invalid actions do not mutate the state; input arrays/lists are copied.
- Pending paid effort and unfinished work remain in the terminal ledger.
  No terminal value, fully settled cost or policy performance is claimed.
- No stochastic seed, prior scientific CRN, checkpoint or patient environment
  is executed. This is not the roadmap's headroom or learnability gate.
- Source and config must match a local commit before the standalone checker
  writes a new, non-overwritable output root. Ten prior source/output files are
  checked against integration commit 5638587 before and after execution.

Output root: `reports/2026-09-28-service-effort-mechanics/fixture/`.
The checker records claim, 24 raw transition rows, summary, status and inventory.
Its separate audit path recomputes timing, work balance and charges from rows
without calling the simulator transition function.

## Next decision, not yet executed

Determine which real preparation/support operation is controllable and whether
progress and eligible workload are measured. Choose coherent overflow and
terminal settlement semantics before connecting to the patient simulator.
Then specify a repeated-decision comparison with adaptive rules and ID-MPC,
including genuine network constraints. Do not train DDPG on this easy,
fully measured coefficient fixture or present its success as paper evidence.
