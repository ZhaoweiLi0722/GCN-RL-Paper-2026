# Cost sensitivity: feedback value survives, online-RL attribution remains open

Date: 2026-09-29. Local, post-hoc analysis of previously recorded synthetic
trajectories. This is not a new manufacturing experiment or a training result.

## Result and decision

Changing the relative labor/switching costs does not reveal a convincing reason
to train DDPG on this finite queue fixture. In 66 of 72 repriced cell/settings,
at least one archived non-neural controller path still attains the recomputed
known-law feedback optimum. The other six have gaps of only 0.125 to 0.375
synthetic cost units, before any MPC replanning for the alternative objective.

These are **72 dependent sensitivity records, not 72 independent experiments**.
The batch and booked-flow cells are previously verified offset duplicates. The
best path is chosen descriptively per cell, not a deployed hybrid controller.
Only four actions, five decisions, a fixed terminal continuation and the original
full-progress observation view are covered. This is not global optimality.

| Labor multiplier | Switching multiplier | Cells with a matching archived path / 8 | Largest best-path gap |
|---|---|---|---|
| 0.5 | 0.5 | 6 | 0.375 |
| 0.5 | 1 | 6 | 0.125 |
| 0.5 | 2 | 6 | 0.250 |
| 1 | 0.5 | 8 | 0 |
| 1 | 1 | 8 | 0 |
| 1 | 2 | 8 | 0 |
| 2 | 0.5 | 8 | 0 |
| 2 | 1 | 8 | 0 |
| 2 | 2 | 8 | 0 |

Holding costs stay fixed. Labor scaling includes both its linear and quadratic
components. At unit multipliers, all original comparator and bound values are
recovered. Every changed-response cell retains positive feedback value over the
best open-loop sequence across all nine settings, ranging from 2.0 to 9.5 cost
units. Unchanged-response cells have zero such feedback value.

The unmatched records are all nonbinding-downstream cells with half-price labor:
persistent-change cells at switching multipliers 0.5 and 1, and no-change cells
at switching multiplier 2. The no-change exception itself illustrates why a cost
reoptimization gap is not evidence of a need for online adaptation.

**Decision:** retain this fixture as a control/information-accounting check;
do not select a favorable cost ratio or start an RL performance campaign on it.
The result supports studying feedback under persistent changes, not attributing
that benefit to online neural updates. It does not rule out practical online-RL
benefit in a different, justified setting.

## What to design next

The next useful question is whether a finite, competently pretrained history
policy fails to adapt under uncertain response changes, and whether online
updates improve on it at an acceptable cost. A mathematically optimal history
map can always be described as fixed; its existence is not a proof that every
finite pretrained network will approximate it adequately.

Prepare one explicitly hypothetical **recurrent support-service scenario**, not
another cost sweep, with the following design dependencies:

1. **Persistent, uncertain response.** Replace immediate noiseless recovery of
   one of two deterministic response worlds with repeated, noisy completion
   evidence. The uncertainty must be stated in the model and shared by all
   controllers, not hidden only from the frozen arm. Persistence and noise ranges
   need an ex-ante justification, not selection after seeing RL scores.
2. **Genuinely recurring decisions.** New work must sometimes arrive after
   support capacity could have become idle, and downstream contention must
   actually change transitions. Repeating the prior release-time offset would
   not introduce another mechanism. Include unchanged and rapidly fluctuating
   response controls; do not assume all uncertainty is learnable.
3. **Causal measurement and feasible action.** Observe releases, completion/
   blocking events, paid/delivered availability and queue occupancy; do not
   reveal latent productivity, future outcomes or exact unmeasured remaining
   work. Treat unfinished work as censored, and lack of eligible work as lack
   of exposure, not measured zero productivity. Specify whether resource booking
   is continuous aggregate time or integer crews; do not invent continuity to
   fit DDPG or accelerate biological growth/testing.
4. **Strong matched-information alternatives.** Pretrain a competent frozen
   history-aware policy over the declared development distribution, then make
   tensor-identical frozen/online forks. Both retain the same causal estimator
   interface. Include adaptive rules and completion-data identification + MPC;
   an estimator from exact remaining work is not an information-matched baseline.
   Charge exploration, identification and resource commitments to the objectives
   and disclose offline data, model queries and measured latency.
5. **Gated execution.** First show action leverage, state-dependent decisions
   and reproducible safe headroom on an independent development split. If cheap
   adaptive control already meets the operational objective, report that result.
   Only then lock the full online-vs-frozen pilot, replication axes and stopping
   rules. No positive outcome or publication acceptance can be guaranteed.

This is a design direction, not a runnable scenario specification or launch
authorization. A question about synthetic mechanism research versus site-data
calibration was sent during this task and had no reply at this readout. Do not
infer the answer. The twelve E1 operational inputs remain null in
`specs/2026-09-29-qualified-support-contract/measurement_contract.json`.
If a synthetic scope is chosen, put hypothetical assumptions in a separate
versioned specification, not into those fields as purported measured values.

N1-N7's verified learner/collector mechanics can be reused after this design is
settled. They do not certify new features, multi-episode recovery, physical
resource semantics or complete clinical liabilities. Stage E and the current
paper's formal claims remain unchanged.

## Reproducibility and verification

- Protocol/config commit: `db089a9`.
- Recorded source commit: `19cb526c67bc8b2ce47b6294cbf5fe8be5b12017`.
- Primary output: `reports/2026-09-29-queue-cost-sensitivity/run/`.
- Independent repetition: `reports/2026-09-29-queue-cost-sensitivity/repeat/`.
- Summary SHA256: `7defedeaa477d8e47aa199db8370d39fca509e848757824b6b1910f263bf65b6`.
- Full sensitivity SHA256: `456b683ec272c47be545b65b95d7ff1d156176d2bfe400434e0d0e7332dd6892`.

The upstream independent accounting audit passed: 60 episodes, 300 decision
rows, 16,368 tree edges, 9,088 closures and 59 verified hashes. All paid terminal
commitments are included. Reweighting preserves all recorded information
partitions; terminal continuation costs do not enter current observations.
Known-law policy witnesses replay exactly, and independently enumerated
clairvoyant/open-loop bounds retain their required order.

Both runs exit 0. All six JSON files are byte-identical across runs; both
inventories verify. Eight current source/config/protocol locks and eight input
file hashes remain unchanged after calculation. A separate direct summation
from archived decision rows plus episode settlements reproduces all 360
method-by-cell/setting expected costs, without using the repriced tree edges.
Eleven new tests and the combined 68-test related regression suite pass; full
Python compilation and diff checks pass. No new environment or planner
transition queries, training, scientific CRNs, remote operations or manuscript
result changes occurred.

Run from the persistent worktree, using a new output directory:

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m evaluation.audit_queue_cost_sensitivity --output /private/tmp/queue-cost-new-analysis
"$PY" -m unittest tests.test_queue_cost_sensitivity tests.test_service_queue_evidence tests.test_service_queue_observation_contract tests.test_service_queue_completion_rule tests.test_service_queue_network tests.test_service_effort_decisions tests.test_service_effort_mechanics
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Scientific JSON is deterministic. A later execution commit changes provenance
and its inventory hash even if all scientific values remain identical. Existing
outputs cannot be overwritten.
