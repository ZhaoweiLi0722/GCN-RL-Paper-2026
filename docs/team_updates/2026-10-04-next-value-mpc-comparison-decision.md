# Next Value-MPC Comparison: Recorded Direction

Recorded 2026-10-04T23:18:16Z at entry HEAD
`8c98ba80f89d936143ecad486c3207e5a68ed4e1`.
Status: Zhaowei endorsed the comparison design and requested this durable record.
This is not a frozen numerical protocol or authorization for new scientific calls.

## Evidence And Question

The completed five-block, 600-world comparison is documented in
[the terminal readout](../../specs/2026-10-04-value-mpc-comparison/recovery1/terminal-readout.md).
Under persistent change, graph/plain mean paired savings were 0.606871%, with
an absolute-cost interval crossing zero and only 3/5 positive blocks. The primary
strong-baseline screen failed. Graph/flat mean paired savings were 1.951634%; its
absolute-cost secondary screen passed with 5/5 positive blocks, but the percentage
interval crossed zero. Some blocks had adverse patient-loss differences. These
are development results, not robust MPC superiority, isolated edge causality or
clinical safety. The earlier 13.7493% updated/initial result used a weak initializer
in a different study and must not be substituted for the strong-baseline result.

Next question: can improved value learning produce a reproducible cost/patient
benefit over both the existing value-MPC method and a competent plain MPC,
after accounting for training data and planning computation?

One candidate explanation is a mismatch between trajectory-based value fitting
and the candidate terminal states/rankings used by the planner. This is a
hypothesis, not an established cause. Choose and prospectively specify one
decision-relevant improvement; do not change several mechanisms at once.

## One End-To-End Comparison

| Arm | Purpose |
|---|---|
| Plain MPC | Strong no-learned-value reference under the common planning setup |
| Existing GCN-value-MPC | Reference implementation separating the new change from the existing method |
| Improved GCN-value-MPC | One explicitly specified value-learning modification |
| Matched-data direct-return value regression | Separate bootstrapped TD targets from direct realized-return targets and extra data |
| Longer-horizon plain MPC | Performance/decision-time reference for spending more computation on planning |

The direct-return arm is not automatically a non-RL arm: Monte Carlo return
regression can itself be policy evaluation. Define the actual estimand as TD
versus direct-return learning under controlled data, not RL versus all supervision.
Where valid, TD and direct-return learners receive identical saved training
trajectories, splits, public inputs and matched optimization budgets. If the
improvement requires additional simulator labels, provide/count comparable data
access or explicitly retain it as part of the treatment. Policy-dependent
continuation data must not be called identical merely because seeds match.

Use common information, action support, objective and evaluation worlds across
arms. Specify whether existing models are reused or the reference is newly trained;
do not compare incompatible initialization/training budgets without disclosure.
The longer-horizon arm is a compute/performance reference, not an equal-query
comparison when it uses more queries. Report total training computation, planning
queries and per-decision latency, including relevant tail latency and hardware.

Seal all selected models/configs before opening new, unused evaluation worlds.
Prespecify the main contrasts, independent training blocks, paired evaluation,
uncertainty method, cost components and patient outcomes. Report every condition,
block and adverse trade-off; no favorable-seed selection or post-test tuning.
Attach value calibration and candidate-ranking diagnostics to this same run.
They explain the result and must not become a separate toy-fit or audit campaign.

## Reward And Scenario Decision

Zhaowei clarified: keep reward unchanged now, but allow a justified change when
necessary. This is NOT a permanent prohibition on reward redesign.

For the next proposed method comparison, retain the current reward weights and
scenarios so that the new learning mechanism has an interpretable incremental
effect. A future reward amendment is warranted by a demonstrated accounting or
terminal-liability defect, an omitted decision-relevant consequence, or an
explicitly revised scientific/operational objective, not merely an unfavorable
result. Document the evidence, old/new objective and affected comparisons, freeze
the amendment before execution, and retain common raw cost/patient reporting.
Signal scaling/shaping and changing the objective are distinct; neither is
automatically effective. Necessary reward changes remain available through an
explicit bounded amendment, not outcome-driven weight search.

New scenarios are also optional, not the immediate remedy. If justified later,
consider advance shared-capacity commitments, persistent supply delays with
perishable reagents, or production/QC queue coupling. Explain the physical
mechanism and delayed consequence; do not weaken MPC or add arbitrary randomness
to manufacture an RL advantage. Preserve patient identity, biology/QC requirements
and honest synthetic assumptions. Missing E1 field data remain missing. Graph
representation attribution and deployment-time online adaptation each require
their own suitable comparisons; this plan first targets simulation value learning.

## Next Action And Boundaries

Produce ONE complete bounded numerical proposal with the exact improvement,
initialization/data sharing, horizons, seeds, sample sizes, training/validation/test
separation, optimizer/environment/query counts, per-arm/phase and total time/IO
caps, stopping/failure rules and a minimal implementation plan. State the concrete
remaining blocker and realistic preparation estimate. Reuse valid facilities and
checks; do not repeat historical audits or add preliminary fitting screens.

Direction/preparation is endorsed. The unproposed numerical package is not yet
approved or frozen. After its approval and necessary committed locks, proceed
directly through preparation, the single experiment and readout without a second
routine launch question. Do not guarantee positive RL results or publication.

All completed/failed attempts, source locks and historical manuscript evidence
remain unchanged. No new run, model call, reward modification, retry, holdout use,
Stage E reopening, remote action or Howard approval follows from this note.
The existing monitor remains paused; scheduling is not experimental progress.
