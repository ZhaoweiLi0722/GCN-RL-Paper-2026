# Dynamic Capacity: One Precision-Correction Attempt

## Authority And Scientific Scope

Zhaowei replied `继续推进` after the complete correction question and the
request to keep preparation short. Record the literal and context, without
claiming Howard's approval. This authorizes one separately budgeted correction
attempt, not a search or automatic retry. Preserve the first terminal failure.

The original pilot-proposal.json and pilot-proposal.md remain byte-identical.
Same three blocks, three operating conditions and six controllers; 288 total
trajectories including 216 evaluation trajectories. Same model initialization,
reward, patient weights, new synthetic labor accounting, observations, seeds,
sample size, action support, comparison metrics and original decision rule.
This is an engineering correction, not independent scientific confirmation.
Two training-world steps were previously observed; no prior model fit or test
world was accessed. E1 calibration remains missing. Stage E remains closed.

## Exact Implementation Correction

The public-only regression fixture reproduces candidate6/quantile0.1/horizon7
at public epoch2. The unfinished head has 8.881784197001252e-16 work after
two 4-unit jobs exhaust 8 units of capacity. Adding this positive remainder
to 8 rounds to 8 in binary64, incorrectly rejecting the strict inequality.

The versioned planner uses signed math.fsum residuals for prefix feasibility,
coordinate hull arithmetic and midpoint service. It preserves positive work,
does not round it away, and adds no tolerance or new policy parameter. The live
completion filter is unchanged; only the planner's internal numerical evaluation
is corrected. Old module, launcher, failure, frozen inputs and archives remain
unchanged. The new entry injects the corrected controller into the existing
serial runner. Snapshot format distinguishes the corrected controller.

## Single-Attempt Budgets

- Native steps 18,432; operations including construction/reset 19,008.
- Actor plus critic optimizer calls 7,872; neural forwards 25,824.
- Approximate planner epochs 1,327,104; filter transitions 1,843,200.
- Total wall time 5,400 seconds including I/O, analysis and archive.
- Combined storage 2,147,483,648 bytes.
- All phase, owner, batch and sub-count caps are unchanged from the original
  proposal. No transfer of unused budget from the first attempt.

Use results/dynamic_capacity_adaptation_20261003_recovery1 only. Freeze source,
runtime, original inputs, this amendment and exact user intent before launch.
Random streams deliberately match the failed pre-fit attempt; bind that old
packet and failure records separately, and exclude only these named attempt
roots from the unrelated-stream collision check. Do not call them fresh streams.

After necessary arithmetic/real-public-entry regression tests and compileall,
execute the whole serial comparison without another routine-stage approval.
All three offline models must seal before final test access. Preserve all cost
and patient outcomes, including unfavorable effects. No automatic retry, model
selection, extra training, reward adjustment, holdout, remote action or export.

## Exit

Report online versus both frozen controls, adaptive rule, ID-MPC and fixed
allocation separately by condition/block. A changed action or training loss is
not performance improvement. Apply the existing prespecified signal rule and
report any cost/patient trade-off. On failure, preserve and close this attempt;
no new run without a separate decision. Retain the visible paused workflow at
handoff. Do not repeat the historical result audit or archive.
