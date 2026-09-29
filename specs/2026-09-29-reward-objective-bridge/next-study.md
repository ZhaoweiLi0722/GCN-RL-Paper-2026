# Next decision: test consistency before changing the task

Status: non-executable design for review, not scientific launch authorization.
No new rollout, optimizer budget, seed stream or reward intervention is assigned.
Howard's approval is not asserted. Stage E and all earlier evidence stay closed.

September 29 update: Zhaowei accepted the recommended legacy-simulator route.
The subsequent archived-cost readout in
`specs/2026-09-29-g1-decision-cost/readout.md` distinguishes costly action
selection errors from exact top-1 disagreement. It is not a reward intervention
or an approval to train. Recommend the original 52-step finite-window endpoint
for the next protocol draft. After the explicit endpoint question, Zhaowei
requested continuation; the finite-window design is now recorded in
`specs/2026-09-29-fixed-window-value-contract/protocol.md`. This is design
acceptance, not approval to launch its proposed data-only pilot or training.

## Separate two routes

| Route | Question | What is still needed |
| --- | --- | --- |
| Existing simulator, fixed published cost weights | Can a consistently specified learner add value over its own frozen actor in the existing modeled task? | A separate development protocol, data/initialization/termination decisions, action-leverage acceptance, fresh streams and bounded budget. Monetary/clinical calibration is not required to describe a simulator-only result. |
| New qualified-support/capacity task | Does learning improve a defensible new operational decision? | E1 task, measurements, resources, cost units and closure inputs, plus the same learning/comparator requirements. Do not fill its missing inputs by copying the old simulator. |

Recommend the first route for a limited feasibility decision, **not** another
large performance campaign. The new E1 staffing channel should not be a blanket
blocker for all legacy-task research. Equally, an old simulator's weights do not
calibrate the new channel. R1 does not pass any old headroom/learnability gate.

## What a reward-consistency study would and would not isolate

The primary contrast remains corrected-online minus its tensor-matched frozen
actor, not corrected-online minus a historical result from another experiment.
Both receive the same information and state-derived reference interface, and
can change actions through feedback. Frozen refers to weights, not constant
decisions. Use identical paired exogenous worlds and charge online exploration
and all new offline/critic-preparation work.

Keep the existing simulator, action constraints, architecture, gate, business
cost weights and external reported metrics fixed. Do not change action geometry,
add a new capacity channel, alter regularization and reshape rewards in the same
contrast. The historical M2 audit is evidence of inconsistency, not proof that
fixing it will improve a policy. Jointly correcting reward/replay/target semantics
tests a coherent repair package; it does not identify which defect caused a null.

Use one absolute negative-cost definition in new offline data, calibration,
online replay and targets. Independent teacher counterfactual rows must remain
one-step records, never concatenated trajectories. Historical critic/Adam/replay
state cannot simply be carried into a claim of a clean objective. Preserve a
competent actor checkpoint and define a common, separately audited critic
initialization/calibration before any online/frozen fork. Gate/actor changes in
that common preparation require their own declaration and frozen re-evaluation.
Do not treat the random N7 engineering actor/gate as a pretrained comparator.

### Resolve the endpoint explicitly

There are two different estimands; do not slide between them:

- A fixed-horizon modeled cost excludes post-horizon liabilities by definition.
  Gamma=1 and terminal value zero can represent that finite-horizon objective,
  but the result must be labeled finite-window cost, not complete operational
  welfare. The observation/critic must represent the remaining horizon. This
  would be a separately approved departure from N5's complete-settlement design.
- A fully settled finite task follows N5: all accepted work and commitments,
  losses/expiry and closure costs are charged before absorbing termination.
  A horizon cutoff with pending obligations is not a zero-value terminal state.
  Implementing closure would change the evaluated task and requires a separate
  contract, not merely setting `done=True` or switching gamma.

R1's six-step mechanics receipts cannot choose between these. A zero active
patient count does not certify inventory/commitment settlement or future demand
closure. Do not use these six traces to choose a favorable endpoint or weight.

## Smallest meaningful next scientific gate

Before actor training, freeze a bounded, independent development check of the
corrected target pipeline and local executed-action value. It must use the
actual deployed discrete projection, gate and observation support, not only a
continuous critic output. Historical G0/G1 failures remain relevant; a reward
change does not remove integer plateaus or create replicated label signal.

Record real temporal lineage, complete component costs for the chosen endpoint,
and held-out action-ranking accuracy/stability. Report absolute target errors
alongside centered action differences: learning state cost without learning
action differences is not enough for DDPG. Any critic-fitting validation is a
new scientific computation requiring authorization, not part of this R1 audit.
Do not use the formal holdout or cached labels of unverified horizon provenance.

Only if a prespecified signal requirement passes should a bounded online/frozen
study proceed. Include competent adaptive control with matched information;
keep unchanged-condition retention cases. Report gradient norms and directions
for separately weighted losses, executed action changes above quantization,
replay source composition, adaptation cost and end-of-budget performance. Do
not infer gradient domination from loss coefficients, or select the best
checkpoint after seeing evaluation outcomes.

Potential-based shaping is a later, separate shaped/unshaped attribution study
under an unchanged external objective. It is not the first repair and does not
license cost-weight tuning. All negative, failed and inconclusive runs remain.

## Decisions needed before execution

1. Legacy simulator-consistency route accepted by Zhaowei; new operational
   scenario remains separate and uncalibrated. This is not Howard's sign-off.
2. Finite-window endpoint chosen for the R3 design; retain its post-window
   liability limitation. The fully settled task is not silently substituted.
3. Identify acceptable actor/gate initialization and corrected critic data with
   verifiable provenance; no automatic migration of the old mixed replay.
4. Fix an independent ranking gate, practical effect margin, streams, sample/
   query/update/wall-time caps and stopping rules before outcomes are observed.
5. Record explicit scientific authorization and commit that protocol before
   any new environment data or critic/actor fitting. No automatic Stage D or
   formal confirmation follows a favorable development result.

Existing N1-N7 code is reusable engineering preparation, not a reason to build
another training framework. This document does not authorize executing any
bullet above. The current paper should retain its supported package-level
benefit and its lack of separately established online-update gain.
