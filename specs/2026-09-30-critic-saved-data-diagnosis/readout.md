# R6 saved-data diagnosis and next decision

## Bottom line

The saved evidence supports **two distinct problems**: the fitted critic fails
to generalize action rankings, and the current scalar cost can prefer a choice
with more patient losses. Reward arithmetic is internally consistent; clinical
objective alignment is a separate scientific issue. Neither changing weights
nor longer DDPG training is an established remedy.

This analysis used only the completed R6 packet: 18 parent trajectories,
72 saved states, 3456 conditional outcomes. It generated **zero new simulator
steps, model inferences, optimizer updates, or choices**. All results below are
post-run descriptive diagnostics, not a new independent confirmation or an
online-RL benefit. The R6 attempt and Stage E remain closed.

## 1. Noise matters, but does not explain everything

Each action pair uses the same eight saved future starts. We compared its mean
advantage difference on draws 0-3 versus 4-7. Exact ties are separate.

| Partition | Non-tied pairs | Opposite half-sample signs | Same-sign pairs | Critic correct on same-sign pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 405 | 130 (32.1%) | 275 | 227/275 (82.5%) |
| Test | 215 | 70 (32.6%) | 145 | 62/145 (42.8%) |

There are also 21 training and 19 test pairs tied in both halves, with none
tied in just one half. Across all non-tied test pairs, accuracy is 89/215
(41.4%). Per-model test accuracies reproduce R6: 32.0%, 46.7%, 46.3%.

The same-sign test subset still ranks poorly, so sign instability alone is not
a sufficient account of the failure. It does **not** identify how much error
comes from estimation, limited coverage, optimization or representation.
These are dependent pairs within states and trajectories, not 215 independent
experiments. Four-versus-four is not a fresh validation sample. Do not claim
statistically below chance or use this subset as a revised success criterion.

## 2. The request-to-execution distinction is visible

Forty certified action pairs have identical recorded first-step physical/RNG/
cost/event identities in all eight starts. Thirty also have identical quantized
requests and equal saved predictions. The remaining ten pairs are all in one
state, `seed62/test1/t39`: actions 1-5 produce the same recorded execution and
continuation costs, but their quantized requests differ. Their saved predicted
value gap reaches **1,339,949.26 simulator cost units** (actions 4 versus 5).

This is a concrete sampled example of a critic assigning value differences to
requests that did not change execution. It supports investigating an
execution-aware representation, but is not proof that all requests are useless
or that changing the representation fixes generalization. Those tied labels
were already excluded from pairwise accuracy, so they do not explain away the
poor non-tied ranking result.

## 3. Coverage and input representation remain hypotheses

- Each critic has only four independent training parents and 16 training
  states, compared with 238433 learnable parameters. Six action labels in one
  state are not six independent operating contexts.
- The locked transform creates 21 nodes x 36 features (756 coordinates),
  including the capacity hub and MDL-2 base-action features. Input normalization
  is disabled. Existing training-feature magnitudes reach 210, while other
  coordinates are survival fractions, time and small action signals.
- Exact graph-input duplicates were absent within each model's 24 states.
  This does not establish a sufficient/Markov observation.
- The fixed train-SD nearest-neighbor calculation is numerically fragile:
  some nonzero SDs are only 2.10e-9 to 5.77e-8 in almost-constant coordinates.
  Several test distances therefore explode. Preserve those outputs, but do
  **not** interpret the large distances as a valid OOD score or a causal
  coverage diagnosis. Four same-time training parents are also too few for
  range exceedances alone to be decisive.

Source inspection establishes information summarization, not its causal cost:
`PatientConditionCapacityEnv._patient_summary` retains counts, mean survival,
near-expiry counts, survival bins and a few routing summaries. It does not
expose complete individual age/risk/shock histories or queue ordering. Some
hidden variables (such as sampled future deterioration epochs) would be oracle
information and must not be added as deployable features.

The graph transform directly retains idle/total bioreactors rather than every
busy stage. However, its appended MDL-2 action features use next-stage capacity,
so claiming that all stage information disappears would be incorrect. These
observations motivate a public-information audit, not unrestricted hidden-state
inputs or a claim that the GCN itself caused the failure.

Source anchors: `src/env/patient_capacity_planning.py:211`,
`src/env/patient_condition.py:35`, `src/models/graph_features.py:324`,
`src/models/graph_features.py:521`, `src/baselines/heuristics.py:845`.

## 4. There is an objective trade-off worth resolving

All 24 sealed test choices were compared with frozen using their original raw
traces. Three choices have lower mean total cost **and** greater mean patient
loss. Components below are millions of original simulator cost units, not
validated real-world dollars. Negative cost is favorable.

| State | Total cost delta | Patient-loss cost delta | Capacity-shortage cost delta | Mean additional patients lost |
| --- | ---: | ---: | ---: | ---: |
| seed60/test1/t39 | -2.064360 | +0.125000 | -1.984550 | +0.25 |
| seed61/test1/t39 | -5.209031 | +0.750000 | -5.750043 | +1.50 |
| seed62/test1/t26 | -0.557240 | +2.375000 | -3.724017 | +4.75 |

The full output retains all eleven components, all eight paired draws and all
24 choices. Independent reconciliation covers 192 chosen-versus-frozen draws.
The patient-loss charge is exactly 500000 per lost patient. Capacity-shortage
cost is a per-epoch backlog/capacity-gap penalty; its accumulated reduction can
outweigh the loss charge. Other components complete each row's arithmetic.

This demonstrates a **scalar-objective conflict**, not a reward-sign/scaling
bug or evidence that the agent deliberately harms patients. Seven of the 24
sealed choices increase mean loss; only the three shown also reduce cost.
These are conditional remaining-window means, not deployed full-policy
clinical outcomes or evidence of clinical significance/safety.

Source anchors: `src/env/patient_capacity_planning.py:460` and `:534`,
`src/env/capacity_planning.py:1221`.

## Decision and next bounded proposal

Do not retrain the full actor or search reward weights now. Separate the work:

1. **Scientific objective:** decide with the team whether loss/completion are
   primary clinical constraints and cost is secondary. If so, prospectively
   define margins, uncertainty treatment and comparison rules from domain
   justification. Do not pick weights/margins to rescue the current outcomes.
   No clinical constraint or reward has been changed in this packet.
2. **Engineering diagnosis:** the smallest next controlled fit would compare
   the raw input against the codebase's existing config-derived graph-feature
   normalization, using only R6's four original training parents per model.
   Use leave-one-parent-out folds, matched initialization and targets, with
   actor/gate/reward/actions fixed. This is a retrospective learning diagnostic,
   not independent test confirmation or proof of online DDPG value.

Proposed ceiling, **not approved or launched**: 3 models x 4 folds x 2 input
arms = 24 fresh critics, at most 1000 updates each, 60 minutes, one attempt,
zero new simulation. No use of the old R6 test labels for fitting or model
selection, no hyperparameter search, actor update, or automatic next phase.
Only if an identifiable learning problem is established should a separately
approved prospective test or execution-aware feature study be considered.

For the paper, preserve the established graph/offline evidence and accurately
report the online-attribution limitation. This packet offers a more specific
explanation to investigate; it does not strengthen the online-RL performance
claim and does not establish that the frozen policy is globally near-optimal.

## Reproducibility

- Analysis commit `42d4792`; `evaluation/diagnose_saved_critic.py`.
- `reports/2026-09-30-critic-saved-data-diagnosis/diagnosis.v2.json` and
  `diagnosis.repeat.json` are byte-identical, SHA256
  `18c00b8e1c37f4c07eeb5e1924d8c2dbf0fe46d1c524be835a461eb51296b665`.
- Independent verifier: `evaluation/verify_saved_critic_diagnosis.py`;
  `verification.json` checks all 660 pairs, 40 execution equivalences and
  192 selected-choice paired component totals against raw evidence.
  `verification.recheck.json` additionally checks every reported choice against
  the original prediction seal and reproduces the same verification result.
- All 72 related unit tests and full repository compileall pass. The full
  saved-data run is the analysis smoke test; no simulator smoke was needed.
- R6 inventory (3649 files) and all 83 recorded source locks were checked
  unchanged before and after analysis; raw R6 execution was not repeated.
- Initial report serialization at `51eb657` failed on a NumPy integer;
  the invalid partial `diagnosis.json` and failure explanation are retained.
  A regression-tested serialization-only repair preceded the v2 calculation.
  Do not use the partial file as evidence.
