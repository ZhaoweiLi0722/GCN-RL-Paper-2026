# Time Baseline and Reward Pivot

## Recommendation
Yes: one bounded time-baseline comparison is reasonable, not a promise of patient gain.
The saved diagnosis shows strong remaining-time structure and weak collection-time value fit;
time already reaches the critic. Reuse [diagnosis:18-40,68-75][diagnosis] and
[source contract:45-80][contract]; no numeric reanalysis or new model work is needed here.
Do not launch another baseline search if the single comparison disappoints.

Coordinator scope: replace only the advantage baseline with training-only
`A(e,t) = G(e,t) - b_minus_e(t)`, excluding the entire target episode from baseline estimation.
Keep actor/start, support, raw rewards, critic return targets, entropy, rollout normalization
and evaluation rule unchanged. No test labels or held-out episode returns enter its baseline.
The saved scalar comparison measures signal/residual variance, not actual policy-gradient
variance, new actions or patient benefit. Lower variance is not itself sufficient success.

## Fast Decision Rule
| Finding at the declared comparison boundary | Next finite action | Do not claim/do |
| --- | --- | --- |
| In the single complete comparison, baseline residual variance improves but raw patient/cost outcomes do not | Close the tested baseline-only route; prepare one state-dependent signal-shaping option below if it adds information beyond time, or a justified objective amendment if indicated. | Do not equate cleaner advantages with patient improvement or repeatedly extend training. |
| Baseline remains poor | Stop baseline tuning after the single comparison; use existing diagnostics to distinguish residual state variation from a concrete implementation error, then prepare one reward-direction decision. | No automatic network/weight sweep; poor baseline fit alone does not prove a bad objective. |
| True objective mismatch is established against the intended care/cost definition | Prioritize an explicit revised objective and prospective comparison; retain old-objective reporting for transparency. | Do not silently fix historical rewards or choose penalties to make PPO win. |
| Useful decision headroom remains unknown | Keep headroom unknown; reuse existing support/outcome evidence and propose one bounded decision only if needed. | Sampled class-return associations are not same-state counterfactuals; neither baseline failure nor shaping creates headroom. |

Actual patient gain cannot be judged by saved advantage arithmetic. Do not add a separate
scientific fit gate; a patient-level null closes only the tested intervention.
There is no new reviewer chain or fit-only prerequisite before a complete comparison proposal.

## Direction A: Preserve the Objective
Use time-indexed, action-independent public-state potential shaping in **scaled reward units**:
`r'_t = r_t + gamma * Phi_(t+1)(s_(t+1)) - Phi_t(s_t)`.
One concrete candidate is `Phi_t(s) = -k * (1 - t/T) * B(s)`, where B is a fixed,
public-observation backlog/near-expiry pressure score, not hidden patient outcomes or oracle Q.
Choose its feature definition and scale prospectively using training information only;
freeze the same potential throughout the comparison. No weights or k are selected here.
Current contract: `r_t = -1e-9 * cost_t`, gamma=lambda=1, T=52, terminal bootstrap=0
([frozen proposal:412-421][proposal]; [source contract:49-55][contract]).

Require `Phi_T(s)=0` for **every** terminal state, including unresolved patients.
Then the discounted episode sum is `G'_0 = G_0 - Phi_0(s_0)`; initial-state sampling
must be policy-independent. At the last step the shaping term is `-Phi_(T-1)`.
Do not drop that cancellation or leave a backlog-dependent Phi_T: either changes rankings.
At any genuine nonterminal truncation, retain the potential and consistent transformed
bootstrap `V' = V - Phi`; do not confuse a rollout cutoff with this study's true terminal.
Keep raw cost/patient evaluation and separate raw/shaping receipts; do not rewrite saved JSON.

Important limit: here full-episode MC gives `G'_t = G_t - Phi_t`. With the correspondingly
transformed value `V'_t = V_t - Phi_t`, advantages are identical. With an unchanged V,
this instead acts like changing the advantage baseline to `V + Phi`.
Thus a time-only potential repeats the baseline idea; shaping is not an independent cure.
Consider a state-dependent potential only for an explicit remaining signal problem;
it preserves the return objective, not necessarily finite-sample PPO training behavior.

## Direction B: Explicitly Amend the Objective
Current horizon accounting reports unfinished identities but adds no terminal cost
([candidate_collection_boundary.py:133-146,195-218][boundary]). This is a declared
finite-window objective, not a proven coding defect or full-lifecycle valuation.
`terminal_active` counts unique waiting, production, specimen-transit and finished-return
patients; lost/delivered identities are separate ([candidate_pilot_verification.py:60-107,245-248][verify]).
Do not value every active patient as a death. Waiting patient-steps are summed end-of-epoch
queue occupancy, not mean completed-patient waiting time or an already added cost
([dynamic_candidate_verification.py:162-188][dynamicverify]).

If the intended objective covers later care obligations, propose a residual terminal
liability `C_T(s_T)` by stage/risk and add `-1e-9 * C_T` once to the final reward.
It must represent future burden not already charged, with separately justified treatment
of remaining resource use and patient risk. This changes the objective; it is not invariant
potential shaping. Keep component outcomes visible and avoid rewarding premature loss merely
for reducing backlog. Calibration/domain justification remains missing, not implicitly zero.

Loss/expiry semantics require care: `patients_lost` already includes expiry losses;
`material_wasted` additionally covers transport loss and discarded manufacturing therapies.
The source charges `weight_patient_lost * patients_lost` plus
`weight_expiry * material_wasted`; the latter is broader than the reported `expiry_losses`
metric ([patient_capacity_planning.py:468-488,534-541][env]; [dynamic verification:186-187][dynamicverify]).
An expiry event can therefore incur both patient harm and material waste charges. This is
overlap of consequences, not proof of duplicate valuation. Waiting expiry versus ineligibility
is mutually exclusive in the event branch ([patient env:765-786][env]).
Do not add total losses and expiry losses as disjoint patient counts. If domain definitions
show the two weights price the **same** harm, amend to nonoverlapping clinical/material costs,
accounting also for already charged purchases/operations; do not simply delete expiry cost.
No such double-counted-harm defect is established by the inspected source alone.

## Bounds
For any later reward trial, success thresholds, patient/cost tolerances, potential weights
and terminal valuations are **not chosen**. Freeze its complete one-attempt package before
new execution; no automatic reward tuning, retry, model call or training is permitted.
The current baseline proposal has its own prospective development screen, not a reward permit.
This file is preparation only: source/report reads, no scientific imports or numeric reruns;
only this memo is authored. Coordinator owns scalar implementation, integration and approvals.

[diagnosis]: ../../reports/2026-10-02-saved-return-ranking/readout.md
[contract]: ../../reports/2026-10-02-saved-return-ranking/source-contract.md
[proposal]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-proposal.json
[boundary]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/candidate_collection_boundary.py
[verify]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/candidate_pilot_verification.py
[dynamicverify]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_verification.py
[env]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/env/patient_capacity_planning.py
