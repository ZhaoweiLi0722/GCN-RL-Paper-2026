# Stage G1 Legal-Action Critic Feasibility Results

> **Scenario reconstruction correction (2026-08-18).** This post-hoc audit
> reconstructed each run from the stored pretraining reference `env`
> (`routing_nominal_history`) rather than its multiscenario online assignment.
> The fitted rows and hashes remain immutable nominal-history diagnostics, but
> the former persistent-hotspot and per-cluster interpretation is superseded.
> No corrected 156-state G1 rerun has been performed. A corrected 27-state
> prospective screen independently failed to authorize escalation; see
> `patient_indexed_specimen_routing_ddpg_problem_redefinition_review.md`.

## Decision

Stage G1 completed on implementation commit `bffd2ae`. The audit covered all
156 frozen-pretrain states, both prespecified horizons, all five distinct
executed legal actions, and independent discovery and validation replication
streams. All 1,560 label rows and 1,560 leave-one-seed-out prediction rows were
unique and finite. The three actor hashes were unchanged, no fitted checkpoint
was saved, and every immutable F1 and G0 input hash matched after execution.

The locked classification is
`unstable_counterfactual_labels_close_extension`.
`legal_action_ranker_feasibility_passed`,
`actor_transfer_smoke_design_authorized`, `online_training_authorized`, and
`formal_confirmation_authorized` are all false.

## Counterfactual label stability

Validation showed material legal-action headroom in 87/156 states (`55.77%`),
so the negative result is not explained by an absence of alternatives.
However, discovery and validation selected the same best remaining-horizon
action in only 85/156 states (`54.49%`), below the prospective `70%` stability
gate. Agreement was `63.46%`, `48.08%`, and `51.92%` for seeds 60, 61, and 62.

Pairwise cost-order signs were more stable (`82.52%` across 1,287 material
pairs), indicating that much of the instability is concentrated near the
top-action decision rather than every pair. That distinction matters for
deployment: DDPG needs a reliable policy-improvement choice, not merely a weak
average ordering signal.

## Leave-one-seed-out ranker result

The fitted critic reached 47/156 validation top-1 decisions (`30.13%`), versus
46/156 (`29.49%`) for the frozen critic. The one-state gain is below the locked
`40%` absolute gate and provides no meaningful improvement over baseline.
Validation pairwise accuracy rose from `53.58%` to `57.74%`, but remained below
the `65%` gate.

| Held-out seed | Frozen top-1 | Fitted top-1 | Gain |
| --- | ---: | ---: | ---: |
| 60 | 16/52 (`30.77%`) | 18/52 (`34.62%`) | `+3.85` points |
| 61 | 12/52 (`23.08%`) | 14/52 (`26.92%`) | `+3.85` points |
| 62 | 18/52 (`34.62%`) | 15/52 (`28.85%`) | `-5.77` points |

No seed reached the required `+5`-point gain, and seeds 61 and 62 missed the
per-seed `30%` fitted-accuracy floor. The fixed training objective reduced its
loss in all three folds and changed each critic by a finite amount, but that
in-sample optimization did not produce the required unseen-seed ranking.

## Mechanistic conclusion

The evidence now closes both links considered after Stage F1:

1. Stage G0 showed that the online paired critic did not generalize to
   remaining-horizon legal-action ranking, while most raw actor changes
   disappeared through correction composition and integer-lot execution.
2. Stage G1 showed that even a stronger offline, all-legal-action,
   remaining-horizon, within-state-normalized critic objective did not meet
   prospective unseen-seed accuracy or label-stability gates.

Therefore, another online episode budget, actor learning-rate change, paired
loss reweighting, or proto-action selector is not scientifically authorized.
Those changes would optimize against an action target that is not yet stable
or demonstrably generalizable.

## Manuscript implication

The paper should not claim that online DDPG updates improve endpoint
performance. The defensible result is that the graph-aware, offline-pretrained,
MDL-2-anchored policy is clinically noninferior, while prospective paired
attribution and two mechanism audits did not establish an incremental online
DDPG benefit under this finite-horizon stochastic simulator.

DDPG may remain in the method lineage and implementation description, but the
source of demonstrated performance must be attributed to the validated
components rather than to online adaptation. Stage F1, G0, and G1 should be
reported as an ablation and limitation, not hidden or converted into a
positive claim. Any future attempt would require a new scientific design with
more stable long-horizon counterfactual targets or a redefined action/reward
problem; it is not a continuation of the current extension.

## Evidence

- Config SHA256:
  `556e5f5b58b7ce1b807f1d5890ccf85eda1d581391141af2020e8da828f1b9c0`
- Summary SHA256:
  `9e96deec850d6033fb5a2fb9e2f5090f8a040468df9faf8634015a14875d1c41`
- Legal-action rows SHA256:
  `160450c7752d3097f37319635b0a8cb810a9ac95c0c985bd43a8aedbd41dfeb3`
- Fold-prediction rows SHA256:
  `96d84e7a1f458227214690f1816388c142a207821373a8eea3d362d39ee47af9`
- Dataset arrays SHA256:
  `837f010a323e90a115a0ddf8dbdf832b675dc30f8bcc04087eb76de6a7bbbb2b`
- Curated directory:
  `experiments/evidence/patient_indexed_specimen_routing_ddpg_legal_action_ranker_g1/`
