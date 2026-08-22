# Routing-primary DDPG hyperparameter development record

Status date: 2026-08-10

## Scope and interpretation

This record covers development-stage parameter sensitivity and mechanism
screening for the routing-primary AFR-GCN-DDPG controller. These screens are
not formal paper results. They used development training seeds and CRN streams
to choose a candidate protocol; those seeds and streams must not be described
as independent confirmation evidence.

The routing mechanics, patient identity rules, four routing scenarios, MDL-2
anchor, teacher-data contract, action bounds, and feasibility projection were
held fixed unless a screen explicitly targeted one of those mechanisms. GCN
and flat policies must receive the same tuning budget in the confirmatory
comparison.

## Completed development screens

| Component | Values or variants examined | Development decision |
| --- | --- | --- |
| Residual deployment magnitude | Anchor-only fallback and residual multipliers 0.10, 0.25, and 0.50 | Retain a bounded 0.10 specimen residual; larger residuals were less stable. |
| Environment update frequency | Every step (1) versus every fourth step (4) | Retain 1. The frequency-4 candidate looked favorable in the seed-0 screen but did not generalize in the three-seed confirmation. |
| Actor update frequency | 2, 4, and 8 critic steps per actor update | Retain 2 for the current candidate. |
| Online actor learning rate | 1e-5 and 3e-5 | Retain 1e-5. |
| Exploration noise | OU sigma 0.005, 0.02, and 0.05 | Retain 0.005. |
| Imitation regularization | Weights 1 and 2 | Retain 1 for the current candidate. |
| Frozen-pretrain reference penalty | Weights 0, 250, and 500 | Retain 500 outside empirically favorable online samples. |
| Pretraining and critic warm start | Short smoke pretraining versus 300 epochs; 0 versus 500 offline critic updates | Retain 300 pretraining epochs and 500 offline critic updates. |
| Specimen action realization | Continuous proposal and integer-lot straight-through quantization | Retain integer-lot critic actions with a straight-through actor gradient. |
| Online reward target | Immediate anchor-relative and complete four-step anchor-relative return | Retain the four-step target. |
| Online release/self-imitation filter | Unfiltered, positive full-horizon, immediate-plus-full dual confirmation, and material dual confirmation | Retain dual confirmation with a 0.0005 scaled-return threshold, equal to one configured patient-loss cost after reward scaling. |
| Deployment gate | Development validation threshold grid with anchor-only fallback | Retain the fixed 0.575 deployment threshold for the current candidate. |

The update-frequency experiment is the clearest reason not to rely on one
training seed. Frequency 4 passed its seed-0 screening gate, but the later
three-seed routing confirmation showed substantial between-seed heterogeneity
and did not support promotion. Accordingly, parameter choice and confirmation
must be separated by both training seeds and CRN streams.

## Current development candidate

The selected development configuration is:

`experiments/configs/patient_indexed_specimen_routing_mac_mps_bounded_gain01_specimenonly_ddpg_sharedpreonline_quantizedspecimenste_anchorrelreward_nstep4_materialadv_dualconfirm_balancedcycle8.json`

Its principal settings are:

| Setting | Value |
| --- | --- |
| Algorithm | GCN residual DDPG around routing MDL-2 |
| Online development episodes | 8 |
| Episode horizon | 52 decision epochs |
| Batch size | 64 |
| Pretraining epochs | 300 |
| Offline critic updates | 500 |
| Actor learning rate | 1e-5 |
| Critic learning rate | 3e-4 (shared benchmark setting) |
| Environment update frequency | 1 |
| Actor update frequency | 2 |
| OU exploration sigma | 0.005 (shared benchmark setting) |
| Specimen residual scale | 0.10 |
| Other residual action groups | Disabled in this specimen-only screen |
| Frozen-pretrain reference weight | 500 |
| Online self-imitation weight | 1 |
| Online reward | Four-step exact-CRN MDL-2-relative return |
| Release criterion | Positive immediate and complete four-step return, each at least 0.0005 |
| Deployment threshold | 0.575 with anchor-only fallback |

On the fresh CRN961 development evaluation, the episode-8 policy had a lower
cost point estimate than both MDL-2 and its frozen-pretraining checkpoint. The
episode-8 versus frozen-pretraining paired mean was -251,985 cost units
(-0.009504%), with a 95% interval of [-615,612, 71,948], 51 wins, and 14 ties
among 100 pairs. The interval crosses zero, so this is a promising development
signal rather than verified online-learning gain.

## Independent confirmation and episode budget

Independent confirmation is required. Seed 0 has been used extensively for
development, and seeds 1 and 2 have appeared in earlier routing sensitivity
work. The routing-primary confirmation should therefore use five untouched
training seeds, provisionally 10-14, for both GCN and parameter-matched flat
DDPG. Five seeds are preferred to three because prior routing runs exhibited
large between-seed differences and occasional sign reversals.

Training to 300 episodes is not automatically required. Each episode contains
52 simulator transitions, so 100 episodes provide approximately 5,200 online
transitions per seed after a strong pretrained initialization. The defensible
budget is staged:

1. Train every GCN and flat confirmation run to 100 episodes, saving complete
   state every five episodes.
2. Inspect prespecified development learning curves at episodes 25, 50, 75,
   and 100 without opening the final holdout.
3. If either architecture has not plateaued, extend every confirmation run to
   200 episodes under the same settings.
4. Extend all runs to the common maximum of 300 only if the 100-200 curves show
   continuing, multi-seed improvement without deterioration in the clinical
   guardrails. Never extend only the better-looking method or seed.
5. If learning has plateaued by 100 or 200 episodes, spend additional compute
   on independent seeds and paired evaluation replications rather than longer
   trajectories.

The extension decision is global and based only on the frozen development
stream. Final policy comparisons use untouched paired CRNs and include final
versus frozen-pretraining, GCN versus flat, and GCN versus routing MDL-2.
The executable first-boundary configuration is
`experiments/configs/patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100.json`.
The confirmation config locks the shared development teacher by SHA256
(`9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92`);
the training entry point verifies this digest before creating any learned-policy
output. Reusing this frozen teacher for both architectures preserves the matched
comparison and does not reuse learned checkpoints or online training state.

## Remaining sensitivity versus HPO

No broad automated HPO is planned. The completed work is a targeted sequential
screen, not a claim that a global optimum was found. For the manuscript, the
main settings above are reported together with their tested ranges. Any
additional sensitivity must be declared before execution, use a separate
development stream, give GCN and flat the same search budget, and leave the
confirmatory seeds and holdout untouched.
