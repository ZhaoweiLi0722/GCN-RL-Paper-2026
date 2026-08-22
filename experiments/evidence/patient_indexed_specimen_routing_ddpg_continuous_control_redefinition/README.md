# DDPG Continuous-Control Redefinition Evidence

This directory freezes the development-only evidence used to decide whether
the routing problem should be redefined around DDPG-compatible continuous
resource controls. No model was trained, no actor was updated, no checkpoint
was selected, and the formal holdout seed `91100000` was not used.

## Audits

- `h0_persistent/`: corrected 27-state screen on the three persistent-hotspot
  assignments used by Stage F1.
- `h1_nonstationary/`: independent 27-state screen on abrupt regime shift,
  regional drift, and compound regional stress.
- `superseded_nominal/`: the first H0 execution. It is preserved only for
  auditability because it reconstructed `routing_nominal_history` while its
  rows were labeled as persistent-hotspot scenarios. It must not support a
  scenario-specific scientific claim.

Each corrected audit used nine fixed decision states per seed, one anchor plus
twelve prespecified actions per state, three discovery replications, five
independent validation replications, and the remaining episode horizon. The
candidate groups were specimen transfer, reagent transfer, and joint
reagent/capacity transfer, each at `+/-0.05` and `+/-0.10`. A material
opportunity required at least 1 million modeled objective units of improvement
without lower completion service, more patients lost, or more manufacturing
ineligibility.

Both corrected audits classified the continuous geometry as unsupported. No
156-state extension, online training campaign, or formal confirmation was
authorized.

## Integrity hashes

- H0 persistent summary:
  `f533452b92d169796bceaa6966ad7925c67b59de5cfa6edf39ef6d29ec025f74`
- H0 persistent rows:
  `4ce85c796da6021f32e09f5014c11e738b027cd05b04c02f84ba9f3866ac0121`
- H1 nonstationary summary:
  `0dbb140148ae96755a94c24f2e74e2a1d644b6b909bbf78ffc31df370bef401e`
- H1 nonstationary rows:
  `1529a0242e6a3d0ba4464226f2dad0c9a1ada652b625574a2c55f4a737b3d046`
- Superseded nominal summary:
  `7b0e0c13b088a84a64fba5784b1abed7269727878b4f6349e34855cdbe7bcd44`
- Superseded nominal rows:
  `a791d32680e1e377f09043ce2432a06460d130ceeb9b7d6df9fd0a18e223b3ad`

The corresponding implementation and locked configurations are:

- `evaluation/audit_ddpg_continuous_control_headroom.py`
- `experiments/configs/patient_indexed_specimen_routing_ddpg_continuous_control_headroom_h0.json`
- `experiments/configs/patient_indexed_specimen_routing_ddpg_continuous_control_nonstationary_h1.json`
