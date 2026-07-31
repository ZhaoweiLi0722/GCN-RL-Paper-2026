# Multi-Scenario AFR-GCN-DDPG Attribution Pilot

## Decision question

Can a graph-aware, MDL-2-anchored residual DDPG controller learn a deployable
multi-scenario network policy, and do online DDPG updates add value beyond the
same frozen advantage-filtered pretraining?

This is a progression experiment, not a final manuscript experiment. Its
settings and gates are fixed before the RTX 4090 run.

## Why this experiment is next

The regional specialist established a graph-policy advantage, but the
five-scenario TD3 diagnostic showed that specialist checkpoints do not
generalize reliably. DDPG remains the canonical backbone because it beat TD3
in the matched 300-episode, five-seed replenishment comparison. The missing
attribution is therefore a matched, multi-scenario DDPG final-versus-frozen
test.

## Locked training protocol

- Methods: `gcn_residual_mdl2_network_ddpg_afd` and the parameter-matched
  `flat_residual_mdl2_network_ddpg_afd`.
- Expected trainable-module counts under the locked observation contract:
  approximately 524,775 for GCN and 524,604 for flat (0.033% difference).
- Training seeds: 0, 1, and 2.
- Online budget: 100 complete 52-week episodes per method and seed.
- Episode schedule: round-robin over nominal history, abrupt regional regime
  shift, gradual regional drift, and compound regional stress.
- Per-episode randomization: demand multiplier in [0.9, 1.1] and forecast
  error in [0.0, 0.2]. Compound-scenario shock and disruption centers remain
  stochastic.
- Anchor: MDL-2.
- Teacher: the same four-scenario advantage-filtered cache for graph and flat
  policies.
- Residual actions: reagent and idle-bioreactor transfers only; specimen
  transfer and replenishment corrections are disabled.
- Action safeguard: one source and one destination per transfer group,
  minimum normalized endpoint magnitude 0.04.
- Optimizer: actor learning rate 1e-5, critic learning rate 3e-4, 500 critic
  warm-up updates, actor update every two eligible updates, exploration sigma
  0.005.
- Persistent teacher regularization weight: 0.25.
- No holdout-based checkpoint, scale, threshold, or seed selection.

The four scenarios are the primary in-distribution factor mixture for this
pilot. Severe global demand drift remains an extrapolation diagnostic and is
not pooled into the progression gate.

## Teacher-data provenance

| Artifact | SHA256 |
|---|---|
| `teacher_train.npz` | `49b21ac92c743f1489b6e914cf17795c214ff3f4d061faec4561db5d91477953` |
| `teacher_validation.npz` | `6070d9a8b959252109e1a5ea597604ae804b873d6a1883686c6d3fd637fb6be3` |
| `manifest.json` | `7b2fb4ff6cd1cede6bf15d5e9570de533b3becac7f092ff7b15f55925f921160` |

## Locked evaluation protocol

Each run produces a frozen post-AFD checkpoint and a final post-online-DDPG
checkpoint. Both variants receive the same fixed deployment:

- residual scale 0.1;
- correction-gate thresholds [0.0, 0.0, 1.0];
- top-1 endpoint projection with minimum magnitude 0.04.

Final and frozen variants use identical common random numbers:

- 100 holdout replications per scenario and training seed;
- holdout seed 59,900,000;
- 52-week horizon;
- hierarchical paired bootstrap with 20,000 resamples.

Primary comparisons:

1. final AFR-GCN-DDPG versus MDL-2;
2. final AFR-GCN-DDPG versus final AFR-Flat-DDPG;
3. final AFR-GCN-DDPG versus frozen AFR-GCN pretraining;
4. graph-minus-flat final/frozen difference-in-differences.

Clinical noninferiority is assessed for completion service, patients lost, and
manufacturing-period ineligibility. The comparison tool also audits CRN keys,
anchor outcomes, residual use, online update counts, finite diagnostics, and
actor drift.

## Progression gates

Advance to the five-seed confirmation only if all conditions hold:

1. final graph cost CI is below zero versus MDL-2;
2. final graph cost CI is below zero versus matched flat;
3. final graph cost CI is below zero versus frozen graph pretraining;
4. graph-minus-flat difference-in-differences cost CI is below zero;
5. final graph is clinically noninferior to MDL-2 and frozen pretraining;
6. at least two of three graph seeds deploy a nonzero residual.

If the graph and anchor gates pass but online-attribution gates fail, retain
the graph-policy result and report online DDPG attribution as not established.
Do not increase the same budget blindly. Diagnose critic calibration and
scenario-specific increments first.

If the complete pilot passes, run 300-500 episodes across five seeds followed
by 500 paired replications and the full heuristic/robust-heuristic benchmark.
The manuscript is not updated until this pilot produces a verified archive.
