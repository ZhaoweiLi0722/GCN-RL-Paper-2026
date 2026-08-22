# DDPG Structured-Exploration Screen: Terminal Evidence

Status: complete and closed.

This directory preserves the compact, publication-reviewable evidence for the
Stage C3 support-matched structured specimen exploration screen. The full
development output remains immutable under
`results/patient_indexed_specimen_routing_ddpg_structured_exploration_development/`
on the execution host and is covered by the committed 828-entry artifact
inventory.

## Locked result

The structured-exploration mechanism worked as an intervention but did not
improve the deterministic DDPG policy:

- all 12 training runs and 1,200 online episodes completed on Apple MPS;
- each candidate run observed all five prespecified options;
- candidate selection rates were approximately 19%, with behaviorally distinct
  specimen actions in approximately 17% of decisions;
- GCN candidate final minus frozen-pretrain cost was `+84,712` (`+0.00416%`),
  with paired 95% CI `[-563,653, +833,767]`;
- two of three GCN seeds were favorable and clinical noninferiority passed, but
  the preregistered effect gate failed;
- GCN candidate minus unchanged control at episode 100 was `-18,204`
  (`-0.00089%`), with paired 95% CI `[-278,652, +271,812]`;
- the locked classification is
  `close_structured_exploration_ddpg_candidate`; no formal confirmation was
  launched.

The independently validated headroom audit retained 58 of 156 states with a
clinically noninferior lower-cost option, including 51 specimen corrections.
This establishes local single-decision headroom, not cumulative episode-level
performance. Those state-level advantages must not be summed or presented as
publication performance evidence.

## Publication decision

Stage C3 closes the bounded DDPG mechanism search. Together with the failed
support-alignment and persistent-shift candidates, it provides no basis for
additional hyperparameter tuning, selective seed extension, or another online
mechanism screen. The paper retains the completed five-seed 100-episode DDPG
protocol as the primary method and narrows the empirical claim to graph-aware
advantage-filtered residual control. Online final-versus-frozen attribution is
reported as not established.

## Evidence files

- `comparison/structured_exploration_summary.json`: paired checkpoint-curve,
  clinical, candidate-versus-control, and advancement-gate results.
- `headroom_audit/summary.json`: reproducible frozen-policy route-headroom
  diagnostic with independent discovery and validation streams.
- `control/evaluation/checkpoint_curve_summary.json`: unchanged control curve.
- `candidate/evaluation/checkpoint_curve_summary.json`: structured-exploration
  candidate curve.
- `launcher/claim.json`: execution environment, commit, and locked asset hashes.
- `launcher/status.json`: completed phase, training, evaluation, and decision
  audits.
- `launcher/artifact_sha256.json`: SHA256 inventory for all 828 full-result
  artifacts.

## Provenance

- Branch: `rl-attribution-refinement`
- Execution commit: `ac313052b702f4956ed725daf400b81de1f17e91`
- Development training seeds: 50, 51, 52
- Execution-only validation seed: 95700000
- Development evaluation seed: 95800000
- Bootstrap seed: 95850000
- Formal holdout seed 91100000: not used
- Comparison SHA256:
  `30d1a39785a8e4bc9f17d68e6cf615f4e29ab5b2a64284e1105601bc87fecda4`
- Status SHA256:
  `b42cd665a0d8a8eb012a2cc26e1d513fc1d0700141c564663324e76156bd4eae`
- Claim SHA256:
  `85a3d32f1e4bc29a2300207d038859bb77c87e804cad1bd906cd6ed783898e3a`
- Artifact inventory SHA256:
  `f07f69c82ce37bdc5f687389f80c5860c2c9df4483db190cff9bded499001834`
