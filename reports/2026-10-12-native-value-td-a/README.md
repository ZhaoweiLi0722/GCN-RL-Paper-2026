# Native Value-TD A: Completed Twenty-Site Comparison

**Status:** scientific execution complete; descriptive synthetic evidence, not a
clinical or deployment validation. Prepared for Howard's review on 2026-10-12.
The manuscript and running B/family experiments are unchanged by this release.

## Bottom Line

The main stage-level result is **positive cost evidence for the current proposed
method**: Graph Value-TD reduces mean simulated cost by 3.4951% versus MDL2,
0.8808% versus the same-planner nonlearning Native-H8, and 4.7124% versus the
matched Flat Value-TD architecture. This is the tested fixed-policy-class recipe,
not a claim that the best possible learned strategy has been found.

Patient losses also improve relative to MDL2 and Flat, but increase relative to
H8. That tradeoff does not erase the measured cost improvement. It does mean the
original predeclared joint learning-benefit screen is not met. The user's current
cost-first emphasis does not retrospectively change that frozen support rule.

This supports the complete proposed controller package over MDL2 and its
graph-versus-global-MLP architecture comparison. It does not establish
unconditional RL superiority, pure edge causality, clinical safety or readiness
for deployment or publication.

## Completed Design

- Original twenty-site network, four original conditions, 52-epoch endpoint;
  reagent transshipment, idle-bioreactor relocation and identity-preserving
  specimen routing remain available under the original physical/cost rules.
- Four roles: original MDL2, nonlearning Native-H8, Graph Value-TD and Flat Value-TD.
- A uses the existing finite 14-option action class, not the later B sequence/CEM
  planner or unrestricted continuous flow optimization. Capacity corrections in
  this class are coupled with reagent corrections.
- H8 and learned-value roles share the same candidate generator, information
  access and eight-step rollout machinery. Learning adds a terminal cost estimate.
  This is value learning in simulation, not actor training or online adaptation
  during evaluation.
- Five independent training blocks, two learned architectures, 96 trajectories
  per model: **960/960 training trajectories and 10/10 sealed final models**.
- Five blocks x four conditions x 12 fresh worlds: **240/240 paired worlds and
  960/960 evaluation trajectories**; zero evaluation optimizer updates.
- All four conditions have equal weight. The existing analysis uses 2,000
  hierarchical bootstrap draws. Intervals below are descriptive and are **not
  multiplicity-adjusted**. The model-seed/block count is five, not 240 independent
  trained models.

The unchanged [original protocol](protocol.md) is preserved as a historical
pre-launch document. Its opening proposed/not-launched label records that earlier
state, not the current run status. [Provenance](provenance.json) records the
completed execution, the later one-time evaluation handoff and final archive.

## Main Contribution Contrasts

Cost saving is mean(reference cost - candidate cost) / mean(reference cost).
Extra patient losses are candidate minus reference, averaged per simulated world;
negative values favor the candidate.

| Graph Compared With | Mean Cost Saving | Descriptive 95% Interval | Mean Extra Patient Losses | Descriptive 95% Interval | Predeclared Support |
| --- | ---: | ---: | ---: | ---: | --- |
| MDL2 | 3.4951% | [2.8844%, 4.2500%] | -71.2750 | [-88.4385, -53.5116] | Operational: passes |
| Native-H8 | 0.8808% | [0.3954%, 1.4277%] | +30.6708 | [+16.6854, +44.3228] | Learning: fails |
| Flat Value-TD | 4.7124% | [3.9188%, 5.5765%] | -62.2208 | [-87.1341, -39.1039] | Architecture: passes |

The frozen support rule requires at least 0.5% mean saving, positive cost means
in all five blocks, a positive descriptive lower cost bound, and nonpositive
mean additional patient losses overall **and in every condition**.

### What The H8 Result Means

It is inaccurate to summarize this as merely "the RL effect is not statistically
significant." The aggregate descriptive cost interval excludes zero, but the
patient-loss interval is positive too. All four condition mean patient-loss
differences favor H8; Graph causes additional simulated losses of 29.8000,
18.1833, 34.4000 and 40.3000 per world in conditions 0, 1, 2 and 3 respectively.
The condition-level cost intervals cross zero for conditions 0 and 3.

The supported statement is narrower: the learned tail offers a modest aggregate
cost reduction but fails the predeclared patient-loss requirement. The ongoing B
sequence-planning and family/learning-target studies test whether other already
authorized recipes produce stronger gains. That is a live research hypothesis,
not a result or a reason to change their frozen evaluation rules. This experiment
does not show that more training or a different algorithm will necessarily improve
cost further or resolve the patient-loss tradeoff. Unfinished packages cannot be
combined with A to claim a completed larger learning contribution.

## All Six Contrasts And Harms

Each harm denominator is 240 paired evaluation worlds. Positive average savings
do not mean every world benefits.

| Candidate / Reference | Mean Cost Saving | Mean Extra Patient Losses | Cost-Harmed Worlds | Patient-Harmed Worlds | Support Rule |
| --- | ---: | ---: | ---: | ---: | --- |
| Graph / MDL2 | 3.4951% | -71.2750 | 17/240 | 38/240 | Pass |
| Graph / H8 | 0.8808% | +30.6708 | 84/240 | 177/240 | Fail |
| Graph / Flat | 4.7124% | -62.2208 | 13/240 | 66/240 | Pass |
| Flat / MDL2 | -1.2776% | -9.0542 | 180/240 | 107/240 | Fail |
| Flat / H8 | -4.0211% | +92.8917 | 218/240 | 206/240 | Fail |
| H8 / MDL2 | 2.6375% | -101.9458 | 36/240 | 11/240 | Pass |

[Machine-readable evidence](evidence.json) preserves all four role tables and all
six pairs, including all block, condition, block-condition and individual-world
savings/loss differences, harm counts, compute totals and interpretation flags.
It is a field-preserving export of the existing completed comparison, not a new
bootstrap, statistical test, trajectory reconciliation or scientific run.

## Execution And Archive

Scientific terminal: 2026-10-12T05:12:33.307241Z, child exit code 0. All 22 claimed
supervisor/coordinator/worker PIDs were confirmed absent by approved host process
inspection. All 99,840 canonical native rows were reconciled by the existing
scientific coordinator before it created the single final archive.

The earlier serial A root was administratively stopped for its explicitly
authorized evaluation-only handoff. Its failed administrative terminal remains
separate from this successful scientific terminal. The continuation imported 42
completed evaluation-role identities and completed the remaining 918, without
replaying training. One abandoned Graph prefix remains charged separately:
33 native steps, 7,592 prediction steps and zero optimizer updates. No refunds.

Unique archive: 1,220,337,199 bytes; 9,706 members fully read and hash-verified.
SHA256: `ad1ed79b8a37b4eb72f5d642aa61e7ee0df1f933c239c557fe756f47e1ace270`.

The existing Dropbox project delivery contains the full archive, comparison,
inventory, terminal receipts and authority records under
`Research Artifacts/delivery_20261011/final/native_value_td_20261010_reexecution1_eval_parallel1/`.
All 29 original final-delivery files (1,260,242,770 bytes) were verified locally.
Cloud synchronization and collaborator access are separate states and were not
verified at publication time. Large raw trajectories/checkpoints and host-specific
authorization/inspection records are not copied into this public GitHub report.
Frozen implementation and runtime evidence remain in the archived payload.

## Review Boundary

Please assess whether the operational and architecture statements are adequately
bounded and whether the H8 patient-loss tradeoff is communicated clearly. This
review does not authorize another run, reward/target tuning, main-branch merge,
manuscript claim change or removal of unfavorable results.

These are synthetic, finite-window results with unresolved patients retained at
cutoff, not lifetime patient outcomes or clinical noninferiority evidence.
Author review is pending; automated source matching is not author approval.
The repository's existing evidence-bound scientific-writing guidance was reused
as documented in its 2026-10-06 publication note; no new literature-based claim is
made here.
