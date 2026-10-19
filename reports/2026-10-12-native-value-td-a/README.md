# Native Value-TD A: Completed Twenty-Site Comparison

**Status:** scientific execution complete; descriptive synthetic evidence, not a
clinical or deployment validation. Prepared on 2026-10-12; interpretation revised
on 2026-10-17 in response to
[Howard's review](https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026/pull/13#pullrequestreview-5558638284).
Only this report's wording changes. Scientific evidence, the historical protocol,
manuscript and all experiment artifacts remain unchanged. References in the
initial release to ongoing B/family work were a 2026-10-12 snapshot, not current status.

## Bottom Line

The main stage-level result is **positive cost evidence for the current proposed
method**: Graph Value-TD reduces mean simulated cost by 3.4951% versus MDL2,
0.8808% versus the same-planner nonlearning Native-H8, and 4.7124% versus the
approximately parameter-matched Flat Value-TD architecture. This is the tested fixed-policy-class recipe,
not a claim that the best possible learned strategy has been found.

Patient losses also improve relative to MDL2 and Flat, but increase relative to
H8. That tradeoff does not erase the measured cost improvement. It does mean the
original predeclared joint learning-benefit screen is not met. The user's current
cost-first emphasis does not retrospectively change that frozen support rule.

The frozen operational and architecture screens pass, but the architecture gap
also includes Flat's deterioration relative to nonlearning H8. It is evidence
about these two implementations under this training recipe, not a pure
message-passing ablation. It does not establish
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

The saved mean cost components explain the accounting tradeoff:

| Component | Graph Minus H8, Million Simulated Cost Units Per World |
| --- | ---: |
| Bioreactor shortage | -40.9481 |
| Patient loss | +15.3354 |
| Expiry | +3.0892 |
| All remaining components, net | -0.4185 |
| Total | -22.9420 |

These are differences of the existing `table[*].metrics_mean` fields in
[evidence.json](evidence.json), not a new rollout or statistical analysis.
They are simulated objective units, not actual medical expenditure. Shortage
penalty savings outweigh increased patient-loss and expiry costs. This supports
an improvement in the fixed scalar cost objective with worse patient outcomes,
not an unqualified better policy. Accounting alone does not identify the full
causal chain from increased patient loss to reduced shortage penalties.

The learned tail therefore fails the predeclared joint learning-benefit rule.
More training or a different algorithm is not established to fix that tradeoff.
Later methods and evaluations are separate studies; their gains cannot be
spliced into A to claim a larger learning contribution.

### What The Architecture Result Means

Flat costs 4.0211% more than its behaviorally equivalent zero-tail H8 reference
and 1.2776% more than MDL2. Graph's 4.7124% saving versus Flat must not be presented
as all-new learning beyond H8: Graph's measured H8-relative saving is 0.8808%.
The percentages use different reference denominators and are not an additive
decomposition. Keep `claims.architecture.passed=true` as the frozen screen's
record while narrowing its interpretation to this particular recipe.

Graph and Flat have 7,873 and 7,859 parameters, respectively (about 0.18%
difference). FLOPs, weight sharing and optimization difficulty are not matched.
Graph avoids the observed Flat cost degradation, but not H8-relative patient
degradation. Neither universal GCN superiority nor a pure edge effect follows.
Later Graph/Self identity-adjacency comparisons address a different, more tightly
matched question; they do not replace or delete this historical Flat evidence.

### Conditions And Training-Seed Variation

Graph/H8 cost-saving point estimates are 1.1412%, 1.2374%, 1.3338% and 0.1763%
in conditions 0 through 3. All are positive, but only conditions 1 and 2 have
descriptive intervals wholly above zero. Five block-level mean savings range
from about 5.24 to 42.47 million simulated cost units per world. This is substantial
variation across only five trained model pairs, not 240 independent model seeds.
Retain all block/condition/world results and the unadjusted descriptive intervals.

### Transit Expiry Is A Separate Outcome

| Role | Mean Patients Lost To Transit Expiry Per World |
| --- | ---: |
| MDL2 | 36.3250 |
| Native-H8 | 109.3875 |
| Graph Value-TD | 121.6208 |
| Flat Value-TD | 103.6042 |

Source: `table[*].metrics_mean["flows.patients_lost_transit_expired"]`.
All three rollout roles have roughly three times MDL2's transit-expiry losses.
This pattern is already present in nonlearning H8; it is not entirely attributable
to learned values. Increased routing exposure is a plausible explanation, not a
separately identified causal effect. Aggregate patient improvements versus MDL2
must not hide this adverse component.

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

Howard could not inspect implementation `c39c704` or execution `6146d71` on a
remote branch. This results-only revision does not resolve that source-access
gap: commit identifiers and archive hashes do not establish accessible source,
collaborator access, or independent reproducibility. No new cloud verification
is claimed.

## Review Boundary

Please assess whether the operational and architecture statements are adequately
bounded and whether the H8 patient-loss tradeoff is communicated clearly. This
review does not authorize another run, reward/target tuning, main-branch merge,
manuscript claim change or removal of unfavorable results.

These are synthetic, finite-window results with unresolved patients retained at
cutoff, not lifetime patient outcomes or clinical noninferiority evidence.
Howard accepted the initial numbers as a results record and requested narrower
interpretations. That review is not manuscript approval, merge authority or
verification of the revised wording. Automated source matching is not author approval.
The repository's existing evidence-bound scientific-writing guidance was reused
as documented in its 2026-10-06 publication note; no new literature-based claim is
made here.
