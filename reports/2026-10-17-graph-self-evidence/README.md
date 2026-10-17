# Later Graph/Self Evidence: Matched Message-Passing Ablation

Results-only report, 2026-10-17. DEVELOPMENT, synthetic-only, three trained blocks.
This report consolidates completed records; it introduces no new experiment,
model selection, bootstrap calculation or pooled estimate. It concerns the later
Localv2 learned-correction models, not the original Native Value-TD A experiment
reported in PR #13. The original A results remain a separate historical record.

## Main Finding

The completed 72-world panel supports a modest benefit from neighbor message
passing within the tested planning-assisted learned-correction controller:
Graph reduced cost by **1.0846%** and mean simulated patient losses by **25.75**
relative to matched Self. The later 24-world panel reused the **same three
Graph/Self model pairs**, producing **1.0530%** lower cost and **22.33** fewer
mean simulated patient losses. These are two evaluation panels of the same
trained models, not six independent training blocks. The latter panel's
paired-world patient interval includes zero. [E1-E4, M3]

Neither panel establishes a 3% Graph-over-Self improvement. The new joint-proposal
PPO pilot did not improve on old Graph and did not pass its fixed promotion rule.
Its negative result is retained below, alongside the inconclusive Graph/Flat
comparison. [E2, E4]

## Why Self Is The Primary GCN Control

The attribution question is: **does explicit neural neighbor aggregation add
value when the rest of this planning-assisted controller is held fixed?**
The matched Self ablation addresses that question more directly than changing
the entire network to a flattened MLP. This distinction was specified in the
Graph/Self protocol before its evaluation, not introduced after observing these
results. [M1; E2: `primary_pairs`, `secondary_pairs`]

| Property | Graph | Matched Self |
| --- | --- | --- |
| Neural message adjacency | Graph adjacency | Identity matrix |
| Backbone and readout | 64/64 backbone, raw projection, flattened global readout | Same |
| Trainable parameters | 10,881 | 10,881 |
| Inputs | 21 x 41 public-state, candidate and H8 features | Same |
| Planning and physical network | Shared H8, same candidates and physical transfer rules | Same |
| Training recipe | 96 worlds; three rounds of 1,024 updates; batch 64; Adam 1e-4 | Same |
| Initialization | Original Graph initial trainable tensors and private batch RNG | Matched to Graph within each block |

Self removes cross-node neural messages; it does **not** remove the physical
network, shared H8 planning, public inputs or global readout. It is therefore
neither topology-blind nor decentralized, and is not a controller without all
network information. After training starts, the policies and their on-policy
data can diverge. The experiment matches the recipe and initial conditions,
not every subsequently collected training row. [M1; E2: `initial_matches`]

Flat answers a different question: how does this graph architecture compare
with a globally informed non-message-passing MLP under the shared planner?
Its architecture and parameterization differ from Self. Flat remains a
prespecified auxiliary comparator, with all results retained. A positive
Graph/Self result does not establish superiority over Flat or all non-GCN
methods. The main narrative can focus on message-passing attribution without
discarding this stronger alternative-architecture comparison. [M1, E2]

The old Graph/Self controllers must not be renamed PPO. The separately trained
PPO proposers below are a different intervention with different training and
action interfaces. [M1, M2]

## Panel 1: Primary 72-World Descriptive Comparison

Three existing training blocks, four conditions and six fresh evaluation
replicates per block-condition supplied 72 paired worlds. Five roles completed
360 zero-update, 52-step trajectories. Three Self models completed 288 training
trajectories in total, with 9,216 optimizer updates; six sealed Graph/Flat models
were inherited unchanged. The completed recovery record, rather than an earlier
interrupted run, is the source of these totals. [E1, E2]

Positive cost saving favors the candidate. Patient difference is candidate minus
reference, in **simulated patient losses per complete world**; negative favors
the candidate. Costs use the original simulated accounting units, not observed
clinical expenditure. Percentages below are the saved `saving_fraction` multiplied
by 100 and rounded for display. All intervals are saved descriptive 95% intervals,
not multiplicity-adjusted tests or clinical safety bounds. [E2]

| Graph versus | Cost saving, % [95% interval] | Patient difference [95% interval] | Cost-harmed worlds | Patient-harmed worlds | Saved joint screen |
| --- | --- | --- | --- | --- | --- |
| Self | 1.0846 [0.2686, 2.1775] | -25.750 [-52.251, -4.999] | 21/72 | 24/72 | Passed |
| Structured H8 | 2.7536 [1.8615, 3.5874] | -21.639 [-43.167, -1.068] | 13/72 | 29/72 | Failed |
| MDL2 | 4.5567 [3.6872, 5.5454] | -87.806 [-118.891, -60.663] | 4/72 | 13/72 | Passed |
| Flat | 0.3264 [-0.5139, 1.6201] | +2.528 [-23.263, 20.501] | 43/72 | 46/72 | Failed |

The three primary questions remain distinct: Graph/Self concerns message passing;
Graph/H8 concerns the learned correction beyond the common planner; Graph/MDL2
concerns overall operational performance. Graph/Flat is the auxiliary practical
architecture comparison. The four percentages have different reference
denominators and must not be added or subtracted as an exact contribution
decomposition. [E2]

### Heterogeneity And Adverse Outcomes

- Graph/Self block cost savings were 0.2846%, 0.8078% and 2.1842%; corresponding
  patient differences were -16.708, -7.333 and -53.208. All four condition means
  favored Graph for both outcomes, but the condition-0 and condition-2 cost
  intervals included zero. Twenty-four individual worlds had increased patient
  losses despite the favorable overall mean. [E2: `local_graph/local_self`]
- Graph/H8 passed its cost screen but failed its joint patient screen:
  condition 2 had **+3.333** additional patient losses per world. Lower average
  cost and fewer average losses do not erase that condition-specific harm.
  [E2: `local_graph/structured_h8`, `claims.learning`]
- Graph/Flat's cost interval included zero; Graph cost more in two of the three
  blocks and had a positive overall patient-loss difference. This is not
  evidence of a reliable architecture advantage. [E2: `local_graph/local_flat`]

The saved screen outcomes are descriptive protocol decisions, not corrected
statistical significance or clinical noninferiority. All ten pairwise records,
including the six contrasts not displayed in the table, remain in `evidence.json`.

## Panel 2: 24 Fresh Worlds, Same Old Graph/Self Models

The later joint-proposal pilot evaluated six roles on 24 fresh paired worlds:
three blocks x four conditions x two replicates, totaling 144 zero-update,
52-step trajectories. Its old Graph and old Self arms used the same six sealed
models as Panel 1. The recorded checkpoint hashes and byte counts agree across
the two source records; this report did not reopen checkpoints. [E3, E4, M3]

| Old Graph versus old Self | Saved estimate |
| --- | --- |
| Cost saving | 1.0530% |
| Paired-world cost interval | [0.2768%, 1.9532%] |
| Training-block cost interval | [0.2060%, 1.8084%] |
| Mean patient difference | -22.333 per world |
| Paired-world patient interval | [-50.676, +5.084] |
| Training-block patient interval | [-50.250, -2.625] |
| Harmed worlds | Cost: 8/24; patients: 9/24 |

Both saved interval types are reported because they summarize different
resampling units. With only three trained blocks, neither should be treated as
well-resolved independent-training uncertainty. The patient paired-world interval
includes zero; selecting only the more favorable block interval would obscure
this uncertainty. [E4: `old_graph/old_self`]

Block cost savings were 0.2060%, 1.8084% and 1.1579%, with patient differences
-2.625, -50.250 and -14.125. Condition 1 nevertheless had **+13.833** additional
patient losses per world. The other three condition means were favorable. Thus,
this panel supports a similar average cost pattern, not a new blanket
cost-and-patient success across conditions. [E4]

Panel 2 did not include the original `structured_h8`, `mdl2` or `local_flat`
roles. Its `null_proposal_h8` adds a stochastic untrained proposal and is not the
original H8 arm; its `global_proposal` is not the old Flat model. Consequently,
the other three old-Graph contrasts must not be invented or borrowed from Panel 1
as though they came from these same 24 worlds. The two panels are not pooled,
and fresh evaluation worlds do not create additional independent trained models.

## Negative Follow-Up: Joint-Proposal PPO

The pilot trained nine new policy/critic endpoints: three blocks each of Graph,
Local and global-MLP proposers, with 24 training worlds per model, totaling 216
training trajectories. Three zero-update proposal controls and the six old models
were also bound before the 144-trajectory evaluation. Evaluation updates were
zero. [E3, E4]

The following table uses **paired-world** 95% descriptive intervals throughout;
the separately saved training-block intervals, all conditions, blocks and world
records remain in the companion evidence. [E4]

| New Graph proposer versus | Cost saving, % [95% interval] | Patient difference [95% interval] | Cost/patient harmed worlds |
| --- | --- | --- | --- |
| New Local proposer | 0.3245 [-0.2990, 0.8421] | -5.250 [-21.543, 15.501] | 5/24; 5/24 |
| New global-MLP proposer | 0.1229 [-0.3737, 0.6622] | -10.042 [-30.836, 9.793] | 10/24; 9/24 |
| Zero-update proposal + H8 | 0.0730 [-0.1814, 0.3151] | -3.625 [-14.000, 5.333] | 9/24; 8/24 |
| Old Graph | -2.9352 [-4.5468, -1.5944] | +17.250 [-18.167, 57.051] | 21/24; 12/24 |
| Old Self | -1.8513 [-3.3939, -0.5272] | -5.083 [-44.209, 37.256] | 17/24; 10/24 |

The new Graph controller cost **2.9352% more** than old Graph and added **17.25**
mean patient losses per world. It failed the fixed promotion rule, and the saved
3% Graph/Local flag was false. This report does not promote it. [E4]

The Graph/Local proposer comparison changes access to neighbor messages within
matched local-head actors, conditional on centralized training and common H8;
it is not equal-information architecture superiority. Both complete controllers
still use H8. New versus old also changes the learning algorithm, training budget
(24 versus 96 worlds per model) and maximum candidate count (15 versus 14).
This pilot cannot isolate a general PPO-versus-old-method effect. [M2, E3, E4]

The saved Graph-proposal selection metric averages **1.4167 of 52 decisions per
world**. Sparse use may limit how much the new proposer affects outcomes, but
that observation alone does not establish the cause of its performance.
No new causal analysis was performed here. [E4: `table`,
`metrics_mean["proposal_selection.selected"]`]

## Evidence And Availability

[evidence.json](evidence.json) preserves all **10 Panel-1 and 15 Panel-2 pair
records**, including their saved intervals, condition, block, block-condition and
individual-world records, plus full role summary tables. It also includes model
identity bindings, saved screen decisions and a source manifest with full SHA256
digests and byte sizes. Large integer seed identities are retained exactly.
No missing interval is imputed, and no comparison or bootstrap was rerun.

The paths below are **repository-relative local historical sources**. Their
presence in the author's workspace does not establish that they, their raw
trajectories or their model files are included in a published PR or accessible
to a collaborator. The report and companion JSON are review artifacts, not a
complete reproducibility package. Publication status is managed separately by
the coordinating author; extraction-stage scope fields in the JSON do not
describe later parent publication actions.

| ID | Local historical source | Bytes | SHA256 prefix (full digest in JSON) |
| --- | --- | ---: | --- |
| E1 | `specs/2026-10-15-gcn-self-attribution/parallel-recovery1/terminal-readout.json` | 54,780 | `3233a32478ad` |
| E2 | `results/native_gcn_self_20261015_parallel_recovery1/payload/comparison.json` | 740,966 | `7fcee1e611ca0` |
| E3 | `specs/2026-10-17-gcn-joint-proposal/terminal-readout.json` | 54,403 | `04f7eddae5bb5` |
| E4 | `results/native_gcn_joint_proposal_20261017/payload/comparison.json` | 475,045 | `8653b65a560a8` |
| M1 | `specs/2026-10-15-gcn-self-attribution/protocol.md` | 6,335 | `960644eaab52` |
| M2 | `specs/2026-10-17-gcn-joint-proposal/protocol-draft.md` | 14,565 | `66942dcc6e71` |
| M3 | `specs/2026-10-17-gcn-joint-proposal/input-bindings.json` | 2,888 | `5d0cf02060f8` |

M2 retains its historical pre-launch heading; completed status here comes from
E3/E4, not that heading. Archive status is quoted from the existing terminal
readouts only: no archive/member verification, re-archiving, cloud synchronization
or collaborator-access check was repeated for this report. Likewise, matching
model-binding records are provenance evidence, not a new checkpoint audit.

## Conclusion And Separate Next-Step Plan

The defensible result is a **modest, controller-specific message-passing benefit**
over matched Self, accompanied by world- and condition-level harms and limited
three-block uncertainty. It is not a 3% improvement, universal GCN superiority,
clinical safety claim or guaranteed improvement from additional model changes.
Graph/Flat remains inconclusive, Graph/H8 has a condition-specific patient
trade-off, and the new PPO attempt did not improve the retained old controller.

The separately maintained [next-step plan](next-step.md) explains the proposed
research direction. This evidence consolidation neither starts that work nor
changes historical models, the manuscript or the original A experiment.
