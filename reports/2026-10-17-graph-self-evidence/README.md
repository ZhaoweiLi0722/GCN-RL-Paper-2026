# Later Graph/Self Evidence: Matched Message-Passing Ablation

Results-only report, revised 2026-10-19. DEVELOPMENT, synthetic-only. Panels 1/2
reuse three trained blocks; Panel 3 contains five independently retrained blocks.
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

The subsequently completed five-block replication gives a smaller **0.4071%**
Graph/Self cost saving, with a saved descriptive interval **[-0.0867%, 1.0407%]**,
and **9.26 fewer** mean patient losses, with an interval crossing zero. It fails
the fixed joint screen. The earlier approximately 1% results therefore must not
be presented as a stable effect across independent retraining. [E5, E6]

No panel establishes a 3% Graph-over-Self improvement. The new joint-proposal
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

Flat answers a different, necessary question: how does this graph architecture compare
with a globally informed non-message-passing MLP under the shared planner?
Its architecture and parameterization differ from Self. Flat remains a
prespecified auxiliary comparator, with all results retained. A positive
Graph/Self result does not establish superiority over Flat or all non-GCN
methods. Graph/Self alone cannot distinguish a graph-topology advantage from
the more general benefit of hidden cross-node mixing. Graph/Flat tests this GCN
against the tested global MLP, but also changes architecture and parameterization;
it is not a perfectly isolated topology intervention either. Both contrasts
belong in the main interpretation, even though the original protocol labeled
Flat auxiliary. The tested implementations do not establish that every hidden
cross-node mixer would help. [M1, E2]

The old Graph/Self controllers use native-return cost-correction learning with
Monte Carlo policy evaluation and policy improvement, not the earlier scalar
Value-TD target and not PPO. The separately trained
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

The missing comparisons are important to that interpretation:

| Candidate versus reference | Cost saving, % [95% interval] | Patient difference [95% interval] |
| --- | --- | --- |
| Flat / Self | 0.7607 [0.1435, 1.2831] | -28.278 [-43.792, -14.440] |
| Structured H8 / Self | -1.7162 [-2.5895, -0.8228] | -4.111 [-24.739, 16.950] |

Thus Self reduces cost relative to H8 but has a positive mean patient-loss
difference; both tested cross-node architectures improve cost over Self.
Graph's additional advantage over Flat remains uncertain. For an H8-denominator
presentation, Self/H8 saves **1.6873%**, Flat/H8 **2.4351%**, and Graph/H8
**2.7536%**. The first two differ slightly from simply changing the signs of
H8/Self and H8/Flat: for saved saving `s=(C_ref-C_cand)/C_ref`, the reversed
saving is `-s/(1-s)`. Flat/Self above uses this conversion and reverses the
saved interval endpoints monotonically. No bootstrap was repeated. Original
pair records remain unchanged in E2's export. [E2, E7]

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
  intervals included zero. Condition-1 and condition-2 patient intervals also
  cross zero: [-50.058, +0.611] and [-49.504, +19.444]. Twenty-four individual worlds had increased patient
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
including every contrast not displayed above, remain in `evidence.json`.

Block 2 supplies about 66% of the net Panel-1 Graph/Self cost saving. The block
means are 7.361, 20.748 and 55.223 million simulated cost units saved per world,
respectively. Two of twelve block-condition cells have higher Graph cost:
block 0/condition 0 saves -0.2880% (patient difference -6.667), and block
1/condition 2 saves -0.2769% (patient difference +20.500). The other ten cells
and all individual-world harms remain visible in the saved pair record.
No unfavorable cell is excluded from the aggregate. [E2]

### Transfers And Transit Expiry

| Role | Specimen transfer arrivals per world | Transit-expired patients per world |
| --- | ---: | ---: |
| MDL2 | 201.8194 | 39.2083 |
| Structured H8 | 232.8472 | 41.6806 |
| Self | 391.7500 | 66.2222 |
| Graph | 436.5556 | 75.4722 |
| Flat | 404.7639 | 67.6667 |

These are saved `flows.specimen_transfer_arrivals` and
`flows.patients_lost_transit_expired` means, not a causal decomposition.
Relative to structured H8, the learned controllers have more transfer arrivals
and transit-expired patients. Their aggregate loss improvements do not remove
that adverse subcomponent. PR #13's **Native-H8 is a different controller**:
its corresponding means are approximately 540 arrivals and 109 transit-expired
patients. Do not label both controllers simply H8 in a cross-study comparison
or splice their contributions. [E2, E7; PR #13 saved role table]

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
resampling units. In this panel the training-block interval endpoints equal
the minimum and maximum of the three block estimates exactly, for both cost
and patients. With three blocks, the percentile bootstrap is highly discrete;
these endpoints must not be interpreted as well-resolved uncertainty about new
independent training runs. They are retained to explain the historical output,
not preferred over the world-based interval. The patient paired-world interval
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

## Panel 3: Five New Training Blocks

The old Graph/Self recipe was subsequently frozen and independently retrained:
960 training trajectories, 30,720 updates and ten sealed final models. After an
implementation failure at checkpoint admission, a separately authorized
evaluation-only recovery reused all training and finals, completing 480
zero-update trajectories on 120 new four-role paired worlds. There was no
retraining or replacement of the failed root. The recovery completed and its
unique local archive was verified on 2026-10-18. [E5, E6]

| Candidate / reference | Cost saving, % [95% interval] | Patient difference [95% interval] | Cost/patient harmed worlds | Joint screen |
| --- | --- | --- | --- | --- |
| Graph / Self | 0.4071 [-0.0867, 1.0407] | -9.258 [-28.298, 6.052] | 51/120; 53/120 | Failed |
| Graph / Structured H8 | 2.1364 [1.6290, 2.5988] | -2.775 [-14.809, 9.528] | 27/120; 63/120 | Failed |
| Graph / MDL2 | 4.8450 [4.3977, 5.2684] | -94.567 [-111.744, -79.050] | 6/120; 16/120 | Passed |
| Self / Structured H8 | 1.7364 [0.9278, 2.4165] | +6.483 [-12.451, 30.926] | 36/120; 68/120 | Failed |
| Self / MDL2 | 4.4561 [3.6423, 5.1048] | -85.308 [-108.492, -59.474] | 9/120; 19/120 | Passed |
| Structured H8 / MDL2 | 2.7677 [2.2971, 3.2417] | -91.792 [-107.577, -77.466] | 10/120; 4/120 | Passed |

Graph/Self block savings are **1.6115%, 0.1378%, 0.2890%, -0.0378%, 0.0465%**;
patient differences are **-45.250, +1.458, -8.583, +1.833, +4.250**. Block 0
contributes about 78.5% of net cost saving. Both overall Graph/Self intervals
cross zero; this is a small positive point estimate, not a demonstrated stable
GCN gain. Graph/H8 retains a cost benefit, but conditions 1 and 2 add **4.2**
and **26.033** patient losses/world, respectively. All six full pair records,
including all block-condition-world harms, are in E6's export. [E5, E6]

Flat was not retrained in this panel. Its absence is an unresolved architecture
benchmark gap, not evidence that Self supersedes Flat. The package is closed;
a Flat extension requires its own bounded protocol and authorization. These
120 worlds have now been read and cannot be called an untouched confirmation
panel for any subsequently selected recipe. The three panels are not pooled.

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

### Protocol And Selection Chronology

The exact [M1 protocol](../../specs/2026-10-15-gcn-self-attribution/protocol.md)
is now included in this PR, unchanged: 6,335 bytes, full SHA256 in E7. The
Localv2 recipe was frozen at **2026-10-14 14:57:14 UTC**; M1's separate namespace
and 7,634 fresh allocations were frozen at **2026-10-15 10:08:30 UTC**. The
saved historical collision audit passed against 2,219 files. A read-only check
also finds no fresh-allocation overlap with the five predecessor packets or
Localv2, and all 72 comparison worlds match M1 exactly across recovery. This
supports nonreuse for old-Graph selection within the recorded lineage. It is
not proof about undocumented external work, and M1 remains DEVELOPMENT rather
than an independent confirmatory test. [M1, E7]

The count depends on what constitutes a recipe. Before the Localv2 recipe freeze,
the directly traced **October Native lineage contains five predecessor campaign
designs and 14 configured learned arms**:

| Predecessor campaign | Learned arms counted |
| --- | --- |
| Native Value-TD A | Graph, Flat: 2 |
| Original B CEM sequence | Graph, Flat: 2 |
| Five-family development | DDPG, TD3, SAC, PPO, TD8, lambda: 6 |
| Revised B structured TD | Graph, Flat: 2 |
| Candidate-Advantage v1 | Graph, Flat: 2 |

Each arm is a fixed learner/architecture/target/planner recipe, not a model seed,
recovery attempt or inherited control reevaluation. This is a development lineage,
not a randomized search or a causal comparison of those recipes. Family's original
analysis failed; its separately authorized saved-data analysis preceded Localv2.
The later relation, interaction, neighbor-correction, ranking, retention, precision
and PPO follow-ups occurred **after** Localv2 and do not count as recipes tried
before its selection. Earlier DDPG and other project campaigns also exist;
**14 is not an exhaustive project-wide search count**. That wider selection
history remains a manuscript disclosure gap, not a resolved multiplicity claim.
Packet dates, roles, hashes and the count definition are recorded in E7.

### Exported Evidence

[evidence.json](evidence.json) preserves all **10 Panel-1 and 15 Panel-2 pair
records**, including their saved intervals, condition, block, block-condition and
individual-world records, plus full role summary tables. It also includes model
identity bindings, saved screen decisions and a source manifest with full SHA256
digests and byte sizes. Large integer seed identities are retained exactly.
No missing interval is imputed, and no comparison or bootstrap was rerun.
The reversed display rows use only the explicit algebraic conversion above.
[replication-evidence.json](replication-evidence.json) preserves all six new
pair records and role/condition tables exactly. [review-supplement.json](review-supplement.json)
binds the protocol, chronology and deterministic table conversions. Original
`evidence.json` has not changed.

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

E5 is the completed five-block recovery terminal readout,
`specs/2026-10-17-graph-self-replication/evaluation-recovery1/terminal-readout.json`.
E6 is its existing `results/native_graph_self_replication_20261017_eval_recovery1/payload/comparison.json`.
Their full hashes and byte sizes are included with the E6 export. E7 is the
local `review-supplement.json` and its source manifest. The export helper
`export_revision.py --check` verifies exact local source equality, without
loading models or importing the research runtime. The original source datasets
are still needed to rerun that helper; publishing it is not a full source/data release.

M2 retains its historical pre-launch heading; completed status here comes from
E3/E4, not that heading. Archive status is quoted from the existing terminal
readouts only: no archive/member verification, re-archiving, cloud synchronization
or collaborator-access check was repeated for this report. Likewise, matching
model-binding records are provenance evidence, not a new checkpoint audit.

## Conclusion And Separate Next-Step Plan

The historical panels show a modest controller-specific message-passing benefit;
the five-block replication shows a **smaller, uncertain Graph/Self advantage**.
Both statements belong in the result, with all world- and condition-level harms.
It is not a 3% improvement, universal GCN superiority,
clinical safety claim or guaranteed improvement from additional model changes.
Graph/Flat remains inconclusive, Graph/H8 has a condition-specific patient
trade-off, and the new PPO attempt did not improve the retained old controller.

The separately maintained [next-step plan](next-step.md) explains the proposed
research direction. This evidence consolidation neither starts that work nor
changes historical models, the manuscript or the original A experiment.
