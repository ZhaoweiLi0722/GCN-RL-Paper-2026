# Archived AFR-GCN-DDPG Method Record

Recorded 2026-10-11 for the local EAAI manuscript milestone. By the latest
user-authorized editorial decision, the proposed method is now native Value-TD;
DDPG is retained as historical, secondary evidence. The target is EAAI, not
AAAI. This changes manuscript positioning, not historical protocols or results.
Earlier instructions in the sources to retain DDPG as primary describe their
then-current decision, not the current manuscript assignment.

This is a source-based archival summary, not a new analysis, model evaluation,
or certification of submission readiness. All numbers below are transcribed
from the saved records; no historical source is modified.

## Original Mechanism

AFR-GCN-DDPG was an advantage-filtered residual controller around MDL-2, not
a from-scratch unconstrained policy. Separate graph encoders supported a
deterministic actor and action-value critic. The actor's shared edge heads
proposed gated, scaled corrections to normalized MDL-2 requests. Clipping was
followed by the simulator's identity-aware decoder and feasibility rules;
the critic evaluated quantized requests, not final donor-receiver events.
Multiple requests could therefore produce identical executed actions. [H1]

Common-random-number look-ahead comparisons against MDL-2 supplied
advantage-filtered teacher targets for supervised initialization, followed by
offline replay updates and critic calibration. The subsequent DDPG phase used
OU exploration, replay, bootstrapped critic targets, delayed actor updates and
soft target-network updates. Teacher, frozen-reference and self-imitation
regularization remained in the actor objective. Actor optimization used soft
detached gates and straight-through specimen quantization, not differentiation
through discrete patient matching. Online replay used up-to-four-step,
anchor-relative scaled rewards; evaluation retained the original accumulated
cost and patient outcomes. Formal evaluation fixed the final episode-100 and
frozen-pretraining checkpoints, without holdout checkpoint selection or
automatic anchor fallback. Here "online" means interaction during simulation
training, not learning during clinical deployment. [H1]

## Restricted Controller, Not Restricted Physics

The formal routing-primary learner had specimen residual scale 0.10 and zero
reagent-transfer, capacity-transfer and replenishment residual scales. Those
resource decisions still came from MDL-2; their physical channels were not
removed. The original twenty-site problem permits reagent transshipment,
idle-bioreactor relocation, one intact patient-bound specimen move before
manufacturing, and return of finished product to that same patient. It forbids
splitting, pooling, substitution, cross-patient reassignment and in-process
transfer. Product return is a lifecycle transition, not an actor decision. [H1, H2]

Thus these results are not evidence of learning all transfer decisions. The
current full-task method must retain specimen, reagent and idle-bioreactor
transfer choices under the original physical rules; a historical zero residual
scale must not silently become a problem constraint. [R]

## Positive Evidence Retained

The formal routing-primary protocol used five training seeds (10-14), four
scenarios and 100 paired replications per seed/scenario: 2,000 paired outcomes
per comparison, not 2,000 independent policy-training runs. Cost differences
below are GCN minus comparator, in millions of modeled weighted-objective
units; negative values favor GCN. [E, H3]

| Comparator | Cost difference | Relative difference | Reported two-level paired 95% interval | Mean simulated losses difference |
| --- | ---: | ---: | ---: | ---: |
| Routing MDL-2 | -17.690 | -0.658% | [-19.361, -15.725] | -25.263 |
| AFR-Flat-DDPG | -8.604 | -0.321% | [-11.065, -6.428] | -16.375 |

Completion and manufacturing-ineligibility metrics were also favorable; all
20 GCN seed-by-scenario mean costs were below MDL-2. Most cost improvement came
from modeled patient-loss cost, not cheaper transportation. Specimen-transfer
cost increased and was already included in base cost, so adding it again would
double count it. These are synthetic objective units, not dollars or clinical
benefits established in patients. [E]

The architecture result supports the complete graph-aware package over the
implemented parameter-matched flat package. Derived features, shared edge
heads and proposal-conditioned gates were not held identical, so it does not
isolate message passing. Routing was admissible for all compared policies;
the comparison does not estimate the value of enabling routing itself. [H1, H3, R]

## Incremental Learning Not Established

Formal GCN final minus frozen pretraining was +72,903 objective units
(+0.00273%), not a demonstrated additional benefit. The historical manuscript's
separate post-hoc crossed seed/world sensitivity reports +0.073 million with
95% interval [-0.180, +0.313] million. This interval supplements, rather than
replaces, the declared primary analysis; it does not establish equivalence.
The retained evidence points to advantage-filtered pretraining/distillation as
the main source of observed value, not an independently established endpoint
gain from the subsequent actor-critic updates. [E, H3]

Matched TD3 development evidence likewise favored its graph controller over
MDL-2 and flat TD3, while final versus frozen did not establish an increment.
The structured-exploration and later F1 paired-critic screens did not pass their
advancement gates. These are bounded findings, not proof that DDPG, TD3 or all
actor-critic learning cannot work. [E, H3]

The archived implementation also mixed absolute-cost offline and
anchor-relative online replay targets, joined some nonadjacent cached teacher
transitions, used inconsistent multi-step calibration discounts, and included
teacher actions outside the specimen-only actor support. These documented
limitations qualify the null result; they are not demonstrated causes of it,
nor evidence that correcting them would produce a positive result. [H4]

## Transport Sensitivity

Frozen-policy specimen availability lead 0 reversed the advantage: GCN cost
was +0.633% versus MDL-2 and +0.255% versus flat. A one-epoch finished-product
return delay instead gave -1.050% and -0.303%, respectively. Epoch-level
availability and continuous geographic travel time are distinct assumptions.
The record therefore does not support general transport-timing robustness. [E, H3]

## Why Secondary Now

The editorial move to native Value-TD addresses the unresolved learning
contribution on the original task; it does not erase the positive historical
controller evidence or declare a new empirical winner. The roadmap separates
same-task learning attribution (T1), final-method graph attribution (T2), and
engineering/manuscript readiness (T3). Four-site support-hour Value-TD findings
motivate a learning mechanism, but neither rank it against twenty-site DDPG nor
complete the original-task evidence. Historical DDPG graph-package gains cannot
be combined with another task's TD gains to prove both contributions for the
new method. No matched native Value-TD-versus-DDPG superiority result is
established by the sources summarized here. [R]

## Chinese Meeting Explanation

旧 DDPG 不是“没有效果”：完整图控制器在历史受限动作实验中优于 MDL-2 和 flat
控制器，但没有证明后续 actor-critic 更新带来了额外收益。旧实验主要学习标本
残差，试剂和生物反应器转移仍由 MDL-2 决策，并非物理系统禁止这些转移。现在按
授权将原二十站点任务的 native Value-TD 作为 EAAI 主线，DDPG 保留为历史证据；
这不是宣称 TD 已击败 DDPG，也不能把不同任务的图优势与学习优势拼成一个结论。

## Source Locators

Locators refer to files read at worktree HEAD
`40c648c476127d7a350aca3879910d961c782435`. Section names and LaTeX labels are
included so the references remain usable if line numbers later move.

- **H1:** [Historical manuscript](../historical-research-record.tex), lines 517-730: graph implementation; `Executed graph--flat interface`; `DDPG Component`; `Advantage-filtered residual initialization and deployment`; `Reward function`; `Action decoding and feasibility projection`. Exact labels: `eq:routing-critic-target`, `eq:routing-actor-loss`, `alg:ddpg_gcn`.
- **H2:** [Historical physical problem](../historical-research-record.tex), lines 192-199 and 271-276, `Autologous identity constraint and admissible transfers` (`sec:problem-identity`).
- **H3:** [Historical results and attribution](../historical-research-record.tex), lines 1500-1694, `sec:routing_primary_confirmation`, `tab:routing_primary_pairs`, `tab:routing_crossed_cost`, and `Prospective online-DDPG attribution and action-geometry audits`.
- **H4:** [Historical training limitations](../historical-research-record.tex), lines 2653-2670, source-and-cache audit paragraph in `sec:conclusion`.
- **E:** [Stage E evidence synthesis](../../../docs/patient_indexed_specimen_routing_stage_e_evidence_synthesis.md), lines 20-119: `Formal routing-primary result`, `Cost interpretation`, `Online attribution`, `Transport timing`, `TD3 development ablation`, `Claim boundary`. Its reproduction section, lines 121-139, identifies the frozen evidence map and generated artifacts.
- **R:** [Publication gap roadmap](../../../docs/publication_gap_roadmap.md), lines 8-29 and 54-126: `Current Decision`, `Reusable Twenty-Site Evidence`, `Shortest Remaining Path`. Historical status/authorization statements are not live experiment or manuscript authority; this record follows the user's newer, bounded editorial assignment.
