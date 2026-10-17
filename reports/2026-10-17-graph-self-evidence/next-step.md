# Next GCN Step: Establish A Repeatable Mechanism Before Another Modification

2026-10-17. Research recommendation, not a frozen protocol or an execution
authorization. No new training, evaluation, forward call, label, seed or archive
is produced by this document. Original Graph/Self and all negative results remain
unchanged. The current monitor stays paused.

## Decision And Its Basis

Publish the completed evidence now; do not wait for a 3% gap. Keep the old
planning-assisted learned cost-correction recipe as the reference method. The
next recommended numerical investment is an independently trained Graph/Self
replication, with decision and cost-component reporting in the same package.
This tests repeatability; it is NOT itself an improvement to the model.

The old Graph/Self gap is about 1.08% on the original 72-world panel and 1.05%
on the later 24-world panel. Both panels reuse the same three trained blocks.
The later panel's patient-loss interval crosses zero and some worlds and one
condition have increased patient losses. These facts support a specific,
limited positive lead, not a guaranteed 3% gain or universal safety. Exact
values, denominators, harms and intervals are in [README.md](README.md) and
[evidence.json](evidence.json).

Finite training-run uncertainty matters in RL evaluation; additional worlds
for the same trained models do not create additional independent training
replicates. This is consistent with [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
That methodological reference does not validate this project's effect size.

## Why More Layers Are Not The Current Recommendation

Graph structure supplies a relational inductive bias, not a theorem that its
realized cost must beat every non-graph model. [Battaglia et al. (2018)](https://arxiv.org/abs/1806.01261)
motivates structured relational computations and generalization, not a fixed
minimum percentage gain in this simulator.

Here, both matched arms already have public inputs, candidate features, a global
readout and H8 planning scores. With scores written as
`S_theta(s,a) = Q_H8(s,a) + R_theta(s,a)`, neighbor messages improve decisions
only if their effect on relative candidate scores changes a choice beneficially,
or helps maintain good choices under new training/world randomness. A better
representation or regression metric alone need not improve the executed policy.
The exact contribution being tested is learned message passing on top of this
shared planner, not access to all network information.

Existing attempts restrict the plausible next investments:

| Attempt | Evidence-based reason not to extend it now |
| --- | --- |
| Typed relations, state-action interactions and neighbor correction | Did not establish stable improvements; some worsened Graph while Self improved. Extra capacity is not a demonstrated remedy. |
| D1 joint endpoint/action messages | Failed to improve over its old coverage Graph in every block of the fixed short screen. |
| D2 root-wise soft-regret and earlier ranking/retention | Mixed blocks and patient harms; a sampled decision-loss gain did not establish a better whole policy. |
| D3 two-to-four-future labels | Label disagreement exists, but the fixed development screen failed in all three blocks; no fresh episode panel was admitted. |
| D4 scheduled-arrival features | Arrival reconstruction succeeded, but the hypothesized error concentration did not repeat across blocks/views. No D4 model was trained; efficacy remains untested, not disproved. |
| New joint-proposal PPO | Promotion failed and cost exceeded old Graph by about 2.94% on that panel. Sparse proposal use is a clue, not proof that forced selection or more PPO updates would help. |

The preceding facts come from existing records, not newly pooled statistics.
Local source locators (not all included in this results-only remote publication):
`specs/2026-10-17-gcn-direction-series/prior-attempts.md`,
`specs/2026-10-17-gcn-direction-series/terminal-readout.md`,
`reports/2026-10-17-d3-follow-up/readout.md`,
`reports/2026-10-17-d4-existing-data/readout.md`, and
`specs/2026-10-17-gcn-joint-proposal/terminal-readout.json`.

## Concrete Replication Design To Budget Next

1. Freeze the old architecture, features, candidate set, H8, target, optimizer,
   update count and 96-world training recipe. Train matched Graph/Self pairs
   from new paired initializations, not by continuing or replacing old finals.
   A proposed design is five new training blocks: 960 training trajectories,
   ten new models. This count is a design proposal, not a completed power analysis.
2. After all models seal, evaluate four roles (Graph, Self, H8, MDL2) on five
   blocks x four conditions x six fresh worlds: 120 paired worlds / 480 zero-update
   trajectories. No checkpoint selection, old-test tuning or optional stopping.
   Historical Flat stays published as auxiliary evidence; omitting another Flat
   retraining is a prospective question/cost choice, not deletion of its results.
3. Prespecify Graph/Self as the message-passing contrast. Graph/H8 and Graph/MDL2
   answer learning and operational questions separately. Keep all six pairwise
   contrasts, cost components, transit expiry, patient losses, unresolved patients
   and block/condition/world harms. Do not combine this panel with the existing
   panels and call all worlds independent training samples.
4. Use normal execution logs to describe transfer/replenishment requests,
   shortage costs and patient outcomes. The two policies can visit different
   states, so cross-trajectory action differences do not isolate a causal
   mechanism. Same-state counterfactual scoring would require separately counted
   model calls within an explicitly approved package; it is not free monitoring.
5. Retain a prospective joint screen such as the existing 0.5% mean cost threshold,
   positive block means and descriptive lower bound, with no mean patient-loss
   increase overall or in any condition. Fix its exact definition before execution.
   The 3% aspiration is reported separately, never used to weaken Self or move a
   threshold after outcomes. No confirmatory or multiplicity-adjusted claim is
   implied by this proposed descriptive screen.

Before approval this proposal still needs exact operation/seed/phase caps,
resource and wall-time accounting, isolation checks and a realistic estimate
from the SAME old training workload. The 59-minute PPO run is not a valid timing
proxy for the old branch-label training pipeline. Do not launch from the counts
above or treat available workers as permission to enlarge the budget.

## The Improvement Question After Replication

The most defensible performance hypothesis remains whether the old GCN can
better estimate the long-term opportunity cost of sharing scarce resources:
benefit at a receiving site versus delayed shortages and patient losses at a
donor. This is a hypothesis about decisions, not evidence that another neighbor
layer, arrival feature or larger training budget will help.

Proceed to ONE targeted change only if the replication exposes a repeatable,
cross-training-block error pattern tied to actual cost and patient harm. For
example, a stable misranking involving donor resource shortages would motivate
a specifically bounded calibration of that existing residual, not a new actor
or a change in reward weights. Such a change must first be distinguished from
the failed ranking, retention and precision recipes above. If no distinctive
pattern is supported, retain the old method and end this optimization line.

An improved Graph must beat the old Graph as well as an unchanged old Self
reference. If features, capacity, labels or training budgets change, a new
matched no-message arm is additionally required for an isolated GCN claim;
the weaker or stale comparator alone is insufficient. Report all failed and
harmful cases, not only favorable strata.

## Stop And Release Rules

- If independent replication fails the fixed joint rule, report it and do not
  automatically expand seeds, train longer, tune rewards or search for 3%.
- If it supports approximately 1% rather than 3%, publish that measured effect
  with uncertainty; a modest reproducible effect is different from no effect.
- Do not reopen D3/D4/PPO, alter horizons or create an easier Self to force a gap.
- New computation needs one complete bounded approval; no second launch-only
  gate after that approval. This document grants none.
- Results reports can be reviewed now. Manuscript integration, complete source
  availability and collaborator archive access remain distinct follow-ups.

Source inspection and report drafting were assisted by Codex. Accountable human
authors have not been marked as having approved these interpretations.
