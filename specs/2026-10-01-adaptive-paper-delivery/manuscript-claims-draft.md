# Paper-facing claim draft

Local drafting only. Reuses the existing
[September 29 evidence checkpoint](../../docs/team_updates/2026-09-29-manuscript-evidence-checkpoint.md);
no new outcome analysis, patient episode, fitting or formal confirmation.
This is not a replacement of the frozen historical methods or results.

## Methods: historical study

We study patient-indexed decisions in a distributed manufacturing network.
The evaluated controller combines a pretrained graph-aware policy with a
restricted specimen-routing residual. Reagent, capacity and replenishment
operations remain part of the simulator, but this does not imply that the
learned residual independently optimizes every operation. The executed training
path uses the request representation and four-step relative online targets
documented in the formal method contract. Deployment uses fixed parameters.
Training by interaction with a simulator is distinct from continued parameter
adaptation after deployment.

## Results: supported historical statement

The complete graph-aware controller reduced pooled modeled cost relative to
MDL-2 by 17.689540 million objective units (0.658474%). The post-hoc crossed
95% interval for the cost difference was [-19.805786, -15.354809] million.
Relative to the implemented flat package, the difference was -8.603865 million
units (-0.321357%; interval [-11.284344, -6.087716] million). These comparisons
support package-level performance, not isolated attribution to message passing:
the graph and flat packages differ in their features, heads and gates.

The final graph-aware policy did not establish an incremental cost benefit
over its frozen pretrained counterpart. The final-minus-frozen difference was
+0.072903 million units, with interval [-0.180322, +0.313172] million. This is
not proof of equivalence, nor a conclusion that all online RL is ineffective.
Five independent trained seeds underlie the formal analysis; the 16,000
projected outcome rows are not 16,000 independent trained policies. Costs are
modeled objective units, not demonstrated monetary savings. The intervals are
the existing crossed-audit sensitivity results, not a newly accessed holdout.

## Prospective mechanism study: not yet executed

The proposed comparison asks whether interaction-based learning improves on
its own competent initializer, beyond simply continuing imitation. A dynamic
candidate policy and separate value baseline would use the same public
information and candidate support across frozen, PPO and BC-CONTINUE branches.
R4 and full MDL-2 remain reference controllers. The draft uses three blocks,
with all learned policies sealed before independent paired test worlds.

The proposed message graph is explicitly the specimen-transport relation. The
candidate bank contains R4, MDL-2 and four prespecified specimen corrections;
non-specimen request components must agree within each bank. The policy can
respond to cross-facility consequences without being a general joint controller
for every resource. This restricted comparison is a mechanism test, not the
permanent scope of a final method. Positive results would support only the
tested intervention, while negative results would delimit that intervention.
Neither outcome alone establishes a general deployment-adaptation claim.

## Limitations and next decision

Historical replay/target and graph-contract limitations are documented, but
they do not identify a single causal explanation for the null RL increment.
Engineering tests and exact restoration are not performance results. The new
graph-only pilot cannot establish isolated GCN value or superiority over a
corrected DDPG comparator. A later graph/self-only/flat comparison must match
information, support, gates and relevant model capacity explicitly.

If the restricted candidate space has no useful learnable headroom, a future
study may consider specimen routing coupled with a physically justified
flexible-capacity decision. This requires a separate bounded design, not
automatic reward tuning or extra trials until a positive result appears.
If headroom exists but the learner fails to realize it, the next intervention
should address the demonstrated learning bottleneck instead. Stronger
adaptive-control comparisons, independent confirmation, domain calibration and
cost/patient trade-off justification remain necessary for broader claims.
E1's missing field inputs remain missing; no clinical noninferiority or
deployment readiness is inferred from cost improvements.

## Traceability

- Historical methods: `specs/2026-09-29-formal-method-contract/readout.md`.
- Cost intervals: `specs/2026-09-29-formal-crossed-audit/readout.md`.
- Attribution limitations: `specs/2026-09-29-formal-graph-contract/readout.md`
  and `specs/2026-09-29-formal-replay-contract/readout.md`.
- New mechanism scope: this directory's `plan.md`, `workflow.json` and
  preserved unapproved `pilot-budget-draft.json`.

No claim here guarantees a positive RL result or journal acceptance.
