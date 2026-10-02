# Paper-facing claim draft

Local drafting only. Reuses the existing
[September 29 evidence checkpoint](../../docs/team_updates/2026-09-29-manuscript-evidence-checkpoint.md)
and the separately authorized completed development evaluations linked below.
This text introduces no additional patient episode, fitting or formal confirmation.
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

## Completed Development Mechanism Study

The separately authorized continuation study completed on 2026-10-02 using
three qualified saved initializers, matched frozen/PPO/BC-CONTINUE branches,
R4 and full MDL-2 references. All nine learned artifacts were sealed before
180 paired development-test controller episodes. It completed all 192 continuation
episodes and 1,152 optimizer calls; the old failed preparation run was not restarted.
The [canonical readout](../../reports/2026-10-01-s1-continuation-recovery/readout.md)
reports exactly zero PPO-minus-frozen and PPO-minus-BC cost and patient differences
on every evaluated world. No incremental greedy PPO benefit was established.

The post-hoc saved-record description shows changed policy probabilities but
unchanged greedy requests on all 1,872 shared test decisions. Nonreference
classes were sampled in 73-80% of PPO training decisions. These observations
do not establish why return optimization failed to change the ranking, that
the reference is globally optimal, or that rewards must be changed. The [0, 0]
primary bootstrap interval reflects resampling identical observed outcomes,
not population equivalence or confirmatory evidence.

PPO's equal-block cost difference versus full MDL-2 was -13.503781 million
modeled units (-0.521671%; descriptive 95% relative interval [-0.755552%, -0.213307%]).
Frozen, BC and R4 shared the same evaluated behavior, so that advantage is inherited,
not an RL continuation gain. Terminal-active burden increased 15.666667 patients
per episode on average; block 61 also had fewer completions and more waiting.
Report the cost/service trade-off rather than overall or clinical superiority.

The evaluated message graph is explicitly the specimen-transport relation. The
candidate bank contains R4, MDL-2 and four prespecified specimen corrections;
non-specimen request components must agree within each bank. The policy can
respond to cross-facility consequences without being a general joint controller
for every resource. This restricted comparison is a mechanism test, not the
permanent scope of a final method. Positive results would support only the
tested intervention, while negative results would delimit that intervention.
Neither outcome alone establishes a general deployment-adaptation claim.
This new development result does not overwrite the separate historical formal
comparisons above or add evidence to their holdout sample.

## Limitations and next decision

A subsequent saved-record diagnosis reconstructed 4,992 PPO training decisions
without new simulation or model evaluation. On the last collected rollout in
each block, the value baseline explained only 0.106%-0.282% of return variance,
while pooled return-to-go correlated strongly with episode step (0.959-0.975).
This identifies weak collection-time baseline discrimination, not its causal
contribution to the null result or the final checkpoint's value accuracy.
Sampled action-return associations are state- and policy-confounded; they do
not establish absent headroom. Separate policy/entropy parameter gradients were
not retained, so their causal contributions cannot be decomposed retrospectively.
See the [saved-return diagnosis](../../reports/2026-10-02-saved-return-ranking/readout.md).
A separately authorized, prospectively controlled time-baseline comparison has
now completed all 216 development evaluation episodes. It replaces only the
policy-advantage baseline with a training-only leave-one-episode-out time
baseline, not the reward, critic targets or deployment rule. Three blocks each
trained original PPO, time-baseline PPO and BC for 32 episodes from identical
qualified starts; all 12 learned artifacts were sealed before evaluation.
The independent raw verifier reports exactly zero time-baseline PPO differences
from frozen, original PPO, BC and R4 for cost and all recorded patient outcomes.
Greedy requests remained unchanged; the prespecified development screen failed.
This delimits the tested baseline intervention, not all RL or the global
optimality of R4. See the [time-baseline readout](../../reports/2026-10-02-time-baseline-comparison/readout.md).

Relative to full MDL-2, the equal-block cost difference was -14.867150 million
modeled objective units (-0.564443%), inherited from the unchanged reference
behavior rather than RL continuation. Mean completions increased 7.5 and losses
decreased 23.416667, but terminal-active patients increased 15.916667 per episode;
block 61 had fewer completions and more waiting. These are cost/service trade-offs,
not clinical superiority, and the three-block development sample is not an
independent confirmation of the historical formal result.

The tested baseline route is closed without more epochs, seeds or alternative
baselines. The next objective-definition question is whether the intended
evaluation prices only the 52-step window or also residual patient obligations
beyond it. The current finite-window objective is not thereby a proven bug.
Any terminal-liability amendment needs justified valuation and a new prospective
authorization; unfinished patients must not be equated to deaths. The subsequent
saved-data diagnosis favors explicit fixed-cohort follow-up over an invented
per-patient terminal penalty, as detailed below.
Full-MC potential shaping can simply shift the advantage baseline, so it is not
an independent breakthrough by default. No reward change or new training follows
automatically from this result.

### Terminal-cohort interpretation and prospective objective design

A saved-only review of all 216 time-baseline evaluation episodes verified 433
source files unchanged. Original PPO, time-baseline PPO, BC and R4 each have
identical full final states to frozen in all 36 paired worlds. Thus any common
deterministic terminal valuation applied to these states preserves zero RL
increment; retrospective reward reweighting cannot establish a learning benefit.

The R4-like versus MDL-2 terminal-active difference also requires care: with
identical enrollment, active+15.916667 exactly reconciles with completed+7.5
and lost-23.416667. More active patients alone do not demonstrate clinical harm.
Final-step entrants average122.5 in both groups, making up17.2853% of the
reference-like terminal-active count but none of the between-arm difference.
These entrants arrive after that step's processing opportunities. Terminal-active
counts measure unresolved obligations at an administrative cutoff, not observed
future deaths or a complete clinical endpoint. Preserve the earlier conservative
reporting flag and block61 concern without overstating either as proven harm.

The subsequent design fixes the52step enrollment window and follows the same
cohort to modeled infusion/loss under a common declared continuation rule.
It records subsequent primitive costs rather than selecting penalty weights to
favor RL. This changes the estimand to closed-cohort prefix control under that
rule; it is not an indefinitely operating network or deployment adaptation.
On2026-10-02 the numerical package was explicitly approved and frozen, and its
single execution started. Real preflight completed18cohorts, original-engine
prefix parity and both restoration boundaries. All288training cohorts and1920
optimizer calls completed and12models sealed. The first evaluation task exceeded
its90second owner limit after11complete frozen-policy worlds and17steps of the
next world. The attempt terminated; no comparative performance result exists.
This is an operational time-budget failure,not a null effect of the amended
objective. A separately authorized evaluation-only recovery is proposed. See the
[terminal diagnosis](../../reports/2026-10-02-terminal-obligation/readout.md)
and [frozen cohort protocol](../2026-10-02-terminal-obligation/protocol.md).

### Methods Draft: Fixed-Cohort Objective Comparison

We preserve the original52-step enrollment, observation and candidate-action
process in the nominal-history simulator. At the administrative cutoff, new
enrollment ceases. A common full-MDL-2 rule with zero future demand and demand-rate
priors resolves the existing cohort; after all patients resolve, no new transfers
or replenishment orders are placed. All controllers retain the same fixed
11-transition follow-up and63-step economic endpoint, including costs after
earlier patient resolution. Existing resource stocks, supplier evolution and
primitive cost coefficients are retained. We record residual stocks and clipped
overflow without inventing a salvage credit or assigning deaths to unfinished
patients. This is finite-window economic accounting, not lifetime valuation.

For each of three training blocks, frozen, window-PPO, cohort-PPO and continued-
imitation branches share identical qualified initial weights. The two PPO arms
retain the same collected-value baseline, optimizer settings, candidate support,
entropy term and training allocation. Window-PPO uses the original52negative-cost
rewards. Cohort-PPO adds the negative sum of the11follow-up primitive costs once
to the final prefix training reward. Original recorded rewards remain immutable;
follow-up actions supply neither PPO decisions nor imitation examples. Thus the
sole PPO-arm intervention is whether common post-window costs enter the return,
not an arbitrary penalty, extra decision authority or longer learned policy.

Each continuation arm receives32complete cohorts per block. All12learned
artifacts must be sealed before evaluation on12new starts per block across six
controllers, including R4 and full MDL-2. The primary comparisons are cohort-PPO
against its frozen start, window-PPO and continued imitation. We report raw63-step
cost, original52-step outcomes, cohort resolution/losses, resource effects and
greedy-request changes. The prespecified development screen requires strictly
lower mean cost in every block and pooled, with no observed block-level loss
increase against each primary comparator. Paired block-then-world intervals
remain descriptive with only three independently trained blocks; the screen is
neither clinical noninferiority nor independent confirmation. Shared starting
seeds do not guarantee identical exogenous paths after state-dependent random
number consumption diverges.

Execution provenance: implementation8a45c72269d23a99d36c2d6ca16c3eacb1b99e1a,
execution098b83f3e0457a3ec65568cbba90147670215bc6, separate
[authorization](../2026-10-02-terminal-obligation/authorization.json) and
[frozen bindings](../2026-10-02-terminal-obligation/frozen.json). Current outputs
are terminal/incomplete and must not be substituted for the closed historical
studies. The primary screen was not evaluated; neither benefit nor failure to
benefit from the amended objective has been established.

Historical replay/target and graph-contract limitations are documented, but
they do not identify a single causal explanation for the null RL increment.
Engineering tests and exact restoration are not performance results. The
completed graph-only pilot cannot establish isolated GCN value or superiority over a
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
- Completed mechanism comparison: `continuation-recovery-authorization.json`,
  `continuation-recovery-frozen.json` and the canonical continuation readout.
- Earlier unapproved `pilot-budget-draft.json` remains historical provenance,
  not the current execution permit or result.

No claim here guarantees a positive RL result or journal acceptance.
