# Stage G1 Legal-Action-Aligned GCN-DDPG Design Review

> **Scenario reconstruction correction (2026-08-18).** The executed G1 audit
> used the stored nominal-history reference environment while labeling rows by
> the intended persistent-hotspot assignment. Its numeric evidence is retained
> for auditability but cannot support the original scenario-specific claim.
> The proposed online extension remains unauthorized; see
> `patient_indexed_specimen_routing_ddpg_problem_redefinition_review.md` for the
> corrected action-geometry screen and current decision.

## Status

This document records the final scientifically defensible route considered for
testing an online DDPG contribution. Stage G1 executed the required offline
leave-one-seed-out feasibility gate and did not pass: validation top-1 accuracy
was `30.13%`, pairwise accuracy was `57.74%`, and discovery-validation best
action agreement was `54.49%`. The locked classification is
`unstable_counterfactual_labels_close_extension`.

The proposed actor-transfer gate and online comparison below are therefore not
authorized. They are retained as prospective design documentation, not as work
to execute. See `docs/patient_indexed_specimen_routing_stage_g1_results.md`.

## Proposed method boundary

Retain the GCN-DDPG backbone, deterministic residual actor, target networks,
replay, and MDL-2 safety anchor. Replace the weak single behavior-versus-anchor
auxiliary signal with a legal-action advantage objective aligned to the
executed action space and endpoint horizon.

At selected replay states, enumerate the same five specimen actions that can
actually execute: MDL-2 and `+/-0.05`, `+/-0.10` specimen corrections. Estimate
paired-CRN counterfactual costs for both a short horizon and a long horizon.
Define within-state normalized advantages

`z(s,a,h) = (C(s,a_anchor,h) - C(s,a,h)) / scale(s,h)`,

where `scale(s,h)` is a prespecified robust within-state scale with a fixed
positive floor. Train an auxiliary advantage head with:

- a Huber regression term on `z`;
- a pairwise margin-ranking term for every behaviorally distinct legal pair;
- separate short- and long-horizon outputs or an explicit horizon input; and
- equal per-state weighting, so states with more extreme costs do not dominate.

The standard Bellman critic remains present. The legal-action head is an
auxiliary policy-improvement signal, not a replacement for the DDPG return
critic.

## Why this still qualifies as DDPG-based

The actor remains deterministic and continuous, and standard DDPG still learns
the base state-action value and target-policy update. The extension changes how
the simulator's known discrete execution geometry supervises and deploys the
residual action. It should be named transparently, for example
`legal-action-aligned GCN-DDPG`, rather than described as unchanged DDPG.

Only after the legal-action head passes should the actor interface be reviewed.
The lowest-risk option is a proto-action policy: the deterministic actor emits
a continuous residual, nearby legal executed candidates are generated, and the
validated legal-action head chooses among them. A quantization-aware
advantage-weighted actor loss may be tested only if proto-action selection
demonstrates that continuous actor movement survives execution.

## Required offline feasibility gate

Before any online training comparison:

1. construct a fresh development-only legal-action dataset with all five
   actions, prespecified short and long horizons, exact CRNs, and no formal
   holdout reuse;
2. evaluate by leave-one-training-seed-out folds, never by random row splits;
3. require long-horizon top-1 accuracy of at least `40%` in aggregate and at
   least `30%` in every held-out seed;
4. require long-horizon pairwise accuracy of at least `65%` in aggregate;
5. require a material improvement over the frozen F1 critic in at least two of
   three held-out seeds; and
6. verify that improvement persists under a second fixed replication stream.

Failure closes further DDPG attribution work. Passing only authorizes a locked
actor-transfer smoke, not a full training campaign.

## Required actor-transfer gate

On fixed states unseen by the ranker fit:

- at least `30%` of selected executed actions must differ from the frozen
  policy;
- the selected action must have positive independently rolled-out advantage in
  at least `60%` of changed states;
- mean independently rolled-out advantage among changed states must be
  positive; and
- clinical noninferiority must hold.

Only then should one tensor-matched online control/candidate campaign be frozen.
The candidate would differ solely in the legal-action advantage objective and
its prespecified action-transfer rule. Episode count, exploration support,
scenarios, actor/critic width, and all evaluation CRNs would remain fixed.

## What not to do

- Do not increase online episodes before ranking transfer is demonstrated.
- Do not tune loss weight against final performance; define target
  normalization so a fixed unit weight has meaning.
- Do not select the best intermediate checkpoint after seeing evaluation.
- Do not use the formal holdout for development.
- Do not call an oracle-selected or simulator-planned action ordinary DDPG;
  report the auxiliary counterfactual policy-improvement mechanism explicitly.
