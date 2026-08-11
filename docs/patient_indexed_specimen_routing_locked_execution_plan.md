# Patient-Indexed Specimen Routing: Locked Publication Execution Plan

Plan version: 1.0

Status date: 2026-08-11

Branch: `patient-indexed-specimen-routing`

Authority: the committed version of this document is the cross-session source
of truth for the routing-primary publication campaign.

## 1. Purpose and change control

This plan preserves the scientific main line across context compaction, new
Codex tasks, machine changes, and stage boundaries. It supersedes ad hoc chat
summaries but does not erase earlier evidence or protocols.

The main line may be amended only when all of the following are true:

1. The current stage is complete or has a terminal documented failure.
2. The stage evidence has been audited and summarized.
3. The user explicitly approves the proposed amendment.
4. The reason, affected stages, data-contamination implications, and replacement
   decision are appended to the amendment log in this file.
5. The amended plan is committed before any changed experiment is launched.

The following are not valid reasons to alter the plan: context loss, a new
agent, an isolated favorable seed, a temporary platform failure, convenience,
or unused compute. Existing outputs are immutable and are never overwritten,
reclassified, or silently pooled with a later protocol.

At the start of every routing-primary task, the agent must:

1. Read this file.
2. Inspect the actual repository commit and experiment evidence.
3. Identify the current stage and its gate.
4. Continue that stage without introducing a new scientific branch.

## 2. Locked research claims

The publication campaign separates four claims:

1. Graph attribution: routing AFR-GCN is better than parameter-matched routing
   AFR-Flat under the same teacher, action space, budget, scenarios, and CRNs.
2. Anchor improvement: routing AFR-GCN is better than routing MDL-2.
3. Online-learning attribution: a final actor-critic checkpoint is better than
   its tensor-matched frozen-pretraining checkpoint.
4. Robustness: the primary comparisons remain directionally stable under
   regional stress and prespecified transport-availability assumptions.

Patient-indexed specimen routing is a default capability of the environment,
not the treatment whose value the paper is trying to prove. No-routing remains
available only for mechanics regression or a separately approved appendix.

## 3. Immutable completed evidence

The following campaign is complete and must remain immutable:

- Training commit: `9c0718b0da49d34f7f878ed2f036e96bf7bb3693`.
- Training seeds: 10, 11, 12, 13, and 14 for routing GCN and matched flat DDPG.
- Training budget: 100 online episodes per run, with full state every five
  episodes.
- Formal evaluation commit: `62a344e7c8f501862c6ae1dcf3be5ef59964ea0d`.
- Formal evaluation: four routing scenarios and 100 paired holdout CRN
  replications per scenario and training seed.
- Formal holdout seed: `91100000`.
- Training manifest SHA256:
  `0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2`.
- Teacher SHA256:
  `9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92`.

Verified pooled final-policy findings are:

- AFR-GCN-DDPG versus routing MDL-2: total cost -17.690 million cost
  units (-0.658%); the two-level 95% interval excludes zero.
- AFR-GCN-DDPG versus matched AFR-Flat-DDPG: total cost -8.604 million
  cost units (-0.321%); the two-level 95% interval excludes zero.
- GCN cost is lower than MDL-2 in all 20 training-seed by scenario cells.
- GCN final versus frozen pretraining: +72,903 cost units (+0.0027%).
  Online DDPG improvement is therefore not established.
- Flat final versus frozen pretraining: -139,378 cost units (-0.0052%).
  This is also too small and heterogeneous to establish an online increment.

These results support graph attribution and anchor improvement. They do not
currently support a claim that online actor-critic updates independently
created the gain.

## 4. Cost interpretation locked for reporting

The MDL-2 mean total objective is approximately 2,686.44 million cost units.
The dominant components are bioreactor-shortage cost (42.09%), patient-loss
cost (44.84%), and expiry cost (8.97%). These sum to about 95.9% of total cost.

This is a structural cost base created by fixed physical capacity and
exogenous demand, not a conventional accounting fixed cost. The manuscript
must not call `base_cost` a fixed cost: in the implementation it is the sum of
operating-cost components and can respond to actions.

The GCN-minus-MDL-2 savings decompose approximately as follows:

- Patient-loss cost: -12.631 million.
- Expiry cost: -2.806 million.
- Operating and shortage cost: -2.100 million.
- Urgency cost: -0.152 million.
- Specimen-transfer cost: +0.174 million.

The manuscript must report absolute and relative effects together, followed by
clinical outcomes and the component decomposition. It must not report only an
absolute number, call modeled cost units dollars without calibration, or invent
an "avoidable-cost percentage" without a defensible oracle or lower bound.

## 5. Fixed execution sequence

### Stage A: transport-availability sensitivity

Status: completed and audited on 2026-08-11. The frozen-policy lead-0 and
return-1 evaluations each completed all 10 algorithm-by-seed runs with 100
paired CRN replications per run. See
`docs/patient_indexed_specimen_routing_stage_a_review.md`.

Run two frozen-final-policy sensitivities with no retraining, checkpoint
selection, or deployment retuning:

1. Specimen availability lead 0 epochs versus the primary lead of 1 epoch.
2. Finished-product return lead 1 epoch versus the primary lead of 0 epochs.

Use GCN and matched flat seeds 10-14, routing MDL-2, and 100 paired CRN
replications. Require exact row counts, matching scenarios and CRNs, nonzero
routing, nonzero learned residual use, finite metrics, clean stderr, and locked
hashes. These tests are transport-timing sensitivities, not geography
sensitivities.

### Stage B: DDPG online-attribution diagnostic

Status: active; protocol preparation is in progress.

Do not tune on formal holdout seed `91100000`. On a newly locked development
CRN stream, evaluate the existing frozen-pretrain and episode 25, 50, 75, and
100 checkpoints. Report cost, clinical guardrails, residual usage, actor drift,
and per-seed trajectories.

This diagnostic decides whether the DDPG online curve is improving, plateaued,
or deteriorating. It is mechanism evidence, not a new independent confirmation.

### Stage C: bounded post-formal algorithm development

The completed seeds 10-14 and their formal holdout cannot regain untouched
status. Any changed training protocol uses a fresh development output root,
fresh development training seeds, and a fresh development CRN stream locked in
a stage-specific spec before launch.

Apply this decision tree:

1. If multi-seed DDPG development curves still improve near 100 episodes, test
   a matched 200-episode DDPG development candidate for both GCN and flat.
2. Extend a matched DDPG candidate to 300 episodes only if the 100-200 segment
   continues to improve across seeds without clinical deterioration.
3. If DDPG is plateaued or deteriorating, do not run a blind 300-episode
   extension. Run a 100-episode, three-development-seed matched AFR-GCN-TD3 and
   AFR-Flat-TD3 screen from the same teacher and residual-policy contract.
4. If the manuscript retains a backbone research question, TD3 remains the
   principal controlled backbone ablation. SAC and PPO are not promoted to
   formal routing-primary training unless a later approved amendment changes
   the paper scope.
5. Promote at most one revised training protocol to independent confirmation.

No broad automated HPO is allowed. A new experiment must be mechanism-driven,
single-factor or explicitly matched, use a separate development stream, and
give graph and flat architectures the same budget.

### Stage D: independent confirmation of a revised candidate

If Stage C changes the backbone, update rule, or episode budget, lock that
candidate before confirmation. Use at least five fresh training seeds and a
fresh holdout CRN stream that has never been used for development. Compare:

1. Final GCN versus matched final flat.
2. Final GCN versus routing MDL-2.
3. Final GCN versus frozen GCN pretraining.
4. Final flat versus frozen flat pretraining.

Use a hierarchical paired bootstrap over training seeds and CRN replications.
An online-learning claim requires a favorable final-minus-frozen cost interval,
consistent seed direction, and no material clinical deterioration. A point
estimate alone is insufficient.

If no revised candidate passes, retain the completed 100-episode DDPG result
and narrow the manuscript claim to graph-aware advantage-filtered residual
control. Do not continue open-ended algorithm search.

### Stage E: manuscript and reproducibility freeze

After the final experiment decision:

1. Generate pooled and per-scenario paired tables.
2. Generate the cost-component figure and absolute-plus-relative effect table.
3. Report graph, anchor, and online attribution as separate claims.
4. State negative or inconclusive online attribution transparently.
5. Update routing, transport timing, parameter provenance, limitations, and
   terminology throughout the manuscript.
6. Freeze source, configs, manifests, checkpoints, hashes, provenance, archive,
   and the exact manuscript evidence map.

## 6. Geography sensitivity decision

A broad formal sensitivity over transport speed or handling time is not part of
the required main line.

The environment is already geography-aware: permitted edges are geographic,
edge distances are computed from facility locations, and transfer time is
`fixed_handling_hours + distance / speed_mph`. The primary values are 500 mph
and 0.5 handling hours. Specimen inventory availability, however, is controlled
by the separate integer lead-time setting, and the primary benchmark disables
the continuous-time viability hook. Under the present model, changing speed
mostly changes recorded transport time and a very small transfer-cost term; it
does not create a proportionate change in inventory timing or clinical
feasibility.

Required geography work is therefore limited to:

1. Cite or document a defensible empirical source for the primary speed and
   handling-time assumptions.
2. Audit the resulting edge-distance and transport-time ranges for plausibility.
3. Describe clearly that lead 0/1 is an epoch-level availability assumption,
   distinct from continuous geographic transport time.
4. Avoid claiming robustness to transport speed unless it is actually tested.

A frozen-policy speed/cost sensitivity may be added only through change control
if a reviewer requests it, if external calibration shows substantial parameter
uncertainty, or if a future model enables continuous-time viability/expiry so
that speed materially changes patient outcomes. It is not a prerequisite for
the current paper.

## 7. Stage review checklist

At the end of each stage, record:

1. Exact commit, configs, seeds, CRNs, output roots, and SHA256 values.
2. Process and hardware evidence for training or evaluation.
3. Row counts, checkpoint counts, finite-metric and anomaly audits.
4. Primary, clinical, and attribution results with uncertainty.
5. Whether the stage gate passed.
6. Whether the next locked stage should proceed unchanged.

If an amendment is proposed, stop before launching it and complete the change
control requirements in Section 1.

## 8. Amendment log

- Version 1.0, 2026-08-11: Initial publication execution lock. Geography-speed
  sensitivity classified as optional; lead/return timing sensitivity retained
  as required. DDPG checkpoint attribution precedes the conditional DDPG
  extension or matched TD3 screen.
