# Patient-Indexed Specimen Routing: Locked Publication Execution Plan

Plan version: 1.2

Status date: 2026-08-12

Branch: `patient-indexed-specimen-routing`

Authority: the committed version of this document is the cross-session source
of truth for the routing-primary publication campaign.

## Current verified position

- Current publication gate: Stage C matched TD3 development screen.
- Independent execution owner: Howard, on Mac MPS, using the locked Stage C
  assets at commit `8a7768e3495be6d63e2091445ac6a7aa28ec0558`.
- Parallel core-team work: one user-approved, single-factor DDPG
  support-alignment development experiment on a separate Mac MPS worktree,
  locked at commit `470b56b0422704d09b4cfe36bdc01be0bb711fc1`.
  It may not change or interfere with Howard's Stage C protocol.
- Verified diagnosis: DDPG online replay ranking improves early and then
  erodes while critic advantage scale becomes severely miscalibrated. The
  teacher ranking regularizer also includes 41.72% of rows whose winning option
  is outside the specimen-only residual policy support.
- The support-alignment experiment is the only concurrent DDPG training
  authorized while Stage C is unresolved. Do not reuse the formal holdout,
  reopen broad HPO, add another DDPG factor, or alter its frozen protocol.
- Immediate next gate: jointly review Howard's matched TD3 result and the
  paired DDPG support-alignment result. At most one revised protocol may proceed
  to fresh independent confirmation; neither may be selected from a point
  estimate alone.

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

Status: completed and audited on 2026-08-11.

The locked checkpoint-curve evaluation completed pretraining and episodes 25,
50, 75, and 100 for both routing GCN and matched flat DDPG. It used five
training seeds, four routing scenarios, 50 paired development CRN replications
per scenario, and produced 50 valid run summaries, 10,000 learned-policy rows,
and 10,000 MDL-2 anchor rows. The formal holdout stream was not reused.

The late GCN segment from episode 75 to episode 100 changed mean total cost by
+68,613 cost units (+0.00256%), with a two-level 95% interval of approximately
[-344,648, +501,630]. Three of five seeds improved and clinical
noninferiority passed, but the cost interval did not exclude zero. The locked
classification is therefore `plateaued_or_inconclusive`, and the next step is
the matched 100-episode, three-development-seed TD3 screen in Stage C. This is
the prespecified decision-tree outcome, not a post-hoc protocol amendment.

- Stage B execution commit:
  `2ba1b3e01cb684fc3580273d1cbac02e17ab8dfb`.
- Checkpoint-curve summary SHA256:
  `0016e3bb8fa779984de1a21015bcade6f377d72a231b7581adc14ffc5a652ecf`.

Do not tune on formal holdout seed `91100000`. On a newly locked development
CRN stream, evaluate the existing frozen-pretrain and episode 25, 50, 75, and
100 checkpoints. Report cost, clinical guardrails, residual usage, actor drift,
and per-seed trajectories.

This diagnostic decides whether the DDPG online curve is improving, plateaued,
or deteriorating. It is mechanism evidence, not a new independent confirmation.

### Stage C: bounded post-formal algorithm development

Status: execution ready. The matched TD3 screen has been selected and its Mac
MPS training/evaluation assets, frozen teacher, fresh development streams,
parameter match, tests, hashes, and output roots are recorded in
`experiments/configs/patient_indexed_specimen_routing_stage_c_td3_execution.json`.
Howard is the independent execution owner. The handoff is in
`docs/howard_stage_c_td3_handoff.md`.

An explicit version-1.2 exception authorizes one concurrent core-team DDPG
mechanism experiment because it uses separate compute, seeds, CRNs, output
roots, and a single diagnosis-derived factor. It compares a fresh unfiltered
DDPG control against a support-aligned candidate for both GCN and matched flat.
Each candidate is forked from a byte-audited episode-0 state payload shared
with its control; only the critic teacher-support contract differs. This is
development evidence, not a second formal confirmation and not an invitation
to parallel HPO. TD3 and support-aligned DDPG must be reviewed together before
either can advance.

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
- Status update, 2026-08-11: Stage B completed with the locked
  `plateaued_or_inconclusive` decision. The existing Stage C decision tree
  therefore selects a matched three-seed TD3 development screen. No scientific
  protocol amendment was made.
- Status update, 2026-08-11: Stage C implementation and execution assets were
  locked for Mac MPS. The screen uses fresh seeds 20-22, six serial 100-episode
  runs, matched parameter budgets, and fresh development CRNs. Howard may
  launch from the supplied commit without a separate approval round.
- Status update, 2026-08-12: The read-only Stage B mechanism diagnosis was
  completed. It identified online replay-ranking erosion, critic scale
  miscalibration, and a teacher-policy support mismatch. This did not amend
  Stage C or authorize a new DDPG experiment.
- Version 1.2, 2026-08-12: The user explicitly approved one exception to the
  sequential Stage C gate so independent Mac compute would not remain idle.
  A paired support-aligned DDPG experiment is authorized concurrently with
  Howard's unchanged TD3 screen. The amendment is limited to one teacher
  support factor, fresh seeds 30-32, fresh development CRNs 95000000 and
  95100000, and disjoint outputs. It does not authorize broad HPO, formal
  holdout reuse, cross-campaign pooling, or automatic winner selection. Both
  results feed one joint gate and at most one protocol may advance.

## 9. Append-only progress ledger

This ledger is the compact-safe recovery point for future tasks. Add one entry
after every material stage result, terminal failure, protocol decision, or
publication-scope decision. Never edit or delete an older entry. Each new entry
must record the date, stage, exact commit, immutable evidence, decision, and
next gate. Detailed reports may live elsewhere, but must be linked here.

### 2026-08-12: Stage B diagnosis opened in parallel with Stage C execution

- Stage: post-Stage-B DDPG online-attribution mechanism diagnosis.
- Source branch/commit: `patient-indexed-specimen-routing` at
  `8a7768e3495be6d63e2091445ac6a7aa28ec0558`.
- Diagnostic branch: `rl-attribution-refinement`.
- Immutable evidence: Stage B checkpoint-curve summary SHA256
  `0016e3bb8fa779984de1a21015bcade6f377d72a231b7581adc14ffc5a652ecf`,
  training manifest SHA256
  `0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2`,
  and teacher SHA256
  `9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92`.
- Verified starting result: GCN improves weakly through episode 50 on the
  development checkpoint curve, then loses that point-estimate gain by episode
  100; the late segment remains statistically inconclusive. Online updates and
  learned residual use are nonzero, so this is not an inactive-policy failure.
- Decision: Howard proceeds independently with the already locked matched TD3
  Stage C screen. The core team performs read-only critic/credit-assignment
  diagnosis only; no new training or formal-holdout evaluation is authorized.
- Next gate: commit a reproducible diagnostic report and use it together with
  the Stage C outcome to decide whether one bounded DDPG refinement is
  scientifically justified.

### 2026-08-12: Stage B DDPG mechanism diagnosis completed

- Stage: read-only critic, replay, teacher-support, and cost-component audit.
- Diagnostic implementation:
  `evaluation/diagnose_ddpg_online_attribution.py`.
- Detailed report:
  `docs/patient_indexed_specimen_routing_rl_attribution_diagnosis.md`.
- Deterministic diagnostic JSON SHA256:
  `15e8f726f3266c9b5fbbb223e1dc261c0e09e082c6c1dd106912ff099b192420`.
- Verified result: GCN replay-ranking Spearman rises from 0.191 at pretrain to
  0.381 at episode 25, then declines to 0.159 at episode 100. Its critic
  calibration slope falls from 0.371 to 0.023 while prediction scale expands
  far beyond the stored replay-return scale. Flat ranking approaches chance.
- Support audit: 131 of 314 teacher rows, or 41.72%, have a globally winning
  option outside the specimen-only policy support. This is a contributing
  mismatch, not a complete explanation of replay-ranking erosion.
- Decision: Stage C remains unchanged because twin critics and clipped double-Q
  targets directly test the leading instability mechanism. No new DDPG run is
  authorized while Howard's matched TD3 screen is unresolved.
- Conditional fallback: only if Stage C fails, prepare one matched
  support-aligned DDPG candidate with `allowed_option_groups` restricted to
  `specimen_transfer`, fresh development seeds/CRNs, and no broad HPO.
- Next gate: audit the completed Stage C result and choose exactly one path:
  TD3 promotion or the single bounded DDPG fallback.

### 2026-08-12: Paired DDPG support-alignment experiment locked

- Stage: concurrent, single-factor post-Stage-B DDPG mechanism development.
- Execution asset commit:
  `470b56b0422704d09b4cfe36bdc01be0bb711fc1` on
  `rl-attribution-refinement`.
- Execution spec:
  `experiments/configs/patient_indexed_specimen_routing_ddpg_support_alignment_execution.json`,
  SHA256
  `bd69ad13ad83ccf4d21439fee4b58aa58117d12038d73ac73999ebfa437331c7`.
- Design: GCN and matched-flat DDPG, seeds 30-32, 100 online episodes per
  control and candidate run, checkpoint every five episodes, and a common
  frozen routing teacher. The candidate differs only by restricting critic
  teacher calibration to policy-supported anchor plus specimen-transfer rows.
- Pairing: six calibration-free, zero-trajectory full states are generated
  once. Control and candidate contract clones retain identical model,
  optimizer, replay, environment, and RNG payloads. The runner audits these
  payloads before accepting either branch.
- Evaluation: final and frozen-pretrain policies use four routing scenarios,
  50 paired replications per scenario, validation seed `95000000`, and
  development holdout seed `95100000`. Formal seed `91100000`, Stage B seed
  `93100000`, and Howard's Stage C streams are forbidden.
- Verification before lock: `python -m compileall` passed; focused tests passed
  74/74; the full repository suite passed 629/629; Howard's Stage C hashes
  remained unchanged.
- Decision: this is development evidence only. No further DDPG factor may be
  introduced from this run. Advancement requires paired cost uncertainty,
  clinical noninferiority, positive final-versus-pretrain attribution, and a
  joint review with the independently executed TD3 screen.
- Next gate: perform one no-trajectory MPS preflight, launch the one-claim
  detached campaign, audit completion, then review it together with Stage C.
