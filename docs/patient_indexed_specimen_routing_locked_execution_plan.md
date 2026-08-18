# Patient-Indexed Specimen Routing: Locked Publication Execution Plan

Plan version: 1.5

Status date: 2026-08-17

Branch: `patient-indexed-specimen-routing`

Authority: the committed version of this document is the cross-session source
of truth for the routing-primary publication campaign.

## Current verified position

- Current publication gate: Stage E manuscript and reproducibility freeze.
- The completed five-seed 100-episode AFR-GCN-DDPG protocol remains the primary
  method. Its formal evidence supports graph attribution and improvement over
  routing MDL-2, but not a separate online final-versus-frozen gain.
- Howard's matched three-seed TD3 development screen completed successfully
  after a post-training CSV-audit recovery. GCN-TD3 beat matched flat TD3 and
  MDL-2, while final and frozen-pretrain performance were indistinguishable.
  TD3 is retained as a controlled backbone ablation, not a replacement primary
  method and not formal holdout evidence.
- The bounded DDPG support-alignment, persistent-shift critic-realignment, and
  structured-exploration candidates all reached audited terminal negative
  decisions. Structured exploration produced the intended behaviorally
  distinct action coverage but still did not establish online gain.
- No revised candidate passed into Stage D. Additional HPO, selective seed
  extension, another online mechanism screen, and formal-holdout reuse are
  closed. The remaining work is evidence integration, manuscript reporting,
  reproducibility packaging, and final branch freeze.

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

The GCN-minus-MDL-2 savings decompose approximately into the following
additive top-level objective components:

- Patient-loss cost: -12.631 million.
- Expiry cost: -2.806 million.
- Base operating cost: -2.100 million.
- Urgency cost: -0.152 million.

Within base operating cost, specimen-transfer cost increases by 0.174 million,
while the other operating and shortage subcomponents decrease by 2.275
million. The specimen-transfer amount is a subcomponent of `base_cost` and
must not be added to the four top-level values a second time.

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

Status: completed and audited. Howard's matched TD3 screen completed six
100-episode Mac MPS runs and paired final/frozen-pretrain development
evaluation. Its recovery status is `completed` with exit code 0, all 54 focused
tests passed, and all 245 inventoried artifacts were independently reconciled.
GCN-TD3 improved on MDL-2 by approximately 0.735% and outperformed matched flat
TD3, but final and frozen-pretrain checkpoints were statistically
indistinguishable. The handoff remains in `docs/howard_stage_c_td3_handoff.md`;
curated evidence is in
`experiments/evidence/patient_indexed_specimen_routing_stage_c_td3/`.

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

### Stage C2: persistent-shift DDPG online-attribution screen

Status: completed and closed. The persistent-shift candidate failed its locked
advancement gate; the terminal result and hashes are recorded in the progress
ledger below. The historical protocol remains here for provenance.

The mechanism hypothesis is that the original episode-round-robin curriculum
suppressed online attribution: a frozen actor could react to observable demand
history, while online updates repeatedly mixed alternating regimes. Stage C2
therefore tests adaptation to a persistent, geographically redistributed demand
regime. This is a geography-of-demand experiment, not a transport-speed sweep.

The protocol is locked as follows:

1. Use fresh development seeds 40, 41, and 42. Assign each seed before launch to
   one persistent hotspot map (clusters 1, 2, and 3 respectively). Every map
   preserves network-wide expected demand and changes only its geographic
   distribution. The three seed-map pairs are development replicates, not
   evidence for map-specific claims.
2. Train matched GCN and flat DDPG for 100 online episodes. Fork each standard
   control and realigned candidate from the same audited episode-0 state. The
   frozen comparator is that tensor-matched episode-0 actor.
3. The candidate may differ from control only at the online boundary: zero the
   action-input columns of the critic's first linear layer, copy critic to target
   critic, clear critic optimizer state, and withhold actor updates for the first
   500 online critic updates. Preserve actor tensors, replay, environment, RNG,
   teacher, residual scale 0.1, gate threshold 0.5, and every other parameter.
4. Audit fixed checkpoints at episodes 0, 10, 25, 50, 75, and 100. Episode 100
   is the deployment checkpoint; intermediate checkpoints diagnose adaptation
   and cannot be selected post hoc.
5. Use fresh execution-only validation CRN seed 95200000 and development CRN
   seed 95300000. Formal seed 91100000 and all earlier development streams are
   forbidden. Evaluate with paired CRNs and report absolute cost first, percent
   cost second, clinical guardrails, residual use, actor drift, and cumulative
   adaptation regret.
6. Advancement requires episode-100 final-versus-frozen improvement of at
   least 0.02% in pooled mean cost or a two-level 95% interval wholly below
   zero, favorable direction in at least two of three seed-map pairs, no
   material clinical deterioration, and nonzero routing, residual use, and
   online updates. The realigned candidate must also be no worse than standard
   DDPG in pooled point estimate to be selected over it.
7. Promote at most one DDPG protocol. A favorable development result requires
   fresh five-seed confirmation and a new holdout stream under Stage D. A failed
   result closes this candidate; a second DDPG mechanism requires another
   explicit amendment. Howard's outputs must not be pooled with Stage C2.

Every preflight must exercise training-state save/restore, scenario assignment,
all downstream parsers, comparator and inventory generation, including a
synthetic CSV field larger than 131,072 bytes. Expensive execution remains
strictly serial on Mac MPS with CPU fallback disabled.

### Stage D: independent confirmation of a revised candidate

Status: not triggered. No Stage C, C2, or C3 revised candidate passed its
preregistered development gate. No new holdout stream was opened.

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

Status: current stage as of 2026-08-17.

After the final experiment decision:

1. Generate pooled and per-scenario paired tables.
2. Generate the cost-component figure and absolute-plus-relative effect table.
3. Report graph, anchor, and online attribution as separate claims.
4. State negative or inconclusive online attribution transparently.
5. Update routing, transport timing, parameter provenance, limitations, and
   terminology throughout the manuscript.
6. Freeze source, configs, manifests, checkpoints, hashes, provenance, archive,
   and the exact manuscript evidence map.

### Stage F: post-freeze online DDPG identifiability

Status: Stage F0 completed on 2026-08-17. The user authorized the single Stage
F1 paired-advantage development experiment, conditional on its locked tests,
CPU smoke, MPS probe, episode-0 clone, and hash preflight. Formal confirmation
remains unauthorized.

Stage F is an optional post-freeze extension. It does not delay or invalidate
the Stage E publication path. Its objective is to determine whether one
mechanistically identified correction could make final online DDPG improve on
its own tensor-matched frozen pretraining checkpoint.

Stage F0 uses immutable Stage C3 replay buffers and checkpoints plus fresh
development-only paired-CRN rollouts. It tests temporal target consistency,
critic ranking on legal integer patient-lot specimen actions, and the validity
of the actor's straight-through critic gradient. Formal seed 91100000 and all
prior CRN streams are forbidden. No training or checkpoint selection is
permitted.

The locked Stage F0 protocol is
`docs/patient_indexed_specimen_routing_stage_f0_protocol.md`. A passing F0 gate
may justify review of one paired Stage F1 protocol. It never launches Stage F1
automatically. A failed F0 gate closes this online-DDPG extension and preserves
the Stage E manuscript decision unchanged.

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
- Version 1.3, 2026-08-12: After the support-alignment experiment reached an
  audited terminal negative result, the user explicitly approved continued
  work on DDPG without waiting for Howard's independent TD3 screen. One bounded
  Stage C2 experiment is authorized to test persistent geographic demand shift
  plus online-boundary critic realignment. It uses fresh seeds 40-42, fresh CRNs
  95200000/95300000, fixed episode-100 deployment, a preregistered 0.02% or
  interval-based advancement gate, and disjoint outputs. It does not reopen
  support alignment, authorize HPO, or expose the formal holdout.
- Version 1.4, 2026-08-17: Howard's matched TD3 screen and the core team's
  persistent-shift and structured-exploration DDPG screens all reached audited
  terminal decisions. No revised candidate passed the Stage D gate. The user
  approved proceeding to Stage E: retain the completed five-seed DDPG protocol
  as primary, retain TD3 as a development-only backbone ablation, report online
  attribution as not established, stop algorithm search, and freeze the
  manuscript evidence package before any merge to `main`.
- Version 1.5, 2026-08-17: Stage E compact evidence, publication tables,
  source-backed figure, claim map, and manuscript synthesis were assembled on
  `stage-e-publication-freeze`. A component audit corrected a reporting-only
  double-count risk: `specimen_transfer_cost` is inside `base_cost`, so it is
  shown as a base-cost detail but not added again to the top-level objective
  decomposition. No scientific result, protocol, or stage decision changed.
- Version 1.6, 2026-08-17: The user authorized a post-freeze, read-only Stage
  F0 identifiability audit because a verified online-DDPG contribution would
  strengthen the paper. The audit is limited to immutable Stage C3 replay and
  checkpoints, fresh diagnostic CRNs beginning at 95900000, and prespecified
  target-sign, legal-action-ranking, gradient, and headroom gates. It does not
  authorize training, formal holdout access, checkpoint selection, broad HPO,
  or delay of the Stage E publication path.

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

### 2026-08-12: Support-alignment launcher failure and Recovery 1

- Superseded execution commit:
  `8688500e8899dda1ec91cc16af147cc859b0ed62` on
  `rl-attribution-refinement`.
- Failure boundary: all six calibration-free episode-0 states completed, but
  the first control GCN seed-30 process exited before episode 0 was executed.
  Training and evaluation consumed zero trajectories and produced no online
  checkpoint.
- Root cause: the metadata-only contract clone changed `num_episodes` from 0
  to 100 but did not synchronize the mechanically derived
  `history_screen.online_episodes` and `history_screen.pretrain_only` fields.
  The full-state safety validator correctly rejected that inconsistent clone;
  this is not evidence of a DDPG, routing, MPS, or numerical failure.
- Immutable failure evidence: status SHA256
  `1984a0b5270e3eb2c84a3b2578f2c9f37471d9bc330d12e3f5190b601d0b58f7`,
  detached stderr SHA256
  `37cd88457b7c66e24c4847c8cdea7d44904e6efddba481e43e7394efef834732`,
  and phase stderr SHA256
  `b2dd7989f34197c9dc20b149997957952225cae9b2c1d6645f2e5a9b5fb7652f`.
- Recovery 1 rule: preserve the superseded root; reuse only its six immutable
  zero-trajectory states and matched pretrain checkpoints. Their 13-file tree
  SHA256 is
  `abcc3a3a831bd46d62f1a826a87b39b2fa1c8e72d85f92152b89f70d2b020c2f`.
  Generate fresh contract clones and every online/evaluation artifact under
  `patient_indexed_specimen_routing_ddpg_support_alignment_development/recovery1`.
- Scientific design remains unchanged. The repair synchronizes only the two
  budget-derived contract fields and rewrites output paths; actor, critic,
  optimizer, replay, environment, RNG, teacher, seeds, CRNs, and all training
  hyperparameters remain locked.
- Next gate: focused and full tests, compile validation, no-trajectory MPS
  preflight, then one detached Recovery 1 launch and five-episode monitoring.

### 2026-08-12: Recovery 1 controls completed; Recovery 2 continuation locked

- Recovery 1 execution commit:
  `c14bf379735a7d88bcea35e6f9adfb6e07e8a185` on
  `rl-attribution-refinement`.
- Completed scientific boundary: all six control runs (GCN and matched flat,
  seeds 30-32) completed 100 online episodes, for 600 total episodes. The
  audited outputs contain 262,016 specimen-routing decisions, 31,200 online
  RL updates, finite persisted metrics, all 120 five-episode policy
  checkpoints, and six final atomic training states. The GCN/flat parameter
  gap is 0.9699%, below the locked 1% maximum.
- Terminal launcher boundary: after the sixth control process exited 0, the
  runner's post-training audit raised Python `_csv.Error` because one
  serialized metric field exceeded the default 131,072-byte CSV field limit.
  No candidate training or evaluation process was started. This is an audit
  implementation failure, not a DDPG, routing, MPS, or numerical failure.
- Immutable Recovery 1 evidence: status SHA256
  `076511140f57ec1f2c0ae6e067c9f759a35a229ce119f26dd32a789c2b33f22e`,
  detached stderr SHA256
  `a779b412d142b2caffa08d29043b283dd086ab08e684c84a7626f1ad5fcb78e6`,
  control manifest SHA256
  `5a67ef4d139b1ec3b1e48692db3eba232de58098d415820be0fbfc34521dd2c4`,
  151-file control tree SHA256
  `99c6ef49705c8e00343401f1f97ddbf93ca8253545c2fc1d1353d14e5570c3a0`,
  and 13-file paired-state tree SHA256
  `ed7dabff2cc2124a2807cfc99fd94dce4b1073ac41c9c52a1e07991caa109cf3`.
- Recovery 2 implementation commit:
  `17ad3bc00a799a18424c7f7ab28616b09fcd407c`. It raises the CSV parser limit,
  revalidates every immutable Recovery 1 control and paired-state artifact,
  reads the completed control manifest without modification, and writes all
  candidate/evaluation/comparison outputs under a fresh `recovery2` root.
- Scientific design remains unchanged. Recovery 2 reuses the exact six
  never-consumed candidate episode-0 contract clones and runs only the six
  support-aligned candidates. It does not retrain controls, reuse candidate
  trajectories, change a hyperparameter, or access a forbidden CRN stream.
- Verification: compileall passed; the locked focused gate passed 59/59; the
  full repository suite ran 634 tests with 632 passing and only the two known,
  unrelated MPS/CPU fixture and PPO-threshold failures. Howard's Stage C files
  remained byte-identical to their handoff commit.
- Next gate: commit this append-only ledger update, run one clean-commit
  no-trajectory Recovery 2 preflight, then issue exactly one detached launch.
  After six candidate runs and four evaluation phases complete, review the
  paired support effect jointly with Howard's Stage C result.

### 2026-08-12: Recovery 2 evidence completed; support alignment did not pass

- Recovery 2 execution commit:
  `0a7cd1ef2edf305be9327cc947557766d68a1858` on
  `rl-attribution-refinement`.
- Completed scientific boundary: six immutable Recovery 1 controls and six
  support-aligned candidates each contain 100 online episodes, for 12 runs and
  1,200 episodes total. Candidate evidence contains 120 five-episode policy
  checkpoints, six atomic states, 262,080 specimen-routing decisions, 31,200
  online updates, finite persisted losses, and a 0.9699% GCN/flat parameter
  gap. All four evaluation phases completed 24 runs and exactly 4,800 learned
  plus 4,800 MDL-2 anchor rows.
- Terminal launcher boundary: training and evaluation exited successfully, but
  the comparison reader used Python's default 131,072-byte CSV field limit and
  failed on `specimen_route_events_json`. This was a post-processing defect;
  it did not invalidate or alter any trajectory, checkpoint, or evaluation.
  Immutable failed status SHA256:
  `52938af461f7e8336bad742f6367a0cbe5bb64a1510a4b402051d0f22996c73f`.
- Post-processing-only fix commit:
  `e08ba5a8ec84cc19a34d12e459d58fde9a595c29`. The comparator now raises the
  supported CSV field limit before reading, and a regression test exercises a
  200,000-byte serialized metric field. The focused support-alignment suite
  passed 11/11, repository-wide `compileall` passed, and the full real-data
  comparison completed with 20,000 bootstrap resamples. No training or
  evaluation process was relaunched.
- Recovered evidence: comparison SHA256
  `56f405e5975a54a5e532805ca3a8e261f47cdcdbafb668945e2725b126f7fa63`,
  76-file evaluation-input tree SHA256
  `b50888d0ea6373c0bc4e0ce9cb9da876310a7e9626b275fa79fc44f1f65e8d9a`,
  and post-processing inventory SHA256
  `e430b542c65083b036a5112581c7efae97ce0b6089eb8fc7fb0973ce0f9171f1`.
- Result: support alignment failed the preregistered advancement gate for both
  matched flat and GCN DDPG. Candidate-final cost was 0.00138% higher than
  control for flat (95% CI for absolute difference
  `[-304685, 398625]`) and 0.00175% higher for GCN
  (`[-130336, 203526]`). Final-versus-pretrain cost also increased by 0.00665%
  for flat and 0.01017% for GCN, with both confidence intervals spanning zero.
  GCN preserved clinical noninferiority, but neither architecture established
  lower cost or positive online attribution.
- Decision: do not promote support-aligned DDPG, add another DDPG factor, or
  selectively extend seeds. Await Howard's unchanged matched TD3 result and
  perform the locked joint review.
- Execution safeguard: every expensive campaign must run every downstream
  parser, comparator, and inventory writer end-to-end during preflight using a
  synthetic CSV field larger than 131,072 bytes. Post-processing phases must
  remain recoverable from immutable upstream artifacts without retraining.

### 2026-08-12: Persistent-shift DDPG Stage C2 authorized

- Starting branch/commit: `rl-attribution-refinement` at
  `fbbdbc627bfc93d3671d40e1ad0f8412d1d11254`.
- Evidence basis: the original DDPG checkpoint curve was
  `plateaued_or_inconclusive`; support alignment subsequently failed without
  improving control. Code audit showed the original online phase alternated
  four regimes and its 500-update actor warmup had already been consumed by
  offline pretraining, leaving no online-only critic adaptation interval.
- Decision: preserve DDPG as the intended primary method and test exactly one
  mechanism-matched alternative in which an observable but persistent
  geographic demand redistribution creates a genuine adaptation problem, while
  critic action dependence is reset at the online boundary before actor updates.
- Contamination boundary: seeds 40-42 and CRNs 95200000/95300000 are fresh;
  formal holdout 91100000, Howard's Stage C streams, prior support streams, and
  all immutable training/evaluation artifacts remain untouched.
- Next gate: implement and test Stage C2, commit the locked execution assets,
  run a no-trajectory end-to-end preflight, and only then launch one detached
  strictly serial development campaign.

### 2026-08-13: Persistent-shift Stage C2 closed; action coverage is the next DDPG gate

- Execution commit: `ccf7138688a62dbc0e28689f4dad3c1a127ed0e6` on
  `rl-attribution-refinement`. The campaign completed 12/12 training runs,
  1,200/1,200 online episodes, 240 policy checkpoints, 12 atomic states, and
  72/72 checkpoint-curve evaluation runs with 3,600 learned plus 3,600 anchor
  rows. Status SHA256 is
  `378348e7912e3a6b60be4661b9f18f067694ff60d52f76c543dafe392e013c91`;
  comparison SHA256 is
  `ec479ff6d2abb31740070729db4ca9c8763422c91c65fb20117765df94f2e3be`;
  the 824-file inventory SHA256 is
  `e79f783566878fd1ca533fcf4e92d7a63846ff7ecd12a89eab9fe8d332f2f151`.
- Locked result: the persistent-shift realignment candidate did not pass.
  GCN episode 10 improved on frozen pretrain by `258348` cost units
  (`0.01273%`), but its 95% CI `[-837548, 249208]` crossed zero. At episode
  100 it was `507246` cost units (`0.02498%`) worse than frozen, with 95% CI
  `[-445243, 1538629]`; only one of three seeds was favorable and the patient
  loss noninferiority check failed. The locked classification is
  `close_persistent_shift_ddpg_candidate`; no formal confirmation was launched.
- Frozen-policy context: the same 150 paired persistent-shift rows show that
  frozen GCN pretrain is already `10871216` cost units (`0.53262%`) better than
  MDL-2, with 95% CI `[-14795906, -6489364]` and all three seeds favorable.
  The reduction is dominated by `19.77` fewer patients lost and `2129333`
  lower expiry cost, not by direct transport-cost savings. Online attribution
  is therefore an incremental-gain problem above a strong frozen policy, not
  evidence that the learned routing policy itself is ineffective.
- Development diagnosis: a no-training route-only oracle probe used one frozen
  GCN trajectory for each persistent hotspot, legal candidates MDL-2 and
  specimen residuals `+/-0.05` and `+/-0.10`, discovery CRN `95500000`, and an
  independent five-replication validation CRN `95600000`. Of 156 frozen-policy
  states, 61 (`39.10%`) retained a lower-cost clinically noninferior action on
  independent validation; 49 were specimen-transfer corrections. Those 49
  validated corrections had median future-cost advantage `2718118` and 43/49
  exceeded one million. This is a single-decision development probe, not a
  cumulative episode estimate or publication result, and its temporary outputs
  must be converted into a committed reproducible audit before citation.
- Mechanistic conclusion: the current OU exploration has network-output
  `sigma=0.005`; after the specimen residual scale `0.1`, its initial normalized
  action perturbation is about `0.0005`. The environment's 120-patient routing
  scale produces a one-patient action grid of `1/120 = 0.00833`, while the
  independently validated useful alternatives are `0.05` or `0.10`. The
  correction gate also remains active during exploration. Online replay thus
  receives little behaviorally distinct specimen-action coverage, explaining
  the collapsed critic action slope, very small actor drift, transient episode
  10 gain, and later regression despite genuine route-only headroom.
- Decision: preserve DDPG as the intended primary method, close critic
  realignment and support alignment, and do not tune actor/critic learning rates
  or expose the formal holdout. The only authorized next DDPG mechanism screen
  is support-matched structured specimen-option exploration: during online
  collection, explore the fixed legal set `{MDL-2, +/-0.05, +/-0.10}` for the
  specimen group while retaining the existing DDPG critic, deterministic actor,
  reward, self-imitation rule, gate, residual scale, and all other scientific
  settings.
- Next gate: implement a reproducible frozen-route-headroom audit and the
  single-factor structured-exploration mechanism with unit tests and a
  zero-trajectory MPS preflight. Then lock fresh development seeds and CRNs for
  one paired control/candidate screen. Advancement requires positive
  final-versus-frozen attribution, at least two favorable seeds, clinical
  noninferiority, and no regression relative to the unchanged DDPG control.

### 2026-08-13: Structured specimen exploration Stage C3 locked for execution

- Scientific question: does behaviorally distinct, support-matched online
  coverage let the existing DDPG critic and actor learn incremental routing
  value that ordinary sub-grid OU exploration misses? This is a mechanism
  screen above the already strong frozen DDPG policy, not a new deployment
  policy, geography calibration, or hyperparameter sweep.
- Before training, rerun the committed frozen-route-headroom audit on the
  immutable Stage C2 control pretrain checkpoints for GCN seeds 40-42. It must
  reproduce 156 persistent-hotspot states with disjoint discovery and
  validation streams 95500000 and 95600000 and retain at least one independently
  validated clinically noninferior specimen correction. These rows are
  mechanistic development evidence only; state-level advantages must never be
  summed into an episode or publication performance claim.
- Fresh paired training uses seeds 50, 51, and 52, assigned once to persistent
  hotspot clusters 1, 2, and 3. Six zero-trajectory episode-0 states are
  created, then metadata-only cloned for GCN and matched-flat control/candidate
  arms. Actor, critic, optimizers, replay, environment, ordinary OU state, all
  RNG state, reward, self-imitation, gate, residual scale, update schedule, and
  every other scientific field remain identical at the fork.
- The only arm difference is
  `residual_action.structured_exploration.enabled`. The control leaves it off.
  During candidate online data collection, after actor output, ordinary OU,
  correction gating, and policy projection, each decision has fixed probability
  0.20 of replacing only the specimen-transfer slice by one uniformly sampled
  legal option from `{MDL-2, -0.05, +0.05, -0.10, +0.10}` around the current
  MDL-2 anchor. Thus the preregistered expected rates are 4% anchor replacement
  and 16% non-anchor specimen correction. Other action groups are untouched,
  and evaluation always uses the deterministic frozen checkpoint policy with
  structured exploration disabled by `explore=False`.
- Train controls first and candidates second, strictly serial, for 100 online
  episodes each: 12 runs, 1,200 online episodes, policy and atomic state every
  five episodes. Require MPS with fallback disabled, nonzero routing and online
  updates, finite persisted metrics, GCN/flat parameter gap at most 1%, exact
  explorer RNG resumption, all five options observed in each candidate run, and
  behaviorally distinct specimen actions. The forced-probability two-step CPU
  smoke is execution validation only and is not evidence.
- Evaluate both arms at pretrain, episode 10, 25, 50, 75, and 100. Each of the
  72 fixed checkpoint runs uses its seed-assigned persistent hotspot and 50
  paired development replications, producing 3,600 learned and 3,600 MDL-2
  anchor rows per arm. Seed 95700000 with one replication is execution-only;
  seed 95800000 is the development evaluation stream; bootstrap seed 95850000
  is fixed. Formal holdout 91100000 and every prior training/evaluation stream
  remain forbidden.
- Candidate advancement requires final GCN cost improvement versus its own
  frozen pretrain of at least 0.02% or a paired 95% cost CI wholly below zero,
  at least two of three favorable training seeds, all preregistered clinical
  noninferiority checks, and no mean-cost or clinical regression versus the
  unchanged fresh-seed control. The checkpoint curve and adaptation AUC are
  supporting diagnostics, never checkpoint-selection devices. Failure closes
  this structured-exploration candidate; success authorizes a separately
  reviewed fresh confirmation only. No formal confirmation launches
  automatically.
- Implementation gate: unit and integration tests, full relevant regression
  tests, compile-all, reproducible asset hashes, clean-worktree preflight, and a
  zero-trajectory host MPS probe must all pass before the single detached run is
  launched. Howard's TD3 campaign and all prior support-alignment and Stage C2
  evidence remain untouched and must not be pooled into this comparison.

### 2026-08-15: Howard matched TD3 Stage C completed; retain as backbone ablation

- Recovery execution commit:
  `8373f1bf4cf509c9ac3189753b1759e1f7adaa0a`. The original executor completed
  all six training runs but failed its post-training CSV reader at Python's
  default 131,072-byte field limit. Recovery raised the supported field limit,
  reused the immutable training root, and reran only focused tests and fresh
  final/pretrain evaluation.
- Completed evidence: six runs, 600 episodes, 120 five-episode checkpoints,
  31,200 critic updates, 15,600 delayed actor updates, 2,400 final plus 2,400
  frozen-pretrain learned rows, and matching MDL-2 anchor rows. Recovery status
  SHA256 is
  `2d0c87a09b63478bcbf94181bf210aaf9e454d649a57c9ce719c7056ad8c05e3`;
  the 245-file inventory SHA256 is
  `1b7afaba19d5b1f8a668caffc0cf7b3ef5885f466a7bdad4fa9dcfd5c6247042`.
- Result: final GCN-TD3 improved total cost versus MDL-2 by approximately
  `0.735%`, with the 95% interval wholly favorable, and improved on matched flat
  TD3 by approximately `0.414%`. Frozen-pretrain GCN was approximately `0.736%`
  better than MDL-2. Final-minus-frozen GCN was approximately `+0.0008%` worse,
  with its paired interval crossing zero; graph-specific online difference-in-
  differences also crossed zero.
- Decision: TD3 corroborates graph and anchor value but does not establish
  online-learning attribution. Retain it as a development-only controlled
  backbone ablation. Do not promote it to formal confirmation or replace the
  completed DDPG primary protocol.
- Curated evidence:
  `experiments/evidence/patient_indexed_specimen_routing_stage_c_td3/`.

### 2026-08-17: Structured exploration Stage C3 closed; enter Stage E

- Execution commit: `ac313052b702f4956ed725daf400b81de1f17e91` on
  `rl-attribution-refinement`. The campaign completed the reproducible
  156-state headroom audit, 12/12 training runs, 1,200/1,200 online episodes,
  240 policy checkpoints, 12 atomic states, and both 36-run checkpoint curves
  with 1,800 learned and 1,800 anchor rows per role.
- Mechanism validation: all candidate runs observed every legal exploration
  option. Candidate selection rates were approximately 19%, and approximately
  17% of decisions were behaviorally distinct. The headroom audit independently
  validated 58/156 lower-cost clinically noninferior local actions, including
  51 specimen corrections. Exploration coverage and local action headroom were
  therefore present.
- Result: GCN candidate final-minus-frozen cost was `+84,712` (`+0.00416%`),
  with paired 95% CI `[-563,653, +833,767]`. Two of three seeds were favorable
  and clinical noninferiority passed, but the effect gate failed. Candidate
  minus unchanged GCN control was `-18,204` (`-0.00089%`), with paired 95% CI
  `[-278,652, +271,812]`. The locked classification is
  `close_structured_exploration_ddpg_candidate`.
- Immutable compact evidence: comparison SHA256
  `30d1a39785a8e4bc9f17d68e6cf615f4e29ab5b2a64284e1105601bc87fecda4`,
  status SHA256
  `b42cd665a0d8a8eb012a2cc26e1d513fc1d0700141c564663324e76156bd4eae`,
  and 828-file inventory SHA256
  `f07f69c82ce37bdc5f687389f80c5860c2c9df4483db190cff9bded499001834`.
- Decision: no revised candidate enters Stage D, no new formal stream is
  opened, and open-ended algorithm search ends. Retain the completed five-seed
  DDPG result as primary, report online attribution as not established, and
  proceed to Stage E manuscript and reproducibility freeze.
- Curated evidence:
  `experiments/evidence/patient_indexed_specimen_routing_ddpg_structured_exploration/`.

### 2026-08-17: Stage E publication evidence package assembled

- Integration branch: `stage-e-publication-freeze`, based on
  `patient-indexed-specimen-routing` with the completed
  `rl-attribution-refinement` branch and Howard's Stage C TD3 evidence commits.
- Primary formal summary SHA256:
  `7f762fd1ab16f95e908b56c3925a0f6bed2bf68f12fa6fd609c13dbb18b6f3a4`.
  Recomputed component-summary SHA256:
  `3f72c363ef75e84fcb220947799e273a8a804b3777ab5e16ec51196a62ac0c31`.
  Publication evidence-map SHA256:
  `efca2fc354a7667326f0e90ecf90aa75cb9c88510f34910703ee9ae93cd748b4`.
- Component audit: 2,000 learned and 2,000 anchor formal rows reconciled across
  five training seeds and four scenarios. Every row satisfied
  `total_cost = base_cost + patient_loss_cost + expiry_cost + urgency_cost`,
  and every base-cost row reconciled to its operating subcomponents.
- Reporting correction: the additive GCN-minus-MDL-2 decomposition is base
  operating cost `-2.100M`, patient-loss cost `-12.631M`, expiry cost
  `-2.806M`, and urgency cost `-0.152M`, totaling `-17.690M`.
  Specimen-transfer cost `+0.174M` is reported within base cost, not added
  separately.
- Decision: retain AFR-GCN-DDPG as the formal primary method; retain TD3 as a
  development-only backbone ablation; state that online learning attribution
  is not established and transport-timing robustness is asymmetric. No new
  experiment or formal holdout use is authorized.
- Artifacts: `docs/patient_indexed_specimen_routing_stage_e_evidence_synthesis.md`,
  `reports/patient_indexed_specimen_routing/`, the routing-primary cost-effect
  and cost-component figures, and
  `experiments/evidence/patient_indexed_specimen_routing_publication_evidence_map.json`.
- Next gate: complete tests and manuscript compilation, review the Stage E PR,
  and merge only after the evidence and paper diff are accepted.

### 2026-08-17: Stage F0 online DDPG identifiability audit opened

- Stage: post-freeze read-only target, critic-ranking, and quantized-gradient
  diagnosis. No new training or formal evaluation is authorized.
- Source branch: `codex/stage-f-ddpg-identifiability-audit`, based on Stage E
  evidence commit `c3b8d8a53e09b9208d86757bab3a87fd3e286a01`.
- Immutable inputs: Stage C3 828-file inventory SHA256
  `f07f69c82ce37bdc5f687389f80c5860c2c9df4483db190cff9bded499001834`
  and comparison SHA256
  `30d1a39785a8e4bc9f17d68e6cf615f4e29ab5b2a64284e1105601bc87fecda4`.
- Fresh diagnostic streams: trajectory base 95900000 and paired-rollout base
  96200000. Formal 91100000 and all previous streams remain forbidden.
- Locked question: determine whether short-horizon target sign, critic ranking
  on executed legal specimen actions, or the straight-through action gradient
  provides a reproducible mechanism linked to final-versus-frozen degradation.
- Next gate: complete the full three-seed audit. Only a passing prespecified F0
  gate can justify design review for one direct counterfactual-advantage critic
  candidate; F1 remains unauthorized.

### 2026-08-17: Stage F0 legal-action ranking branch passed

- Stage: completed read-only audit of six control replay buffers and the GCN
  legal specimen-action manifold at pretrain, episode 25, and final.
- Replay integrity: 31,200 online transitions were finite, and every persisted
  n-step target and discount multiplier reconstructed within `1e-6`. Among the
  positive one-step plus positive four-step GCN self-imitation samples, only
  0.74%, 1.05%, and 0.39% had a negative behavior-trajectory remainder for
  seeds 50-52. The temporal-sign branch did not pass.
- Ranking result: remaining-horizon critic pairwise accuracy declined from
  0.575/0.575/0.650 at pretrain to 0.475/0.4625/0.5125 at final. Final critic
  versus rollout Spearman correlation was 0.101/-0.012/-0.021. All five legal
  executed actions remained distinct, and at least 55.6% of frozen states in
  every seed retained material headroom.
- Behavioral link: all three seeds passed the legal-ranking mismatch threshold;
  seed 51 also had locked final-minus-frozen cost degradation of +715,787.
- Immutable F0 evidence: config SHA256
  `2b7d44556cf10bb0632b6ccb26ddea8d89b42bf3c6e060f6893d4194da23248c`,
  summary SHA256
  `1ed22e07109a8fa93e0130ee527cedffd7bb832c32a42cb22b9940c4a6cc4008`,
  and 1,215-row manifold SHA256
  `980b547034ba8999bb233401a1b5ab01e0db9a21aced74fda02e049c75ff5094`.
- Decision: the legal-action ranking branch passes the Stage F0 design gate.
  Stage F1 training remains unauthorized. Freeze and review exactly one paired
  direct counterfactual-advantage critic protocol; do not change actor,
  exploration, gate, scale, scenario distribution, or episode budget.
- Detailed report:
  `docs/patient_indexed_specimen_routing_stage_f0_results.md`.

### 2026-08-17: Stage F1 single paired-advantage candidate frozen

- Scientific question: can direct within-state supervision of the online DDPG
  critic convert already verified legal-action coverage into a favorable final
  policy relative to its own frozen pretraining checkpoint?
- Both arms share the locked Stage C3 structured specimen exploration support.
  The only episode-0 fork difference is
  `online_paired_advantage_critic.enabled`. The candidate fits
  `Q(s,a_behavior)-Q(s,a_MDL2)` to an exact-CRN four-step return difference;
  both branches use MDL-2 after the first action. The fixed loss weight is 3.0,
  matching the existing online teacher-ranking weight and not selected from a
  result.
- Fresh development seeds 60-62 map to persistent hotspot clusters 1-3. The
  execution-only, development, and bootstrap streams are 96600000, 96700000,
  and 96750000. Formal holdout 91100000 and all prior streams remain forbidden.
- Primary advancement requires a pooled GCN final-minus-frozen paired 95% cost
  interval wholly below zero, at least two favorable seeds, clinical
  noninferiority, and no regression versus the tensor-matched control. Curves,
  candidate-minus-control, and flat difference-in-differences are supporting
  diagnostics only.
- Full training remains gated on focused tests, CPU smoke, zero-trajectory MPS
  construction, exact episode-0 clone audit, and locked hashes. No formal
  confirmation launches automatically.
- Protocol:
  `docs/patient_indexed_specimen_routing_stage_f1_protocol.md`.
- Implementation status: the replay schema, finite-horizon paired rollout,
  GCN/flat critic loss, strict episode-0 fork, prospective comparator, and
  serial executor are implemented on the isolated Stage F branch. The real
  ten-step candidate smoke produced six finite paired targets, a nonzero
  sampled paired loss, zero cloned first-step reward error, and no persisted
  non-finite metrics. The matched control smoke is part of the final locked
  preflight and must persist zero paired targets.

### 2026-08-17: Stage F1 preflight recovery 1

- Execution commit `a8d5e9595edad27c8ddd0683824ba984f95a3de0` stopped during
  the host focused-test gate, before claim creation, smoke, pre-online state
  preparation, or any training job.
- Cause: one legacy unit-test fixture constructed CPU tensors while its agent
  auto-selected an available MPS device. This was a test-device mismatch, not
  an algorithm, numerical, fallback, or scientific-contract failure.
- Recovery scope: preserve the original launcher logs, make that fixture
  explicitly CPU-bound, refresh the locked hash manifest, and rerun the entire
  preflight under a new recovery launcher root. No scientific configuration,
  seed, option, loss, scenario, or budget may change.

### 2026-08-17: Stage F1 post-training audit recovery 2 authorized

- Recovery 1 completed all 12 control/candidate training jobs, 1,200/1,200
  online episodes, 240 policy checkpoints, and 12 full states. Routing and
  online updates were nonzero, persisted metrics and paired targets were
  finite, cloned first-step reward error was zero, and the GCN/flat parameter
  gap was 0.970%. No checkpoint-curve evaluation or comparison was launched.
- The terminal error `Candidate paired distinct count mismatch` was caused by
  a post-training audit comparing two different valid counters. The structured
  explorer counter covers only explorer-selected behavior changes; the paired
  critic counter covers every executed action distinct from MDL-2, including
  ordinary policy/OU residuals. Across the six candidate runs, the latter
  exactly matched the persisted paired replay-target count.
- Corrected lock: paired contexts equal all online decisions, paired distinct
  actions equal replay paired targets, and structured-explorer distinct actions
  remain a separate support diagnostic. A regression fixture deliberately
  makes these counts unequal so the old audit cannot return.
- Recovery 2 is evaluation-only. It must hash-lock the immutable Recovery 1
  status, claim, logs, manifests, pre-online states, paired forks, and both
  complete training trees before running the two prespecified 36-run curves
  and comparator. Its phase whitelist contains no preparation, training, or
  resume command. Formal confirmation remains unauthorized.

### 2026-08-18: Stage F1 completed and closed

- Recovery 2 completed both prespecified 36-run curves using the immutable F1
  training trees: 72 evaluation runs, 3,600 learned rows, and 3,600 MDL-2
  anchor rows. No new training or formal confirmation was launched.
- Candidate GCN final minus frozen pretrain was `-21,297.91`
  (`-0.001067%`), with paired 95% CI `[-715,370.98, +527,374.97]`; only one
  of three seeds was favorable. Candidate minus matched control was
  `+4,080.57` (`+0.000205%`), with paired 95% CI
  `[-95,056.15, +95,036.33]` and 131/150 exact ties.
- Clinical noninferiority passed, but both the strict online-gain gate and the
  no-regression-versus-control condition failed. The locked classification is
  `close_online_ddpg_attribution_extension`.
- Comparison SHA256:
  `8f4967115c584900969a7dba93e29da5ab70dfc65caa27d81423805be85c010a`.
  Recovery 2 status SHA256:
  `0097b4c84f9f9ab2403eccd670543b640109657eb03a86764d423c5a0496d380`.
- Detailed report:
  `docs/patient_indexed_specimen_routing_stage_f1_results.md`.

### 2026-08-18: Stage G0 actor-projection transfer audit opened

- Stage G0 is the final read-only mechanism gate before considering any more
  online DDPG development. No new training, tuning, checkpoint selection, or
  formal stream is authorized.
- It evaluates immutable F1 control/candidate GCN checkpoints on 27 fixed
  frozen-pretrain trajectory states and five legal specimen actions using
  fresh CRNs. It separately measures legal-action critic ranking,
  straight-through gradient alignment, raw actor movement, and survival through
  projection and integer-lot quantization.
- Only a prespecified critic-to-gradient or actor-to-execution transfer failure
  can justify review of one action-aligned DDPG design. Failure or ambiguity
  ends the extension and preserves the Stage E conclusion that online learning
  attribution is not established.
- Protocol:
  `docs/patient_indexed_specimen_routing_stage_g0_protocol.md`.
