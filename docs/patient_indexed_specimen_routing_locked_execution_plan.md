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

### 2026-08-18: Stage G0 completed; actor redesign not authorized

- The locked audit completed 27 fixed states, 270 independently valued legal
  action rows, and 1,620 checkpoint-transfer rows. All rows were unique and
  finite, all five actions remained behaviorally distinct, and immutable F1
  training-tree hashes matched before and after execution.
- Remaining-horizon material headroom was present in 17/27 states, but the
  candidate final critic selected the best legal action in only 3/27 states,
  versus 2/27 for control. The locked classification is
  `paired_critic_did_not_generalize` and the action-aligned actor design gate
  did not pass.
- Candidate raw actor output differed at all 27 states, yet only 2/27 executed
  integer-lot actions changed and 25/27 raw changes collapsed. This confirms a
  downstream projection bottleneck but does not supersede the failed critic
  gate.
- Four-step and remaining-horizon best actions agreed in only 14/27 states.
  The next permissible work is a critic-only design review for multi-horizon,
  within-state-normalized, pairwise legal-action ranking. No online training or
  formal confirmation is authorized.
- Summary SHA256:
  `738bf04c8f5716e284a2f3006fe0a102516e83ae81dc1b87bbd37c1513f82524`.
  Detailed result:
  `docs/patient_indexed_specimen_routing_stage_g0_results.md`.
  Design review:
  `docs/patient_indexed_specimen_routing_stage_g1_design_review.md`.

### 2026-08-18: Stage G1 offline critic feasibility gate authorized

- User authorization is limited to a development-only, leave-one-seed-out
  critic audit. No online episode, actor update, formal stream, or fitted-model
  deployment is authorized.
- The audit expands the fixed diagnostic manifold to all 156 frozen-pretrain
  states and evaluates all five executed legal specimen actions under separate
  discovery and validation CRNs.
- Each fold initializes the held-out seed's frozen-pretrain critic, fits only on
  the other two seeds, and evaluates on the full unseen seed. The single locked
  objective combines within-state normalized remaining-horizon regression with
  pairwise legal-action margin ranking. No hyperparameter or epoch selection is
  permitted.
- A full prospective ranker gate must pass before an actor-transfer smoke can
  even be designed. Online training remains unauthorized.
- Protocol:
  `docs/patient_indexed_specimen_routing_stage_g1_protocol.md`.

### 2026-08-18: Stage G1 completed; DDPG attribution extension closed

- The full audit completed on implementation commit `bffd2ae`: 156 frozen
  states, 1,560 legal-action label rows, and 1,560 leave-one-seed-out prediction
  rows. All rows and arrays were unique where required and finite, every state
  retained five distinct executed actions, and actor hashes were unchanged.
- Discovery and validation agreed on the remaining-horizon best action in only
  85/156 states (`54.49%`), below the locked `70%` label-stability gate.
- The fitted critic reached 47/156 (`30.13%`) validation top-1 accuracy and
  `57.74%` pairwise accuracy, below the `40%` and `65%` gates. Per-seed top-1
  was `34.62%`, `26.92%`, and `28.85%`; no seed improved by the required five
  percentage points over its frozen critic.
- The locked classification is
  `unstable_counterfactual_labels_close_extension`. Actor-transfer smoke,
  online training, and formal confirmation are all unauthorized. The manuscript
  must not attribute endpoint improvement to online DDPG updates.
- Summary SHA256:
  `9e96deec850d6033fb5a2fb9e2f5090f8a040468df9faf8634015a14875d1c41`.
  Detailed report:
  `docs/patient_indexed_specimen_routing_stage_g1_results.md`.

### 2026-08-29: Continuous overtime control follow-up study opened (new study, Stage E0 complete)

- Howard and Zhaowei approved the step-0 specification
  `specs/2026-08-29-continuous-overtime-control/` (sign-offs recorded in its
  README). This opens a NEW STUDY motivated by the closed online-attribution
  extension; it does not amend, reopen, or reinterpret any routing-primary
  stage, result, or the formal holdout, all of which remain closed at Stage E.
- Authorized scope: flag-gated environment implementation (Stage E1: overtime
  capacity control behind `enable_overtime_control`, dormant
  `enable_production_throttle` rejected at validation), extended overtime
  heuristic comparators, and the three evaluation-only screens E2 (headroom),
  E3 (label stability), E4 (critic ranking) with the prospective gates and
  one-round remediations fixed in that spec's plan.md.
- Not authorized: any RL or distillation training on the new channel (Stage E5
  requires its own specification and sign-off), any reuse of the routing-primary
  formal holdout streams, and any change to existing result roots, checkpoints,
  teacher artifacts, or CRN streams.
- Mandatory invariant: with the new flags disabled, environment behavior is
  bit-identical to the current environment under fixed seeds, enforced by a
  committed regression test before any screen runs.

### 2026-08-29: Stage E2 completed; gate amendment proposed, E3 not authorized

- The overtime headroom screen ran on implementation commit `9effc17`
  (2,592 rows, 27 states x 12 arms x 8 CRN replications). All rows were
  unique, finite, count-checked, and passed a live-environment provenance
  assertion.
- The literal E2 gate PASSED (`overtime_headroom_established`): both
  non-nominal scenarios cleared the prospective 30% state fraction
  (abrupt shift 1.00, compound stress 0.33; nominal 0.89).
- The screen nevertheless produced a disqualifying diagnostic. The optimum is
  a corner solution: `u_1.00` is best in 26/27 states and mean cost falls
  monotonically to the ladder boundary. Marginal overtime cost never exceeds
  30,000 against a 50,274 shortage benefit and a 500,000 patient-loss
  benefit, so the optimum is always the upper bound.
- A follow-up analysis of the same rows measured the quantity the gate failed
  to ask about. Overtime is worth 303.2M over the anchor, but the best single
  constant rung captures essentially all of it. Selecting per-state arms
  PROSPECTIVELY (chosen on discovery, scored on held-out validation, with the
  constant chosen the same way) is **worse than the constant by 10.20M, i.e.
  -0.083% of anchor cost**. An in-sample oracle reads +0.0054%, but it selects
  on the stream it is scored on and so capitalizes on replication noise; the
  honest out-of-sample value is negative. A state-dependent policy fitted to
  real data on this channel loses to a one-line constant, so there is no
  budget for any learned policy to capture.
- Classification of the channel as configured:
  `channel_captured_by_constant_policy`. **Stage E3 is not authorized.**
- No re-tune or re-run was performed. Re-running after observing a result
  requires this change-control entry, a new config name, and a new output
  root.
- Amendment APPROVED by Howard 2026-08-29 (Zhaowei's review still pending;
  recorded here so the approval trail is exact). Makes the PROSPECTIVE value
  of state-dependence the PRIMARY E2 criterion (threshold 0.005 of anchor
  cost, ~100x the noise floor, plus interior-best-arm fraction 0.30),
  demoting the headroom criterion to a necessary precondition; only then
  re-calibrate the overtime cost weights and re-run. If no calibration clears
  the amended gate, reject the overtime channel and report that.
- Measure and gate implemented read-only in
  `evaluation/state_dependence_value.py` (15 tests). It consumes rows already
  collected and trains nothing. The gate refuses any report lacking a
  prospective value rather than falling back to the optimistic in-sample
  figure; a regression test drives pure noise through both and asserts the
  oracle looks spuriously positive while the prospective value does not.
- Evidence: `specs/2026-08-29-continuous-overtime-control/results.md`,
  `results/continuous_overtime_headroom_e2/`,
  `experiments/evidence/continuous_overtime_headroom_e2/`
  (summary carries config/plan/rows SHA256; `state_dependence.json` carries
  the report and gate decision).

### 2026-09-28: Separate service-effort mechanics fixture, no campaign reopening

- Zhaowei requested continuation of the local online-adaptation roadmap. The
  next bounded packet is `specs/2026-09-28-service-effort-mechanics/protocol.md`:
  deterministic software fixtures for delayed continuous effort, conserved
  service work and a receipt-only response estimator.
- This module is isolated from patient production and is not a new scientific
  routing experiment. It uses no patient trajectories, teacher/checkpoint,
  scientific CRNs, neural training or formal confirmation. Howard's sign-off
  is not asserted. The protocol and source are committed before its recorded
  fixture execution; prior campaign evidence remains immutable.
- The completed September 28 disruption pilot did not test unknown resource
  effectiveness. The fixture does not establish engineering calibration,
  sequential headroom, clinical noninferiority, graph benefit or online RL
  gain. A scientifically justified process/measurement contract is required
  before integrating any new mechanism into patient simulation.

### 2026-09-29: Service-effort comparator software, no scientific reopening

- Continued local progress adds the deterministic decision fixture specified
  in `specs/2026-09-29-service-effort-decisions/protocol.md`: fixed and adaptive
  rules, fixed-model and identification MPC, plus a same-grid exact diagnostic.
- This is generic service-work software validation, not a patient experiment,
  a new clinical headroom claim, or permission to train DDPG. Source/config and
  terminal closure are committed before the recorded run. All signs of the
  adaptive-baseline comparison are retained.
- Literature supports examining personnel/resource scheduling, but does not
  calibrate the synthetic response model. Domain measurement decisions and
  collaborator sign-offs are not fabricated. Formal Stage E remains closed;
  old source/evidence, scientific CRNs, checkpoints and teachers are untouched.

### 2026-09-29: Bounded support-queue mechanism check, local only

- Zhaowei explicitly requested continued stepwise work without waiting for
  routine local confirmations. The bounded scope is now documented in
  `specs/2026-09-29-service-queue-boundary/protocol.md`: booked arrivals and a
  mandatory downstream workstation, eight predeclared synthetic cells, five
  common-information non-neural controllers, and finite-grid diagnostics.
- This is isolated generic queue software and a synthetic mechanism boundary
  check. Divisible support work, productivity and measurement assumptions are
  not clinically calibrated. No integration into patient dynamics, DDPG launch,
  formal confirmation, remote push, main merge or Howard approval is inferred.
- Commit the source/config/protocol before recorded comparisons. Retain all
  cells and negative outcomes without post-result retuning. Finish and charge
  every booked job and prepaid commitment under a common declared continuation;
  do not improve the apparent objective by dropping terminal liabilities.
- Preserve the prior service-effort result where best open loop equaled optimal
  feedback. Any new feedback benefit is not evidence of online weight-update
  value: the exact nonanticipative policy itself is a frozen history-action map.

### 2026-09-29: Existing formal-row recovery and reporting sensitivity

- Continued local Stage E reporting work located the original final/pretrain
  CSVs in the persistent patient-routing worktree. Existing compact evidence
  verifies both summary hashes and ten GCN-final CSV hashes. The other thirty
  CSVs require current hashes and reconciliation to all historical cell means;
  historical byte-level provenance is not asserted for those files.
- The post-hoc reporting protocol is
  `specs/2026-09-29-formal-crossed-audit/protocol.md`. It uses Howard's unchanged
  crossed-bootstrap tool to audit the five existing total-cost contrasts. It
  does not run new evaluation, choose policies/scenarios, tune on the holdout,
  change clinical margins, or replace the original primary statistical method.
- Preserve every old file. A new local cost-only projection and dated analysis
  are additive reproducibility artifacts. Formal Stage E remains closed; no
  new scientific launch, external publication, or collaborator approval is
  implied by this reporting task.

### 2026-09-29: Approved bounded prospective collector engineering

- Following the N1/N2/N3 contracts and 154 passing tests, Zhaowei answered
  "continue" to the explicit request for default-off real collector/agent
  integration and small engineering verification. The bounded protocol is
  `specs/2026-09-29-prospective-collector-engineering/protocol.md`.
- This opens tiny real-environment mechanics and frozen neural forwards only,
  with zero optimizer updates. Protocol/config are committed before execution;
  source is committed before the recorded smoke. Preserve all historical files
  and defaults; do not use a teacher, historical checkpoint or scientific CRN.
- This does not authorize a patient scenario campaign, performance evaluation,
  formal confirmation, remote actions or imply Howard's sign-off. Raw-observation
  equality and return plumbing are engineering acceptance, not proof of RL gain.

### 2026-09-29: Prospective design/count preflight after N4

- N4's 36 real engineering steps, repeated report and 171 tests passed, without
  updates or performance claims. Zhaowei requested continuation of the next
  proposed step: specify the small development design before training.
- `specs/2026-09-29-prospective-development-design/protocol.md` fixes an
  outcome-independent component-count search and separates online-weight,
  message-passing and representation contrasts. Run only frozen module/shape
  checks on the locked N4 schema; commit source/config before recording them.
- No environment rollout, optimizer update, scientific seed allocation or
  launch is authorized by this preflight. Scenario/calibration/closure and
  scientific launch evidence remain unfilled, not assumed. Old results,
  checkpoints, teacher, streams and Howard's work remain unchanged.

### 2026-09-29: Bounded prospective update/resume engineering after N5

- N5 completed component-count/operator checks and a non-executable development
  design. Zhaowei explicitly requested its next named task: actual DDPG update
  and checkpoint/resume mechanics, documented in
  `specs/2026-09-29-prospective-learner-engineering/protocol.md`.
- Permit the declared tiny optimizer checks on invented numeric records only.
  Hash-lock prior schema/count evidence, commit protocol/config and source
  before recorded execution, and preserve every old result and default.
- This does not reopen Stage E or permit environment training, performance
  evaluation, formal data use, remote operations or a claim of online gain.
  Kernel-boundary resumption is distinct from full environment/collector resume.

### 2026-09-29: Bounded live closed-loop/resume engineering after N6

- N6 passed exact synthetic CPU update-boundary recovery. Zhaowei requested the
  next named small real-collector closed-loop and complete trajectory-resume check.
  Protocol: `specs/2026-09-29-prospective-closed-loop-engineering/protocol.md`.
- Permit only its six bounded CPU cases, existing unit fixture and finite
  optimizer updates, with a fresh-process interrupted path and one repeat.
  Commit config/protocol first and implementation before recorded execution.
- This is engineering acceptance, not a scientific gain screen or new scenario.
  Preserve old evidence/defaults; do not use formal streams, tune outcomes,
  infer clinical terminal adequacy or Howard approval, or perform remote actions.

### 2026-09-29: Existing-row queue cost sensitivity after N7

- Following N7, Zhaowei requested the next step. Review the solved synthetic
  queue's dependence on cost ratios with the additive post-hoc protocol in
  `specs/2026-09-29-queue-cost-sensitivity/protocol.md`, without new simulation,
  training, evaluation trajectories, scenario selection or reward tuning.
- Reprice all recorded decision and closure costs over nine declared positive
  multiplier pairs. Recompute finite-tree bounds and unchanged archived paths;
  never describe an archived MPC path as reoptimized for another objective.
- Preserve every original file and Stage E. This is not a scientific launch,
  calibrated patient evidence, online-RL benefit or collaborator approval.
  Commit protocol/config and source before recording this local analysis.

### 2026-09-29: Completion-feedback synthetic mechanism after cost sensitivity

- The cost sensitivity is complete: 66/72 dependent records still have an
  optimal archived non-neural path; no online-RL opportunity was established.
  Zhaowei requested continuation of the next design. Scope is the isolated
  hypothetical mechanism and bounded checks in
  `specs/2026-09-29-completion-feedback-mechanics/protocol.md`.
- Add noisy completion-only service, recurrent booked work, fixed downstream
  obligations and a shared receipt-only likelihood filter. Freeze source/config
  before the 16-case mechanics matrix and its single repeat. No neural update,
  policy performance comparison or patient-simulator amendment is authorized.
- Keep all old evidence and defaults, E1's missing domain data, and Stage E
  unchanged. This does not assert clinical realism, Howard sign-off, approval
  for a calibrated study, headroom or any guaranteed benefit/publication result.

### 2026-09-29: S2 bounded conventional-control screen

- S1 mechanics and positive/failed closure fixtures passed; no learning benefit
  or safe headroom was established. Zhaowei requested continuation of the named
  conventional-control comparison. The local synthetic scope is fixed in
  `specs/2026-09-29-completion-control-screen/protocol.md`.
- Preserve S1. Use fresh namespaced development worlds with unchanged, persistent
  and fast-independent response families, equal information/actions and full
  settlement. Compare fixed public rules and identical fixed/identified rollout
  planners at two budgets; add preselected depth checks and retain every result.
- Commit protocol/config and source before the finite 288-episode screen. Stop
  at its declared wall/closure caps without automatic retuning or relaunch.
  No neural campaign, old holdout use, source-default change, calibrated
  manufacturing claim, remote action or collaborator approval is inferred.

### 2026-09-29: S2 completion and decision

- Source commit `b0b3e315a629b9fca44e742120075981d5187c01` completed all 288
  declared episodes and six probes with zero retries; all 9,364 scalar intervals,
  full settlement, 60 paired contrast records and 29 file locks reverify.
  Readout: `specs/2026-09-29-completion-control-screen/readout.md`.
- Four of six preselected probes change first request with forecast budget.
  Identification-plus-rollout does not show replicated persistent-shift gain;
  the favorable slots4 discovery mean reverses on the fixed replication split.
- Do not launch DDPG or claim certified frozen-policy headroom. Any next
  independent action-ranking/continuation study needs a separate committed
  protocol. Preserve this completed evidence, prior results and closed Stage E.

### 2026-09-29: S3 and automatic routine continuation

- After the audited S2 readout, Zhaowei explicitly asked to continue the named
  independent action-ranking check and avoid manually prompting every routine
  step. Scope: `specs/2026-09-29-completion-action-ranking/protocol.md`.
- Authorize implementation/tests, one bounded conditional-model forecast
  diagnostic on all six archived probes (four unique public contexts), audit,
  readout and a continuation-adequacy decision memo. All planned sample budgets
  and independent blocks are fixed before execution; no outcome-based extension.
- Commit protocol/config then source before recording. Preserve S1/S2 and
  earlier results, disclose conditional model assumptions and duplicated probes.
  A same-task scheduled continuation may carry routine stages forward without
  asking again, updating the local queue with evidence and next actions.
- This does not authorize a neural campaign, arbitrary new scenario/reward
  changes, formal holdout use, remote writes or Howard sign-off. Stop at a
  terminal failure or the declared decision boundary rather than relaunching
  or manufacturing new experimental scope to keep an automation running.

### 2026-09-29: S3 audited completion and scope boundary

- Source `a6d450906f57d1fac1718562ba99145b175e8af7` completed the one locked
  forecast matrix: four unique public contexts, four independent blocks,
  491,520 model paths and zero new actual episodes/optimizer updates. All raw
  cost hashes, 5,040 paired contrasts and 960 bounded scalar replays verify;
  140 relevant tests pass. No recorded failure, retry or protocol change.
- Readout: `specs/2026-09-29-completion-action-ranking/readout.md`. Two unchanged
  contexts agree on the simple booked rule. Both changed public contexts have
  block argmin disagreement; the one non-rule selection reverses its advantage
  across validation blocks. No stable increment over the rule or online-RL
  benefit is established by this conditional-model exercise.
- The finite S3 chain is complete. `decision.md` proposes, but does not authorize,
  a bounded feedback-continuation comparison. Remove this continuation
  automation at handoff. New scope needs explicit approval and a separately
  committed protocol; preserve S3/S2/S1, missing E1 data and closed Stage E.

### 2026-09-29: Approved S4 continuation comparison and reward diagnosis

- After S3's completed audit and the named continuation-comparison proposal,
  Zhaowei requested continuation and consideration of the reward function if
  progress remains absent. Scope is fixed in
  `specs/2026-09-29-completion-continuation-comparison/protocol.md`.
- Authorize that finite model-only implementation/test/one-run/audit packet on
  the four archived public contexts, followed by a read-only reward/objective
  diagnosis if no stable conditional signal emerges. Commit protocol/config
  and then source before recording. Preserve all old code and evidence.
- Do not change reward weights, run neural training or new actual/patient
  episodes, expand/retry after outcomes, use formal holdout, perform remote
  operations or imply Howard approval. Stage E remains closed. Reward revision
  is a separate scientific decision, not an automatic response to a null result.

### 2026-09-29: S4 completed; reward diagnosis retained without intervention

- Source `6409afef49b667b7abef146ea117e59b7bb4cc5c` completed the one S4 matrix:
  345,600 inner + 1,966,080 outer model paths, 49,204,865 transition queries,
  exit 0 in 35.638s. Independent arithmetic, all 436 arrays, 321 file locks and
  9,240 bounded scalar paths verify. The 154-test suite and compileall pass.
- No context passes the declared stable conditional-signal check. Tree128
  matches or loses to booked in all validation contexts/blocks; it beats the
  weaker reservation continuation. Inner actions change in 64/180 entries
  between budgets, so no optimality or strong-MPC bound is claimed.
- `specs/2026-09-29-completion-continuation-comparison/reward-diagnosis.md`
  separates historical reward/return inconsistency, missing operational cost
  calibration and untested neural learning signal. Read-only telescoping on
  all 288 archived S2 episodes gives a common +2 reward offset, no ranking
  change and no new simulator queries. No reward or historical result changed.
- The finite S4 packet is closed. Do not train on an unpassed headroom screen,
  tune costs to rescue it, or automatically repeat it. A corrected reward
  intervention/development study needs new scope and a prospective design;
  existing N1-N7 engineering is preparation, not performance evidence.

### 2026-09-29: Continued read-only reward-objective preparation (R1)

- Following S4, Zhaowei requested further progress. Continue read-only reward
  accounting and development-design preparation in
  `specs/2026-09-29-reward-objective-bridge/protocol.md`, not a scientific launch.
- Add an independent JSON-only check of all six N7 case receipts and resumed
  duplicates, including disjoint costs, raw rewards, scaling, lineage, short
  tails and engineering-vs-proposed objective differences. No model or binary
  checkpoint is loaded; source and evidence hashes are preserved.
- Prepare a non-executable decision separating legacy-simulator consistency
  work from the uncalibrated E1 new-task proposal. Neither is authorized to run
  by this entry. Do not change costs, old sources/results, Stage E or holdout.

### 2026-09-29: R1 accounting complete; no new performance result

- Source `541b73d` audited six archived cases / 36 receipts and six resumed
  duplicates; all incurred-cost/raw-reward/lineage/tail checks passed. Nine
  input files and 34 historical source locks remain unchanged. A stdout repeat
  is byte-identical, and separate Decimal recurrence verifies the arithmetic.
- Fifteen new/48 relevant tests and full compileall pass. No environment query,
  optimizer update, source reward change or Q-model reload was performed.
- Readout: `specs/2026-09-29-reward-objective-bridge/readout.md`. N7's mechanics
  objective is not silently promoted to N5's complete-settlement objective.
  `next-study.md` separates fixed-weight legacy-task consistency research from
  the E1-dependent new operational task. Endpoint, data, budget and scientific
  launch remain explicit future decisions, not approved by a successful audit.

### 2026-09-29: Continued archived decision-cost diagnosis (R2)

- Zhaowei accepted the recommended legacy-task direction and asked to see the
  next result. Proceed with existing-evidence arithmetic in
  `specs/2026-09-29-g1-decision-cost/protocol.md`. No new simulation, fitting,
  reward intervention or scientific campaign is authorized by this entry.
- Reconcile the immutable G1 labels/predictions and quantify independent
  validation cost consequences, rather than relying only on top-1 accuracy.
  Respect the published nominal-history reconstruction correction, overlapping
  cost horizons and unavailable replication-level rows. No replacement gate.
- Preserve all old failures, code, costs and decisions. Source/config commit
  precedes the recorded audit; local artifacts only, no remote action.

### 2026-09-29: R2 complete; cost-sensitive diagnostics do not reopen G1

- Auditor `1441c32` reconciles all 1,560 cost / 1,560 prediction rows and NPZ,
  reproducing G1 ranking and label-stability metrics. Independent verifier
  `dd62d6f` checks all 624 choices and cost summaries using Decimal. Seven input
  locks, 14 new/29 combined tests, whole-repository compileall and diff checks
  pass. Both recorded processes exit 0; zero new simulation or fitting.
- `specs/2026-09-29-g1-decision-cost/readout.md` reports favorable descriptive
  validation cost for discovery-cost-selected actions, but unfavorable mean
  cost for both critic selectors relative to MDL-2. Forty of 71 best-action
  disagreements exceed the old 1M reporting threshold. This does not prove
  true headroom, clinical noninferiority, a deployable selector or online gain.
- Keep nominal-history provenance, mean-only/no-CI limits, failed G1/H0/H1
  decisions and closed Stage E. The next prospective endpoint/data/budget/gate
  decision remains separate; no reward-weight change or neural launch follows.

### 2026-09-29: R3 finite-window design and synthetic acceptance checks

- Following the explicit 52-step endpoint question and R2 readout, Zhaowei
  requested continuation. Adopt that fixed-weight endpoint for the bounded
  design in `specs/2026-09-29-fixed-window-value-contract/protocol.md`; this is
  not approval for new trajectories, fitting, or a neural campaign.
- Specify negative absolute step costs, one scale, gamma=1, one-step TD,
  objective-terminal masking and named frozen-policy continuation. Preserve
  the original actor/gate rather than substituting the N4-N7 random prototype.
- Test only invented numeric paths with existing return helpers; no model,
  environment or checkpoint loaded. Proposed 1,152 continuation records /
  37,596 transitions / 3,600 seconds are caps for a future data-only pilot, not
  consumed budget. F1 checkpoint provenance, runtime parity, new streams,
  implementation and explicit launch approval remain outstanding.

### 2026-09-29: R4 replacement-baseline rebuild authorized after missing artifacts

- Zhaowei explicitly permits retraining when original files cannot be located
  and requests Dropbox preservation. The scoped local/archive/Dropbox search
  found no F1 weights. Follow the bounded R4 protocol in
  `specs/2026-09-29-frozen-baseline-rebuild/protocol.md`.
- Rebuild only three GCN pretraining baselines using the original fixed recipe,
  seeds 60-62, locked teacher, zero online episodes, no new performance test.
  Initial heuristic demonstrations are training data, not zero trajectories.
  Preserve current source identity; do not claim recovery of historical bytes.
- Save in the persistent worktree, hash-verify per-seed archives and copies to
  the existing Dropbox project directory. No sharing-permission changes or
  remote Git action. Dropbox copy and cloud verification remain distinct.
- No reward tuning, fresh scientific scenario, formal stream, R3 collection,
  or online-training campaign is opened. Old conclusions and Stage E remain
  unchanged. Commit source/config/protocol before the single bounded attempt.

### 2026-09-29: R4 completed and archived, not a new performance result

- Production execution `1845339` completed the three authorized GCN seeds in
  398.235 seconds, exit 0. Each used 1,040 heuristic demonstration steps,
  300 distillation epochs and 500 offline updates; zero online episodes.
- Three finite actor/gate checkpoints exactly match their full saved states;
  MPS RNG sidecars retained. All three actor-only digests differ from archived
  G1, so the outputs remain replacement baselines, not historical recovery.
- Independent verification checks all members of the 10/17/24-file cumulative
  archives and Dropbox local copies. Final archive SHA256:
  `5959575f4b99ab98dcd12a9b893a942fc25f0872e783b496c628af4fbc7bf731`.
- `specs/2026-09-29-frozen-baseline-rebuild/readout.md` and the companion
  reports preserve provenance and the preflight-verifier failure. No production
  retry occurred. Cloud sync and Howard access are not verified; no sharing
  permissions or remote Git state changed.
- R3 must explicitly adopt replacement baselines before use; its collection
  and online learning remain outside this completed packet. Stage E unchanged.

### 2026-09-29: R5 replacement-policy engineering preparation

- After the archived R4 result, Zhaowei asked to continue progressing. Prepare
  strict actor/gate loading and archived-observation compatibility under
  `specs/2026-09-29-replacement-policy-compatibility/protocol.md`.
- Inspect all three R4 replacements on fixed saved observations only, with
  CPU/full-state and CPU/MPS output checks. No new environment step or optimizer
  update. Preserve R3's historical design and label this proposed substitution
  explicitly; it is not recovered F1 or a new performance result.
- Commit protocol/config/source before the recorded check. Scientific R3
  collection, fresh scenario/reward changes, online training, remote operations
  and formal holdout remain outside scope. Do not infer Howard approval.

### 2026-09-29: R5 reporting repair, no inference retry

- The engineering checker at `f089fd0` saved all seed60 raw outputs, then failed
  only on scalar gate-margin formatting. No seed61/62 inference was attempted.
  Preserve the failure and raw hashes in R5 `summary-failure.md` and the new
  unvisited-seed continuation config.
- Under the ongoing routine engineering authorization, fix/test the scalar
  reducer, reuse seed60's completed inference without replay, and finish only
  seeds61/62 after a new source commit. No extra observation, tolerance change,
  simulation, fitting or scientific retry is allowed by this completion.

### 2026-09-29: R5 saved-observation compatibility complete

- All 48 fixed archived observations across three R4 replacements pass exact
  CPU policy/full-state agreement and prespecified CPU/MPS tolerance checks.
  Hard gate and rounded request decisions match; max normalized request
  difference is 2.981e-8. All 24 input files and 83 existing source locks match.
- The stdlib raw verifier independently reproduces comparisons and request-lot
  arithmetic. Twenty-three new/69 related tests and full compileall pass.
  Failed summary evidence remains, with seed60 reused without new inference.
- This is not actual executed routing, performance, clinical safety, F1 recovery
  or an online benefit. R3 collection needs its distinct explicit approval and
  collector/action/stream acceptance. A question is pending; silence is not
  approval. Preserve all scientific boundaries and closed Stage E.

### 2026-09-29: R3 replacement-baseline data-only pilot approved

- Zhaowei replied "continue" to the explicit 12-state, 8+8-draw, at-most
  37,596-step/one-hour data-only pilot question. This is direct scope approval,
  not silence or a claimed Howard sign-off. Original R3 remains historical.
- Follow `specs/2026-09-29-replacement-fixed-window-pilot/protocol.md` and its
  separate config. R4 replacements are not recovered F1 and their competence
  remains unmeasured. Use unchanged nominal-history dynamics and absolute cost.
- Freeze source, pass collector tests and the distinct 24-step clone/truncation
  engineering smoke, then one scientific attempt. Save seed namespace audit,
  raw costs, actual-action aliases, snapshots and discovery-before-validation
  selections. No fitting, reward change, extra scenario, retry, or formal CRN.
- Independently verify calculations and preserve an archive plus local Dropbox
  checksum receipts. Finish with a decision memo, not automatic online training.
  Historical results, Stage E and remote Git state remain unchanged.

### 2026-09-29: R3 replacement pilot complete; conditional headroom only

- Source `195d0eb` passes12 new/81 related tests and24-step exact-clone smoke.
  The single scientific attempt exits0:12 states,1,152 logical records,
  24,348 steps,600.139s, zero optimizer updates. No retry or expansion.
- Independent stdlib verification reconciles raw costs, chronology,272 safe
  execution aliases, discovery selections and paired validation differences.
  All24 R4 input files and83 prior source hashes remain unchanged.
- Eight of nine noninitial selections lower mean validation cost, but three
  raise mean patient loss and two have uncertified action-map witnesses.
  Three non-anchor rows have favorable cost/clinical directions and certified
  maps. These are conditional descriptive observations, not a trained-policy,
  safety, optimality, or online-RL attribution result. Report all12 rows.
- Readout: `specs/2026-09-29-replacement-fixed-window-pilot/readout.md`.
  The1215-file archive and Dropbox local copies verify; cloud sync and Howard
  access are unverified. Archive SHA256:
  `3e0ca30f1b0db3a5d6dfa7987e5a4e40c27526f9aad529b9483a52c31d5dce43`.
- Stop here. A clean-critic generalization study needs separate prospective
  sampling, clinical rules, compute caps and authorization. Do not fit on these
  states then call new draws on the same states generalization. No reward
  tuning, actor updates, formal holdout, remote Git or reopening Stage E.

### 2026-09-30: R6 bounded clean-critic diagnostic explicitly approved

- Zhaowei answered the concrete scope question: "Approve this bounded critic
  diagnostic". This authorizes18 fresh frozen-policy parent trajectories,
  four training/two test per replacement model, and three fresh critics with
  at most1000 updates each. It does not represent Howard approval.
- Follow `specs/2026-09-30-clean-critic-generalization/protocol.md` and
  `experiments/configs/clean_critic_generalization_20260930.json`. Commit first;
  single attempt, at most113256 simulator steps/7200 seconds, no expansion.
- Preserve actor/gate/reward/dynamics and all old evidence. New trajectories
  are disjoint from R3 and from each other's partition. Seal predictions before
  opening test labels; select constant baseline on training labels only.
- This is supervised frozen-continuation critic ranking, not online DDPG or a
  full-episode improved policy. Retain all clinical harms and null results.
  No automatic actor training regardless of triage; new scope needs approval.
- Independently audit and archive raw data, full critic/RNG state and source;
  checksum the authorized Dropbox local copy. Cloud sync/access separately
  unverified. No remote Git, messages, formal holdout or Stage E reopening.

### 2026-09-30: R6 critic-only diagnostic complete; generalization gates fail

- Single source97c79a6 attempt completed exit0:18 parent trajectories,72 states,
  3456 outcomes,73232 actual simulator steps,1895.705s,three critics at1000
  supervised updates each. No actor/DDPG update, retry, or sample expansion.
- Independent raw arithmetic/lineage/848 alias/seal/selection verification
  passes. Fresh initial weights reproduce, all Adam steps equal1000, final
  saved-weight MPS predictions exactly match. R4/R3 inputs/evidence unchanged,
  stderr empty and no experiment/verifier remains.63 final related tests and
  compileall pass, including the read-only post-run error-accounting helper.
- Test pairwise accuracy32.0%,46.7%,46.3%, versus training78.7%,78.8%,67.8%.
  Test errors exceed the zero-advantage prediction reference for all models.
  Cost improves over frozen in only2/6 test trajectories;2/6 have an adverse
  clinical mean. These are dependent remaining-window diagnostic contrasts,
  not improved full-policy episodes or formal statistical/safety conclusions.
- All prespecified triage components fail. Preserve the limited R3 headroom
  finding but do not convert it to an online-DDPG claim. Generalization and
  clinical-objective alignment need separate diagnosis; reward changes are
  not an established remedy and are not automatically authorized.
- Readout: `specs/2026-09-30-clean-critic-generalization/readout.md`. The3659-file
  archive and Dropbox local archive/manifest copies hash-verify; cloud/access
  unverified. ArchiveSHA256:
  `bce84b741c3dc714a9439cc49f8226e8be7dee646450a4ca53ca23eaca0977f0`.
- Finite packet closed; no actor follow-on, new experiment, remote Git, formal
  holdout or Stage E reopening. No Howard sign-off is inferred.

### 2026-09-30: R6 saved-data postmortem, no experiment extension

- Following Zhaowei's continuation request, fixed post-run diagnostics used
  only saved R6 labels/states/predictions/traces. Zero new simulation, model
  inference, fitting or action selection; R6 and Stage E remain closed.
- 70/215 non-tied test pairs reverse sign between fixed four-draw halves.
  On145 same-sign pairs, the saved critic ranks62 correctly (42.8%). This is
  descriptive and dependent, not a new validation or statistical test.
- Three of24 sealed choices reduce mean cost while increasing mean patient
  loss. Independent raw component reconciliation shows capacity-shortage
  savings can outweigh patient-loss charges. No reward weights were changed.
- Forty sampled execution-equivalent action pairs verified; ten in one test
  state have differing quantized requests and unequal saved value predictions.
  This motivates investigation, not a proven global representation diagnosis.
- Readout: `specs/2026-09-30-critic-saved-data-diagnosis/readout.md`.
  Identical repeated output and independent660-pair/192-contrast verification
  pass. Retain the initial JSON-serialization failure and its corrected v2.
- A24-critic, zero-simulation input-normalization control is a bounded proposal
  only. New fitting, clinical-objective changes or prospective data collection
  require explicit approval. No automatic launch, Howard sign-off or remote Git.

### 2026-09-30: Algorithm agnostic GCN and RL research direction

- Zhaowei explicitly stated that DDPG is not mandatory: the objective is a
  rigorous, publishable GCN-plus-RL solution, while assessing whether DDPG's
  previous limitations reflect its implementation or its decision interface.
  This updates future method-selection priorities, not the frozen historical
  primary method, results, formal holdout or Stage E status.
- Review historical replay/graph audits and R6 evidence before choosing an
  algorithm. Distinguish imitation, RL training in simulation, and additional
  deployment-time parameter adaptation. A failure of the third is not a proof
  that the second is impossible; imitation gains are not renamed RL gains.
- Local evidence/literature review and software-contract preparation are open.
  The preferred comparison is a clean DDPG reference versus a feasible discrete
  graph-policy route, not an unlimited algorithm or reward search. PPO is a
  proposal, not an executed or selected winning method. A changed representation
  and optimizer jointly define a controller comparison, not an isolated
  algorithm effect. Cost/service trade-offs remain legitimate when declared.
- Review memo: `docs/team_updates/2026-09-30-gcn-rl-research-direction.md`.
  87 existing synthetic/replay/input/kernel tests and full compileall pass;
  no scientific fitting, new patient trajectory or performance measurement.
- Next prepare public-information action and clean-learner contracts. New
  scientific execution still requires a committed bounded protocol and specific
  authorization. The prior24-critic normalization question was not approved;
  do not launch it implicitly. No Howard sign-off or remote action is inferred.

### 2026-09-30: Routing request-class engineering interface accepted

- Following Zhaowei's continuation request, added the opt-in
  `routing_candidate_contract.py` and24 synthetic unit tests. No existing agent,
  dynamics, reward, config, historical output or experimental scope changed.
- The contract canonicalizes exact integer decoder requests, preserves original
  submitted actions and reference/specimen-anchor provenance, uses one
  categorical probability per unique class and guards replay precision. It is
  not a physical-feasibility oracle, hidden-state mask or complete policy learner.
- A synthetic typed-replay integration check preserves the original request
  and terminal target. Existing continuous Gaussian GCN-PPO is not the proposed
  categorical learner; do not treat a config switch as an implementation.
- Readout: `specs/2026-09-30-routing-candidate-contract/readout.md`.
  All111 related tests and full compileall pass; eight prior source/evidence
  fingerprints match; no related research workload remains. Initial float64
  fixture expectation failures are documented, not erased or scientific retries.
- Next allowed engineering work is synthetic-only candidate scoring and sealed
  on-policy receipts. New scientific trajectories or fitting still require a
  committed bounded protocol and specific execution approval. No broad algorithm
  search, remote Git, Howard sign-off, holdout use or Stage E reopening.

### 2026-09-30: Candidate scoring and on-policy receipts accepted

- Zhaowei's continuation authorizes the next local synthetic-only engineering
  packet. Added graph/self-only/flat candidate scorer and value heads plus sealed
  behavior-policy receipts and closed-segment GAE preparation. Exposed the
  existing precision guard before returning a selected action for collection.
- Shared numerical information and candidate heads are checked; graph/self-only
  fixture parameters match, flat does not. This is not graph superiority, a
  complete PPO learner, exact feasibility certification or a scientific fit.
- Readout: `specs/2026-09-30-candidate-policy-receipts/readout.md`.32 new and111
  prior related tests pass,143 total; full compileall and eight prior fingerprints
  pass. No new scientific trajectory, reward change, historical-agent change or
  experimental performance claim. Development-test failures remain documented.
- Next prepare a bounded update/recovery adapter and source-audited collector
  using invented-data tests. A real categorical model is not the historical
  continuous actor, so specify its no-RL initialization comparator prospectively.
- New scientific execution still needs a committed bounded protocol and explicit
  approval; R6 test evidence is not untouched validation. No automatic training,
  Howard sign-off, remote Git, formal holdout or Stage E reopening.

### 2026-09-30: PPO objective acceptance and finite automatic continuation

- Zhaowei explicitly requested continued work plus automatic continuation. A
  newly scoped in-thread heartbeat `gcn-rl` is active every30min, verified after
  creation. It authorizes a finite local engineering chain, not a new scientific
  campaign. Other automations, including the paused historical monitor, remain
  unchanged; earlier deletion records are preserved.
- Added a checked categorical PPO objective adapted from existing PPO arithmetic:
  clipped surrogate, value MSE, entropy bonus and full-rollout normalization.
  Sealed-support integration uses invented graph/self-only/flat fixtures. The
  shared candidate encoder differs from the legacy separate actor/critic layout.
- Readout: `specs/2026-09-30-candidate-ppo-update/readout.md`. All156 related
  tests pass in4.746s; full compileall and eight preservation fingerprints pass.
  New tests calculate gradients without optimizer steps; existing DDPG fixture
  tests contain bounded optimizer steps. No patient simulation or scientific fit.
- Remaining finite chain: capped PPO update/recovery adapter; public-collector
  source audit and mocked preparation; one bounded pilot decision packet. Update
  the Live checkpoint and make local commits; delete the heartbeat at completion
  or when only new-scope approval remains. Do not invent experiments to continue.
- Scientific execution still requires a committed protocol and specific approval
  of named arms, initialization, metrics, fresh streams and query/update/time caps.
  No automatic reward/scenario/model search, real episodes, research training,
  formal holdout, external compute, remote Git, messages, Howard sign-off or Stage
  E reopening. Inspected R6 test labels are not untouched confirmation.

### 2026-09-30: Capped categorical PPO update/recovery engineering accepted

- Following Zhaowei's continuation, added an opt-in CPU categorical PPO kernel
  with explicit rollout/update/optimizer caps, sealed on-policy receipt checks,
  once-per-rollout normalization, clipped loss, gradient clipping and private
  sampling/shuffle RNGs. Historical DDPG/PPO agents and prior evidence unchanged.
- A complete rollout update publishes atomically from a private copy. Validated
  no-overwrite checkpoints include model, Adam, pending closed segments, consumed
  lineage, update history and both RNGs. This does not restore an environment,
  unfinished collection, mid-minibatch cursor or GPU execution.
- Readout: `specs/2026-09-30-candidate-ppo-kernel/readout.md`.26 new tests bring
  the related suite to182, passing in5.516s; full compileall and eight preserved
  fingerprints pass. Independent first-step arithmetic, failure rollback and
  exact next CPU update/sampling after recovery pass. Fixture failures recorded.
- All optimizer work used invented inputs; no scientific fitting, real patient
  episode, reward change or performance comparison ran. The frozen arm has no
  optimizer; enabling an update kernel is not evidence of online-RL benefit.
- The existing finite `gcn-rl` heartbeat stays active without modification. Next
  source-audit/prepare the public collector with mocks, then draft one bounded
  pilot for explicit execution approval. No automatic experiment, algorithm
  search, remote action, Howard sign-off, formal holdout or Stage E reopening.

### 2026-09-30: Public candidate collector audit and pure boundary accepted

- Following Zhaowei's continuation, source-audited actual observation, anchor,
  routing, cost, horizon and recovery paths. The existing proposal/gate collector
  cannot silently stand in for a categorical collector. Time-limit done does
  not resolve all patients or add a terminal liability; this is an objective
  boundary, not proof of why earlier RL failed.
- Added pure public context/submission/receipt checks without environment
  construction or steps. Preserve the full MDL-2 anchor and exact original
  request; reconcile raw cost, decoded/conserved flow and unfinished identities.
  No new reward, penalty, patient-state mask, dynamics or historical-agent change.
- Readout: `specs/2026-09-30-candidate-collector-audit/readout.md`.17 new tests
  and182 prior related tests pass,199 total in5.595s; full compileall and eight
  preservation fingerprints pass. New tests actively forbid patient constructor,
  reset and step; only metadata shells and invented arrays/receipts are used.
- This is not end-to-end collector acceptance. A combined categorical collector
  recovery driver, real preflight, candidate support/initialization and precise
  finite-window/clinical endpoint objective remain pre-execution requirements.
  Clone checks must count against the future simulator-query budget.
- Raw input parity does not equal full neural feature parity: candidate scoring
  also consumes reference/candidate features. A DDPG comparison would compare
  controller packages unless those differences and initialization are controlled.
- Next draft one bounded pilot packet and ask explicit scientific execution
  approval, then remove the finite heartbeat when only that approval remains.
  No automatic training, scenario/reward search, remote action, Howard sign-off,
  holdout use or Stage E reopening; inspected R6 labels remain development data.
