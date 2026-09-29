# Autonomous local research queue

## Live checkpoint: R2 archived decision-cost diagnosis complete

Updated 2026-09-29 17:36 UTC after the user's acceptance of the recommended
legacy-simulator direction. R2 is retrospective archived-cost analysis, not a
changed experiment, reward intervention or authorization to train.

| Item | Current state |
| --- | --- |
| Workspace | Persistent `worktrees/september-research-integration`, branch `codex/september-research-integration`; local only |
| Last verified stage | R2 complete; R1/S4 and all earlier scientific code/results unchanged |
| Current task | Archived action-cost audit and independent Decimal verification complete |
| Protocol commit | R2 protocol/config `5594989`, auditor `1441c32`, verifier `dd62d6f`; outputs recorded after the corresponding source commit |
| Automatic continuation | `gcn-rl` deleted through the app on 2026-09-29 at 15:38 UTC; deletion confirmed. Finite chain complete, no new experiment created to keep it active |
| Implementation evidence | New read-only CSV/NPZ auditor and separate stdlib Decimal verifier; 14 new/29 combined tests, full compileall and diff checks pass. No production reward/learner/environment source changed |
| Recorded execution | 1,560 cost rows, 1,560 predictions, all 156 archived states; exit 0. Zero environment queries, optimizer updates or reward changes |
| Independent verification | Seven locked inputs unchanged; all 624 selections and pooled/per-seed cost summaries match independent Decimal reconstruction; original G1 metrics reproduce |
| Process/output check | Fresh read-only ps check found no related research Python process; audit and verification exited 0, no background task |
| Scientific result | Descriptive validation cost differences vs MDL-2: frozen critic argmax +4.488M, fitted offline critic +3.181M, discovery-cost selector -0.932M per diagnostic state. These are NOT frozen/online actor comparisons or episode savings |
| Reward diagnosis | 40/71 discovery-validation top-action disagreements exceed the legacy 1M cost threshold. Wrong actions are not uniformly near-ties; no evidence yet that changing cost weights or repairing targets causes online gain |
| Evidence paths | `reports/2026-09-29-g1-decision-cost/{audit,verification}.json`; `specs/2026-09-29-g1-decision-cost/{protocol,readout}.md` |
| Next concrete action | Obtain endpoint decision, then draft a separately bounded fixed-weight reward/replay/TD-consistency feasibility protocol with raw replication costs and clinical outcomes. Reuse N1-N7; do not start training or choose a favorable gate from old validation data |
| Approval needed now | Legacy route accepted. Recommend original 52-step finite-window cost for the next protocol, not full settlement. This endpoint choice is pending; data/initialization/budget and scientific launch still require a committed protocol and authorization |
| Approval boundary | New patient/scientific scenario, reward/model/continuation changes, neural campaign, extra matrix/retry, external compute, push/merge/send or formal evidence use |
| Result guarantee | None; record negative/null/unstable outcomes without tuning toward a desired conclusion |

The user asked for automatic continuation and a continuously updated status.
Keep this checkpoint current after each completed work packet, including
timestamps, current phase, actual process evidence if any, last test/result,
next action and any precise approval need. A running label needs fresh process
and progress evidence, not the existence of the scheduler. Do not duplicate an
active job. Continue adjacent unblocked steps within a run where feasible.

R2 source of truth:
`specs/2026-09-29-g1-decision-cost/protocol.md` and
`experiments/configs/g1_decision_cost_20260929.json`. The finite read-only packet
has completed; no automation remains to disable. G1's nominal-history correction
and failed decision remain in force; its favorable discovery-selector mean is
not grounds to reopen it. Do not fill the queue with invented experiments or
repeat the completed arithmetic as new progress.
Unchanged waits need no repeated message; report actual milestones or decisions.

## Authority and boundaries

Zhaowei requested on September 29 that routine steps continue automatically
without waiting for a reply after each one. Work remains on the persistent local
`codex/september-research-integration` worktree. This is not permission to push,
merge main, send messages, change historical experiments or use the formal
holdout for model selection. Do not assert Howard's sign-off or promise a
positive RL result. The locked execution plan still governs scientific launches.

Each continuation should verify current files and processes, choose the next
unblocked packet, complete implementation or analysis plus its tests, and
record evidence and remaining limitations here. Do not just rewrite the plan.
Use focused tests and full Python compilation for code changes. Keep bounded
tasks within a single continuation where feasible. Do not launch a second copy
of an already active job. Preserve work by others. Local commits are allowed.

The formal raw-row crossed-cost audit and eight-cell support-queue packet are
complete at `f8b965c`. Their results remain immutable. The latter is a solved
restricted synthetic fixture, not an excuse to train DDPG or claim PRM optimality.

## Ordered packets

| ID | Status | Deliverable and completion boundary |
| --- | --- | --- |
| M1 | Complete | Historical source/config audit and manuscript corrections are in `specs/2026-09-29-formal-method-contract/readout.md`. Ten configs and the historical manifest checked; 87 focused tests and full compilation pass. PDF rendering remains unavailable without a TeX engine. No dynamics changes. |
| M2 | Audit complete; findings open | `specs/2026-09-29-formal-replay-contract/readout.md` records mixed reward definitions, discontinuous cached multi-step windows, calibration discount mismatch, teacher support and missing label-horizon provenance. Manuscript limitations updated. No historical fixes/retraining; a corrected campaign needs separate authorization/protocol. |
| G1 | Audit complete; parity gaps open | `specs/2026-09-29-formal-graph-contract/readout.md` records exact component counts, graph/flat input and head differences, and proposal-conditioned gate asymmetry. Formal package-level results preserved; encoder-only attribution and topology generalization are not established. No new experiment. |
| E1 | Contract complete; domain inputs open | `specs/2026-09-29-qualified-support-contract/decision_contract.md` selects a proposed setup-support staffing decision, separates source evidence from assumptions, and specifies time-valid event observations. Companion dictionary has 12 unresolved domain inputs and explicitly forbids treating nulls as calibration or execution permission. No new experiment. |
| C1 | Bounded comparator complete; richer comparisons open | `specs/2026-09-29-completion-count-comparator/readout.md` reports a fixed count-only rule on unchanged recorded trees, with zero new simulator queries. It attains the completion-information bound in changed/nonbinding cells; bottleneck gaps do not identify an RL advantage. Censored-data ID-MPC/history-policy studies remain outside this packet. |
| P1 | Complete; coauthor/domain decisions open | `docs/team_updates/2026-09-29-manuscript-evidence-checkpoint.md` consolidates supported claims, raw-evidence limits, reproducibility commands and remaining decisions. Manuscript wording and future operational scope tightened; no submission-ready or online-gain claim. |

If a packet needs an external scientific/engineering decision, record the
specific missing fact and move to an independent unblocked packet. Ask once
for a necessary decision; do not repeat unchanged requests. When all allowed
packets are complete, or all remaining packets require external input, report
the checkpoint and remove the continuation automation. Do not automatically
invent a new experimental campaign to keep the queue nonempty.

## Renewed local preparation queue

After the completed P1 checkpoint, Zhaowei explicitly asked to continue and
advance automatically. This opens the following **finite software-preparation
queue**, not a new scientific campaign. Existing completion records remain
historical checkpoints. All original no-push/no-training/no-holdout boundaries
still apply. Do not wait for engineering data to complete independent software
contracts, but do not fabricate that data to launch a patient scenario.

| ID | Status | Concrete completion boundary |
| --- | --- | --- |
| N1 | Complete; production integration not authorized | Opt-in `src/rl/validated_returns.py`, 19 synthetic failure/return tests and `specs/2026-09-29-replay-repair-preparation/contract.md`. Combined 109-test suite and full compilation pass. Not wired into old agents. Reject disconnected windows, mixed semantics and implicit legacy migration; one explicit gamma-to-n target reference. |
| N2 | Complete; producer/model integration remains open | Separate `src/models/matched_inputs.py` common-input graph/flat/gate fixture, 21 new synthetic tests and `specs/2026-09-29-matched-input-preparation/contract.md`. Combined 140-test suite and full compilation pass. Explicit shared information/order/proposals, neural-only message ablation and parameter-count scope. No existing agent/default changes or environment/learning run. |
| N3 | Complete; bounded forward integration subsequently approved as N4 | `src/rl/prospective_adapter.py` joins N1/N2 with a typed no-training acceptance harness. Fourteen new tests, combined 154-test suite and full compilation pass. Covers current/bootstrap state and anchor mapping, rewards, short tails and termination/truncation. `specs/2026-09-29-prospective-integration-acceptance/decision_packet.md` records legacy non-migration and the next prospective approval. No experiment seeds, tuning or execution in N3. |
| N4 | Complete; production learning and scientific campaign remain outside scope | Actual patient collection plus frozen graph/flat actor, gate, critic and target forwards. Nine bounded engineering cases/36 steps reproduce exactly, weights unchanged, 171 tests pass. Readout: `specs/2026-09-29-prospective-collector-engineering/readout.md`. Parameter counts are not matched, no optimizer/resume framework, no online benefit claim. |
| N5 | Complete; scientific design remains non-executable | Count-only graph64/flat68 matching on the N4 schema: every component <1%, total gap 0.3340%. Isolated physical/self-only operator check, 15 new/95 combined tests and repeat-identical metadata report. `specs/2026-09-29-prospective-development-design/readout.md`. Eight named launch dependencies remain missing; no environment/update/performance run. |
| N6 | Complete within kernel scope; live-collector resume remains open | Default-off DDPG kernel, 18 new/113 combined tests, three exact fresh-process resume cases and repeat-identical diagnostics. `specs/2026-09-29-prospective-learner-engineering/readout.md`. Two synthetic matrices total 72 kernel updates / 144 Adam steps, plus bounded unit tests; zero environment/performance runs. No scientific-launch gate is automatically passed. |
| N7 | Complete within single-episode CPU scope; scientific launch remains closed | Six exact fresh-process live-session cases, 14 new/144 combined tests, 13 repeat-identical diagnostic JSON files and 18 tensor-identical checkpoints. `specs/2026-09-29-prospective-closed-loop-engineering/readout.md`. Actual collection/OU/pending replay/kernel recover together. No performance comparison or new-task calibration claim. |

For N2/N3, implement only the smallest independent prototype/tests needed to
answer the contract question. Do not create a duplicate full training framework.
If a production integration choice needs scientific approval, record that exact
choice and finish the other independent checks. At completion or external block,
report once and remove the renewed automation rather than adding new packets.

## Renewed progress

- September 29, N7 complete: six bounded online/frozen CPU paths recover exactly
  from a live four-step checkpoint, including two pending records and one online
  update. No tail loss or duplicate insertion; all three boundary masks pass.
  Two matrices total 144 primary plus 144 clone steps and 36 kernel updates,
  with additional bounded unit tests. Fourteen new/144 combined tests and full
  compilation pass. Thirteen repeated diagnostic JSONs are byte-identical;
  eighteen checkpoints match tensor for tensor; 34 source hashes reconcile.
  The next work is resolving operational/synthetic-scope assumptions and action
  leverage, not an automatically authorized learning campaign. Readout:
  `specs/2026-09-29-prospective-closed-loop-engineering/readout.md`. Old evidence,
  manuscript conclusions and remote state are unchanged; no automation added.

- September 29, N6 complete: real actor/critic Adam updates on invented numeric
  windows pass alongside fixed gate/reference and exact fresh-process kernel
  resume for graph physical/self-only and matched flat heads. Repeated reports
  are byte-identical; nine checkpoint payloads agree tensor for tensor and all
  ten source hashes reconcile. Eighteen new/113 combined tests, compilation and
  diff checks pass. Readout and local checkpoint inventory are retained; binary
  checkpoints follow the existing ignored-file policy. No environment training,
  old-evidence changes, performance claim, remote operation or automation.
  The next distinct mechanics question is live collector/exploration/episode
  resume. N5's domain, baseline, objective and scientific-launch gaps remain.

- September 29, N5 complete: following the user's continuation, specified
  primary frozen/online and separate operator/representation contrasts. The
  precommitted shape-only search finds graph64/flat68 with 0.3340% total gap and
  all component gaps below 1%; these are engineering dimensions only. Counts,
  backward participation and operator isolation pass without changing weights.
  Fifteen new tests and the 95-test focused suite, full compilation and diff
  checks pass; two reports are byte-identical and all seven source hashes match.
  No environment, optimizer or performance execution. The proposed finite-task
  objective is not a calibrated launch protocol. Next independent engineering
  dependency is the actual learner/resume contract; operational calibration,
  replicated leverage and scientific locks/authorization remain unresolved.
  No automatic campaign or continuation automation is opened by this packet.

- September 29, N4 complete: the user approved the specifically requested
  bounded actual collector/agent engineering check. Protocol:
  `specs/2026-09-29-prospective-collector-engineering/protocol.md`. The real
  dataflow check is complete locally: 36 exact cloned steps, lossless input
  views, requested-versus-executed actions, endpoint targets and short tails,
  unchanged frozen weights, byte-identical repeat JSON and 28 source hashes.
  Seventeen new tests and the 171-test combined suite pass, as do full
  compilation and diff checks. A wrapper-only virtual-filename failure and its
  tested fix are disclosed in the readout. No training, historical migration,
  new scientific scenario, remote action or automation was started. Stop at
  this approved packet's boundary; scientific launch and domain calibration
  are not inferred from software acceptance.

- September 29, N1: implemented explicit replay semantics, immutable one-step
  records, lineage-checked multi-step returns and a shared numeric Bellman-target
  reference. Independent counterfactuals remain one-step. Reward scaling occurs
  once; actual tail length and explicit terminal/truncation policy determine
  bootstrap. Nineteen new synthetic tests and the combined 109-test regression
  suite pass, as do full compilation and diff checks. Existing model,
  environment, baseline, legacy replay, config and tracked evidence paths are
  unchanged from `04d8c67`. No agent imports the prototype. Lineage assertions
  still require collector verification; this is neither a historical repair
  nor evidence of improved online learning. N2 is the next independent packet.
- September 29, N2: implemented canonical ordered input schemas with graph and
  flat views of exactly the same numerical blocks. Critic includes the anchor;
  both gate views include the proposal with explicit gradient-detach policy.
  Preserves all supplied node-feature slots and rejects implicit broadcasting,
  order/definition mismatches and invalid values. Physical-link metadata remains
  unchanged under neural self-only message ablation. A synthetic forward fixture
  reuses the existing GraphConvolution with a common dense head; normalization
  agrees with the existing helper. Parameter inventory separately reports unique,
  shared and trainable weights. Twenty-one new tests, the combined 140-test suite,
  full compilation and diff checks pass. Existing tracked source, configs and
  evidence are unchanged from `e17474a`; no agent imports the prototype. Actual
  producer provenance, head parity, directed/multirelation support and scientific
  integration remain open. N3 is next; no training/evaluation or remote action.
- September 29, N3: joined the replay and input contracts in an isolated adapter.
  Current critic uses the first recorded action; bootstrap inputs include the
  endpoint's own state and anchor. Closed-segment windows preserve all short
  tails and enforce lineage even for one-step requests. Tensor Bellman targets
  match the NumPy reference and independent backward recursion across float32/
  float64, discount and terminal/truncation cases, with detached target Q.
  Invalid/missing legacy metadata and mixed semantics fail rather than being
  guessed. Fourteen new acceptance tests, the combined 154-test suite, full
  compilation and diff checks pass. Existing tracked source, config, evidence
  and reports are unchanged from `1472083`; no existing agent imports the adapter.
  The decision packet separates actual producer/agent verification, scenario
  calibration and scientific launch approval from these completed prototypes.
  N1/N2/N3 are complete within scope. Ask once about the bounded prospective
  integration/engineering-verification amendment and remove the preparation
  automation; do not infer approval, launch a campaign or create more packets.

## Historical progress

- September 29, M1: initial inspection found generic DDPG prose inconsistent
  with the formal configuration: four-step anchor-relative online returns,
  specimen-only nonzero residual scale, fixed final deployment, separate
  actor/critic graph encoders and OU rather than independent Gaussian noise.
  Historical training manifest hash matches the locked plan. Verification and
  corrections completed in the method text, equations, pseudocode and deployment
  description. A second read-only audit reproduced the JSON byte for byte.
  Eighty-seven focused tests, full compilation and `git diff --check` passed.
  No TeX engine is available, so PDF layout is not verified. No new training or
  evaluation has been launched. Saved in local commit `39604fd`.
- September 29, M2: teacher hash verified; all ten summaries and 1,000 training
  rows audited. Historical numeric GCN/flat replay insertion was reconstructed
  without model or environment imports. Both emit 153/314 windows with
  discontinuous adjacent observations; an independent index recurrence agrees.
  Also documented absolute/relative reward mixing, calibration discount
  mismatch, broader teacher support and unavailable generation-horizon metadata.
  Logged updates and losses are not a causal explanation of the null increment.
  Ninety-four focused tests, full compilation and diff checks passed. A second
  audit reproduced the JSON byte for byte. No training/evaluation launched;
  G1 is the next unblocked packet. Preserve these findings as prerequisites
  for any future corrected development protocol, not a reason to silently
  repair or replace the historical formal campaign.
- September 29, G1: reconstructed all ten historical graph/flat model shapes
  and parameter totals from pinned source without full agents or rollouts.
  Totals 613,286/607,338 match the locked manifest (0.97936% gap). The graph actor
  uses shared edge heads; only graph critic/gate use fixed-order flattening.
  GCN features include derived demand and anchor blocks; flat critic lacks the
  latter. The graph gate consumes proposed action deltas, while the flat gate
  ignores that config field. Physical feasibility is unchanged by the flat
  graph-ablation flag. Added claim-to-code matrix and narrowed manuscript claims
  to the measured package comparison. Corrected M1's overbroad actor-readout
  description. Fifty focused tests, full compilation and diff checks pass;
  repeated audit JSON is byte-identical. TeX rendering remains unavailable.
  E1 is the next independent packet; parity/replay repairs need a separately
  approved prospective scientific protocol, not silent historical retraining.
- September 29, E1: narrowed the proposed lever to booking site-local qualified
  staff for pre-run setup/material-connection support. Preserved nonpreemption,
  qualifications and equipment constraints; no assumption that overtime speeds
  biological growth or mandatory tests. Added a seven-record measurement
  dictionary separating request/acceptance/delivery and event/availability time.
  Future corrections, latent response and exact unmeasured progress cannot leak
  into policy inputs. Literature supports investigating the channel, not its
  calibration or RL headroom. Twelve inputs still need domain evidence; these
  consolidate existing questions and do not block the independent C1 packet.
  No patient model, historical artifact, training or evaluation was changed.
- September 29, C1: committed the untuned count-only rule and protocol before
  recorded-tree replay (`5650ac1`). It uses only epoch and support-stage counts.
  Twelve paths/60 decisions extracted from existing edges; no new simulator or
  planner queries. Changed/nonbinding costs equal the completion-view bounds
  (103.25 batch, 93.25 booked flow); changed/bottleneck gaps are 6.25 synthetic
  units, not evidence for online RL. All 59 upstream hashes/row audits passed
  before and after; three output files reproduce byte for byte; a separate
  direct walk verified all 12 paths. Fifty-seven focused tests and compilation
  passed. P1 is next; do not tune the rule or weaken its information contract.
- September 29, P1: assembled the manuscript claim-to-evidence checkpoint and
  a short unsent coauthor draft. Corrected the integration note's isolated-
  encoder implication; narrowed abstract online-null wording and future
  staffing assumptions, and disclosed the tracked cost-only projection.
  Eighty-five combined focused tests and full compilation passed. A fresh
  independent 40-CSV/five-contrast bootstrap verification exactly matches the
  archived verifier output; all new formal/queue table values reconcile.
  Twenty-eight local links, 47 unique TeX labels, 39 bibliography keys and
  comparator provenance hashes pass. No TeX engine: PDF rendering unverified.
  All finite local packets are complete within scope. Remaining methodology,
  provenance and domain gaps are listed in the checkpoint; no new campaign,
  remote action or sign-off is implied. End the continuation automation rather
  than manufacturing more tasks while those dependencies remain unresolved.
