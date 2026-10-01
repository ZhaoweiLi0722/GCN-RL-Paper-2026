# Single-GCN pilot budget draft

2026-10-01. **PROPOSED; NOT EXECUTION AUTHORIZATION; NOT READY TO LAUNCH.**
Companion: `pilot-budget-draft.json`. Source-reading base:
`7b5a96e9e8b787876bbe2e0bd8ebf7076cecb25f`, branch
`codex/september-research-integration`. No source, frozen configuration, locked
plan, queue, Live checkpoint or experiment artifact is changed by this draft.

## Decision and scope

One future consolidated decision would approve a prospective scientific
amendment and this exact bounded packet after its implementation is frozen.
The locked A1 result remains **failed acceptance: 1/9 complete passes**, with
the numerical attempt closed. The existing A3 prerequisite says A1 must pass.
Therefore a future explicit, committed prospective amendment must replace that
prerequisite for this pilot, explain the scientific reason and retained risk,
and retain the failed result. This draft neither declares A1 passed nor grants
a retry, waiver, patient permit or automatic follow-on. See [S1-S2].

Question: does sampled-return PPO add value beyond its **own competent,
tensor-identical initialized GCN policy**, compared with additional imitation?
Three independent training blocks, labelled 60/61/62 for their corresponding
frozen R4 reference artifacts. These labels are not new environment seeds.
One GCN representation; five final controllers per block: own frozen start,
own PPO, own BC-CONTINUE, R4, and **full MDL-2**. No flat/self-only attribution,
DDPG repair, new scenario, online deployment adaptation or confirmation study.

Keep the nominal 20-facility, 80-request, horizon-52 environment and reward:
negative absolute step cost, scale 1e-9 once, gamma=1, lambda=1, terminal zero
bootstrap, no added terminal cost. Retain all original cost coefficients.
This is a finite-horizon simulator objective, not a complete-lifecycle or
clinical-value claim. Shortage/loss interpretation and unfinished obligations
remain disclosed limitations; no independently justified clinical margin exists.

## Fixed proposed schedules

These constants are selected prospectively from the existing P1 collection,
imitation and PPO schedules, not from a new fit. No tuning, warm-start search,
best-checkpoint selection, early success stopping or replacement seed is allowed.

1. **Runtime/input gate:** one claimed local run, CPU float32, one scientific
   job at a time. Bind the three existing R4 pretrain/config pairs, new source,
   effective config and runtime hashes. No historical R4 critic/optimizer/replay
   restoration. Preparation is not a fresh R4 fit. All real construction and
   inference occur only inside the eventual approved timed attempt.
2. **Prototype integration preflight:** one 52-step sampled episode per block
   with a disposable new-model copy, no updates. Save at t=26 and replay the
   next four transitions on one restored clone; compare decision, original
   request, reward, identity, environment and RNG state. Preserve the untouched
   initialization template. Three full episodes plus 12 clone steps.
3. **Initialization collection:** eight fresh R4-controlled episodes per block,
   416 public-state/R4-class examples per initializer, 24 episodes/1,248 steps
   total. No optional counterfactual, qualification or test labels enter fitting.
4. **Initialization fitting:** one actor fit per block, exactly 256 Adam calls,
   batch 64 sampled with replacement from that block's 416 examples. R4-class
   cross entropy only, lr=0.0003, gradient norm cap=0.5. Exactly 16,384 example
   presentations per fit, 49,152 total; these are reuses, not new observations.
   Critic initialization is deterministic random/zero-output construction,
   **zero critic prefit calls**. No hidden value warmup. Discard imitation Adam
   moments before all continuation forks. Initializer final checkpoint only.
5. **Separate qualification:** two new paired worlds per block, each run once
   with R4 and once with the new initializer greedy controller: 12 episodes,
   624 steps. On the 104 R4-state rows/block, require at least 95% reference
   agreement overall AND at least 95% on multi-class rows, with at least one
   multi-class row. Apply the same checks to the initializer's own 104-state
   trace using its contemporaneous R4 request; singleton-only coverage fails.
   Require finite scores/values/probabilities and exact request/terminal/cost
   accounting. As a conservative **development rejection screen**, also require
   block-mean initial cost/losses/terminal-active not above paired R4, and
   completions not below it. This zero-adverse-direction screen is not a
   clinical margin, equivalence or noninferiority test. Report waiting and all
   raw outcomes regardless. All three blocks must pass; no refit on failure.
   Qualification is never included in initialization or rollout updates.
6. **Same-start forks and real preflight:** copy qualified actor, independent
   critic, preprocessing and buffers exactly into frozen/PPO/BC-CONTINUE;
   verify seals and no shared mutable parameter storage. Fresh, empty Adam
   state for each trainable owner; identical private sampling RNG starts for
   PPO/BC, independent shuffles. Run all five controllers once per block on
   one further paired preflight world: frozen greedy, PPO sampled, BC sampled,
   R4 reference and full MDL-2. Each 52-step session has a four-step restored
   clone at t=26: 15 full episodes plus 60 clone steps. On the 52 frozen-path
   public contexts/block, additionally compare all three fork greedy outputs
   without extra environment steps. Sampler clones are disposable; neither
   preflight nor restoration may advance real training RNG or change weights.
7. **Continuation:** PPO and BC each get exactly 32 episodes/block, horizon 52,
   organized as eight rollouts of four complete episodes (208 rows). Use paired
   world starts across arms, not a claim of identical future event streams.
   Four epochs/rollout, minibatches 64/64/64/16, hence 16 minibatches/update
   and 128 minibatches/arm/block. No evaluation-triggered stopping.
8. **Seal and evaluate:** seal all nine learned-controller artifacts (three
   frozen, three PPO, three BC) before any test. Twelve new paired worlds/block,
   all five controllers, 180 episodes/9,360 steps. Greedy learned policies,
   unchanged R4 and full MDL-2, final checkpoint only; zero optimizer calls.
   These are development worlds, not formal holdout and not 180 independently
   trained policies. No extra stochastic evaluation mode.
9. **Readback and local preservation:** use saved receipts only to reconcile
   costs/outcomes, all charged calls, optimizer counters, seeds, RNG, final
   seals and failures. One saved-data paired analysis; local archive and
   per-file/member verification. No replay through a model or environment,
   Dropbox copy, cloud-sync claim or external communication.

### Learner and optimizer accounting

Proposed new model: request-conditioned shared scalar scorer, graph encoder
width 16 and hidden head width 32; learned reference/anchor indicators and
no fixed reference log-prior. Use the existing public matched-information
schema, physical/shared-relations graph and canonical request support (up to
six original requests, specimen options +/-0.05 and +/-0.10). Preserve original
float64 requests for submission. A restricted specimen anchor is **not** full
MDL-2. Separate state-value model, own graph encoder/parameters/optimizer, same
public information; no hidden state or future outcomes. Linear final scalar
outputs, no fixed output clipping. Exact new initialization, preprocessing,
parameter counts and full restore schema must be source-frozen, not guessed
from the old 59,602-parameter joint model. No architecture variants are budgeted.

All Adam instances: lr=0.0003, betas=(0.9,0.999), eps=1e-8,
weight_decay=0, foreach=false. PPO actor loss uses ratio clip 0.2 and entropy
coefficient 0.01; critic loss is 0.5*MSE on fixed sampled-return targets.
Normalize detached advantages once over all 208 rollout rows with population
std + 1e-8. Gamma=lambda=1 makes terminal returns Monte Carlo returns;
no oracle Q table or counterfactual result enters the learner. For each
minibatch: actor backward/clip(0.5)/charged Adam call, then independent critic
backward/clip(0.5)/charged Adam call, using the same indices and fixed pre-update
advantages/targets. No shared gradients, KL-driven epoch reduction or extra
critic steps. This changes the old joint-optimizer implementation [S5-S6].

BC-CONTINUE samples its current categorical policy, learns the R4 class at
each current public state, uses CE only and the same batch/epoch/lr/clip
schedule. Its critic stays byte-identical to the initializer. Frozen has no
training collection and no optimizer. Each continuation arm has 6,656
example presentations/block (1,664 distinct step observations x four epochs).

| Adam calls | Per block | Three blocks |
| --- | ---: | ---: |
| Initialization actor, 256 x 1 | 256 | 768 |
| Initialization critic | 0 | 0 |
| PPO actor, 8 x 4 x ceil(208/64) | 128 | 384 |
| PPO independent critic, same schedule | 128 | 384 |
| BC-CONTINUE actor, same schedule | 128 | 384 |
| All other owners/phases, including optional diagnostics | 0 | 0 |
| **Total** | **640** | **1,920** |

BC matches actor-update and environment exposure, not PPO's total compute;
PPO has an additional independent critic. Report that difference explicitly.

## Complete resource envelope

An environment call below means **one env.step**, including restored/counterfactual
branches, not a policy forward or a reset. All constructors/resets, inference,
receipt validation, checkpointing and I/O consume the phase/global clocks.
Cap fresh episode constructions at 426 plus three backend layout preparations
(429 build_env calls); zero unscheduled reset calls. Eighteen mandatory restored
session clones and, only if selected, at most 72 one-step diagnostic clones.
Copies/restores may not perform hidden simulation or optimization. Initial RNG
use in backend preparation is declared under preflight, never called unseen.

| Phase | Full episodes | Main trajectory steps | Mandatory restore steps | Adam calls | Phase seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Runtime/input binding | 0 | 0 | 0 | 0 | 600 |
| Prototype preflight, 3 x (52+4) | 3 | 156 | 12 | 0 | 600 |
| Initialization collection, 3 x 8 x 52 | 24 | 1,248 | 0 | 0 | 900 |
| Initialization fit, 3 x 256 | 0 | 0 | 0 | 768 | 1,800 |
| Qualification, 3 x 2 worlds x 2 controllers x 52 | 12 | 624 | 0 | 0 | 900 |
| Fork/controller preflight, 3 x 5 x (52+4) | 15 | 780 | 60 | 0 | 1,200 |
| PPO, 3 x 32 x 52 | 96 | 4,992 | 0 | 768 | 4,500 |
| BC-CONTINUE, 3 x 32 x 52 | 96 | 4,992 | 0 | 384 | 2,700 |
| All-model seal barrier | 0 | 0 | 0 | 0 | 300 |
| Final evaluation, 3 x 12 x 5 x 52 | 180 | 9,360 | 0 | 0 | 4,500 |
| Saved-data verification/analysis | 0 | 0 | 0 | 0 | 1,200 |
| Project-local archive/member verification | 0 | 0 | 0 | 0 | 1,200 |
| Supervisor, dispatch, terminal preservation/closure | 0 | 0 | 0 | 0 | 900 |
| **Main total** | **426** | **22,152** | **72** | **1,920** | **21,300** |
| Optional one-step counterfactual appendix | 0 | 0 | separate <=72 | 0 | <=300 |
| **Maximum combined** | **426** | **22,152** | **144** | **1,920** | **21,600** |

Main env.step cap: **22,224 = 19,344 + 1,248 + 624 + 168 + 840**.
Maximum including the separately selected diagnostic: **22,296**.
Per initializer: 600 seconds. Per PPO block: 1,500 seconds/1,664 steps/
128 actor+128 critic calls. Per BC block: 900 seconds/1,664 steps/128 actor
calls. Per final controller/block: 300 seconds/624 steps. Global elapsed cap:
**21,600 seconds (6 hours)** from claim before runtime/model/environment setup
through verification, archive and closure. Each phase cap is also mandatory.
Main workload clocks sum to 21,300 seconds; the remaining 300 belongs only to
the optional appendix and is unusable by main work. These are conservative
proposed ceilings, not benchmarked runtime estimates or a success promise.

### Optional means separately elected, never hidden

Default `include_optional_counterfactual=false`; effective optional call/time
caps are then **0/0**. A single future consolidated approval may elect the
predefined appendix before any numerical call. It uses the frozen-policy fork
preflight snapshots immediately before steps t=13,26,39,51, for each block.
For each snapshot submit one saved original request from each canonical class
(maximum six), once, on an isolated restored environment for one transition:
3 x 4 x 6 x 1 = **72 calls maximum**, no optimizer or model rescoring.
Use canonical order, do not pad aliases, and burn unused class slots. Record
requested versus realized differences and one-step costs only; no long-run
headroom/optimality claim. These outputs never feed fitting, qualification,
checkpoint choice or the primary test; no test-world clones. Mandatory restore
clones above remain main-budget operations even if this appendix is disabled.

### Accounting and terminal behavior

Before EVERY env.step and EVERY individual optimizer.step, durably charge its
owner, phase, resource subtype and global ledger, and check its numerical time
deadline. Check the deadline before forward/backward/restore operations too;
an external watchdog must stop an overlong call. Partial/failed calls retain
their charges. Check both PPO call slots before beginning its minibatch and
still debit each call immediately before execution. A failure between actor
and critic steps closes the entire attempt, with partial states preserved.

No refunds, transfers between arms/blocks/phases, counter rollback from a model
checkpoint, cap expansion, changed seeds, dropped failing block or retry.
One scientific claim and one attempt. Qualification/contract failure, timeout,
nonfinite quantity, exhausted resource, source drift, incomplete ledger or
lost receipt closes it. Save partial weights, optimizer/RNG, ledger and stderr;
do not fit or simulate during failure preservation. The 900-second closure
slice is not extra science. If the global deadline prevents complete archiving,
report preservation incomplete; a later saved-data-only preservation decision
does not resume the numerical attempt. Mock pre-execution code repairs are not
permission for a numerical trial. Nothing is run by this draft.

## Prospective seed allocation, not an audit

Proposed namespace `APD-single-gcn-pilot-20261001-v1`; environment seed is
`(int.from_bytes(sha256(namespace)[:12], 'big') << 16) + ordinal`, following
[S8]. The ranges below allocate **168 distinct starts**; intentional arm
pairing references the same start, not a second seed allocation.

| Split | Block 60 ordinals | Block 61 ordinals | Block 62 ordinals | Uses/start |
| --- | --- | --- | --- | ---: |
| Prototype preflight | 0 | 1 | 2 | 1 (+layout setup, no step) |
| Fork preflight | 3 | 4 | 5 | 5 controllers |
| Initialization demonstration | 6-13 | 14-21 | 22-29 | 1 |
| Qualification | 30-31 | 32-33 | 34-35 | 2 controllers |
| Training | 36-67 | 68-99 | 100-131 | 2 learned arms |
| Development evaluation | 132-143 | 144-155 | 156-167 | 5 controllers |

Neural/private analysis seeds use the first eight SHA256 bytes of
`namespace + '/' + role_path`, big-endian modulo 2**63. Fixed role paths:
`block{b}/graph/actor_initialization`, `critic_initialization`,
`bc_init/shuffle`, `continuation/sample`, `ppo/shuffle`, `bc_continue/shuffle`
(each suffix under `block{b}/graph/`); plus `analysis/bootstrap`. Same
continuation sampler seed is deliberately copied into PPO and BC, while
preflight consumes disposable RNG copies. Separate actor/critic initialization
streams; PPO actor/critic share each recorded minibatch permutation, not weights.

**Allocation proposal only: no seed collision audit has been performed.**
Before freeze/approval, the coordinator must validate generated seeds against
the applicable local prior stream inventory and document coverage gaps; never
relabel an inspected or initialized stream as fresh. This task performs no
broad historical audit and consumes no seeds. Collision requires a prospective
allocation revision, not automatic resampling within the attempt.

## Readout and unresolved integration

Primary: paired raw total cost, PPO minus own frozen; negative is better.
Report all 36 world differences, all three block means and their equal-weight
mean. For block b define relative change as 100 * mean(C_ppo-C_frozen) /
mean(C_frozen), then average these three percentages equally; a nonpositive
denominator makes the relative screen undefined, not passed. Proposed practical
screening threshold is 1% mean relative cost reduction and favorable cost
direction in all three blocks;
this is an explicitly proposed development rule, not a validated clinical
margin or inherited approval. Use one fixed 10,000-draw hierarchical paired
bootstrap (resample three blocks; within each sampled block resample twelve
paired worlds) for descriptive uncertainty; three blocks give weak precision.
No additional model evaluations. Report PPO vs BC, R4 and full MDL-2 as
secondary, disclose initializer-vs-R4 differences rather than attribute them
to RL. Always include losses, completions, waiting, terminal-active obligations,
requested/executed action changes and compute costs. A cost gain with an
adverse patient-outcome direction is a trade-off, not overall success or
noninferiority. No guaranteed RL gain, graph attribution or publication claim.

**Single concrete implementation blocker:** there is no source-frozen,
contract-accepted dynamic-candidate GCN plus independent-critic runner for
this schedule. Specifically, current PPO owns one joint Adam, imitation/PPO
have exact old-policy type guards, qualification sessions prohibit greedy
selection, prototype preflight assumes a `self_only` sibling, and budget/job
builders do not express these extra phases or separate Adam owners [S4-S9].
Exact model initialization/preprocessing/parameter counts, new restore
acceptance and these orchestration extensions remain unresolved, not silently
inherited from toy tests. A future implementation must resolve them without
fitted searches and fit the fixed envelope; otherwise return one bounded
amendment question, not made-up readiness. Budget arithmetic can be complete
while executable readiness, clinical justification and authorization are not.

No new diagnostic fitting to rescue A1 is included. Gates before launch remain
the prospective A1 amendment, accepted new implementation/zero-update contracts,
explicit finite-horizon objective decision, seed/input/runtime freeze and one
consolidated user authorization. Real preflight/qualification are inside that
one approved attempt, not uncharged prerequisites conducted beforehand.

## Source references and safe validation

Paths below are relative to this worktree; line numbers identify the inspected
source, not claims that these files already implement this draft.

- **S1:** `docs/patient_indexed_specimen_routing_locked_execution_plan.md:2317-2349`,
  failed A1 and closed attempt; no patient reopening.
- **S2:** `specs/2026-10-01-rl-improvement-workflow/plan.md:95-129`, old A3
  condition and explicitly incomplete 19,344-step subtotal.
- **S3:** `specs/2026-10-01-rl-improvement-workflow/candidate-contract-readout.md:32-52`,
  dynamic scorer, independent critic, competency and new-model acceptance;
  `reward-decision.md` in that directory, unchanged finite-horizon objective.
- **S4:** `experiments/configs/candidate_return_pilot_20260930.json:68-148`,
  eight demos, two qualification worlds, 256 init calls and 32-episode schedules;
  `src/rl/candidate_imitation.py:68-160`, CE, replacement batches, calls/guards.
- **S5:** `src/rl/candidate_ppo_kernel.py:75-126,182-238`, joint Adam and
  epoch/minibatch count; this draft doubles PPO calls for independent owners.
- **S6:** `src/rl/candidate_ppo_objective.py:36-45,59-90`, normalization and
  fixed-target losses; `src/models/candidate_policy.py:50-59,89-105`, current
  widths, shared representation and dynamic scalar-score interface.
- **S7:** `src/rl/candidate_pilot_driver.py:48-104,395-415`, initialization,
  exact forks, serial sequence, seal barrier and reference-row qualification.
- **S8:** `src/rl/candidate_pilot_resources.py:19-54,102-225`, stream derivation,
  phase/owner caps and durable charge-before-call/no-refund convention.
- **S9:** `src/rl/candidate_pilot_campaign.py:55-90,210-317,377-425`, actual
  backend/preflight/qualification/evaluation/archive paths; the current archive
  path includes external copying and must NOT be reused unchanged here.
- **S10:** `src/rl/candidate_patient_session.py:27-35,75-85,115-159`, full
  MDL-2, split restrictions, original submission and retained failure charges.

Permitted draft validation uses only stdlib `json`, `math`, `hashlib` and
arithmetic assertions: parse the JSON, sum rows, verify 32/4=8 and
8*4*ceil(208/64)=128, confirm 426*52+72=22,224, optimizer total 1,920, time
21,300+300=21,600, and ordinal coverage 0..167 without cross-purpose overlap.
No project imports, tensor/checkpoint loads, neural forwards, fitting,
environment construction/steps, holdout or remote actions are needed. This is
documentation/data only; coordinator owns any final repository compilation
and integration checks under AGENTS.md. Arithmetic is not model acceptance.

Draft check performed: stdlib-only assertions PASS for all phase/optimizer/time
sums, replacement/epoch counts, optional cap, 168 nonoverlapping ordinal slots,
426 controller-episode uses, 19 distinct neural role paths and existence of
the cited source paths. This checks proposed allocation arithmetic, **not**
historical seed freshness. No model or project module was imported. No
compileall, tests, numerical pilot, remote action or commit was performed by
this sidecar; the coordinator owns integration validation.
