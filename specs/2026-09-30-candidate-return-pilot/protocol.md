# P1: bounded candidate-policy return-training pilot

Date: 2026-09-30. **Draft for Zhaowei's execution decision; NOT authorized or
executed.** Engineering base: `1f31765d921e577612b2ef3b9181da876b2d52b9`.
Machine-readable proposal: `experiments/configs/candidate_return_pilot_20260930.json`.
Only the persistent September integration worktree is eligible. Stage E remains
closed. No Howard approval, positive outcome or publication readiness is implied.

## Question and scope

Does training a graph candidate-routing policy on actual trajectory returns
improve its fixed-weight evaluation performance beyond its own imitation-only
initialization, continued imitation, the unchanged R4 replacement controller,
and full MDL-2? This tests **RL training in simulation**, not deployment-time
parameter adaptation, unknown capacity shifts, global optimality or PPO's
intrinsic superiority to DDPG. No new scenario or reward is proposed here.

The existing 52-step nominal-history objective is retained. It is observed
cost within a time window, NOT full patient-lifecycle cost. Outstanding patients
are separately counted, never assigned a newly invented penalty or treated as
deaths. A gain that depends on poorer completion or deferring obligations is
not sufficient to advance a patient-benefit claim.

## Fixed environment, references and information

- Use `routing_nominal_history`, 20 facilities, horizon52, raw state561,
  action80, transfer scale120, from the three hash-locked R4 effective configs.
  Compare the complete effective environment dictionaries and realized topology
  before use. Unsupported public-producer layout/relation differences STOP P1;
  do not silently collapse edges or change the scenario to fit the implementation.
- R4 seeds60/61/62 are **replacement pretrains**, not recovered F1 models and
  not proven near-optimal. Their policy/config hashes are in the proposal. Load
  only inference actor/gates through `StrictFrozenPolicy` on CPU; no historical
  critic, optimizer, replay or labels initialize the new learner. Preserve R4's
  inherited distillation/data/compute provenance in the report.
- Per decision, use raw public observation, declared configuration and physical
  links only. Reference inference must work from raw state without a live env
  argument. MDL-2 and four +/-0.05, +/-0.10 specimen options use
  `residual_option_actions_from_state`, not the hidden/live-env helper or lookahead.
  Preserve the exact original request, with only existing box clipping in that
  helper. The R4 request and all options must equal full MDL-2 outside the
  specimen slice; fail instead of projecting a changed reference.
- At most six original requests: R4, MDL-2 and four public specimen options.
  Canonicalize integer requested-net classes, retain reference/anchor provenance
  and one categorical probability per class. No outcome-based merging, oracle
  ranking, hidden feasibility mask, future reward or patient registry in inputs.
  Distinct classes need not yield distinct realized routing or useful outcomes.
- Reference requests, class features and role flags are inputs to every new
  candidate model. R4/MDL-2 are complete controller baselines, not feature-matched
  candidate architectures. The reference itself contains a GCN; self-only and
  flat candidates do not remove that shared inherited graph computation.

## Arms, initialization and estimands

There are three independent development blocks, linked to R4 seeds60/61/62.
Each block has three candidate representations: graph/physical, graph/self-only,
and flat/self-only. Widths are encoder16/head32, CPUfloat32, no architecture or
normalization search. With the declared20x28+1 input these have59,602,59,602 and
238,658 parameters. Counts are verified without patient simulation; actual
runtime schema and counts must reproduce them. Flat is NOT parameter-matched.

Graph/self-only starts with identical tensor values, differing only in the
message operator. Each is then imitation-trained separately, so their post-BC
weights need not match. Physical links remain available in all views. This is
an incremental encoder-message comparison conditional on a common graph-based
reference, not removal of all graph information from the entire system.

Collect8 fresh52-step R4 demonstration episodes per block (416 states). On
those same states train each new representation for exactly256 Adam minibatches
of64, learning_rate0.0003, betas(0.9,0.999), eps1e-8, weight_decay0,
foreachFalse, max_grad_norm0.5. Target is the R4 reference class, cross entropy
only; no Monte Carlo labels, rewards or fitted value targets in BC. Sample
minibatches with replacement from the416 rows using a private fixed RNG. The
value head is not fitted; shared-encoder changes are retained and disclosed.

On2 separate R4 episodes/block, check greedy reference-class agreement >=95%
for **each** of the9 initializations. Include single-class states, and separately
report multi-class agreement and support counts. If any fails, close as an
initialization failure, without increasing BC updates or proceeding to RL.
This qualification is action imitation, not a clinical/performance guarantee.

Fork every passing initializer into the following tensor-identical roles.
Save initial weights, contract, reference hash and sampling states before forks.
Both trainable roles get fresh optimizer moments; discard BC optimizer state
for both, not just one. Frozen has no optimizer.

| Role per representation/block | New interaction | New optimization | Evaluation |
| --- | --- | --- | --- |
| FROZEN | None | None | Greedy fixed initialized candidate policy |
| PPO |32 complete episodes |8 four-episode updates,128 Adam steps | Greedy final policy; no online test updates |
| BC-CONTINUE |32 complete episodes |8 four-episode CE updates,128 Adam steps | Greedy final policy; no online test updates |
| R4 reference (once/block) | None | None | Original frozen request, unchanged |
| MDL-2 (once/block) | None | None | Full heuristic, unchanged |

BC-CONTINUE samples its current categorical policy while collecting and labels
each encountered public state with the same R4 reference class. It matches PPO
in interactions, optimizer count, batch size, LR and epoch count, but learns
from teacher actions rather than return. It is an imitation/data-aggregation
control, not RL. Its state distribution can legitimately diverge after updates.

PPO: four whole episodes/rollout (208 rows), four epochs, batch64 including the
last16, clip0.2, value coefficient0.5, entropy0.01, grad norm0.5, LR0.0003,
gamma1, GAE lambda1, once-per-rollout population advantage normalization.
Keep every epoch, fixed LR, no tuning/early checkpoint selection. All rewards
are raw `-total_cost`, scaled once by1e-9. Endpoint52 is terminal with value0;
no bootstrap across resets. Interrupted/incomplete episodes are failures, not
training data or completed test outcomes. Never call a new categorical actor
tensor-identical to the historical continuous R4 actor.

During training sample from the categorical distribution, including in
BC-CONTINUE. At evaluation use greedy class choice with lexicographic integer
class-key tie breaking, identical for every candidate representation/role.

Within each block, initial raw model seeds are30930060/61/62 respectively;
graph and self-only use the same seed. Separate named sampling, shuffle and BC
streams are derived prospectively. Clone sampling streams across PPO/BC-CONTINUE
at fork, with no claim that subsequent trajectories remain equal.

## Clean DDPG boundary

**Zero new DDPG fits/episodes in P1.** The repaired prospective DDPG kernel is
not the historical R4 model and has different actor/gate/critic and input paths.
It lacks the candidate/reference feature path; it chooses continuous requests
instead of this restricted bank. Giving it random weights and comparing against
a distilled candidate would not be a fair exclusion of DDPG.

A later separately approved DDPG package comparison would require its own
same-start frozen pair, demonstrated competent initialization from the same
training-only data, declared gate semantics/support, consistent absolute reward,
contiguous returns, matching total query budget and raw public information, and
disclosed remaining feature/parameter differences. P1 cannot say PPO beats
repaired DDPG or resolve whether DDPG's gradients are useful. Do not add a DDPG
arm opportunistically after seeing P1 outcomes.

## Split and recording order

1. Before any new step: approval, complete implementation, tests, commit,
   source/input/runtime locks, byte-verified R4 inputs, collision-checked named
   RNG allocation, exclusive claim and append-only budget ledger.
2. One bounded real preflight: public-input/no-env-inference parity, original
   requests, components, and whole collector/environment recovery. Then BC data,
   initial fitting and independent action qualification. No reward selection.
3. For each block and representation, finish PPO and BC-CONTINUE on32 worlds
   each. Use the same allocated training start seeds across representations and
   roles within a block. Save every four-episode boundary; evaluate only the
   prescribed final checkpoint. Finish/seal ALL final models before test data.
4. Evaluate11 policies/block on12 fresh52-step worlds each. Same test starting
   RNG states within a block; different test worlds across blocks. Frozen and
   reference outcomes can be reused as paired comparators in reporting, never
   counted as independently rerun evidence. No test-time updates, lookahead,
   warm starts from R6, checkpoint choice or partial-result tuning.

Use namespace `P1-candidate-return-pilot-20260930-v1`. Allocate environment
starts from SHA256(namespace)[:12] interpreted big-endian, shifted16 bits,
plus a fixed ordinal: preflight0..11; demonstration12..35; qualification36..41;
training42..137 (32/block); test138..173 (12/block). Within each group iterate
block60,61,62 then episode index. Reusing a training/test start between declared
arms is intentional pairing, not duplicate independent evidence. Preflight's
624-step cap includes originals, clones and restore checks, not624 plus clones.
Neural RNGs use SHA256(namespace + '/' + role_path)[:8] modulo2**63; save full
names/values and reject collisions, including initializer seeds. Statistical
bootstrap uses its own derived `analysis/bootstrap` RNG with no simulator use.
Fixed role paths are `block{b}/{representation}/bc_init/shuffle`,
`block{b}/{representation}/continuation/sample` (intentionally cloned into both
trainable roles), and `block{b}/{representation}/{ppo|bc_continue}/shuffle`.
Representations are `graph`, `self_only`, `flat`. Training order is block,
representation in that order, then PPO followed by BC-CONTINUE; test order is
block, representation, FROZEN/PPO/BC-CONTINUE, then R4 and MDL-2. All loops use
ascending episode index. Cloned RNG roles are declared aliases, not collisions.

Audit namespaces/numbers against committed seed declarations AND local prior
run seed manifests before execution; document scope and any unknown external
streams. Collision or unverifiable required local manifests stops readiness,
not an excuse to quietly allocate another namespace. The namespace is proposed,
not yet claimed collision-free. R6 test labels remain inspected development
data; historical formal holdout stays unused.

Matching initial RNG state does not guarantee identical patient/event draws
after action-dependent RNG consumption. Save arrivals and denominators per arm,
and report this limitation instead of asserting perfect event-level CRNs.

## Fixed resource caps

| Work | Simulator steps | Optimizer steps | Time ceiling |
| --- | ---: | ---: | --- |
| Real preflight, including clones/recovery |624 total |0 |600s total |
| R4 demonstration collection,8 episodes/block |1,248 total |0 |600s total |
| Nine BC initializers |0 |2,304 (256/model) |120s/model |
| Qualification,2 episodes/block shared as public inputs |312 total |0 |600s total |
| Nine PPO continuations,32 episodes/model |14,976 (1,664/model) |1,152 (128/model) |600s/model |
| Nine BC continuations,32 episodes/model |14,976 (1,664/model) |1,152 (128/model) |600s/model |
|33 fixed evaluation policies x12 episodes |20,592 (624/policy) |0 |120s/policy |
| **Global hard cap** |**52,728** |**4,608** |**21,600s (6h)** |

Serial execution, one attempt. Caps are maxima, not a target to spend. Unused
allowance cannot move between phases/arms; global wall time includes I/O and
archiving. No free smoke/clone/evaluation steps; debit BEFORE every env.step.
Qualification/evaluation inference does not fit models. Unit tests use invented
fixtures and are separately identified, not disguised research trajectories.

## Outcomes and decision rules

Primary contrast is **graph-PPO minus its graph-FROZEN initializer** in raw
52-step total cost. For each block, average paired world differences over12
worlds; grand mean weights the3 blocks equally. Also report block relative gain
`100 * (mean_cost_frozen - mean_cost_ppo) / mean_cost_frozen`, never an average
of unstable per-patient ratios. Require positive finite cost denominators.

Report all36 paired world differences, all3 block estimates, a descriptive95%
t interval over the3 block means (df2; assumptions and low power explicit), and
within-block paired-world percentile bootstrap intervals (10,000 draws). The
bootstrap does NOT create more than3 independent training blocks. No confirmatory
p-value, population-level clinical safety or EAAI-readiness claim from this pilot.

Mandatory secondary contrasts: PPO vs BC-CONTINUE, R4 and MDL-2; self-only/flat
PPO vs their own FROZEN/BC controls; graph vs self-only; flat as unmatched-capacity
sensitivity. Report them all, not only a winning model. No post-hoc primary swap.

For every arm save component costs, unique enrolled/initial patient counts,
losses (with causes when available), completed treatments, identity conservation,
terminal active waiting/production/specimen-transit/finished-return compartments,
service denominators, route counts, blocked requests, inference time, request
class diversity, reference selection and distinct executed flow. Show cost,
loss, completion and unresolved differences together. Last-step arrivals and
unequal action-dependent enrollment denominators must remain visible. Unfinished
patients are not deaths; no extra drain simulation or tail-value fit is allowed.

Predeclared triage, not clinically validated noninferiority margins:
- Promising enough to DISCUSS a larger study only if graph-PPO improves cost
  in all3 blocks, grand relative improvement >=1%, is not worse in mean cost
  than BC-CONTINUE/R4/MDL-2, and shows no adverse mean direction per block in
  losses, completions or terminal active counts versus these comparators and
  its FROZEN pair. Confidence intervals and individual harms still accompany it.
- If cost improves but a service/terminal comparison worsens, report a trade-off,
  not a reward bug or an unqualified patient benefit. No automatic retuning.
- If intervals are wide, effects inconsistent, initialization fails, action
  support is rarely distinct, or optimization does not improve cost, conclude
  limited/inconclusive/negative evidence for THIS budget and design. Do not infer
  universal RL failure or a near-optimal baseline from a small failed trial.
- None of these outcomes automatically opens another experiment or formal holdout.

## Execution readiness and failure policy

The pure collector audit is NOT an integrated real-environment driver. Before
P1 launch, implement and freeze a categorical driver that jointly saves/restores
the environment including RNG, current raw observation and cursor, unresolved
segment, requests/receipts, private sampling/shuffle/BC RNG, model/optimizer,
normalization contract, update counts and phase/budget ledger. Fake-env tests
must reproduce the next decision/update and an interrupted transaction. Recovery
acceptance is not permission to resume a failed scientific run automatically.

Real preflight is inside the approved budget, with NO fitting. Restore/copy
checks must reproduce public observations, decisions, exact next reward/route
and identity state; every clone step counts. Re-evaluate observation-only options
with hidden fields inaccessible. Verify legacy reward mode is not accidentally
used: new data is absolute reward only. Test data remains absent until all models
and analysis choices are sealed. No partial checkpoint migration or replay reuse.

Fail closed on missing/changed locks, wrong config/schema, hidden data dependency,
invalid costs/identities/receipts, recovery mismatch, NaN/Inf, duplicate process,
budget/time limit or failed qualification. Preserve raw/partial evidence, exit
nonzero, and request a new decision; no repair-and-rerun under this approval.
Unfavorable but valid simulated outcomes are fully recorded, not discarded.

## Preservation and authority

Fresh root only: `results/candidate_return_pilot_20260930`. Save exact source
commit and source bundle, proposal/effective configs, all inherited input hashes,
runtime, named streams, raw per-step/action/outcome receipts, split manifests,
initial/final and four-episode boundary weights/full states, counters, stderr,
partial failure record, independent arithmetic verification and artifact index.
No existing root may be overwritten. Independent verifier reads raw costs and
paired outcomes without importing the learner's summary routines.

After a later approved run, use verified archives and per-file inventories.
Keep originals. Proposed Dropbox destination is a new P1 subdirectory of the
existing Research Artifacts folder, not a change of sharing or a message to
Howard. Local archive verification, Dropbox-local bytes, cloud sync and access
are separate statuses. This preparation performs NO cloud copy.

Before any science: record explicit approval of this bounded scope, append the
locked-plan execution amendment, freeze implementation/config and satisfy the
readiness gates. This draft's `scientific_execution_authorized` remains false.
An approved execution amendment must retain this draft and bind a separately
committed effective execution config, actual implementation hash and approval
record. Changing any scientific setting requires another explicit decision.
No remote Git, external compute, messages, reward/scenario search, DDPG fit,
formal holdout or reopening Stage E. The finite engineering heartbeat ends when
this decision packet is handed off, rather than waiting and re-prompting forever.
