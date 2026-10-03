# Patient-Constrained Capacity Policy Improvement

Status: prospective development proposal; preparation approved, execution NOT
approved. Prepared 2026-10-03 from fa211c6. Numerical companion:
`experiments/configs/patient_constrained_capacity_20261003.json`.

## Decision And Scope

Recovery2 completed and found no reliable deployment-online benefit. In its
persistent-change condition, online DDPG had higher mean cost and more patient
losses than both corrected ID-MPC and fixed allocation. Small savings against
its frozen neural comparators came with additional losses. These observations
motivate a new objective, not more epochs on the consumed attempt or a claim
that all RL is ineffective. See recovery2/terminal-readout.md for the evidence.

Question: can simulation-trained GCN control reduce full settled cost without
increasing expected patient losses relative to a competent, exactly reproducible
starting controller? Does an explicit loss constraint help versus otherwise
identical cost-only learning?

Use the existing continuous capacity system and corrected predictor. Initialize
the GCN actor's final affine layer to zero: its existing `clip(2 + z, 0, 4)`
output is exactly [2,2,2,2], including the shared-budget projection. The frozen
same-start policy is therefore the fixed-allocation comparator, not a weak
imitation of it. Keep corrected ID-MPC as an external strong comparator.
Fixed allocation is not assumed optimal or clinically safe.

This deliberately tests **RL policy training in simulation**, with no parameter
updates in evaluation. Deployment-online adaptation remains unresolved. First
establish whether the learned policy is useful at all; do not advertise this
experiment as a deployment-online result. No isolated GCN contribution follows
without a separate matched representation study. No automated follow-on study.

## What Changes, What Does Not

Change: exact fixed-policy initialization; an additional loss-count critic;
patient-loss constraints and their multiplier updates; fresh paired training
and evaluation streams. This is a new learning design, not an isolated causal
comparison with the historical DDPG package. Within this new design, the two
learned arms differ only in whether the multiplier affects actor optimization.

Unchanged: four sites, 48 control plus 16 settlement epochs, three existing
conditions, patient identity and dynamics, 2-step flexible-hours commitment,
hour caps/shared budget, observations, resource ring, GCN widths, base cost
components/weights, fixed drainage, public filter and corrected ID-MPC recipe.
The old proposal's system/controller definitions are inherited by hash; obsolete
status and scheduling fields are not inherited. The optimizer recipe below
supersedes its learner and budget. No new scenarios, weight sweep or architecture
search. Support labor remains synthetic and uncalibrated; E1 remains missing.

## Objective And Actual Training Constraint

For each training condition c, seek to minimize equally weighted expected full
64-epoch cost C while satisfying:

    E[L(policy) - L(fixed) | c] <= 0

L is the integer count of uniquely identified lost patients after settlement,
not urgency cost, a survival prediction, or a proxy inferred from expenditure.
The reference is the fixed controller on the same training exogenous world.
Zero is a conservative development target of no additional expected losses,
NOT a clinically justified noninferiority margin or a pathwise safety promise.

Keep reward `-C/100000`. Add a separate positive loss-count target. Lump all tail
costs AND tail losses into the last control transition; terminal bootstrapping
is zero. No financial weights change. The actor minimizes:

    mean(-Q_reward(s, actor(s)) + lambda[c] * Q_loss(s, actor(s)))

There are three nonnegative multipliers. Initialize each to 1, cap at 20, and
after each complete constrained training cohort update only its condition:

    lambda[c] = clip(lambda[c] + 0.01 * (L(policy) - L(fixed)), 0, 20)

Then perform exactly 48 actor/dual-critic updates. No multiplier update until
the cohort settles. The cost-only arm fixes all multipliers at zero but trains
the same shadow loss critic for matched computation. Both critics have separate
encoders/optimizers; actor gradients do not update their parameters. Reuse the
existing float32 architectures, Adam settings, gradient cap10, gamma1 and
tau0.005; actor LR0.0001, each critic LR0.0003. No multiplier/learning-rate search.

Training condition IDs are scheduler labels used only to index multipliers and
replay strata. They are absent from actor and critic input features and from
evaluation controller interfaces. Private efficiencies, future random tapes and
change times are never policy inputs. The critic is an approximation from public
history, not an exact condition-specific value oracle. For each update use 32
fixed-reference and 32 learned-arm rows of the current training condition,
sampled with replacement. Both arms have identical sampling/noise seeds and
budgets, though endogenous trajectories can diverge. At warmup, use 64 reference
rows per batch, cycling conditions; no actor updates.

This is approximate Lagrangian constrained DDPG, not CPO, and may violate its
expected constraint. The multiplier uses noisy exploratory training outcomes;
it does not guarantee the final greedy policy is feasible. A capped multiplier
can bind; report this rather than increasing the cap. Test rejection alone is
not the training constraint. The constraint changes optimization explicitly,
even though the physical cost ledger and its reward remain unchanged.

## One Serial Attempt

Three independent initialization seeds. For each block:

1. Collect 12 fresh fixed-reference training worlds, four per condition, with
   exact final loss counts and 576 control transitions. No neural action call.
2. Warm up the cost and loss critics for 256 joint batches with actor frozen.
   Clone the complete model, targets, optimizer and replay/RNG state to two arms;
   only their multiplier rule differs. No BC or historical checkpoint load.
3. Each arm collects the same 12 training world seeds, one complete world at a
   time, with sigma0.25 Gaussian hours followed by the existing projection.
   Weights stay fixed within a rollout. After settlement perform 48 updates.
   Each arm has 576 actor updates and 576 updates to EACH critic. Final only.
4. Seal all six final trained models before opening any evaluation world.
   Evaluate constrained, cost-only, fixed and corrected ID-MPC on 36 fresh
   matched worlds total (3 blocks x3 conditions x4 worlds), greedy and frozen.

No extra smoke episode, qualification fit or checkpoint selection. The first
reference world is the budgeted real preflight and is retained, not repeated.
Mock checks must cover the real persisted metadata and entry point before it.
The fixed-reference replay capacity is576 and each learned replay capacity576;
never share learned trajectories between arms. Store raw requested/projected
hours, full cost components, final patient identities, losses, targets, gradients,
multiplier histories, optimizer receipts and RNG/restore states.

Restore support is for complete snapshots and forensic verification, not a
license to retry a terminated attempt. Each debit precedes dispatch and is not
refundable. No test-driven parameter change, replacement seed, extension or
automatic recovery. Failure preserves evidence and stops this attempt.

## Explicit Budget

| Phase | Worlds | Native steps | Actor steps | Critic steps | Forwards |
|---|---:|---:|---:|---:|---:|
| Reference and shared warmup | 36 | 2304 | 0 | 1536 | 3840 |
| Two learned arms | 72 | 4608 | 3456 | 6912 | 31104 |
| Four-controller evaluation | 144 | 9216 | 0 | 0 | 3456 |
| Total | 252 | 16128 | 3456 | 8448 | 38400 |

There are 11,904 neural optimizer calls and 36 separate scalar multiplier
updates. Warmup uses five module forwards (shared target actor, two target
critics, two current critics); training uses eight (the preceding five plus
actor and two actor-loss critic calls). Training actions add3456 and evaluation
actions add3456. Batch cap64; optimizer example presentations761856.

Count 252 constructors and252 constructor resets separately: total native
operations16632. No explicit resets, clones or simulator branches beyond these.
ID-MPC evaluation:1728decisions,82944candidate/quantile rollouts and663552model
epochs; never count them as free neural decisions. Filter:16128receipt updates,
1612800site-response-pair transitions. No saved-filter reconstruction.

CPU float32, one worker/four threads; total wall ceiling7200seconds including
recording and archive. Phase ceilings: admission300, reference/warmup1200,
training2400, evaluation2400, analysis/archive600, failure flush300. Each
reference block owner<=400s, each learned arm/block<=400s, each evaluation
controller/block/condition owner<=200s, also subject to its phase/global ceiling.
RSS4GiB; raw1GiB plus archive1GiB; max2000raw files. Unused budgets cannot transfer.

These are proposal arithmetic, not measured new-run throughput. Preparation
estimate: 60-120minutes of focused integration; scientific run estimate45-90min
with a hard2hour ceiling. Uncertainty: second-critic and new runner integration.
If preparation exceeds this estimate, report the named blocker and remove
optional reporting rather than adding another diagnostic campaign.

## Readout And Stop Rule

Primary condition: persistent change. Report full settled cost and lost/delivered
counts, all three block means and every paired world, for all conditions.
Primary training-gain contrast: constrained versus exact same-start fixed.
Objective-effect contrast: constrained versus matched cost-only. ID-MPC is an
external performance comparator, not a causal control for neural updates.

Development signal requires positive constrained-versus-fixed mean savings in
each persistent block, descriptive95%hierarchical interval for savings above0,
and nonpositive mean additional losses in EACH condition. Use 2000 bootstrap
resamples, blocks then paired worlds within condition; three independent models
is still a small pilot. This is not an equivalence or clinical safety test.
Also report intervals on loss differences, all violations, multiplier saturation,
tail liabilities, action changes and complete component costs. Training evidence
and old recovery2 test worlds are not independent confirmation.

- Cost down/loss up: trade-off; fail the patient-preserving signal.
- Better than fixed but worse than MPC: limited RL training gain, no demonstrated
  best-controller advantage. Do not hide either comparator.
- Fewer losses but higher cost: constrained trade-off, not cost superiority.
- No useful training gain: close this learner/objective trial; no extra epochs,
  lambda sweep, automatic online experiment or new scenario under this package.
- Useful gain: propose a separate replication and matched deployment-online
  comparison; neither is authorized now. Graph attribution also remains separate.

## Minimum Implementation And Approval

Reuse the corrected native world, public filter/features, projection, cost
reader, serial receipts, archive and watchdog. Add only a versioned two-critic
learner/typed loss labels, paired-reference training schedule and this proposal's
runner/reader. Necessary zero-optimizer fake tests: exact fixed start, identical
forks, tail loss/cost targets, dual sign/cap/condition isolation, gradient ownership,
matched shadow critic, seals-before-test and nonrefundable counters. No real
model loading or forward is allowed during unapproved preparation.

Before admission: obtain approval for the full numerical package, append exact
authorization, commit implementation/config plus source/runtime/input/seed locks
and validate the real entry with fake backends. No scientific run until then.
Only local commits; no push/PR/merge, Dropbox, messages, holdout or StageE reopening.
User's current `继续` approves preparing this proposal, not its unasked budget.

Independent finite design input: Hypatia, completed; coordinator adopted the
exact fixed start and training-only design. Existing efficiency advice reused:
one end-to-end comparison, no new toy gates or historical re-audit.

## Method Context

Separating reward and expected outcome constraints is established constrained-RL
framing, but this proposed approximate Lagrangian implementation inherits no
theoretical guarantees from other algorithms. See [Achiam et al., CPO](https://proceedings.mlr.press/v70/achiam17a.html)
and [Liu et al., CVPO](https://arxiv.org/abs/2201.11927). These references motivate
the formulation; they do not validate this simulator or its patient thresholds.
