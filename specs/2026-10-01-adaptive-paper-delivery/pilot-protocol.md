# S1 dynamic candidate mechanism pilot

Prospective proposal, not execution authorization. This protocol resolves the
declared draft's implementation fields without changing its numerical budgets,
reward, scenario, seeds, threshold or one-attempt rule. The preserved
`pilot-budget-draft.json` stays false/unapproved. The exact generated proposal,
source commit, input hashes and runtime are in `frozen-proposal/proposal.json`
when that file has been created after the implementation commit.

## Question and scope

Does simulator-interaction PPO improve a graph-aware candidate selector beyond
its own competent frozen initializer and continued R4 imitation? This is not
deployment-time parameter adaptation, an isolated GCN effect or PPO-versus-DDPG
superiority. The final research goal remains constrained cross-facility
coordination; the current restricted space is a mechanism test, not a permanent
ceiling. No positive result, clinical noninferiority or publication is promised.

The simulator retains specimen, reagent, capacity and replenishment operations.
The candidate message graph is `specimen_routes`, an explicit prospective
correction of the original `physical_shared_relations` label. The historical
R4 reference retains its own representation. The new actor selects among R4,
MDL-2 and specimen corrections of -0.10,-0.05,+0.05,+0.10. Non-specimen request
components must agree within a bank; aliases are canonicalized, and exact
original float64 requests are submitted. No hidden patient-level feasibility
mask or general independently learned joint control is introduced.

## Fixed design

- Three independent blocks60,61,62; graph-only, CPU float32, deterministic
  runtime, encoder width16 and head width32. Actor31,346 and critic28,721
  parameters are separate. Initial reference bias0.0 is trainable, selected
  prospectively rather than from fitted outcomes. The critic's final output
  layer starts at zero, with no critic prefit or extra warmup.
- One existing 52-step persistent-hotspot scenario. Reward is negative raw
  absolute simulator cost scaled by1e-9; gamma=1, GAE lambda=1, terminal
  bootstrap=0, no added terminal cost or changed cost weights. Patient
  completions, losses, waiting and unfinished terminal obligations are reported
  separately. This is not a full-lifecycle cost or clinical calibration claim.
- Eight R4 demonstration episodes and256 actor-only BC calls per block.
  Qualification uses two fresh paired worlds per block and both R4/greedy own
  paths. Each path requires >=95% overall and multiclass action agreement and
  at least one multiclass row. Block-mean initializer-minus-R4 cost,losses and
  terminal-active must be <=0; completions must be >=0. These are development
  rejection screens, not clinical noninferiority margins. Any block failure
  closes the attempt before continuation or test, without refitting.
- Fork actor,critic and buffers identically into frozen/PPO/BC-CONTINUE.
  Continuation optimizers start fresh; PPO/BC sampling RNG starts identically
  but may diverge with behavior. Thirty-two episodes per continued arm/block,
  eight four-episode rollouts, four epochs of batches64/64/64/16. PPO makes128
  actor and128 critic calls per block; BC-CONTINUE128 actor calls only.
- Adam learning rate0.0003, betas(0.9,0.999), eps1e-8, weight decay0,
  foreach=false; separate owner gradient norm cap0.5; PPO clip0.2, entropy0.01,
  value coefficient0.5. Advantages/returns are fixed from sampled behavior,
  normalized once per complete rollout. No counterfactual target or tuning.
- All nine learned artifacts must be sealed before any test starts. Final-only
  greedy evaluation uses12 fresh paired worlds/block for own-frozen,own-PPO,
  own-BC-CONTINUE,R4 and full-MDL2:180 evaluation episodes. No favorable
  checkpoint or seed selection, extra stochastic mode or test fitting.

## Accounting and closure

The original JSON lists every phase and arm cap. Main totals:426 full episodes,
22,152 trajectory steps plus72 mandatory restored-clone steps =22,224 env.step
calls;1,536 actor plus384 critic calls =1,920 optimizer.step calls. At most429
environment constructions:426 episodes and3 layout preparations. No implicit
reset or optional72-call appendix. There are168 unique world starts and19
private neural/analysis streams in the prospective namespace. Pairing world
starts does not assert identical event-level RNG after different actions.

Phase time allocations (seconds): input/setup600; prototype preflight600;
demonstrations900; initialization1,800; qualification900; same-start
preflight1,200; PPO4,500; continued BC2,700; all-model seal300; evaluation4,500;
raw verification1,200; local archive1,200; supervisor/terminal closure900.
Their sum is21,300. Global elapsed cap is21,600 seconds including initial
binding, I/O, verification and preservation; unused allocations cannot move
between phases or fund optional calls. Owner partitions remain in the JSON.

Every environment/optimizer call is debited before execution. Deadline checks
precede forward/backward and restoration operations; the external same-host
watchdog independently enforces startup, current scope, aggregate phase and
global deadlines, including terminal closure. Polling delay and bounded
termination grace are reported, not hidden as exact real-time guarantees.

The supervisor creates an exclusive fresh result root before any numerical
work. No second claim, resume command, repair-and-rerun or parameter change is
permitted. Checkpoints preserve model/optimizer/RNG/collector state for
inspection and in-owner boundary restoration; they do not authorize restarting
a terminated scientific attempt. On interruption, native crash, failed
qualification, invalid evidence or deadline exhaustion, save available partial
evidence and report failure. An incomplete timeout preservation is not success.

## Endpoints and preservation

Primary: paired raw total cost of PPO minus own-frozen. Relative change is the
equal mean over blocks of100 times the block mean paired cost difference
divided by block mean frozen cost. Nonpositive denominators are undefined.
The prospective development screen is at least1% cost reduction with favorable
cost direction in all3 blocks; patient-adverse direction is reported as a
trade-off, not overall success. Secondary contrasts: PPO minus BC-CONTINUE,
R4 and full-MDL2, plus own-frozen minus R4. Report every world/block, action
changes, raw and executed requests, patient outcomes and computational cost.

Use10,000 hierarchical paired block-then-world bootstrap draws with fixed
analysis RNG. Three trained blocks give low-precision descriptive development
uncertainty, not formal confirmation. No equivalence or clinical margin is
invented. Independent saved-data verification checks identities, costs,
actions, totals, seals and hashes without rerunning patients or learners.

Preserve source,input checkpoints/configs,effective proposal,approval,raw
records,model/full states,ledger and local tar archive with byte/member checks.
The result root is `results/dynamic_candidate_pilot_20261001`. No Dropbox copy,
cloud-sync/access claim, remote push/PR/merge, message, external compute,
formal holdout, Howard sign-off claim or Stage E reopening is included.

## One required decision

Explicit approval must bind this protocol, the final frozen proposal and
implementation commit, and acknowledge three prospective choices:
`specimen_routes`, neutral trainable bias0.0, and replacement of the failed A1
artificial prerequisite by this bounded end-to-end test. A1 stays failed1/9;
it is not renamed a pass. Record the user's actual words and timestamp in the
fixed-path authorization file, append locked-plan change control, commit both,
then execute one attempt. No authorization file with approved=true is created
during preparation. Routine steps within an approved packet need no further
microapproval. Subsequent scientific scope requires a new decision.
