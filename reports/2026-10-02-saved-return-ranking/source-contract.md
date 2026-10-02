# Saved Return Ranking: Frozen Source Contract

## Decision

**Do not diagnose missing time input or minibatch-normalized value targets.**
The dynamic critic receives `normalized_time`, graph/node information and public
request context. PPO fits unnormalized, once-scaled terminal Monte Carlo returns;
only advantages are standardized, once across each complete 208-row rollout.
The next interpretation is therefore about the recorded baseline's fit and
time-dependent credit signal, not a demonstrated missing-input or unit defect.
Raw input scales, two tanh stages, zero-output critic initialization and limited
critic updates are source-supported candidate mechanisms, **not established
causes** of near-constant values or the null greedy-policy increment.

The coordinator reports final-rollout value EV of 0.00282/0.00106/0.00214 for
blocks 60/61/62, return-versus-step correlations 0.959-0.975, advantage-versus-step
correlations 0.885-0.919, and matching first-minibatch loss reconstruction for all
24 rollouts. These numbers are coordinator-supplied saved-JSON findings, not
recomputed here. Crucially, the last rollout's stored values are **behavior
values collected after seven updates and before update eight**, not an evaluation
of the final eight-update checkpoint. See the collection/update ordering in
[dynamic_candidate_campaign.py:403-428][campaign] and sealed evaluation in
[dynamic_candidate_rollout.py:14-30][rollout].

## Scope and Evidence

- Worktree: `/Users/lizhaowei/GCN-RL Paper 2026/worktrees/september-research-integration`;
  branch: `codex/september-research-integration`.
- Read AGENTS.md and the latest locked-plan entry, lines 2567-2600. Reuse the
  [completed comparison readout:5-21,114-130][readout]: PPO/frozen/BC/R4 greedy
  actions and outcomes matched; probability vectors changed and non-R4 classes
  were sampled during training. No comparison or historical audit was repeated.
- All source links below point to the completed run's preserved frozen copy,
  under `results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/`.
  The 20 reviewed source files were compared byte-for-byte with the current worktree
  copies and matched. This was a targeted source-identity check, not an archive
  or whole-history verification. Recorded execution commit:
  `63c2383d33f5795f22ec1173bd75222a30a63a4a` ([closure-index.json:2-4][closure]).
- Only source text and JSON were read. No model/checkpoint deserialization,
  scientific imports, tensor computation, forwards, optimizer/environment calls,
  fitting, result edits, tests, or compileall were run. This is a documentation-only
  deliverable, not a new acceptance gate. Coordinator-owned diagnostic files,
  tests, memo, Live/workflow and frozen sources remain outside this file's ownership.

## Units and Return Construction

| Quantity | Exact contract | Source |
| --- | --- | --- |
| `record.raw_reward` | Negative absolute environment step cost, in modeled objective units, not dollars and not an R4-relative reward. Restore checks `raw_reward == -info.cost`. | [session:255-265,383-389][session]; [proposal:409-422][proposal] |
| Training reward `r_t` | `float32(raw_reward * 1e-9)`. Scaling occurs once in segment preparation, not in the stored raw reward. | [candidate_rollout:230-238][segments] |
| Collected `V_t` | Scalar critic output stored directly; consumed in the same once-scaled return units. No multiplication by reward scale, z-score or inverse transform is applied to V. A unit contract is not evidence that V is accurate. | [dynamic rollout:21-27,49-54][rollout]; [candidate_rollout:231-232][segments] |
| GAE | `delta_t = r_t + gamma * V_next * nonterminal - V_t`; `A_t = delta_t + gamma * lambda * nonterminal * A_next`; target `G_t = A_t + V_t`. | [baselines/ppo:230-240][gae] |
| Frozen settings | `gamma=1`, `lambda=1`, horizon 52, terminal bootstrap 0, added terminal cost 0. In exact arithmetic, `G_t = sum(k=t..51, r_k)` and `A_t = G_t - V_t`. | [proposal:412-422][proposal] |
| Boundaries | Every training episode is a separate closed terminal segment. The generic replay contract permits truncation bootstrap, but this session chooses terminal, passes `bootstrap=None`, and marks step 51 terminated, not truncated. Never accumulate returns across episodes. | [session:166,309-317,387-389][session]; [candidate_rollout:215-238][segments] |
| Precision | The GAE core uses float32 rewards, values and advantage storage even if a caller's policy dtype differs; the completed proposal uses float32. Targets are cast to kernel dtype when updating. | [candidate_rollout:230-242][segments]; [baselines/ppo:230-239][gae]; [dynamic PPO:209-213][ppo]; [proposal:368-369][proposal] |

For saved arithmetic, retain both high-precision mathematical suffix sums and
the source-faithful float32 recurrence if distinguishing numerical differences.
`A + V` has float32 rounding; a float64 suffix sum is not promised to reproduce
every stored loss bitwise. A value unit is 1e9 raw modeled objective units; MSE
has squared scaled-return units. The 1e-9 factor does not change the underlying
cost objective or add terminal obligations.

## Normalization and Old Probabilities

1. **Rollout grouping:** four consecutive 52-step episodes per update, eight
   updates per block. Flatten in episode order then step order. Advantage mean
   and population standard deviation use all 208 rows in that one block/update:
   `A_hat = (A - mean(A)) / (std(A, unbiased=False) + 1e-8)`.
   It is not per episode, time, action class, minibatch, block-wide, or pooled
   across blocks. [proposal:203-249][proposal];
   [continuation:35-45,77-85][continuation]; [dynamic PPO:200-213][ppo];
   [objective:36-46][objective].
2. **No minibatch target normalization:** `returns[indices]` is passed directly
   to `mse_loss(values, returns.detach())`. Minibatches 64/64/64/16 are shuffled
   anew in each of four epochs, but return targets, rollout-normalized advantages
   and old log probabilities remain fixed through all epochs. There is no value
   clipping, return running-stat normalization, GAE recomputation after critic
   updates, or KL early-stop in this update loop.
   [dynamic PPO:209-235][ppo]; [objective:59-90][objective].
3. **Old probability is the chosen canonical class's behavior probability:**
   `old_logp = evaluation.log_probs[choice.class_index]`, detached in
   `rho = exp(current_logp - old_logp)`. Collection stores normalized categorical
   log probabilities, not raw scores. They are not refreshed to each epoch's
   current policy. [candidate_rollout:47-76,100-102][segments];
   [dynamic rollout:21-27][rollout]; [objective:77-84][objective].
4. **Support is frozen per decision, not globally:** reevaluation uses the saved
   observation and that decision's saved candidate bank, checks schema, model
   definition and actor input, and returns current chosen-class logp, V and entropy.
   Admission reproduces behavior receipts before fitting and rejects altered
   returns or repeated consumed records. These are source guards; this review
   did not rerun model-based receipt validation.
   [dynamic rollout:33-54][rollout]; [dynamic PPO:171-190][ppo];
   [candidate kernel:159-170][kernel].
5. **Canonical classes, not option slots:** one categorical logit per unique
   request class. `reference_class = request_to_class[0]` and
   `anchor_class = request_to_class[1]`; either can alias another request. Do not
   assume class 0 is R4 or that class index j means the same request across states.
   `Categorical` normalization is a probability operation, not target normalization.
   [routing contract:104-120,156-177][routing].

## Critic Information and Initialization

**Yes, normalized time reaches the critic in collection AND minibatch updates.**

| Link in the path | What reaches the value branch | Source |
| --- | --- | --- |
| Environment | `clip(t / episode_horizon, 0, 1)` appended as float32; patient summary preserves time as the last raw field. | [capacity env:469-474,1517-1518][capacity]; [patient env:166-182][patient] |
| Public producer | Schema names global feature `normalized_time`; 20 x 28 node features are assembled without feature standardization, raw trailing time becomes `observation.globals`, and specimen-route links become the physical graph. | [producer:73-117,126-144][producer] |
| Update reconstruction | `actor_state` is split back into nodes, globals, links and anchor; global time is not discarded. | [candidate kernel:137-139][kernel]; [adapter:97-106][adapter] |
| Shared input layout, independent weights | Both value and score branches request `role="actor"` deliberately: public nodes, global time, flattened physical links and anchor; an additional R4 reference request is passed separately. This is state-value V(s), not action-value Q(s,a). | [dynamic policy:209-230][policy]; [matched inputs:103-128][inputs] |
| Critic encoder/head | An independent graph linear projection uses normalized adjacency; flattened node encodings pass tanh, then concatenate public context and reference. Value head is `Linear -> Tanh -> Linear`, producing an unbounded scalar. No selected action, per-class candidate features, actor reference-bias scalar or candidate-pool aggregation enters V. | [dynamic policy:33-48,68-77,184-194,222-233][policy]; [gcn:53-62][gcn] |

Direct saved receipt spot-check: training block60/episode00 `events.jsonl` lines
1-2 name `global_feature_names=["normalized_time"]`, 28 node features and a
1041-wide `actor_state`. Its zero-based index 560 is 0 at step 0 and
0.01923076994717121 at step 1 (float32 1/52). Line 52 records terminal step 51.
This checks the actual stored layout without invoking the model. [receipt][events]

The critic's frozen dimensions are 20 x 16 graph encodings + 1 time + 400 links
+ 80 anchor + 80 reference = 881 inputs to a 32-unit hidden value layer;
28,721 critic parameters are declared. [dynamic policy:135-142][policy];
[proposal:365-376][proposal]. Graph degree normalization is
`D^-1/2 (links + I) D^-1/2`, **not node-feature normalization**.
[matched inputs:129-142][inputs].

Source facts relevant to low variance, without causal overreach:

- **Constant-zero start is intentional.** Independent seeded float32 default
  Linear initialization is followed by zeroing only the critic's final weight
  and bias. Imitation initializes the actor but freezes and verifies the entire
  critic; there is no critic prefit or extra warmup. Thus a competent actor does
  not imply a competent baseline. [dynamic policy:80-99,135-147][policy];
  [imitation:49-66,140-144][imitation]; [proposal:328-345,238-244][proposal].
- **First-gradient implication, not an empirical diagnosis:** with the final
  critic weight initially zero, the first backward has zero data gradient to
  earlier critic layers by the chain rule; final weight/bias can learn. Later
  steps are not structurally forced to remain constant. No activation or
  per-layer gradient measurements were made here.
  [dynamic policy:68-77,145-147][policy].
- **Potential scale/saturation issue remains unmeasured.** The path feeds raw
  count/inventory/survival features to a graph linear layer then tanh, while
  normalized time bypasses that first tanh through context and meets a second
  head tanh. There is no adaptive observation standardization or LayerNorm in
  this encoder/head. Different feature scales and many context dimensions can
  motivate examining saturation/time sensitivity in a separately authorized
  model diagnostic; source alone cannot establish either. Time being present
  also does not prove the fitted critic uses it effectively.
  [producer:126-144][producer]; [dynamic policy:33-48,68-77][policy].
- **Optimization is bounded and separate.** Forks copy the qualified policy but
  start fresh Adam moments; critic has 128 minibatch steps per block, learning
  rate 0.0003, loss coefficient 0.5, own gradient cap 0.5, no weight decay.
  These are candidate constraints, not evidence that more steps, a larger rate
  or a different scale would fix learning. [factory:116-140][factory];
  [dynamic PPO:117-141,238-258][ppo]; [proposal:231-249,424-435][proposal].

## Gradient Ownership

For minibatch mean losses, the source implements:

```text
L_policy = -mean(min(rho * A_hat, clip(rho, 0.8, 1.2) * A_hat))
L_value  = mean((V_current - G_fixed)^2)
H        = mean(categorical_entropy_current)
L_actor  = L_policy - 0.01 * H
L_critic = 0.5 * L_value
```

The helper returns a combined `total`, but **dynamic PPO never backpropagates
that total**. It zeros all gradients, backpropagates `L_actor`, checks no critic
gradient, clips only actor gradients, and steps actor Adam. It then zeros all
gradients, backpropagates `L_critic`, checks no actor gradient, clips only critic
gradients, and steps critic Adam. Parameter storage ownership is explicitly
disjoint. V was evaluated before the actor step, which cannot change the
independent value parameters. [objective:77-90][objective];
[dynamic PPO:91-97,117-141,225-266][ppo].

The saved actor norm is for **policy plus entropy together**; the saved critic
norm already includes coefficient 0.5. Saved `policy_loss` excludes entropy;
saved `value_loss` is unweighted MSE; saved `entropy` is unweighted mean entropy
in nats over unique classes, not divided by log(class count). Large critic loss
cannot directly consume the actor's clipping budget. The critic can still
influence future actor updates through newly collected baseline advantages.

## Persisted Diagnostics and Reconstruction Boundary

Actual JSON receipts exist at
`results/dynamic_candidate_continuation_recovery_20261001/payload/phases/ppo_continuation/block{60,61,62}.json`.
Each has eight `updates`, 16 minibatches per update, 208 rollout rows, and final
cumulative counters 128 actor + 128 critic = 256 optimizer steps. This is 384
PPO minibatch records across the three blocks, not 384 independent experiments.
The three BC phase receipts separately have eight updates and 128 actor steps
per block, with `indices`, `cross_entropy`, `grad_norm`, no critic updates.

Concrete line anchors: all three PPO receipt files use lines 552-565 for
job/model/update/first-minibatch fields, 631-632 for first policy/value losses,
1568-1570 for first rollout counters, and 8687-8689 for final counters.
[block60][p60], [block61][p61], [block62][p62]. For example, block60's first
minibatch records policy loss 0.040851056575775146, value loss 2.8151626586914062,
entropy 1.6996777057647705 and clip fraction 0.0. This is an existing receipt,
not a new numerical run.

Persistence is explicit: `Continuation.update()` returns kernel diagnostics;
the campaign appends them to `work.updates`, then writes `work` to the phase
JSON. A kernel checkpoint's `history` itself stores rollout lengths, not these
loss records. [continuation:93-109][continuation]; [dynamic PPO:262-274,311-320][ppo];
[campaign:424-428,559-566][campaign].

| Available without models | Not present as per-minibatch JSON diagnostics |
| --- | --- |
| `epoch`, exact flattened `indices`; current pre-step policy loss, unweighted MSE, mean entropy, clip fraction; actor/critic pre-clip norms; cumulative owner step counts | Current per-row log probabilities/ratios/value predictions, full probability vectors on update rows, KL, policy-only versus entropy-only parameter gradients, per-layer gradients/activations, clipped gradients, Adam parameter deltas |
| Every collection row's raw reward, old V, normalized class logp vector, selected class, candidate support, state/time and boundary flags | Alternate-action outcomes at that same state; final-checkpoint V on the last training rollout |
| Source-faithful behavior GAE/MC targets, rollout advantage mean/std, sampled-class descriptive return summaries, old entropy, baseline residuals/EV, first-minibatch losses at the unchanged behavior start | Exact later-minibatch loss reconstruction from collection JSON alone after weights have changed; gradient attribution or sensitivity of the final fitted model |

`clip_fraction` means the proportion with `abs(rho - 1) > 0.2`; it is not
necessarily the proportion whose clipped branch wins the surrogate minimum.
The diagnostics are measured on pre-step forwards; the logged norms precede
clipping. Within an epoch, use row counts to weight 64/64/64/16 loss summaries
if reporting a row-weighted descriptive mean; those batches were evaluated at
different model states and are not a single-model full-rollout loss.
[objective:81-90][objective]; [dynamic PPO:219-266][ppo].

For update `u` (one-based), flattened index `i` maps to zero-based episode
`4*(u-1) + floor(i/52)` and step `i % 52`. Use the saved `indexes` paths and
trajectory IDs, not filesystem listing order. Data fields are
`event.audit.record.raw_reward`, `.step_index`, `.terminated`, `.truncated`,
`event.audit.decision.evaluation.value`, `.log_probs`, `.actor_state`,
`.contract.replay`, `.candidates`, and `event.audit.decision.choice.class_index`.
`reward_kind` is in `evaluation.contract.replay` and in the record's nested
`semantics`, not a top-level record field.
[campaign:411-446][campaign]; [dynamic PPO:203-213][ppo]; [receipt][events].

## Mismatch and Claim Register

| Suspected explanation or mismatch | Source-supported disposition |
| --- | --- |
| Critic omits time/graph/global inputs | Ruled out as a wiring claim in this path; time is the only named global feature, and both encoded graph nodes and raw physical links reach V. Effective learned use remains unknown. |
| Value target is standardized per minibatch but V used as raw-scaled return | No such transform exists here. The coordinator's 24 first-minibatch reconstructions further support the stated arithmetic; they do not verify every later gradient. |
| Combined helper/docstring implies shared actor/critic gradients | The objective module's lines 5-6 describe shared-encoder use, not this dynamic caller. The dynamic kernel explicitly separates parameters, backward passes, clipping and Adam. This documentation-context mismatch is real, not evidence of gradient leakage. |
| Generic `bootstrap_on_truncation=true` conflicts with terminal-zero proposal | Not a conflict: this campaign terminates complete episodes; termination takes precedence. No continuation value or added terminal-cost label is introduced. |
| Last-rollout low EV is final-checkpoint low EV | Incorrect time attribution: collection values precede the last PPO update. Low EV also measures residual variance, not absolute calibration; constant offsets are invisible to EV alone. |
| Critic saturation, missing time sensitivity, entropy dominance, optimizer starvation caused the null | Unproven. Saved scalar loss magnitudes/norms cannot identify parameter-gradient directions, saturation or a counterfactual training outcome. Model loading plus suitable forwards/autograd would be needed for unlogged activation/sensitivity/gradient diagnostics; such work is outside this task. Even those diagnostics alone would not prove a remedy's causal effect. |
| Mean sampled return by class identifies the best action | Unsupported causal claim. Returns share downstream rewards and vary strongly with remaining horizon, visited state, candidate support, policy version and subsequent sampled actions. A single selected-action return is not a same-state alternate-action Q label. Condition descriptive summaries by block/update/time and actual class role/support; do not rename them an oracle ranking. |
| Lower R4 probability proves entropy harmed policy performance | Probability change is established by the completed diagnostic; its cause is not. Old probability/advantage arithmetic can describe a local categorical signal, but not separate shared-parameter policy/entropy gradients or reconstruct Adam's effect. |
| Zero greedy increment means no learning/headroom, graph failure, or no stochastic-policy benefit | The completed comparison establishes only its observed greedy action/outcome equality. It does not establish these broader claims; model loading alone would not supply missing counterfactual patient outcomes or independent comparisons. |

**Coordinator handoff:** retain the completed null comparison; combine the
saved arithmetic with the proven presence of time and the raw-target contract.
Describe the baseline as failing to remove much of the observed time-related
return variation at collection, using the coordinator's diagnostics, not as
proven architecturally unable to see time. Treat saturation/scale/optimization
explanations as unresolved. No automatic model load, corrective training,
reward change, additional test gate or experiment is authorized by this file.

[readout]: ../2026-10-01-s1-continuation-recovery/readout.md
[closure]: ../2026-10-01-s1-continuation-recovery/closure-index.json
[proposal]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-proposal.json
[continuation]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_continuation.py
[ppo]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_ppo.py
[rollout]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_rollout.py
[objective]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/candidate_ppo_objective.py
[segments]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/candidate_rollout.py
[gae]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/baselines/ppo.py
[session]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_session.py
[campaign]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_campaign.py
[factory]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_factory.py
[policy]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/models/dynamic_candidate_policy.py
[imitation]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/dynamic_candidate_imitation.py
[kernel]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/candidate_ppo_kernel.py
[routing]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/routing_candidate_contract.py
[producer]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/patient_replay_collector.py
[inputs]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/models/matched_inputs.py
[adapter]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/rl/prospective_adapter.py
[gcn]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/models/gcn.py
[capacity]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/env/capacity_planning.py
[patient]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/locks/worktree/src/env/patient_capacity_planning.py
[events]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/episodes/training/block60/graph/own_ppo/episode00/events.jsonl
[p60]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/phases/ppo_continuation/block60.json
[p61]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/phases/ppo_continuation/block61.json
[p62]: ../../results/dynamic_candidate_continuation_recovery_20261001/payload/phases/ppo_continuation/block62.json
