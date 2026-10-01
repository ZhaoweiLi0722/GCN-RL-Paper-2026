# P2 saved-training diagnostic

2026-10-01. Zhaowei replied "continue" after the proposed saved-data-only
follow-up. Scope is posthoc diagnosis of the completed P2, not another trial.
Base a7c891886b3bf67e956442b9a83c9f3430faebd5, persistent September worktree.
This is not a preregistered independent analysis: a preliminary saved-weight
readback already found all nine final scorer range bounds below the prior gap.
The fixed complete readback below quantifies that finding and remaining limits.

Inputs: all nine PPO models, their initial/final weights, 288 training episodes,
72 saved four-episode pre-update collector states, and 1,152 saved minibatch
records. Read the prior closure result only for context. No new test evaluation
or test-state scoring, environment construction, neural forward/backward,
optimizer update, counterfactual rollout or hyperparameter search is allowed.
Preserve source, run, failure records, result and archive hashes. New analysis
code, artificial arithmetic tests, documentation and local commits are allowed.

Fixed checks:
1. Derive an input-independent score-range bound from the implemented
   linear-tanh-linear scorer and saved output weights. Compare 2*L1(output
   weights) with log(9*(K-1)), the reference/nonreference prior margin. Common
   output bias cancels. Report singletons separately; the certificate applies
   only to this architecture, weights and prior, not future fitted weights.
2. Reconcile all recorded probability-clipping fractions, total-gradient norms
   and loss components. A large value loss or total norm alone does not prove
   component-gradient domination, particularly under Adam.
3. Independently sum the once-scaled negative raw costs backward within each
   terminal episode; compare with persisted gamma=lambda=1 GAE returns and
   advantages. Recompute normalization and first minibatch objective from saved
   receipts only. Verify recorded indices cover each rollout once per epoch.
4. Quantify descriptive within-rollout time-position variation in advantages,
   sampled reference/nonreference groups and the reference-logit score-function
   signal. These are not same-state causal action advantages or headroom tests.
5. Bound value outputs using the saved final affine weights/bias; quantify
   observed training targets outside its mathematical range. This is a fixed
   checkpoint representability bound, not an inability of the trainable model
   class to represent those values after additional fitting.
6. Reconcile raw cost-component shares; retain original weights, objectives and
   terminal accounting. Do not tune reward to obtain a positive result.

Numerical tolerances cover known float32 GAE accumulation, not missing data.
No significance claims from repeated steps or minibatches; nine fitted models
are grouped under three original blocks. No success gate is adjusted. Report
observed, mathematical and unresolved causal findings separately, with a
specific next-design decision rather than automatically extending execution.

Freeze diagnostic code after artificial tests and full compilation, then run
once to produce a non-overwriting report outside the original result tree.
Readback-code/schema errors may be corrected with preserved error evidence;
that does not permit replaying the scientific experiment. Verify original
payload/launcher byte inventories before and after. Copy only completed new
diagnostic evidence to a new Dropbox-local subfolder; cloud sync/access remain
unverified. No remote Git action, message, holdout or Stage E reopening.
