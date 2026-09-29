# S3: independent action-ranking diagnostic on archived S2 states

## Authority, question and boundary

S2 completed and its readout was reviewed: four of six paired probe records
changed first request with sampling budget, and identification-plus-rollout
benefit did not replicate. On September 29 Zhaowei requested continuation and
automatic progression of routine local work instead of repeated manual prompts.
This authorizes the finite S3 diagnostic below, including implementation,
tests, one recorded forecast matrix, audit and readout. It is not authorization
for a neural campaign, arbitrary scenario search or a favorable outcome.

Question: **under the existing assumed response model and feedback continuation,
does independent Monte Carlo sampling reproduce the action ranking?** This does
not validate the assumed response model or establish actual policy performance,
strong-planner headroom, clinical benefit, GCN attribution or online-RL value.

Keep S1/S2 source and output bytes unchanged. Add an independent evaluation
module using the existing forecast API, not edits that invalidate S2 locks.
Read the locked plan and S2 audit before implementation. Commit this protocol
and config first; commit implementation/tests before recorded execution.

## Fixed inputs and finite computation

Use all six S2 probe records. Canonical context identity consists only of the
public observation, posterior mean response, downstream capacity and shared
mechanics config. Retain all family aliases. The six records represent four
unique contexts; compute each unique context once, then explicitly map aliases
to it. Identical public contexts must not get different outcomes because their
hidden family labels differ. Do not treat aliases as independent replications.

For every unique context, score all 15 existing quarter-unit requests. Optimize
only the first request, then follow the unchanged S1 booked-backlog **feedback**
rule. Preserve the original decision boundary, forced-zero boundary request,
full settlement and closure cap. Hold response at the archived posterior mean.
Use neither the private actual-world response nor future actual completion tape.
No new actual-world episodes, posterior/model changes or depth2 search occurs.

Four independent forecast blocks are fixed: selection, validation_a,
validation_b and validation_c. Each has 2,048 independent Monte Carlo paths per
action; budgets 128/512/2048 are nested prefixes, not separate samples or
optional stopping rules. All actions in a block share uniforms. Derive seeds
from the fresh namespace, canonical context hash and block label. Keep them
disjoint from S2 and all historical scientific streams. Do not reuse discovery
noise for validation. Chunk by 128 samples without changing RNG ordering.

This is exactly 4 contexts x 4 blocks x 2048 paths x 15 actions = 491,520
forecast paths. Store every action cost on every path, with identities, seed
derivation, array shape, input/source hashes and query counts. These are model
forecasts, not 491,520 independent environment episodes or patient outcomes.
Samples within a block pair actions; nested budgets and duplicated aliases
must not inflate effect sample sizes.

## Prespecified reporting

1. For each context/block/budget report all action means, the exact stable-grid
   argmin, and paired cost differences for all action pairs. Compute uncertainty
   from per-path differences, not by subtracting two marginal standard errors.
   Monte Carlo standard errors concern integration under this fixed model only;
   do not report them as uncertainty over operating conditions or model error.
2. Select the one action per context from the **selection block at 2048 only**.
   Freeze it before reading validation outcomes. In all three validation blocks
   report its full-cost differences versus the archived S2 depth1/16 action,
   archived S2 depth1/64 action, and the public booked-rule request at that state.
   Never choose whichever validation block/action/continuation looks best.
3. Separately show each block's own argmin, budget agreement, validation signs,
   and excess model cost of the frozen selection relative to each validation
   block's minimum. The latter is an in-sample descriptive gap, not unbiased
   optimality regret. Preserve exact ties; do not invent a clinical margin.
4. Mark ranking unresolved when independent argmins or improvement directions
   disagree. Agreement is a conditional diagnostic, never an automatic RL
   launch gate. No p-value, population-power claim, or practical-benefit
   threshold is supplied by this model-only exercise.

No sample expansion, reward tuning, new context selection, estimator change or
fallback replacement is permitted after reading outcomes. Negative, null and
unstable findings are complete, reportable results.

## Engineering, execution and stop rules

Tests must cover raw-cost/scalar parity, delayed action/full-tail accounting,
paired-difference arithmetic, prefix matching, block separation, canonical
duplicate mapping, frozen selection, missing/duplicate rows and failed closure.
Run relevant regression tests and full compileall. Audit all recorded arrays,
identities, counts and recomputed summaries. Independently reproduce bounded
sampled scalar paths and contrast arithmetic; do not rerun the entire matrix.

One local recorded run only, maximum 900 seconds with cooperative checks per
chunk. Fail closed at the existing closure cap or on nonfinite values, hash
mismatch, duplicate records or deadline breach. Preserve partial evidence and
error status, notify, and do not automatically retry or change the protocol.
Before launching, confirm no prior active/completed/failed S3 run is being
overwritten or duplicated. Never infer a running process from an automation.

After the audit, write a readout and a short continuation-adequacy decision
memo. The booked-rule fallback is known to be restricted; ranking convergence
does not make it a strong MPC bound. Design a credible feedback-continuation
comparison or explain why this channel should close, but do not execute a new
comparison or train DDPG without further scientific scope approval.

The automation may advance through all implementation/audit/documentation
steps here without repeated user approval. Stop when S3 and its decision memo
are complete, or a genuine scientific/permission blocker prevents further
authorized work. Ask once for the specific decision, not a generic "continue".
Do not impersonate Howard's approval, fill missing E1 data, push, merge, send
messages, buy compute or reopen Stage E. Scientific improvement is not promised.
