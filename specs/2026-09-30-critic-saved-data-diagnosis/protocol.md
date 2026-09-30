# R6 saved-data diagnosis (post-run, descriptive)

This is the next local analysis requested by Zhaowei's "continue", not a new
experiment or an amendment to the closed R6 attempt. Its results are post hoc.
No simulator reset/step, neural inference, fitting, optimizer update, new RNG
draw, reward change, action reselection, or new holdout is permitted. No Howard
approval is inferred. Stage E remains closed; no remote Git or messages.

## Sources and immutable boundary

Use only `results/clean_critic_generalization_20260930/`: all 72 saved states,
3456 outcomes, paired eight-draw labels, saved training predictions and sealed
test predictions. Check the entire recorded inventory and source locks before
and after analysis. The inventory SHA256 must be
`29ddb322f5da8b003f68008cf8fba95c2cc8489e30f71f0b9bfd27b2cf38bb35`.
Preserve all R6 files and the original 4-train/2-test trajectory split per model.
Derived outputs go to a new report directory and refuse overwrite.

## Fixed diagnostic questions

1. Label stability: for every map-certified action pair in every state,
   contrast draws 0-3 versus 4-7. Separate same nonzero sign, opposite sign,
   exact ties in both halves, and a tie in one half. Compare the already saved
   critic rankings on same-sign and opposite-sign subsets. Report all pairs;
   subsets are diagnostic, not a revised benchmark or independent validation.
   A pair's standard error is calculated from its eight *paired differences*,
   not by assuming action labels independent. Report the descriptive fraction
   with absolute mean exceeding two estimated SEs, without p-values or claims
   of independent pair/state replications.
2. Coverage: reconstruct the unchanged deterministic public graph features
   on CPU, without constructing an environment or network. Describe scales,
   exact duplicate rows and constant training coordinates. Normalize distances
   only for this diagnostic by training-coordinate population SD; exclude zero
   SD coordinates from distances and explicitly count novel constant values.
   For each state compare nearest training state at the same t; exclude the
   entire parent trajectory for training queries. Report test distances versus
   the four training leave-one-parent-out distances at the same t. Small-sample
   range exceedances are not proof of out-of-distribution deployment.
3. Execution equivalence: identify certified action pairs with equal recorded
   execution identities in all eight draws. Compare their existing predicted
   advantage gaps and quantized requests. This is sampled equality, not global
   equivalence; no deployed action is reselected.
4. Objective accounting: decompose every one of the 24 sealed test choices
   against frozen into all logged cost components, restoring exact saved tail
   aliases. Reconcile per-draw sums to raw outcome deltas and patient-loss
   charges to the fixed 500000 weight. Highlight, but do not remove, choices
   with lower mean total cost and higher mean patient loss. Do not search cost
   weights or infer clinical significance from these small conditional means.
5. Representation inspection: document what patient and production-stage
   information the public observation and graph transform retain or summarize.
   Exact saved duplicates may show aliasing; their absence cannot prove a
   sufficient/Markov representation. Source-level omissions alone cannot
   establish that added features would fix R6.

## Verification and completion

Implement isolated analysis helpers and synthetic unit tests, then commit
before deriving the report. Run the deterministic diagnostic twice and require
identical substantive output; independently recalculate headline denominators
and all cheaper-but-more-loss component sums. Run related tests and full
compileall. Record limitations, source hashes, script commit, and a decision
memo in the live checkpoint. This finite analysis does not authorize new
training. Any prospective data/fit/objective experiment requires a concrete
bounded proposal and explicit approval.
