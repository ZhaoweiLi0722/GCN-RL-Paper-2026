# S4: bounded feedback-continuation comparison

## Authority and question

On 2026-09-29, after the audited S3 result and explicit continuation-comparison
proposal, Zhaowei requested continuation and consideration of reward-function
improvements if progress remains absent. This authorizes the proposed finite
model-only packet. It does not authorize reward changes, cost tuning, neural
training, new actual-world/patient episodes, holdout use or remote operations.

Question: do the first-request ranking and conditional total cost change when
the post-request feedback policy is less restricted? This is not an online-RL
attribution study or validation of the response model. Preserve all S1/S2/S3
source and evidence. Commit protocol/config before implementation, and source
before the single recorded run. Audit S3 before proceeding.

## Inputs and policies

Reuse all four canonical public contexts (six aliases), the 15 quarter-unit
actions, constant archived response means, mechanics, decision boundary 32,
one-interval actuation delay, forced-zero boundary request and closure cap 256.
No private family label, actual future completion tape or evolving true rate
is available to a controller. No belief updating is introduced to one arm.

Every method has a candidate first request from the same 15-action grid:

1. `booked`: subsequently use the unchanged S1 booked-backlog rule.
2. `reservation`: subsequently use the existing S2 reservation rule until the
   decision boundary; after forced zero, use the common S1 closure rule.
3. `tree128`: after observing first-interval completion events, choose a second
   request from a frozen event-conditioned table; thereafter use booked feedback.
4. `tree32`: the same tree construction from a nested 32-sample prefix, retained
   solely as a numerical-sensitivity comparator, never a post-hoc substitute.

The tree has only two optimized requests, not a globally optimal feedback
policy or strong MPC bound. First-interval completion depends on *previously
booked* hours, not the first new request. Enumerate its reachable two-site
Bernoulli event branches with their exact probabilities from the public state
and fixed model (at most four). Use scalar physics to obtain each next public
state for every candidate first request. The tree observes that state before
acting; it never sees the next or subsequent completion draw.

For each context/branch/first request, estimate full-settlement costs of all 15
second requests under the booked tail with 128 inner paths and a nested prefix
of 32. Use the stable grid argmin for each budget. No arbitrary terminal value
surrogate is substituted. Record every inner per-path cost, mean and the paired
gap/Monte Carlo SE between each selected action and every alternative. Report
32-vs-128 branch-choice disagreements; small planning budgets are a limitation,
not a convergence certificate. All tree tables freeze before outer sampling.

## Independent assessment and accounting

Seeds derive from the fresh namespace, canonical context hash and either
`tree_fit` plus branch code, or an outer block label. Inner uniforms are common
across candidate first/second requests within a branch; they are independent
of all outer blocks. Hidden family aliases never affect seeds or outcomes.

Four outer blocks are fixed: selection, validation A/B/C. Each has 2048 paired
paths, nested prefixes 128/512/2048, with common uniforms across all first
requests and methods. Chunk by 128 preserving path/time/site ordering. Select
one first request per method/context using selection at 2048 only. Freeze all
selections before any validation. Preserve each path's holding, labor and
switching costs, including full settlement and prepaid obligations.

For every method/context/block/budget report all first-request cost means and
stable argmin. For validation report frozen tree128 minus frozen booked and
reservation continuations, the same contrasts for tree32, tree128 minus tree32,
and tree128 minus the unoptimized first-request versions of both public rules.
Report paired total-cost difference/MC SE and cost-component differences. Also
record continuation-induced argmin changes and budget sensitivity. Never choose
a policy using validation outcomes or pool duplicated aliases as replications.

A context has a direction-stable conditional signal only if both tree budgets
have negative mean differences against both independently frozen comparator
continuations in all three full-budget validation blocks. This descriptive
gate is not significance, a clinical threshold, a population effect or an RL
launch gate. Report every context; do not select an attractive scenario.

There are exactly 1,966,080 outer model paths. Inner paths are at most 460,800;
reachable branches alone determine their count. Maximum total is 2,426,880
paths and 698,941,440 scalar-equivalent transition queries. These are forecasts,
not real environment episodes. Include inner planning cost; do not label
outer evaluation alone as the full computational budget.

## Verification and stop rules

Maximum 900 seconds with cooperative checks between inner tasks and outer
chunks, including the bounded audit. One recorded matrix only, no automatic
retry or post-outcome extension. Fail closed on nonfinite metrics, invalid
branch/table lookup, hash/shape/count mismatch, closure failure or deadline.
Refuse existing output roots and inspect processes before launch. Retain
partial chunks, error and failed status on any terminal failure.

Tests cover reachable branch probabilities, one-step delay, scalar/batched
component parity, old booked forecast parity, reservation-rule parity, causal
tree lookup, no future-noise input to decisions, nested budgets, stream
separation, frozen selection, full tails, missing/duplicate evidence and errors.
Run relevant regression tests, small invented-state smoke and full compileall.

Read-only audit recomputes numerical summaries from raw arrays, verifies all
input/source hashes and event order, and replays scalar outer paths at fixed
indices 0/127/128/2047 for every method and first request. Check inner paths
0/127 for all second actions in every reachable branch and first request.
Replay subsets only, not the full matrix. Preserve query counts separately
from audit replay work and verify exact counts on bounded engineering tests.

## Conditional reward diagnostic and handoff

If no context passes the descriptive signal check, stop escalating this
synthetic channel toward training. Produce a read-only reward/objective
diagnosis using S4 cost components, archived replay-contract findings and
existing cost-sensitivity evidence. Separate objective misalignment from reward
implementation inconsistency, credit assignment, numerical estimation and lack
of control headroom. Reward diagnostics may be considered even with a positive
conditional signal, but changing weights or shaping remains unapproved.

Do not search weights to favor RL or silently rewrite old evidence. Any actual
reward revision requires an explicit operational justification, prospective
protocol, matched reward semantics, unchanged evaluation KPIs and new scope
approval. Close this finite packet with audited results and that decision.
E1 missing operational data and Stage E's closure remain unchanged.
