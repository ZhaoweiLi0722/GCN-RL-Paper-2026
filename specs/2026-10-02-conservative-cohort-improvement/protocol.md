# Two-Round Conservative Cohort Policy Improvement

Status: prospective execution proposal; not approved or launched. Zhaowei's
"启动新实验 我们现在的目的就是优化performance" approves preparing the next
performance-oriented experiment, not numerical limits that had not been presented.
One consolidated execution question was submitted with the complete limits below.

## Decision and Hypothesis

The previous complete comparison trained six actors but the proposed actor had
higher mean cost and patient loss than its same-start frozen and BC controls.
It remains closed, including its separate launcher-archive timeout. Neither more
epochs nor a changed reward follows from that result.

The new hypothesis is that collecting states visited by the learned controller,
evaluating candidate actions under that controller's continuation, and restraining
distribution drift may improve closed-loop generalization. This is a bundled
mechanism test, not identification of which individual change caused an effect.
The old one-step/R4 continuation was not inherently invalid policy improvement;
approximation, sparse state coverage and noisy labels are possible explanations,
not established causes. Four future draws do not establish label reliability.

## Fixed Design

- Blocks 60, 61, 62; three already qualified graph initializers, no initializer
  refit. Graph, public features, candidate support, physical dynamics, reward and
  cost weights remain inherited and unchanged. No new scenario or formal holdout.
- Two rounds per block. At round start freeze the paired-cost actor. Collect two
  complete greedy cohorts under it, saving states after steps 4, 20 and 36.
  Round 1 uses the original initializer, round 2 the round-1 paired-cost actor.
- For every canonical candidate at each state, use four independently allocated
  conditional future seeds. Take that candidate once, then the frozen round-start
  paired-cost policy greedily through step 52, then the existing MDL-2 accounting
  tail through 63. Preserve float64 original requests and raw costs. Same future
  seed is not a claim of event-aligned CRNs after paths diverge.
- Train only the actor on those six current-round states, 64 full-batch Adam
  updates per round. Objective is mean expected remaining raw cost / 1e9 plus
  0.05 times KL(current probabilities || detached round-start probabilities).
  Subtracting the state-specific R4-candidate cost changes no exact gradient.
  No label clipping/filtering, extra entropy, critic update, or reward revision.
- KL is a fixed soft regularizer on observed states, not a hard global trust
  region, calibrated confidence rule or clinical safety guarantee. Greedy
  deployment can still change abruptly. All adverse outcomes must be reported.
- BC-CONTINUE starts from the identical original initializer, trains on exactly
  the same public states, and imitates R4. It keeps its own weights across rounds;
  both arms reset Adam once at each round boundary. BC loss does not use cost
  labels. Equal actor update counts do not make simulator-label costs equal.
- No retrospective choice between rounds, seeds, KL coefficients or policies.
  Only final-round models enter the sealed test comparison.

## Resource Envelope

| Operation | Maximum |
| --- | ---: |
| Fresh reference-layout builds (no steps) | 3 |
| Same-start originals and clones | 3 + 3 complete cohorts, 378 calls |
| Collection | 12 complete cohorts, 36 states, 756 calls |
| Conditional futures | 864 canonical branches, 37,152 calls |
| Actor updates | 3 blocks x 2 rounds x 2 arms x 64 = 768 |
| Critic updates | 0 |
| Final evaluation | 5 controllers x 3 blocks x 12 worlds = 180 cohorts |
| Evaluation calls | 11,340 |
| Total environment calls | 49,626 |
| Wall time, including setup, recording and local archives | 28,800 seconds |

Branch bound is 3 x 2 x 2 x 4 x 6 x ((63-4)+(63-20)+(63-36)).
Aliases consume fewer branches, never additional rounds or samples. Calls are
charged before execution; failed attempts are not refunded. All owner, phase
and global bounds are in `conservative_cohort_improvement_20261002.json`.
The stage time caps sum to the global cap. Exactly one attempt; no automatic
repair/retry, sample expansion, reward adjustment or follow-on experiment.

## Evaluation and Interpretation

All nine learned final controllers (frozen, paired-cost, BC across three blocks)
are sealed before any test-world access. R4 and full MDL-2 are additional controls.
Tests share the 12 prospective worlds per block across controllers. Training and
test seeds are allocated separately and checked against local seed manifests;
R6 and previous test worlds are not independent confirmation data.

Primary contrasts are paired-cost minus own-frozen and paired-cost minus BC.
Report each block and each paired world, mean block-relative cost differences,
patient loss/delivery/expiry/waiting, settled terminal registry, prefix/tail cost,
requested/executed action changes, and all costs of extra simulator labeling.
Turnaround among completed patients is not an improvement when more patients
are lost. A development screen requires >=1% mean block-relative cost reduction,
at least two cost-improved blocks, and no block-level mean loss increase for both
primary contrasts. This is not a statistical/clinical noninferiority guarantee.
With only three trained blocks, no claim of reliable general superiority follows.

This is model-assisted policy training in simulation, not model-free PPO/DDPG
superiority, deployment-time online parameter adaptation, isolated GCN benefit,
or an equal-compute comparison with historical policies. No success is promised.

## Execution and Preservation

Complete the versioned runner and real-persisted-metadata mock path without
scientific forwards or optimizer updates. Commit the implementation/config,
record the exact scope approval, append locked-plan change control, freeze
runtime/source/input/seed locks, then launch one exclusive serial process.
No scientific work is permitted merely by this draft existing.

Preserve every original and failed result. Save boundary state and optimizer/RNG
receipts without duplicating the whole immutable training dataset in every
checkpoint. Archive once per root with member hashes and enough I/O time;
closure must be watchdog-covered, not a second unbounded launcher archive.
No Dropbox, remote push, PR, merge, or external messages in this package.
