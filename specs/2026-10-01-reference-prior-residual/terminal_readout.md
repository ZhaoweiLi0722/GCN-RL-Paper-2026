# P2 terminal readout: completed, no incremental greedy-policy gain

2026-10-01. The one authorized P2 attempt completed, not failed. Scientific
execution and archival succeeded; the efficacy criterion did not. This is
development evidence under the unchanged nominal environment and reward.
There was no retry, expanded sample, reward search or automatic follow-on.

## Execution and acceptance

| Item | Verified result |
| --- | --- |
| Execution commit | `9dc736777393269b13b46680cec9e7ea2e84c052` |
| Implementation | `da5cfbae27b0f1154b10a2c1cc3ac7b7ba4e3735` |
| Effective config SHA256 | `914a751214ce5e9386c349d200cd17faa22b3c3ac9767c998fcf293507df56b9` |
| Terminal | completed, child exit 0, no forced kill; 56/56 jobs |
| Duration | 9,473.37 seconds, approximately 2h38m, including archive |
| Preflight | 9 cases, 504 calls including 36 clones |
| Prior qualification | 6 episodes, 312 calls, all 9 policies pass |
| Training | 18 continued models, 576 episodes, 2,304 Adam steps |
| Final evaluation | all 27 models sealed first; 33 policies, 396 episodes |
| Total | 51,360/51,480 calls, 2,304/2,304 optimizer steps, below 6h |
| Initial acceptance | 315 tests, full compilation and diff check passed |
| Closure acceptance | 19 focused tests, including 7 new arithmetic fixtures; full compilation passed |
| Process | original parent 97343 and child 97654 exited; no duplicate scientific run |

Root: `results/candidate_reference_prior_pilot_20261001`.
The original design-only protocol/config remain unchanged; the separate
execution authorization and effective packet governed this attempt.
Both earlier P1 failures and their hashes remain intact.

## Prespecified result

**Graph-PPO minus its own same-start frozen controller: exactly zero observed
cost difference and zero patient-outcome difference in all 36 paired worlds.**
Graph-PPO also equals continued imitation and R4. Self-only and flat PPO show
the same null incremental outcome. The locked decision is
`limited_negative_or_inconclusive`; the positive-result triage does not pass.

These are three independent training blocks, not 396 independent training
replications. The duplicated frozen controllers do not enlarge the sample.
Identical outcomes on these sampled worlds are not a global equivalence proof,
a near-optimality certificate or a demonstration that no better action exists.
P2 does not compare against repaired DDPG or test deployment-time adaptation.
Flat parameters remain unmatched; all arms retain the graph-based R4 reference.

Against MDL-2, the unchanged R4 behavior yields **0.735498% lower average
52-step cost** (raw mean difference -18,762,829.76). PPO inherits this result;
it is not a new RL gain, nor a fresh isolation of the GCN contribution.
Average paired patient differences versus MDL-2 are:

- 23.4722 fewer losses within the window;
- 9.3333 more completions;
- 14.1389 more active unresolved patients at the boundary.

Terminal active patients must not be called deaths or ignored as free benefit.
The larger terminal obligation fails the prespecified no-adverse-direction
screen; there is no clinical noninferiority claim. Costs retain the protocol's
original component weights and finite horizon, with no posthoc tail charge.

## Saved-data diagnosis

All 9 PPO models have changed weights and 128 saved Adam steps each; training
made nonreference choices on approximately 8.95%-11.24% of steps. Routing is
nonzero. Thus the null result is not explained by absent updates or absent
exploration. It is also not the earlier P1 initialization-gate failure.

In final greedy evaluation, all 5,616 PPO decisions still choose the R4
reference class. Saved reference probabilities range from about 89.9512% to
90.1272%; the smallest reference-versus-best-alternative log-probability margin
is 3.5765. The learned change to that margin is only about -0.0126 to +0.0142.
The 90% prior therefore still dominates the observed final action ranking.
BC-CONTINUE also keeps the same greedy reference choices.

This is a measured ranking description, not a causal test of removing the prior.
It does not yet distinguish limited sample/update budget, weak or noisy action
advantage, objective coupling or genuinely small headroom. Increasing steps,
annealing the prior or changing reward to force a positive result is not
authorized by this null result. Test receipts inspected here are now seen
development data and cannot become a fresh confirmation set.

The post-closure audit uses saved JSON and checkpoint tensor differences only,
with **zero new model forwards, simulator calls or optimizer steps**. It recounts
987 episodes/51,324 recorded transitions plus 36 clones, independently sums raw
cost, checks patient identities, and compares 324 full closed-loop traces
(PPO, BC and frozen) against their matched R4 traces. All comparisons are exact.
MDL-2 receipts deliberately label their own anchor as reference; their local
reference-choice count must not be misread as equality to R4.

The first supplemental readback stopped at a BC checkpoint-schema assumption
(`pending`, a PPO-only field). Its error record and original 18-test acceptance
are preserved. The reader now explicitly handles synchronous BC history/steps;
19 tests pass after the correction. This was a post-run audit-code correction,
not a scientific execution failure or another attempt. Frozen experiment code,
data and archives were not modified.

## Evidence and preservation

- Prespecified raw verification: `payload/independent-verification.json`.
- Terminal/budget receipts: `launcher/{terminal,completed,supervisor}.json`
  and `launcher/budget.jsonl`.
- Additional saved-data readback and script:
  `reports/2026-10-01-reference-prior-integration/terminal-audit.json` and
  `terminal_audit.py`; checks source/runtime/input/prior evidence locks and
  unchanged original payload/launcher inventories.
- Payload archive: 5,254 files, 7,158,089,011 bytes,
  SHA256 `581f2ebda6736aa10cbf0d64bdebfebe20d46ece16a402da8c5069d09225e799`.
- Supplemental terminal-launcher archive: 265 files, 3,680,467 bytes,
  SHA256 `4334b109f8c6f91a029ecc8420bb1677390ec841b10e4ff4f1c7350f89945d6a`.
- Every archive member and local destination byte was verified. New copies
  reside in the existing Dropbox Research Artifacts folder under
  `candidate_reference_prior_pilot_20261001`; original files remain in place.
  **Cloud synchronization and Howard access are unverified.** No sharing change.

## Handoff and next decision

The authorized finite chain is closed. No background training or new research
automation remains. All changes stay local; no push, PR, merge, message,
holdout use, Howard approval claim or Stage E reopening occurred.

Recommended next permission is **saved-training-data diagnosis only**: inspect
already recorded action advantages, clipping/value-loss scales and ranking
changes to decide whether a new learning intervention has a defensible target.
Use no new trajectories, policy fitting, model/scenario/reward search or
additional test evaluation. This is more informative than an immediate blind
retrain, but cannot promise a positive result or publication. A subsequent
scientific attempt would require a separate bounded, prospective approval.

Manuscript interpretation remains: graph-aware reference-policy gains must be
separated from incremental RL gains. P2 adds a transparent null attribution
result and a decision-ranking diagnostic, not evidence of improved online RL.
