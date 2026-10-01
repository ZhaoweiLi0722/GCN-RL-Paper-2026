# Sampled-return control: completed, acceptance gate not passed

2026-10-01. One authorized artificial packet, not a patient experiment.
Implementation: `2bf2403d346c708338a7ee20a3a6ee8bf190589e`.
Execution/authorization commit: `4763e1e1fd4dfd658b4026f1d84ba69f68a839b6`.
The original protocol/config remain unchanged; the later committed
`execution_authorization.json` records the bounded approval.

## Decision

Execution completed with exit code 0; the independent saved-receipt verifier
also completed with exit code 0. Scientific/engineering acceptance is a
different question: only **1 of 9 fits passes all prospective gates**, so this
packet is a **no-go for patient integration/execution**, not a successful
patient-performance result. No retry, extra seed, tuning or reward revision.

All 9 fits improved the stochastic policy's exact expected artificial return
relative to their own frozen start. Only 3 improved the deterministic greedy
decision; 6 remained at 50% accuracy. This is partial learning, not reliable
state-dependent control. Do not select the lone passing flat-101 result.

## Results

Every frozen actor starts at accuracy 0.50, mean expected return
-1.8333319444 and mean greedy return -1.75 on the same eight invented contexts.
Positive return differences below favor the continued policy. They are toy
return units, not clinical/economic percentages or confidence intervals.

| Fit | Final accuracy | Expected-return gain | Greedy-return gain | Minimum winner margin | Critic MSE / constant MSE | Coverage | All gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| graph-101 | 0.50 | 0.068092 | 0 | -7.053585 | 0.438622 | Pass | Fail |
| graph-102 | 0.50 | 0.104938 | 0 | -0.261108 | 0.433706 | Fail | Fail |
| graph-103 | 0.75 | 0.078187 | 0.0625 | -0.295651 | 0.453283 | Pass | Fail |
| self_only-101 | 0.50 | 0.068091 | 0 | -7.053563 | 0.438622 | Pass | Fail |
| self_only-102 | 0.50 | 0.104938 | 0 | -0.261109 | 0.433706 | Fail | Fail |
| self_only-103 | 0.75 | 0.078187 | 0.0625 | -0.295604 | 0.453283 | Pass | Fail |
| flat-101 | 1.00 | 0.098621 | 0.1250 | 0.101493 | 0.459020 | Pass | Pass |
| flat-102 | 0.50 | 0.101196 | 0 | -0.381274 | 0.432643 | Fail | Fail |
| flat-103 | 0.50 | 0.071358 | 0 | -3.339153 | 0.443711 | Pass | Fail |

Counts: 9 artificial fits; 128 actor and 128 critic calls per fit;
2,304 total optimizer calls; 13,824 sampled training action observations;
13.753 numerical seconds, excluding later verification/archive I/O. No patient
calls or fitting. Both runner and verifier stderr are empty. All nine final
models were sealed before test contexts were opened. The verifier checked
9,863 run files, private RNG replay, sampled returns, normalized advantages,
optimizer counters, chronological budget charges, final seals and raw gates.
P2 and prior calibration/actor-control evidence remained unchanged.
Post-run closure validation passed 32 relevant non-fitting tests and full
repository compileall; `closure-validation.json` records the commands.

## What the saved outcomes narrow down

Readback uses saved `heldout.json` logits, `training.json` candidate manifests
and coverage counts, and `terminal.json` metrics, without new model forwards.
Canonical winning classes are request 0 for negative cue and 0.75 for positive
cue. Six held-out policies always choose one of these two requests regardless
of cue: graph/self_only seeds 101 and 102, and flat seeds 102 and 103. Graph and
self_only seed 103 also choose the wrong request for two positive-cue times.
Thus average sampling improves without reliably learning when to switch.

Importantly, **no fit completely missed the winning action at any training
context**. The lowest per-context winning-action count varies from 1 to 29
across fits. Missing coverage pairs are only request 0.25: context 11 in
graph/self_only-102, and contexts 5 and 11 in flat-102. These still fail the
locked coverage gate, but cannot by themselves explain failure to discover
either winner. Merely adding more exploration is not an established remedy.

All critics pass the relative-MSE gate, yet their final RMSE is 0.574-0.591
return units, compared with a within-state action effect of 0.25 and a
time-only baseline span of 2.75. This comparison does not prove that critic
error caused the policy failure. A state-only baseline is a variance-control
device, not an action-value oracle. It does identify sampled credit assignment
and finite-batch variability as a more focused hypothesis than "RL cannot
learn" or "the environment has no headroom". No causal ablation was performed.

## Next decision, not another open-ended search

The completed exact-Q actor control (9/9) and this sampled-return result (1/9)
show that successful full-information optimization did not transfer reliably
to sampled learning under this fixed budget. They differ in supervision and
critic design; this is not a single-component causal comparison.

Prioritize one prespecified comparison that improves the state/time baseline
using training-only data before policy updates, with unchanged task rewards,
same-start frozen controls and explicit equal compute accounting. This would
test learning-signal separation, not change the clinical objective to manufacture
an improvement. It is a proposed next scope, **not authorized or executed**.
Do not simultaneously increase training, alter exploration, change architecture
and tune reward. No new fit starts from this readout. Existing source/mock
dynamic-candidate audits need not be repeated wholesale.

Patient performance is evaluated only after a working dynamic-candidate
interface and a separately approved bounded patient pilot. Continue to report
the existing P2 null greedy increment, rather than substitute toy gains.
The paper's established graph/distillation evidence and the unresolved RL
increment stay distinct. Reward coefficients remain unchanged; the separate
reward-decision document records unresolved domain/terminal-obligation issues.

## Limits and evidence

Graph/self_only actors have 362 parameters; flat has 426; each critic has 25.
The cue bypasses graph aggregation. This is not evidence of flat or GCN
superiority. The nine fits reuse three initialization seeds and eight test
contexts; they are not nine independent clinical replications. Rewards are
one-step deterministic invented outcomes, not 52-step delayed patient returns,
new operating scenarios or deployment-time online adaptation.

- Raw packet: `results/candidate_sampled_return_control_20261001/`.
- Runner/verifier logs: corresponding `-launcher/` directory.
- Independent report: `reports/2026-10-01-sampled-return-control/verification.json`.
- Preservation helper/receipt: same report directory, `preserve.py` and
  `preservation.json` (receipt exists only after verified local archival).
- Local archive: `results/candidate_sampled_return_control_20261001_archive/`.
- No Dropbox export, remote action, holdout use or Stage E reopening.

Progress correction: implementation tests are prerequisites, not paper results.
This turn delivered an actual bounded result and a narrower failure mode;
it did not deliver new patient-level RL benefit or guarantee publication.
