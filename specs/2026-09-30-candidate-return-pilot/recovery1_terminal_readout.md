# P1-R1 terminal readout: initialization gate not met

Recorded2026-10-01T01:25Z (September30 local time). The single explicitly
approved recovery ended at qualification, before any PPO or BC-CONTINUE fit.
This is a failed initialization gate, not a negative RL treatment comparison.
No repair, resumed fitting, changed threshold or second attempt followed.

## Execution and actual work

- Implementation86c57ab6d32b43f4682637d0096fcab42d03683f;
  execution010ca27fec294db27f935828aac711063cbc0f01.
- Effective execution SHA256
  cb1b35f0f22ffbc8e699830f89ffc29566c5cc61509f04bef02757473cca643b.
- Root: `results/candidate_return_pilot_20260930_recovery1`.
- All9 real preflight cases passed public-input/support and exact restore
  checks:504 calls including36 cloned steps.24 demonstration episodes used1248
  calls. Nine initializers completed256 imitation updates each. Six independent
  qualification episodes used312 calls, shared as inputs across representations.
- Totals:2064 environment calls and2304 BC-initialization optimizer steps.
  Zero PPO updates, zero BC-CONTINUE updates, zero final-policy evaluations,
  zero forked continuation models. Test streams were not opened.
- Supervisor: child89438/parent89151, exit1,299.528191 seconds, no forced kill.
  Parent exec also exited1. Post-terminal host checks found neither PID nor
  another P1 runner. The research automation remains deleted.
- Error: `initialization qualification below locked agreement gate; attempt closed`.
  The traceback is in stdout and failure.json. Empty stderr is not success.
- Ledger4392 events, seal
  a8aa38046abfc046d70fc1f79827be9d0b2854efdf3c10dd7c7cd03e44085aeb;
  all phase and global resource arithmetic/caps reconcile.

## All initializer results

Each qualification denominator is104 saved states from2 R4-driven episodes.
All states have5 or6 candidate classes; single-class states cannot inflate these
scores. The gate was>=95% for each model, not a pooled or best-seed average.

| Block | Representation | Demonstration agreement | Qualification agreement | Pass |
| --- | --- | ---: | ---: | --- |
|60 |graph |394/416 (94.71%) |93/104 (89.42%) |no |
|60 |self-only |365/416 (87.74%) |85/104 (81.73%) |no |
|60 |flat |416/416 (100%) |104/104 (100%) |yes |
|61 |graph |416/416 (100%) |104/104 (100%) |yes |
|61 |self-only |408/416 (98.08%) |102/104 (98.08%) |yes |
|61 |flat |416/416 (100%) |104/104 (100%) |yes |
|62 |graph |414/416 (99.52%) |102/104 (98.08%) |yes |
|62 |self-only |416/416 (100%) |103/104 (99.04%) |yes |
|62 |flat |416/416 (100%) |104/104 (100%) |yes |

These are action-imitation diagnostics, not cost/clinical comparisons. Flat has
238,658 parameters versus59,602 for graph/self-only, so these rates do not establish
an architectural superiority. Block60 fails on the demonstration data too; the
problem is not solely a qualification-distribution mismatch. The present data
do not isolate optimizer budget, initialization, representation or calibration
as the unique cause. The two displayed loss endpoints are different sampled
minibatches, not a fixed-dataset loss curve.

## Independent readback

`reports/2026-09-30-candidate-pilot-integration/recovery1_terminal_audit.py`
reopens the failed state and saved initializer envelopes, with inference only.
It does not use the qualification summary routine to compute accuracy. It
directly scores saved examples, recomputes argmax hits/softmax probabilities and
checks every qualification result against the original saved result. All1248
demonstration and312 qualification examples match the original raw event state
and candidate bank. Nine fit envelopes match the failure-state payload and the
qualification kernel hashes. No weights or RNG state in the run are changed.

The existing learner-independent raw verifier also reopens all39 completed
episodes,2028 original event rows. It recounts identity compartments, arrivals,
loss causes, completions, terminal obligations, components/total raw costs,
absolute reward and original request/class/routing arithmetic. Every recomputed
outcome equals its stored outcome; route count is at least300 per episode. The
36 cloned preflight calls are separately accounted for by the preflight/ledger
evidence, not miscounted as additional recorded episodes.

The entire279-file run remains byte-identical before/after audit. Source/runtime,
270 source locks, seven R4 inputs, prior failed evidence and historical locks
reverify under the frozen effective packet. Audit output:
`reports/2026-09-30-candidate-pilot-integration/recovery1-terminal-audit.json`.
No new simulator calls, optimizer steps, final-test evidence or counterfactual
patient outcomes are produced by this analysis.

## Additional actionable limitation

Across the nine models, mean reference-class sampling probability on saved
qualification states is only21.64%-26.34%, even where greedy agreement is100%.
Thus high greedy agreement alone does not ensure that the stochastic policy
used for PPO/BC-CONTINUE collection behaves like the reference. This readback
does NOT show that alternative actions worsen outcomes: those counterfactual
outcomes were not evaluated. It identifies a gap in what the initialization
gate certifies, not a demonstrated downstream PPO failure.

Reward values are not used in this CE-only initialization. Changing the reward
is therefore not supported as a remedy for this particular failed gate. Nor
does this attempt revise historical GCN/distillation evidence, show PPO loses
to DDPG, or show that the baseline is optimal.

## Next decision, not an execution permit

Do not simply lower95%, continue only the passing seeds, transfer unused budget
or increase BC steps until the gate passes. Preserve P1-R1 as closed.

Recommended next design review: a categorical residual controller initialized
to preserve the frozen R4 decision, with an explicit, prospective exploration
distribution and a learned correction to a fixed reference prior. R4 identity
is already a public candidate feature; using it as a prior adds no outcome
oracle. However it changes policy initialization/parameterization and requires
a new approved protocol. A purely deterministic delta prior would prevent
exploration; any smoothing must state the resulting sampling deviation and
coverage instead of claiming exact stochastic equality. A strong prior can
also suppress useful exploration, so it is not a guaranteed improvement.

First prepare the algebra, mock acceptance and same-start frozen/PPO/BC controls,
without new patient simulation, fitting or selecting settings on these results.
Only a separately approved, fully bounded and frozen design may run again.
These qualification trajectories are now seen development data, never a fresh
confirmation set. A future same-start comparison still cannot claim clinical
benefit without actual paired cost, completion, loss and terminal evidence.

Exact next approval question: may we prepare the reference-preserving residual
initialization design and artificial-fixture tests, with zero new patient
trajectories/optimizer fitting and no scientific relaunch? No Howard sign-off,
remote Git action, message, holdout use or Stage E reopening is inferred.

## Preservation and engineering status

The complete279-file failed root is retained and archived without overwrite:
`results/candidate_return_pilot_20260930_recovery1_failure_archive/failed-attempt-010ca27.tar.gz`.
All archive members and unchanged source bytes were read back. Archive size
275,251,375bytes; SHA256
801567b3ff7e5609a964a03761cdf594a8aa196c1ca6fa7f3c8f2259c875a21b.
Manifest SHA256
2563b39fec1c04d46a13420ed9fd00572b2146615d632f9b060841138339e645.

Archive, manifest, auditJSON and audit-script bytes were also verified in the
authorized Dropbox-local `Research Artifacts/candidate_return_pilot_20260930_recovery1`
directory. Cloud sync and Howard access remain unverified; no sharing change or
message occurred. The original P1 failed tree and archive remain unchanged.
Detailed copy receipt: `recovery1-terminal-preservation.json` in the integration
reports directory. This is a failed-attempt archive, not a complete P1 dataset.

Before launch:297 relevant tests and full compileall passed. After closure:
25 focused verification/recovery/imitation tests passed in1.862s. The saved-data
audit completed all reconciliation assertions. No experiment was rerun by tests.
Post-closure full-repository compileall and git diff check also exit0.
