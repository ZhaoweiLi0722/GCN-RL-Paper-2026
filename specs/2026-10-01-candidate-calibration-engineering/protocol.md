# Candidate calibration artificial acceptance

## Authorization and scope

2026-10-01: Zhaowei replied "按照你的思路 继续" to the specific engineering-only
proposal: nine invented fixtures, at most128 optimizer.step calls each/1,152
total,30 minutes of numerical execution, one candidate, no expanded search if
it fails. This is not permission for a patient trajectory or new scientific
pilot. Base07f9766cdc188ffb8aba5e8a1fdb2550ffbfda4e; local worktree only.

P2 is closed. Its final-weight ranking barrier and value underfit motivated this
packet, so it is explicitly development-informed, not independent confirmation.
All old source/config/results remain byte-identical. Only new modules, config,
tests, engineering output and appended project records may be written. No
holdout, message, remote Git operation, Howard sign-off or Stage E reopening.

## One candidate

The original architecture widths, three representations,90% initial reference
mass, Adam rate0.0003 and norm cap0.5 are retained in a new unregistered type.
Its zero output heads retain exact initial reference probabilities and value0.
The following numerical changes are fixed before any optimization:

1. Explicit schema-matched positive divisors for neural input features, reference,
   anchor and candidate coordinates. No learned statistics or clipping. The raw
   observation receipt and submitted request are unchanged. Physical message
   adjacency remains the original operator, not the rescaled context distances.
2. Multiply the residual logits by32 (one declared head-width scale), subtracting
   the common reference residual for stable coordinates. This relaxes the
   observed score-range bottleneck; it does not guarantee a useful ranking.
3. Apply the same head-width gain32 to the value output, and compute value MSE
   in a separate declared return unit by dividing prediction and target by4.
   Output gain and loss unit are distinct, both fixed before fitting. No patient
   reward, cost weight, discount or termination rule is changed.

This is a numerical package, not a component ablation. Gains are not fitted or
swept. The toy input units are not production/patient normalization constants.
Any future real-input scale declaration needs separate justification and review.
Flat parameters remain unmatched. Both graph variants retain identical raw
information, parameter counts and seeded tensors; all controls keep physical
links in their public context.

## Invented fixtures

Nine representation/seed pairs:graph,self-only,flat x101,102,103, CPU float32.
Two nodes, one invented count feature at1000/2000, public time and cue, a symmetric
link of100000 units. Six request classes with reference0, anchor0.25 and options
-0.25,0.5,-0.5,0.75. These are decoder-contract tensors, not a patient environment.
For cue+1,0.75 is the known winner; for cue-1,the reference0 is the winner.
All other actions are worse by0.25. The exact invented conditional returns are:

`Q(time,cue,action) = -0.25 - 2.75*(1-time) - 0.25*I(action != winner(cue))`.

Training coordinates are the six configured times crossed with both cues. Test
coordinates are the four configured interleaved times crossed with both cues.
Freeze all nine final models before evaluating any test coordinate. This small
deterministic interpolation check is not an independent research test set.
Known labels use a public cue, deliberately making this an elementary necessary
learnability check, not evidence of graph advantage, planning or patient benefit.

Every four updates, store exact old categorical probabilities. The artificial
actor advantage is Q minus its exact old-probability expectation. Reuse the
existing clipped PPO objective with advantages multiplied by K*old_probability,
so the uniform enumeration gives the exact expected clipped surrogate rather
than pretending all actions were sampled equally. The value target is the same
old-probability expectation, expressed in output units. This oracle fixture
bypasses stochastic exploration, GAE and real-world credit-assignment errors;
passing it does not establish end-to-end PPO efficacy. Never label it a patient
rollout or actual online policy adaptation.

Record policy/value/shared-encoder gradient norms and shared-gradient cosine,
global clipping, probability clipping, loss terms and counters. This measures
the artificial task, not the unrecorded P2 component gradients. No extra fit,
replay or optimizer call for recovery tests. Initial/final states, Adam moments,
old probabilities, labels, CPU RNG, logs and source/runtime hashes are preserved.

## Fixed gates and stopping

Zero-update contract tests first; optimizer methods are blocked in those tests.
Full compilation and a local implementation/config/protocol commit must precede
the single recorded nine-fixture acceptance. No tuning after seeing its values.
Charge each optimizer.step before calling it, including a failing call. Partial
results and failures are immutable; never refund consumed steps or retry.
The30-minute cap covers the serial recorded numerical acceptance. Each fixture
uses exactly128 calls unless a terminal exception/time budget stops the packet.
Ordinary gate failures do not skip remaining predefined fixtures; no extra work
is appended. No optimization in test evaluation or independent readback.

Initial gate:exact reference greedy choice and probability error<=1e-6, same
frozen/trainable initial tensors. Final heldout gates for every fixture:
100% known winner accuracy, minimum winner-vs-best-other logit margin>=0.1,
value explained variance>=0.90 and RMSE<=0.25 against exact frozen-policy Q
expectation. Report every fixture and every gate; one failure means no-go for
claiming numerical acceptance. No conclusion of scientific efficacy is permitted
even if all pass. Keep P2 comparison scores and history unchanged.

Archive and independently byte-verify this packet and an authorized Dropbox-local
copy without changing sharing; cloud sync/Howard access remain unverified.
Close the finite chain with a readout and one next decision. No automatic pilot.
