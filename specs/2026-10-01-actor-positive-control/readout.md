# Actor-only positive control: engineering pass, not a patient RL result

## Decision

The single authorized packet completed and passed all nine engineering cases.
Execution exit0;1,152 Adam calls;3.140315542 numerical seconds;zero patient
environment calls and zero scientific fits. The finite numerical packet is closed.
No patient pilot, additional artificial fit, integration or search is authorized
by this result. Preserve the previous negative packets without revision.

Frozen execution commit:87621a18e84314f996ffcc2e8081bb99e6e3654c.
Config SHA256:69f377d7400d064b9cf5062217574e32d6115b8211474beb24a675fa49a6c738.
Protocol SHA256:cd8d43d32cb87b5eda7f57be31dddef73e9ad2e2ba0ca6ddf360ef42875a4a6d.
Runtime:CPU float32,Python3.9.6,Torch2.8.0,NumPy2.0.2,one Torch thread.

## Results

| Representation / seed | Frozen accuracy | Updated accuracy | Minimum winner logit margin | Gate |
| --- | ---: | ---: | ---: | --- |
| graph101 |50%|100%|2.630130|pass|
| graph102 |50%|100%|2.628218|pass|
| graph103 |50%|100%|2.627036|pass|
| self-only101 |50%|100%|2.630130|pass|
| self-only102 |50%|100%|2.628218|pass|
| self-only103 |50%|100%|2.627035|pass|
| flat101 |50%|100%|2.665689|pass|
| flat102 |50%|100%|2.664095|pass|
| flat103 |50%|100%|2.664909|pass|

All9 models select the known correct class on the same8 interpolation contexts:
72 correct recorded choices, NOT72 independent research replications. The gate
was100% accuracy and minimum margin0.1 per model. Correct-class probabilities
are at least0.93227. Updated greedy invented return averages-1.625,compared with
-1.75 for its own initial greedy policy,an absolute toy improvement0.125. This
is not a patient-cost reduction or a clinically meaningful effect size.

The same-start frozen comparator starts with exactly the same weights. Its
reference logit is1e-4 and other logits0;initial reference probability is about
1/6,not90%. Only initial greedy choice matches the earlier reference policy.
No claim of continuing the historical sampling policy is made. All nine initial
logit checks have zero observed error. Graph/self-only have362 parameters each;
flat426 is not parameter matched. The common public cue bypasses graph messages,
so these results say nothing about the value of graph representation.

Gradient-norm clipping occurs1,149/1,152times;ratio clipping is nonzero on144
updates. Successful ranking despite extensive norm clipping reinforces that
clipping frequency alone is not a sufficient failure diagnosis. There is no
critic or learned value baseline in this packet.

## What this resolves, and what it does not

The earlier calibration packet retained a permanent90% reference prior and
shared nonlinear actor/value optimization;all nine actors remained at50% on
this invented task. This packet instead uses a direct linear categorical head,
initialization-only small trainable bias and actor-only optimization. It shows
that this simpler parameterization plus exact-label surrogate can learn the
state-dependent ranking. The head,prior and optimizer changed together,so the
result does not isolate which change explains the earlier failure.

Training receives the complete exact Q table and enumerates all actions with
old-policy weights. This is an oracle-assisted engineering test,not ordinary
sampled PPO,DDPG,imitation from patient data or deployment online adaptation.
No noisy trajectory returns,learned-critic error,temporal credit assignment,
exploration shortage or clinical simulator was tested. There is no evidence
here that reward weights should change. Existing P2 incremental null results
and previously documented clinical trade-offs remain unchanged.

The head is bound to a fixed canonical candidate bank. Source inspection of
src/rl/candidate_patient_session.py:27 and src/rl/residual_options.py:139 confirms
that the real collector reconstructs requests from each public state;integer
decoder classes can also alias or reorder. Therefore,the toy's class indices
must NOT be connected directly to real patient requests. Its scientific
collectors remain unregistered and reject it;this is a deliberate boundary.

## Validation and evidence

- 100 prelaunch zero-optimizer tests,including13 new checks;full compileall and
  git diff checks pass. The first engineering test exposed an ndarray RNG
  serialization incompatibility. It was fixed before any fit and preserved in
  reports/2026-10-01-actor-positive-control/engineering-test-history.json.
- Initial/final weights,Adam,RNGs,per-call charge records,old probability tables,
  config and runtime are saved. All9 final checkpoints were sealed before any
  test coordinates were evaluated;there was no checkpoint selection or rerun.
- The independent scalar reader validates2,345 files,all1,152 budget charges,
  Adam counters,model digests,known-answer returns,probabilities,rankings and
  frozen comparisons. It performs no new model forward or optimizer call;
  this is arithmetic/hash verification of logged logits,not prediction replay.
- P2 frozen source/document locks and original payload/launcher remain unchanged;
  the full previous calibration inventory also remains unchanged.

Raw root:results/candidate_actor_positive_control_20261001.
Verification:reports/2026-10-01-actor-positive-control/verification.json.
Preservation receipt,when produced:
reports/2026-10-01-actor-positive-control/preservation.json.
Local archive members require byte verification. The attempted Dropbox copy was
blocked by permission review before process launch:the source/results package
and exact synchronized destination need explicit user confirmation. Only local
preservation proceeds;no Dropbox copy,cloud-sync or Howard-access claim is made.
Details:reports/2026-10-01-actor-positive-control/dropbox-permission-review.json.
No remote Git or collaborator message.

## Proposed next bounded decision (not approved or executed)

Do not launch a large patient retraining on the strength of this toy pass.
Recommended next check:keep this actor and this same invented fixed-bank task,
but train from sampled actions and their observed returns,with an independently
optimized critic. Remove full-action oracle Q access from the learner. Exact Q
remains accessible only to the fixture's reward generator and final evaluator.

Proposed ceiling:9 fits,128 actor calls plus128 separate critic calls each,
2,304 optimizer calls total,13,824 artificial action/reward observations total
(12 per batch),30 numerical minutes,one frozen configuration,no retry/search.
No patient environment,real data,changed reward or new clinical scenario.
Freeze full sampling/return/baseline rules,gates and RNG/budget handling before
execution. A fresh user approval is needed;none of these updates has run.

In parallel planning terms only,the subsequent patient interface needs a
state-conditioned candidate-feature scorer or explicit option-identity mapping
with correct alias aggregation. That is a separate integration design with
zero-update contract tests first,not an implicit right to change this actor
mid-test. Any later patient pilot must compare a NEW policy with its own frozen
same-start policy,retain the historical R4/full MDL-2 baselines,and keep public
information and costs fixed. Passing this sampled artificial check would still
not by itself establish long-horizon learning or authorize a patient pilot.
