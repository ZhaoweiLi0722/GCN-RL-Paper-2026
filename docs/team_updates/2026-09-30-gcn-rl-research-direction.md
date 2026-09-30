# GCN and RL method selection for the paper

Date: 2026-09-30. Source checkout reviewed: `8511e42ed96379ef0c61ba8b5ec3d45e47ec4547`.
Status: completed local evidence and literature review; proposed method direction,
not an executed performance experiment or an executable training protocol.

## Objective and decision

Zhaowei explicitly removed DDPG as a required algorithm. The objective is a
rigorous, useful GCN-plus-RL contribution for a publishable paper, with EAAI as
the target level. Positive results and acceptance cannot be promised. Assess
DDPG fairly before concluding that its algorithm family is unsuitable, but do
not let repeated attempts to rescue online DDPG consume the entire study.

Recommendation: prioritize an execution-aware graph policy for the decisions
the simulator actually exposes, while retaining one clean DDPG reference path.
Do not start a broad DDPG/TD3/SAC/PPO sweep. For discrete routing, the preferred
alternative to develop is a graph candidate-action policy trained with PPO.
That preference is a mechanism-based hypothesis, not demonstrated superiority.

## Three different claims

| Claim | Necessary comparison | What does not establish it |
| --- | --- | --- |
| Graph representation helps | Same numerical information, action support, decision head, training budget and physical links; graph messages versus matched flat/self-only models | Current package contrast by itself |
| RL training improves decisions | Return-optimized policy versus the same imitation-initialized frozen policy, plus a budget-matched supervised continuation where teacher labels exist; evaluate both with fixed weights | Distillation, a fitted conditional action ranker, or teacher quality alone |
| Deployment-time updates add value | Tensor-matched trained policies, one frozen and one updated, with identical history/information and adaptation cost accounting | Feedback actions changing with observations, or a history-based frozen controller adapting its behavior |

RL training by interaction with a simulator followed by frozen deployment is
still RL; it is not necessarily the technical setting called offline RL, which
learns from a fixed dataset. This broadens the possible contribution without
relabeling our existing imitation-dominated gains. The final-versus-pretrain
null result still stands for its measured procedure. New successful training
would need its own prospective no-RL comparison.

## What the existing evidence does and does not say

1. **Historical DDPG is not a clean exclusion of the algorithm.** The replay
   audit found absolute and anchor-relative reward targets mixed in one buffer,
   153 discontinuous cached multi-step windows out of 314 emitted items, and a
   calibration bootstrap convention inconsistent with those multi-step items.
   These are concrete target-contract defects/limitations. Their causal effect
   on policy quality is unmeasured. Existing support-filtering, critic-reset,
   exploration and TD3 studies must not be represented as tests of a jointly
   repaired data path. See [replay audit](../../specs/2026-09-29-formal-replay-contract/readout.md).
2. **Learning requests is not necessarily learning useful execution.** G0's
   corrected interpretation is nominal-history diagnostics, not persistent
   hotspot evidence: 25/27 actor displacements did not change integer execution.
   The later R6 packet independently exhibits unequal predictions for differing
   requests with identical sampled execution at one state. Neither proves that
   all actions are equivalent. G0's critic failed before an actor redesign was
   justified. See [G0 correction and results](../patient_indexed_specimen_routing_stage_g0_results.md)
   and [R6 postmortem](../../specs/2026-09-30-critic-saved-data-diagnosis/readout.md).
3. **R6 is a small critic generalization failure, not an online-control trial.**
   Each 238433-parameter critic had 16 training states from four parent
   trajectories. Correct ranking on the same-sign test subset was 62/145
   (42.8%), with dependent pairs. This identifies poor prediction under that
   design, not global unlearnability or global optimality of the frozen policy.
   Labels use one changed action followed by frozen continuation; they are not
   the returns of an independently deployed multi-step learned controller.
4. **Graph attribution also needs tightening.** Historical graph and flat
   controllers differ in derived inputs, decision heads and gate conditioning.
   The observed package-level advantage remains, but message passing alone is
   not isolated. See [graph audit](../../specs/2026-09-29-formal-graph-contract/readout.md).
5. **Increasing uncertainty is insufficient.** Prior overtime and support-queue
   studies sometimes let a constant rule or ID-MPC capture the accessible gain.
   A restricted planner failing to improve does not certify near-optimality.
   See [overtime](../../specs/2026-08-29-continuous-overtime-control/results.md),
   [queue screen](../../specs/2026-09-29-service-queue-boundary/readout.md) and
   [continuation comparison](../../specs/2026-09-29-completion-continuation-comparison/readout.md).

## Route A Give DDPG a fair but bounded test

The next DDPG study, if separately authorized, must have one consistent reward
definition, real contiguous trajectory returns, consistent gamma-to-n bootstrap
and explicit terminal liabilities. Counterfactual one-step labels stay separate
from trajectory replay. Use aligned public features, documented scaling and
an architecture appropriate to independent state coverage. A frozen encoder
with a small value head is an option to test, not an assumed sufficient state.

Reusable engineering already exists in `validated_returns.py`,
`matched_inputs.py`, `prospective_adapter.py` and `prospective_ddpg_kernel.py`.
The prospective kernel verifies actor gradients through the deployed request
transform, fixed gate semantics and atomic update/resume behavior. It is an
opt-in CPU float32 engineering kernel; these are not repaired historical runs
or evidence of useful gradients through physical execution. Its fixture reward
scale, gamma, dimensions and initial models are not approved scientific settings.

Before an expensive actor campaign, ask whether finite feasible changes along
critic-recommended directions actually improve independent continuation returns
and change physical decisions. DDPG does not require differentiable simulator
dynamics; the concern is reliability of the learned action-value gradient across
projection and quantization. A straight-through gradient is an estimator, not
proof that this concern has been solved. Local ranking failure would stop this
particular actor follow-on, not prove that every RL approach fails.

Do not attribute any improvement from a jointly repaired procedure to one
individual repair without an ablation. Do not perform many small repairs on the
old test set and treat the selected winner as independent confirmation.

## Route B Learn the feasible routing choices directly

For an inherently discrete choice, let the GCN score feasible candidate
allocations and train a categorical policy from trajectory returns using PPO.
A state-value baseline avoids the need for a derivative of Q with respect to
an integer routing request, though it does not eliminate value estimation,
exploration or credit-assignment difficulties. This remains a proposal.

The first engineering contract should use the existing patient scenario and
request support; do not simultaneously invent new clinical dynamics. Include
the anchor and no-change choice. Build feasibility and any canonicalization
from information available at decision time, not future rollout costs or hidden
patient deterioration draws. Only merge choices proven equivalent under the
public deterministic decoder. Sampled identical futures are not a universal
equivalence certificate. If public inputs cannot determine execution, preserve
that uncertainty rather than leaking simulator state to the policy.

For multiple facility decisions, either score complete feasible allocations or
construct them sequentially while updating a shared reservation ledger. An
independent per-facility mask does not guarantee joint capacity feasibility.
Candidate coverage, ordering dependence and fallback behavior must be tested.
The initial policy may remain limited by its candidate set; this is not a
global optimization claim. Broader actions require a separate amendment.

PPO is not guaranteed to be sample-efficient here and old off-policy R6 rows
cannot simply become on-policy PPO trajectories. Use a maintained PPO core
where practical after checking the custom graph/action interface. No production
learner, environment episode or optimizer fit is authorized by this memo.

Changing the action representation and optimizer together compares two
controller packages, not the intrinsic superiority of PPO over DDPG. Keep
reward, public information and physical feasibility comparable; isolate graph
and RL-training effects within the selected representation. Do not require a
large full-factorial experiment merely to claim that one practical package is
better, but state exactly which attribution the comparisons support.

## Reward scenarios and evaluation

Cost/service trade-offs are legitimate. A lower scalar cost with worse patient
loss is not itself a reward bug or automatic evidence that all weights must
change. Report the components; justify units, accumulation and terminal
accounting. Keep current weights fixed in a first method diagnostic. A later
objective change needs prospective rationale, not weights chosen after results.
The user's preference not to wait for Howard does not imply Howard's approval.

Only add a new adaptation scenario after current-action competence is assessed.
The leading candidate remains persistent unknown facility efficiency under
observable feedback and a genuinely limited resource budget. Retain existing
physical lags/coupling where justified; do not stack uncalibrated new mechanisms
to manufacture an RL advantage. Compare against a history-aware frozen policy,
adaptive rules and ID-MPC, not only an uninformed static policy. Missing E1
operational inputs remain missing. Stable/no-shift conditions are necessary
controls; fast independent noise is not automatically a learnable regime.

Prospective evaluation should separate independent training seeds, environment
worlds and scenario shifts. Report uncertainty at the independent-unit level,
total cost and service outcomes, inference latency, environment-query budget
(including counterfactuals/planning), training effort and adaptation losses.
Set the primary endpoint and practical relevance criteria before a new test.
Assess a fully declared episode/terminal objective rather than overlapping
conditional windows. Current R6 test labels have been inspected: they cannot
serve as an untouched confirmation set for the revised methods. Historical
formal holdout remains closed. Negative and unstable outcomes are retained.

## Literature checked and its limited implications

- Lillicrap et al., [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971):
  DDPG is a continuous-action actor-critic method. This motivates checking the
  match to our decision representation, not declaring it impossible on a
  quantized simulator.
- Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html), ICML 2018:
  twin critics and delayed policy updates address value-estimation error. They
  do not provide a remedy for our inconsistent return semantics or prove that
  a switch to TD3/SAC will recover gains. Howard's existing TD3 result is not
  ignored or automatically rerun.
- Zhang et al., [Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://papers.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf), NeurIPS 2020:
  graph representations and PPO learn dispatching decisions; testing greedily
  uses the trained policy, with scale/generalization and computation comparisons.
  This supports the distinction between RL training and deployment updates,
  not expected improvement in our patient network. It uses GIN, not our GCN.
- Wan et al., [An effective multi-agent-based graph reinforcement learning method for solving flexible job shop scheduling problem](https://doi.org/10.1016/j.engappai.2024.109557), EAAI 139, 2025:
  the publisher abstract describes graph encoding, separate sequencing and
  assignment decisions, PPO-family training, and solution-quality/runtime
  comparisons across sizes. Only the publisher abstract was inspected; a
  direct full-page fetch failed. This is a relevant contribution pattern, not
  an endorsement of its full evaluation or a reason to copy its multi-agent
  architecture. Publication there does not guarantee acceptance of our study.

## Next finite work and stop conditions

1. Complete the no-simulation candidate-decision and clean-DDPG integration
   contracts, including public inputs, legal action identity, shared information
   and tests. Reuse existing modules; do not build another full framework.
2. Present one bounded prospective pilot with named arms, data separation,
   scenario, seeds, primary metric, per-arm query/update/wall-time caps and a
   single-attempt rule. Those values are not silently inherited from R6 or the
   earlier engineering fixture. Obtain specific approval before fitting or
   collecting new scientific data; the previous 24-critic normalization question
   was not approved and is not this memo's recommended main experiment.
3. If the RL-trained policy beats its same-start no-RL comparator on fresh
   complete episodes, prioritize graph attribution and strong-baseline testing.
   Deployment adaptation is a separate optional contribution, not a prerequisite
   for calling that genuinely return-trained policy RL.
4. If it does not, preserve the negative result and assess whether graph-guided
   decision quality, efficiency or robustness supports a narrower paper. Do not
   assume it will, relabel distillation, or keep searching for a favorable seed.

## Verification of this review

No experiment, checkpoint or manuscript result was changed. A fresh process
scan found no related Python experiment; the default sandbox process query was
denied, then the approved read-only host query succeeded. Only local preparation
is authorized. No push, PR, merge, message or automatic continuation was started.

At the reviewed checkout, the following 87 existing tests passed in 4.791 s:
`test_formal_replay_contract`, `test_formal_graph_contract`,
`test_validated_returns`, `test_matched_inputs`, `test_prospective_adapter`,
and `test_prospective_learner`. These include invented-tensor optimizer steps,
not scientific fits, new patient episodes or performance observations. Full
`compileall -q .` passed using the project interpreter and external pycache.
All seven local evidence links resolve and all eight fingerprints below were
rechecked unchanged after document preparation. The user's requested long-term
priority was also saved as a separate memory extension note.

Evidence fingerprints at review time (not retroactive archival provenance):

| File | SHA256 |
| --- | --- |
| `specs/2026-09-29-formal-replay-contract/readout.md` | `483c30f4935c9005e489321c71587ae9b096564ee229fd7dd4444aa80d683632` |
| `specs/2026-09-29-formal-graph-contract/readout.md` | `9bff796f6b63e415382c4d61254a41cda4106b5e3505b21ca88d375b95588657` |
| `specs/2026-09-30-clean-critic-generalization/readout.md` | `e513ff3a10abcd8f7c84314d3ba04f21e6f41dc521f2402422768e17a04414b6` |
| `specs/2026-09-30-critic-saved-data-diagnosis/readout.md` | `99f089ccd9eaf4cdf5daf2cdd8b5bf351ed7e1301d3ff81b60e9c9cc1cc057b9` |
| `src/rl/prospective_ddpg_kernel.py` | `ffdb998ff56e5df76be3aed0a06a47e6c68d48b999cd509e455be0dc80a39332` |
| `src/rl/validated_returns.py` | `45de110691e67ec56fcaf56ab80d8ddd0a15870f33186360994bccd12d3c36e2` |
| `src/models/matched_inputs.py` | `7f6b1995c2bb56a82c63161f5f1b63b2e11f9e7e827c0453687d20ed90544f14` |
| `docs/patient_indexed_specimen_routing_stage_g0_results.md` | `6e039a3f2d418775b2f43012c13139becd9dfb20486885a6adb65fa8ce868615` |
