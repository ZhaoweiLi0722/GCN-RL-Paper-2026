# RL improvement and reward decision plan

## Objective and current decision

Prepared 2026-10-01 for Zhaowei. Improve the evidence for a GCN-plus-RL paper,
not the appearance of a positive result. First establish that sampled-return
learning can change useful decisions, then measure its incremental value against
the same frozen policy. Audit objective validity in parallel. Do not change
reward weights just because a run has no improvement.

The user requested a complete plan and automatic continuation. That authorizes
the finite local engineering and read-only preparation below. It does not by
itself approve a new patient experiment, reward revision, or Dropbox export.
The immediately proposed artificial numerical packet is pending one explicit
bounded approval. Scheduling is not an experiment-launch receipt.

Work only in `/Users/lizhaowei/GCN-RL Paper 2026/worktrees/september-research-integration`
on `codex/september-research-integration`. Entry HEAD is
`e9d5db00a5f49d8e5dfcc2fd2003212574247d25`. Preserve historical evidence and
Howard's work. Stage E stays closed. Only local commits are allowed.

## What the evidence currently says

| Evidence | Result | What it does not establish |
| --- | --- | --- |
| Closed P2 simulator pilot | Zero observed incremental greedy cost or patient-outcome change versus its own frozen policy; all 5,616 PPO test choices stayed at R4 | No proof that RL is universally ineffective or R4 globally optimal |
| P2 saved-weight diagnosis | Final score bounds could not overcome the fixed prior; value targets often exceeded value-head range; raw reward and return arithmetic passed | Correct arithmetic is not proof that the scientific objective is appropriate |
| Closed artificial calibration | 0/9 full gates; value fitting improved but actors remained at 50% ranking accuracy | A better critic alone was insufficient in this artificial design |
| Closed actor-only positive control | 9/9 passes, each 100% versus its own frozen 50% on eight invented interpolation contexts | Exact full Q labels, fixed bank and no critic: not ordinary sampled PPO, patient performance, or a GCN advantage |

The P2 benefit of 0.735498% versus MDL-2 was inherited R4 behavior, not new PPO
benefit. Keep the associated patient-loss, completion and terminal-active
trade-offs visible. The actor-only control changed several components together;
it is not a causal single-component ablation. Its fixed class indices cannot
be connected directly to state-dependent patient request classes.

Evidence: `specs/2026-10-01-reference-prior-residual/terminal_readout.md`,
`specs/2026-10-01-p2-training-diagnostic/readout.md`,
`specs/2026-10-01-candidate-calibration-engineering/readout.md`, and
`specs/2026-10-01-actor-positive-control/readout.md`.

## Milestones and performance evaluation

### A1 Sampled learning control

Prepare one artificial fixed-bank diagnostic using the existing successful
actor/task, sampled actions and their observed rewards, and an independent
critic with a separate optimizer. No full oracle Q table may enter learner
inputs, actor advantages or critic targets; the fixture generator and final
verifier may use it. This remains an elementary artificial task, not evidence
about delayed clinical rewards or deployment adaptation.

Proposed hard caps: nine fits; at most 128 actor and 128 critic Adam calls per
fit, 2,304 total; 13,824 invented training action observations; CPU float32;
one fixed configuration; 30 numerical minutes including setup and final
evaluation; no retry, search, early-checkpoint selection or budget expansion.
Freeze sampling schedule, seeds, loss definitions, observation accounting and
all acceptance gates before any optimizer call. Fixed final evaluation is
separate from the training-observation cap and must be explicitly enumerated.

Engineering can proceed automatically: new standalone implementation, synthetic
zero-update contract/gradient tests, bounded runner and independent scalar
verifier. Do not modify the closed positive-control implementation. Artificial
optimization is blocked until the specific approval is recorded. No hidden
trial fitting is allowed while choosing gates or constants.

Acceptance must check action improvement against the same-start frozen actor,
both state-dependent action directions, finite values and uncertainty/coverage,
critic error relative to a declared simple baseline, no oracle leakage,
independent actor/critic gradients, exact sampling probabilities, and budget
and RNG receipts. Use prospective engineering tolerances, not statistical
generalization or graph-superiority claims. Report all nine fits even if only
one fails. Passing permits preparation of the next stage, not a patient launch.

### A2 Dynamic candidate and reward contracts

In parallel with A1 preparation, audit source and write zero-update mock tests.
Specify a request-conditioned scoring interface for variable candidate banks;
do not silently treat toy output slots as stable clinical actions. Any later
integrated model is a new implementation, not the already-tested fixed-bank
actor. No automatic fitted architecture search is allowed.

Required contracts: permutation and alias invariance, changing bank size,
singleton support, stable reference identity, exact original float64 request
submission, no hidden feasibility or future-information leakage, old/new log
probabilities for the actual submitted class, same-start forks, full restore,
and time/terminal accounting. Distinguish requested actions from realized
execution. A change in logits is not proof of a change in executed behavior.

Complete the reward decision audit below from saved receipts and source only.
Do not run the patient environment, collect new trajectories or optimize a
patient-data model. Mock tests can verify algebra without generating research
performance data. Avoid changing historical locked collectors or configs.

### A3 First new simulator performance pilot

This is a separately approved future stage. Start only after A1 passes, A2
contracts are complete, reward validity has a written decision, and the actual
integrated model passes a budgeted real preflight in its own approved protocol.
An artificial pass alone cannot establish patient readiness.

Draft scope for a fast first decision: GCN only, three independent training
blocks, 32 episodes per continued arm, 12 fresh paired evaluation worlds per
block. Compare five controllers: new-policy PPO, its identical frozen start,
its identical-start BC-CONTINUE, R4, and full MDL-2. This implies 192 continued
training episodes and 180 final evaluations, or 19,344 environment steps at
52 steps/episode, BEFORE initialization, qualification, preflight and clones.
This arithmetic is not a launch budget. The complete protocol must add and cap
every extra step, optimizer call and wall-clock phase before asking approval.

Specify initialization explicitly; a near-uniform new actor is not a
well-trained frozen-policy comparator. Preserve useful reference behavior
without a permanently insurmountable score prior, test its expressivity and
qualification prospectively, and compare the new actor to its own exact start.
Do not credit an architecture/initialization change to RL. All learners receive
the same public information and deployment action support. Keep the original
reward and scenario unless the separate objective gate requires amendment.

Primary endpoint: raw total-cost difference, PPO minus its same-start frozen,
on paired worlds; negative is better. Report block effects, relative change,
uncertainty, action changes, losses, completions, waiting and terminal-active
obligations. Prespecify one practical improvement threshold before launch;
the old 1% development screen is a candidate, not a newly approved threshold.
No posthoc choice of favorable seeds, checkpoint, metric or evaluation mode.

### A4 Independent confirmation and manuscript claims

Only a promising, clinically interpretable development result warrants a
separate confirmation design. Determine additional independent training seeds
and evaluation worlds from pilot variance and a prespecified meaningful effect,
not a promise of significance. Three training blocks are a screening study;
180 episodes are not 180 independent trained policies. Intervals must respect
training blocks and paired-world dependence; disclose instability with few
training seeds. Do not use inspected P2/R6 worlds as unseen confirmation.

Add GCN/self-only/flat attribution with matched information and explicit
parameter counts; match capacity where feasible and disclose mismatches.
Include strong task-relevant heuristics/planners with the same available
information. A claim of superiority to repaired DDPG requires its own fair
budget-matched comparison, not historical broken-run comparisons. DDPG is not
mandatory for the paper.

Our near-term estimand is RL training in a simulator followed by frozen
deployment. Unknown persistent capacity or lead-time changes are a separate
future adaptation study: compare online updates versus the same frozen start,
adaptive heuristics, and online system identification plus MPC under identical
information. Do not launch new scenarios just to obtain a positive result.

## When to change reward

### Audit now

Map every cost term to its operational meaning and source; separate operating
cost, patient loss, expiry and urgency. Current P1/P2 definitions use negative
absolute step cost, a single 1e-9 scaling, gamma=1, a 52-step terminal horizon,
zero terminal bootstrap and no added terminal cost. Verify the actual effective
config and receipts, not just the original draft. These choices define a
finite-horizon objective, not a complete patient-lifecycle objective.

Audit sign, units, double counting, discount/horizon alignment, delay to outcome,
and which remaining obligations are visible at episode end. Use saved data to
describe associations, not to invent unseen counterfactual outcomes. Record
missing domain/clinical justification as missing. Never assign a dollar value
or noninferiority margin from our desire to publish.

### Decision rules

| Observation | Decision | Permission boundary |
| --- | --- | --- |
| Wrong sign, duplicate cost, mixed units, broken terminal flag | Stop affected execution, retain evidence, isolate accounting bug | Engineering reproduction allowed; changed research implementation and a new run require an approved amendment |
| Correct objective, actor cannot change actions or value scale is inadequate | Fix learnability, normalization or independent critic first | Artificial engineering only within its approved packet; do not call this a better clinical reward |
| Correct objective, learnable actions, but noisy/delayed credit | Consider a single prespecified shaping or credit-assignment experiment | Draft only until separately approved; no reward-weight sweep |
| Documented mismatch between stated goal and modeled incentives | Propose objective/constraint revision before the next patient pilot, even if learning is not yet solved | Explicit user decision, versioned protocol, new data; no need to waste a pilot on a known invalid objective |
| Cost improvement accompanies worse patient outcomes or deferred workload | Do not declare overall success; identify acceptable trade-offs or reject the candidate | Operational/clinical tolerance must be justified before confirmation; no automatic weight adjustment |
| No replicated actionable headroom under a valid objective | Close this RL route or propose a justified new scenario | No reward modification merely to manufacture a win |

### Preferred forms of improvement

1. Numerical scaling is not objective redesign. A fixed positive reward scale
   preserves mathematical return ordering, but changes optimization and the
   relative effect of entropy/regularization. Record those effects explicitly.
2. If credit assignment is the problem, potential-based shaping is a candidate:
   `r'_t = r_t + gamma * Phi_(t+1)(s_(t+1)) - Phi_t(s_t)`.
   Use public state only, freeze the potential without test data, and for the
   finite horizon set terminal potential to zero (or an appropriate constant).
   Check telescoping and terminal/truncation semantics on mock trajectories.
   Theory preserves the objective under its assumptions, not finite-sample
   neural performance. See [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
3. If cost and patient outcomes genuinely conflict, prefer an explicitly
   constrained/Pareto formulation over silently increasing loss weights until
   RL wins. Propose operating-cost minimization subject to independently
   justified service/outcome limits; report feasibility and each outcome.
   Constrained RL is an established formulation, not a safety guarantee for
   this simulator. See [Achiam et al.](https://proceedings.mlr.press/v70/achiam17a.html).
4. If unfinished obligations distort the intended lifecycle objective, define
   continuation or terminal valuation with domain rationale and no double
   counting. This changes the estimand; it is not a harmless normalization.
   Do not equate active patients with deaths, or treat them as free benefit.

For any approved revision, retain old reward and raw cost/outcome evaluation,
version all changes, freeze coefficients/constraints before outcomes, and use
the same-information frozen/BC controls. Separate `learner change only` from
`reward change only` and their interaction where the approved budget permits.
Do not compare only rescaled/shaped reward totals and label that improvement.
Never change objective, scenario and learner together and attribute everything
to RL. If a multi-change exploratory package is necessary, disclose that limit.

## Automatic workflow

The schedule returns to this thread every 30 minutes. A wake-up continues the
next authorized local task, not a new experiment. Continue multiple unblocked
steps in one turn; do not wait for another heartbeat after each small step.
Local scheduling needs the computer on and app running; it is not a guaranteed
always-on service. See [official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app).

`workflow.json` is an auditable queue, not a scientific execution config or
security boundary. Protocol, user authorization, actual source and runner
guards all remain required. The finite current chain is:

1. A1 implementation/protocol and zero-update tests, with numerical run only
   after its specific approval; commit implementation/config before running.
2. A2 source-level dynamic-candidate contracts and saved-data reward audit;
   no patient calls, fitted architecture variant or reward modification.
3. If A1 is approved, run its single packet, independently verify raw receipts,
   locally archive every file with hashes and close it on success or failure.
4. Deliver an integrated readiness report and one bounded A3 approval question
   if ready, otherwise the exact remaining blocker. Never launch A3 by default.

Every wake-up first reads AGENTS, this plan, workflow.json, the locked plan's
latest changes and the Live checkpoint. Verify actual branch/HEAD, working-tree
changes and relevant process identity. If a matching task is live, inspect its
PID/PPID/command and persisted progress; never start a duplicate. If process
inspection is unavailable, mark it unverified and do not launch computation.
Retain unrelated edits and all failed-run evidence.

States: pending, preparing, ready, approval_required, running, verifying,
passed, failed, closed. Transition to running only with explicit scope, frozen
source/config/runtime/input hashes, test acceptance, a unique claim, unused
attempt and all caps. Charge before calls; no refund or transfer. A terminal
failure preserves partial checkpoints/RNG/ledger/stderr and closes the attempt.
Do not repair and rerun a recorded numerical failure automatically. Pre-run
zero-update engineering failures may be fixed and retested with error history.

After code edits run relevant non-fitting tests and full repository compileall.
Record tests, evidence paths, pending approval, process status and next concrete
action in the Live checkpoint. A running claim requires fresh matching process
evidence and a new boundary or two file-growth samples at least 60 seconds apart.
Archive new results within this project with per-file verification. No Dropbox
copy without explicit payload/destination approval; cloud sync and Howard access
remain separate facts. No remote push/PR/merge, messages, paid compute, formal
holdout, historical evidence replacement or claim of Howard's approval.

Notify only on meaningful progress, completed gates, failures or a required
decision. Do not repeat unchanged waits. When this finite chain is finished,
fails terminally, or only new-scope approvals remain, write the handoff and
delete this automation. Do not invent additional experiments to keep it alive.
If approval arrives later, continue the specifically approved packet; a new
schedule is not itself needed to authorize its scientific scope.

## Time estimates and stopping rules

Engineering target: A1/A2 preparation within 1-2 working days; A1 numerical
work has a 30-minute cap, not a 30-minute end-to-end delivery promise. If those
gates pass and A3 is approved, aim for a first simulator readout in 2-3 working
days from readiness. Historical P2 took about 2h38m; a new runtime must be
measured, not assumed. Formal confirmation takes a separately sized campaign.
These are planning estimates conditional on gates, permissions and machine
availability, not guarantees.

One well-instrumented null or failed result ends the current attempt. Diagnose
with saved data; propose at most the next scientifically justified bounded
decision, rather than automatically changing algorithms or rewards. If online
RL still lacks stable incremental benefit, the manuscript should emphasize
traceable graph/distillation results and report the RL boundary honestly.
Stronger provenance, fair attribution and justified trade-offs improve the
paper; neither positive RL gain nor EAAI acceptance can be guaranteed.

Methodological references: [PPO](https://arxiv.org/abs/1707.06347) motivates
testing interaction-sampled data rather than just oracle-table optimization;
[Agarwal et al.](https://arxiv.org/abs/2108.13264) motivates uncertainty-aware
RL evaluation. Neither paper establishes effectiveness in our environment.
