# AGENTS.md

This is a research codebase for graph-aware deep reinforcement learning in distributed personalized regenerative medicine manufacturing networks. The intended implementation includes GCN-DDPG, flat-state DDPG / MLP-DDPG baselines, graph ablations, multiple random seeds, and Monte Carlo evaluation.

## Research Objective

Zhaowei clarified on 2026-09-30 that the objective is a rigorous, publishable
GCN-plus-reinforcement-learning study, targeting EAAI-level work, not preserving
DDPG or guaranteeing a positive result. DDPG remains a candidate to assess fairly;
other algorithms may be considered when they fit the actual decisions better.
Separate RL policy training in simulation from parameter adaptation during
deployment, and separate both from imitation/distillation. Do not rename an
imitation gain as an RL gain. Favor decisive, bounded comparisons over an
open-ended algorithm search. The locked-plan change control and local-only
execution boundaries still apply; this objective is not a new experiment permit.

## Research Delivery Mode

On 2026-10-01 Zhaowei explicitly requested an independent advancement agent
and a delivery-efficiency evaluator to prevent repetitive audit-only cycles.
Use these two roles for substantive research milestones when their work can
be separated; do not spawn extra reviewers or duplicate agents every turn.
The user further clarified that delivery means completing paper-level
experiments and evidence strong enough to support a defensible submission,
not improving tooling for its own sake. Prioritize missing end-to-end method
comparisons, graph/RL attribution, independent-seed stability, justified
generalization and cost/service trade-offs. Link every engineering diagnostic
to one such gap and an exit back to the corresponding comparison; do not
replace that comparison with an indefinite sequence of toy acceptance tests.
Neither statistical significance alone nor guaranteed positive RL outcomes
is the definition of publication readiness. Acceptance cannot be promised.

On 2026-10-03 UTC Zhaowei reiterated that preparation must not drag and the
priority remains improving model performance. For future authorized comparisons,
state the concrete performance question, minimum remaining preparation and a
realistic preparation-time estimate. If preparation overruns, identify the actual
blocker and defer nonessential work; do not start another audit or toy-test chain.
Move from necessary entry checks directly to the authorized full comparison.
This prospective guidance does not reopen failed attempts or alter frozen locks.

- The advancement agent owns one concrete deliverable with disjoint file
  ownership: usable code, decision-changing saved-data analysis, or an already
  authorized bounded experiment. It must proceed through routine steps inside
  that scope rather than return only a plan. A closed attempt is not reusable.
- The efficiency evaluator is read-only and advisory. Review the current
  critical path once per milestone or after two preparation-only cycles;
  identify up to three avoidable delays and what to stop/reuse. Do not repeat
  numerical verification, full-history hashes or tests, and do not add another
  approval gate. State uncertainty when timing/effort data are absent.
- The coordinating agent owns integration, the Live checkpoint, final scoped
  tests/compileall and local commit. Keep moving on non-overlapping work while
  delegates run. Resolve disagreements without another reviewer chain.
- Reuse valid completed checks; rerun only for changed dependencies, concrete
  defects, or checks explicitly required at an execution/archive boundary.
  Consolidate user decisions at genuine new scientific or external-action
  boundaries; do not ask again for routine already-authorized steps.
- Report progress as: a new supported answer, a usable implementation that
  removes a named blocker, or a completed comparison. Report actual patient
  performance separately. Document/test/commit counts are not research gains.
  A well-supported negative result that changes the next decision is progress.
- Each checkpoint states the artifact, question answered, remaining patient-
  comparison blockers, next action and whether it is authorized. If two work
  cycles only restate preparation, consolidate the handoff and either deliver
  the missing implementation or ask one concrete bounded scope question.
- This is task delegation, not an always-running scheduler. Record actual
  agent states, close finite assignments, and never infer background execution
  from a role definition. No automatic budget expansion, new fit, reward
  search, patient episode, remote action or guarantee of RL improvement.

## Adaptive Paper Workflow

The current delivery roadmap is
`specs/2026-10-01-adaptive-paper-delivery/plan.md`, with task state and decision
history in its `workflow.json`. Zhaowei requested automatic routine progression
and evidence-driven replanning on 2026-10-01. Continue the current finite local
engineering chain without requiring another "continue" for each step. Use one
same-thread schedule, not parallel copies of the campaign. Scientific scope,
frozen parameters, consumed attempts and external-action boundaries are not
automatically editable. A new integrated pilot needs one complete prospective
approval, including the explicit replacement of the failed artificial gate;
that gate must never be reported as passed. Do not append more toy campaigns
to delay the end-to-end comparison. Inspect actual processes and record actual
deliveries, not inferred background progress. See the roadmap for closure and
consolidated-decision rules; it is not itself an experiment execution permit.

On 2026-10-03 Zhaowei explicitly requested preparation AND execution together:
after approval of a complete numerical package, finish necessary implementation,
tests and committed locks and proceed directly to its single scientific attempt.
Do not stop at readiness to ask for a redundant launch approval. Continue routine
training, evaluation and readout within that package. This does not authorize
unasked scientific budgets, changed rewards/scenarios or retries after failure.

## Coding Guidelines

- Keep environment dynamics, graph construction, model architectures, training loops, and evaluation scripts modular.
- Put PRM manufacturing simulation logic under `src/env/`.
- Put graph construction and edge-ablation logic under `src/graph/`.
- Put GCN-DDPG actor/critic models and shared neural network components under `src/models/`.
- Put flat-state DDPG / MLP-DDPG and other non-graph baselines under `src/baselines/`.
- Put reusable logging, seeding, metrics, and config helpers under `src/utils/`.
- Avoid hard-coding experiment parameters inside model files.
- Use config files under `experiments/configs/` whenever possible.
- Keep scripts under `experiments/scripts/` thin: load config, set seeds, call library code, and write logs.
- Do not fabricate experimental results. Tables, plots, and manuscript claims should be traceable to logged outputs.

## Locked Routing-Primary Experiment Plan

Before changing, launching, extending, or interpreting any patient-indexed
specimen-routing experiment, read
`docs/patient_indexed_specimen_routing_locked_execution_plan.md`. That file is
the authoritative cross-session execution plan. Context compaction, a new
Codex task, convenience, an isolated seed result, or available compute is not
a reason to change the main sequence.

Any scientific amendment requires a completed-stage evidence review, explicit
user approval, an appended change-control entry, and a committed plan update
before the amended experiment is launched. Existing result roots, checkpoints,
teacher artifacts, CRN streams, and provenance remain immutable.

## Research Artifact Preservation

- Run research jobs in a persistent project worktree, never a temporary
  `/tmp` checkout. Use a new run directory; preserve failed and old artifacts.
- Freeze source/config before execution. Save the execution commit, runtime
  versions, teacher/input hashes, effective configs, raw logs, summaries,
  checkpoints and full resume state. Record any RNG state absent from the
  existing checkpoint format separately; do not promise bitwise GPU recovery.
- Archive completed run boundaries with per-file and archive SHA256 checks.
  Use `src/utils/research_archive.py` or the campaign's established equivalent;
  verify all archive members and destination bytes, never overwrite a different
  backup. Keep originals after copying.
- Zhaowei requested Dropbox preservation on 2026-09-29. The current project
  destination is documented in the R4 protocol/config. Use versioned run
  subdirectories there for authorized research artifacts. Do not change sharing
  permissions, create public links, or send files to collaborators implicitly.
- Distinguish local archive verified, Dropbox local copy verified, cloud sync
  verified, and collaborator access verified. A sync-folder copy alone is not
  an independent off-device backup. Do not claim old campaigns were archived
  unless their payloads and checksums have actually been verified.

## Validation

Before finishing any coding task, run at least:

```bash
python -m compileall .
```

If the local sandbox cannot write to the default Python cache directory, use:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
```

If tests exist, run the relevant tests. If training or evaluation scripts are modified, run a small smoke test when feasible, using a tiny horizon, few facilities, and a minimal seed count.
