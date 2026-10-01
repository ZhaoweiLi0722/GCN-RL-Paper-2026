# Two-agent delivery: paper evidence is the objective

2026-10-01. Zhaowei requested two complementary agents and clarified that all
work serves a defensible experimental paper targeting EAAI or a comparable
journal. Neither positive RL results nor acceptance is promised.

## Completed assignments

- Descartes: bounded read-only critical-path review; completed and closed.
  No new release gate, tests or numerical re-audit.
- Hegel: implemented `signal_readback.py`, its hand-calculable tests, and ran
  the training-receipt readback once, exit0;15 synthetic tests passed.
  Completed and closed,without new simulations,model forwards or fitting.
- Parent: reviewed code/summary,integrated durable roles in AGENTS.md and the
  Live checkpoint,and owns final compileall/local commit.

These were actual delegated tasks,not perpetual background services. No
automation or new training was launched. Independent tasks with clear outputs
and parent coordination follow the [official orchestration guidance](https://developers.openai.com/api/docs/guides/agents-api/multi-agent).

## New saved-data finding

`signal-readback.json` consumes298 files:the saved config,nine training
manifests and288 training sample receipts. All13,824 existing observations
are included;no held-out outcomes or checkpoint forwards are used. Only
consumed input hashes are recorded;prior archives are not recopied/re-audited.

Within-rollout,observation-weighted between-context variance accounts for
98.91%-99.56% of residual return-minus-old-value variance across the nine fits.
This includes context/time means,action mix and finite sampling;it is not a
pure causal time-effect estimate. Every training context eventually observed
its winning action,but1,464 of3,456 context-rollouts observed none of it in
that rollout's four draws.

The empirical normalized winner-logit score-function proxy is nonpositive
in1,382 of3,456 context-rollouts despite a positive evaluator-only oracle
expected direction. This is a hypothetical independent-logit calculation at
the behavior policy,NOT the shared network's actual gradient,PPO clipped
update or a significance result. Four draws per context are sparse;winner
exposure does not prove sufficient exploration. Fit-pooled variance includes
policy/value drift and is reported separately.

Decision:prioritize one training-only credit-assignment/baseline change over
an exposure-only extension,while treating both as unproven mechanisms. This
readback does not establish patient-performance improvement or authorize a fit.

## Next paper-facing deliverable

One prospective,bounded end-to-end simulator amendment package,not a new
family of toy experiments. Cover the minimum dynamic-candidate implementation,
competent initialization and qualification,then compare RL / identical frozen
start / BC-CONTINUE with R4 and full MDL-2. Keep reward,information and physical
action support fixed unless an explicit approved amendment justifies change.
Budget every initialization,preflight,clone,update,environment query and
evaluation;report raw costs,patient outcomes and independent-seed effects.

Replacing the failed artificial prerequisite is prospective and requires
approval;the current gate remains failed. A repaired full-package comparison
can establish incremental RL value without proving which repair caused it.
Graph-only attribution additionally requires matched information,gates,
support and capacity. Strong baseline/robustness and independent confirmation
remain open. This session did not close those paper-evidence gaps.

Progress means a paper-evidence gap closed,a usable implementation removing
a named experiment blocker,or a completed informative comparison. Do not reward
document/test/commit counts. Reuse verified components and stop preparation-only
loops. A negative result may inform the paper but cannot be relabeled as RL
benefit. The next numerical scope is not approved or running.

Integration validation:parent code review,full repository
`python -m compileall -q .` and `git diff --check` passed. The worker's15
focused synthetic tests were reused;no historical whole-suite rerun or
second readback execution was added. Existing experimental files are unchanged.
