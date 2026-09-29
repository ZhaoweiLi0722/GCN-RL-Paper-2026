# S3 implementation acceptance

Prepared 2026-09-29 before the recorded matrix. Protocol and config remain
unchanged at `b91dd3b`; S2 source/results remain unchanged at `efe5a86`.

- Added only `evaluation/check_completion_action_ranking.py` and its test file.
- 17 new tests, 140 combined relevant tests pass; full compileall passes.
- S2 read-only audit passes for all 288 archived episodes. All 40 input/source
  locks match the S2 result commit. Protocol/config match their protocol commit.
- Tiny unit recording uses an invented initial state, unit-only namespace and
  2/4/8 forecasts. It tests recording, independent audit, corruption rejection,
  partial-failure retention and overwrite refusal. It is not the S3 matrix or
  a new actual-world/patient episode; its temporary files are test fixtures.
- Raw costs use separate `(15, 128)` NumPy arrays per context/block/chunk;
  zero-based path indices, seeds, uniform hashes and query counts are recorded.
  All four selection actions are frozen before any validation block is run.
- The audit recomputes every mean and pairwise difference using Python
  `statistics.fmean/stdev`, distinct from the NumPy production arithmetic.
  It regenerates only uniform tapes, not the full forecast matrix. Bounded
  scalar checks replay indices 0, 127, 128 and 2047 for all 15 actions in every
  block: 960 scalar forecasts planned, not 960 new actual-world episodes.
- The maximum recorded matrix remains 491,520 model paths and 900 seconds.
  Per-chunk deadline checks include the bounded audit. Any recorded failure
  retains raw chunks/error/status and is terminal without automatic retry.

Recorded S3 execution has not occurred at this acceptance checkpoint.

## Commands

From the persistent integration worktree, with the project virtual environment:

```bash
python -m unittest tests.test_completion_action_ranking tests.test_completion_rollout_control tests.test_completion_control_screen tests.test_completion_feedback_queue tests.test_completion_feedback_check tests.test_completion_feedback_closure tests.test_service_queue_network tests.test_service_queue_observation_contract tests.test_service_queue_evidence tests.test_service_queue_completion_rule tests.test_queue_cost_sensitivity tests.test_service_effort_mechanics tests.test_service_effort_decisions
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python -m compileall -q .
```

After committing implementation, the sole recorded command is:

```bash
python -m evaluation.check_completion_action_ranking --output reports/2026-09-29-completion-action-ranking/run
```

Subsequent verification is read-only, not a matrix retry:

```bash
python -m evaluation.check_completion_action_ranking --output reports/2026-09-29-completion-action-ranking/run --audit-only
```
