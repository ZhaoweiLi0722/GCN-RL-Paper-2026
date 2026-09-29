# S4 engineering acceptance before recording

Date: 2026-09-29. Protocol/config frozen at `e4e9fef`.

- Additive modules: `src/env/completion_continuations.py`,
  `src/baselines/completion_feedback_tree.py`, and
  `evaluation/check_completion_continuation_comparison.py`.
- 14 new tests and 154 combined relevant tests pass; full compileall and diff
  checks pass. No S1/S2/S3 source or artifact was edited.
- All 315 historical input/source hashes match `838249d`; S3's independent
  read-only audit passes again. The two S4 protocol/config locks match.
- A tiny invented-state smoke verifies recording, frozen trees/selections,
  independent arithmetic/scalar audit, overwrite refusal and retained partial
  failure evidence. It uses a unit-only namespace and 2/4/8 outer paths, not
  the S4 recorded matrix. Temporary fixtures are not scientific outcomes.
- Scalar/batched parity tests check every first action under all four methods,
  two downstream capacities and several response settings. Old booked total
  costs and query counts agree exactly. Component sums include closure costs.
- Tree causality is tested by changing future draws while preserving the first
  event: the first and second requests do not change. Changing the observable
  event can change the second request. Current requests do not affect the
  first interval's already-booked service completion.
- Raw arrays are compressed NumPy archives. Inner arrays retain every second-
  request cost; outer arrays retain holding/labor/switching per path. The audit
  uses independent Python arithmetic and bounded scalar replays, not a full
  matrix rerun. Both planning and assessment costs are reported.

Recorded S4 execution has not occurred at this acceptance checkpoint.

From the persistent integration worktree and project virtual environment:

```bash
python -m unittest tests.test_completion_continuation_comparison tests.test_completion_action_ranking tests.test_completion_rollout_control tests.test_completion_control_screen tests.test_completion_feedback_queue tests.test_completion_feedback_check tests.test_completion_feedback_closure tests.test_service_queue_network tests.test_service_queue_observation_contract tests.test_service_queue_evidence tests.test_service_queue_completion_rule tests.test_queue_cost_sensitivity tests.test_service_effort_mechanics tests.test_service_effort_decisions
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python -m compileall -q .
```

The only recorded run, after source commit and live process check:

```bash
python -m evaluation.check_completion_continuation_comparison --output reports/2026-09-29-completion-continuation-comparison/run
```

Subsequent verification only:

```bash
python -m evaluation.check_completion_continuation_comparison --output reports/2026-09-29-completion-continuation-comparison/run --audit-only
```
