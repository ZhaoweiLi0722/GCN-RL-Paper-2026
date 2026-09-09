# Pre-Execution Validation

Date: 2026-09-08. No scientific screen seeds were used by these tests.

Runtime: existing project Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0 at
`/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python`.
This is a CPU simulator/heuristic screen, not neural training or MPS execution.

## Checks

- Full `python -m compileall .` passed with a separate temporary bytecode cache.
- 87 focused tests passed across residual authorization, residual mechanics,
  shared-capacity screens, intertemporal dynamics, overtime equivalence,
  overtime contracts, heuristics, and headroom screening.
- An end-to-end fixture used only state seed 123 and world seeds 456/1456,
  two real legal actions, and a three-step horizon. CSV, summary, claim,
  terminal status, and all inventory hashes reconciled.
- An initial fixture used overlapping world families; configuration validation
  correctly rejected it. The fixture was corrected before the passing suite.
  No scientific state generation or production output was involved.
- Failure injection preserved already-written rows and marked terminal failure;
  existing roots refused re-execution. No automatic retry/resume exists.
- Original config remains unauthorized with Howard false. The new exception
  config is authorized by Zhaowei, with Howard still false. Attempts to change
  science, the exception record/hash, or output destination are rejected.
- `--describe` verified 54 states, 81 candidates, 8,748 discovery rows and
  21,870 validation rows without constructing an environment.
- Local process inspection found no related Python campaign; repository search
  found the reserved 997/998/999 seed families only in the unexecuted protocol.
- GitHub PR #12 remains draft/open at
  `118f9978f50433d25de1e392e242cd1655101b43`, with no reviews or comments.

## Execution Command

Run only from the committed isolated worktree. The evaluator claims its
single-use output root before state generation, saves one CSV batch per
completed paired world, and persists status with PID, command and config hash.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -u \
  -m evaluation.screen_intertemporal_residual_allocation_headroom \
  --config experiments/configs/intertemporal_residual_allocation_headroom_user_20260908.json
```

No scientific result is implied by passing preflight. Required terminal audit:
status completed/exit 0; 30,618 unique, finite rows with exact phase, state,
action and CRN cardinality; paired RNG agreement; complete 54-state manifest;
recomputed R0-R2 summary; matching artifact hashes; no training/formal output;
original branch/config/protocol and parent evidence unchanged. Failure is
retained; no automatic downstream action follows a positive or negative gate.

## Post-Execution Checks, 2026-09-09

- The same 87 focused tests passed again after execution (2.775 seconds).
- Full `python -m compileall -q .` passed with the existing venv and separate
  temporary bytecode cache, including the new read-only audit script.
- `audit_result.py` independently reproduced all R0-R2 decisions and primary
  calculations from the retained CSV; [audit.json](audit.json) records its
  output. It imports no experiment code and consumes no simulation seeds.
- Terminal evidence, limits and the uncorrected clock discrepancy are recorded
  in [results.md](results.md). Scientific support failed; execution and artifact
  integrity checks passed. These are different conclusions.

Focused test invocation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' -m unittest \
  tests/test_residual_headroom_user_authorization.py \
  tests/test_intertemporal_residual_headroom.py \
  tests/test_intertemporal_shared_capacity_screen.py \
  tests/test_intertemporal_shared_capacity.py \
  tests/test_overtime_control_equivalence.py \
  tests/test_overtime_control_contract.py \
  tests/test_overtime_heuristics.py tests/test_overtime_headroom_e2.py
```
