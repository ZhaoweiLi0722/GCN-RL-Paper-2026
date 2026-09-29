# R5 complete: replacement-policy load and device compatibility

2026-09-29. This is engineering acceptance on saved observations, not new
performance evidence, recovered F1 parity, or proof of a competent baseline.

## Results

All three R4 seeds pass strict metadata/graph/tensor loading, exact CPU
policy-file versus full-state-module output agreement, and the fixed CPU/MPS
checks on 16 preselected replay observations each (48 total).

| Seed | Max actor difference | Max gate-score difference | Max request difference | Min gate margin |
| --- | ---: | ---: | ---: | ---: |
| 60 | 2.0862e-7 | 7.7486e-7 | 2.2352e-8 | 0.187147 |
| 61 | 2.9802e-7 | 5.9605e-7 | 2.9802e-8 | 0.017330 |
| 62 | 2.0862e-7 | 5.9605e-7 | 2.2352e-8 | 0.050945 |

Hard gate decisions and rounded requested specimen lots match exactly on all
48 observations. Every policy tensor remains unchanged. None of the saved
critic/optimizer/replay states initializes a learner; no optimizer step or
environment reset/step occurred. The wrapper reuses the existing agent's
inference path; its constructor's empty replay and unused optimizers are not a
new training framework. Time is present at observation index560; sampled t/H
ranges from 0 to 51/52. These finite sampled checks are not a global equivalence
proof or certification of clinical routing, feasibility, safety or headroom.

All 24 R4 payload files and 83 archived existing source files match before and
after. Static request quantization does not execute inventory/patient matching.
The nominal-history collector, continuation-state resets, executed-action
deduplication/reachability and fresh RNG streams remain to be validated.

## Preserved reporting failure

Initial source `f089fd0b56b3e7eed36b26b79438234a103addca` evaluated seed60 and
wrote raw outputs, then exited1 because scalar gate-margin summaries assumed
a list. The immutable failed root and traceback remain. This was a summary
bug, not a measured device mismatch or a failed training run.

Repair/continuation source `91a1e5b14a3eb73deaa66cd9660fd25ebbf4e797` reused
seed60 bytes with no new inference and evaluated only unvisited seeds61/62,
exit0. No observation, tolerance or scientific parameter was changed. See
`summary-failure.md` and the committed continuation config for all prior hashes.

## Independent verification

The separate stdlib verifier does not import the model, NumPy or collector. It
recomputes all raw paired errors, threshold decisions, float32 requested-lot
rounding, selection indices, time coordinates and hashes. It also verifies
unchanged R4 inputs/source and retained failure evidence. This verifies saved
arithmetic, not an independently implemented model or environment.

All 23 new loader/verifier tests and 69 related tests pass, together with full
repository compileall and diff checks. No new simulator smoke is necessary for
this no-environment inference packet; the real-agent synthetic roundtrip test
forbids environment reset/step, legacy partial loading and optimizer updates.

```bash
python -m unittest tests.test_strict_frozen_policy tests.test_fixed_window_objective tests.test_validated_returns tests.test_research_archive tests.test_frozen_baseline_rebuild -q
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python -m compileall -q .
python -m evaluation.verify_replacement_policy_compatibility
```

## Next decision

The provenance blocker is narrowed: usable **replacement** policies now have
strict loading and sampled cross-device acceptance. Do not call them original
F1 policies, assume historical performance, pool them with G1, or retrain again.

The proposed next data-only check retains the original 52-step absolute-cost
objective and all R3 caps. First complete collector/action identity and stream
acceptance; then, only with explicit pilot approval, collect 12 dependent states
and 8 discovery + 8 validation draws (max1,152 records, 37,596 steps, one hour).
The primary comparison is alternate first action versus frozen first action,
**both followed by the same frozen actor**, not online learning. Report every
seed and clinical harm, even when cost improves. No automatic training gate.

An approval question was sent in the chat; do not assume its answer. If approved,
append a prospective R3 replacement amendment and freeze implementation/config
before collection. Reward costs, previous null results, Stage E and holdout stay
unchanged. No remote Git or collaborator message was sent.
