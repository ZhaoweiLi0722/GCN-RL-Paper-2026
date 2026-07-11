# Plan — Benchmark Algorithms on the Patient-Condition Env

Ordered, independently-testable task groups. Validate after each
(`compileall` + that group's tests). Do not change existing heuristic behaviour.

## 1. Confirm existing baselines on the patient env

1.1. Sanity tests for MYO, ISO, MDL-1, MDL-2, F-MYO on a 2-clinic patient config:
     actions are valid (correct shape, within `[-1, 1]`), deterministic under a
     fixed seed, and total episode cost beats a random policy.
1.2. Confirm flat-DDPG builds with `env.observation_size` and steps end-to-end on
     the patient env (extends the group-7 smoke into a recorded check).

## 2. Patient-aware heuristic (`umyo`)

2.1. Add `UrgencyAwareMyopicPolicy` (`algorithm = "umyo"`) in
     `src/baselines/heuristics.py`, subclassing `CapacityHeuristicPolicy`.
     Override `_facility_net_action`: take the myopic base action, then for each
     clinic scale up replenishment (`p`) and bias inbound capacity (`q`) by an
     urgency signal from at-risk / near-expiry patient counts. Surge strength is
     config-driven (`urgency_surge`, default modest).
2.2. Gracefully fall back to plain myopic behaviour when the env is not a
     `PatientConditionCapacityEnv` (no patient accessors) — so it can't crash on
     the base env.
2.3. Expose a small public accessor on the patient env for at-risk / near-expiry
     counts (reuse `_at_risk_unserved_counts`; add a near-expiry counterpart) so
     the heuristic does not reach into private internals.
2.4. Register `umyo` in `_HEURISTICS` / `available_heuristics()` /
     `get_heuristic_class()`.
2.5. Tests: valid actions, deterministic, and — the key assertion — uMYO achieves
     a **higher eligibility rate than condition-blind MYO** on a capacity-
     constrained patient config.

## 3. RL-baseline verification on the patient env

3.1. Add a patient-env sanity to the verification harness (or a sibling helper):
     briefly train flat-DDPG on a small patient config and confirm its evaluated
     cost beats a random policy (learning happens on *our* problem, not just LQR).
3.2. Record the already-run LQR-gate results (flat-DDPG / TD3 / SAC / PPO) rather
     than re-running — reference the harness output.

## 4. Provenance table

4.1. Write `specs/2026-07-11-benchmark-algorithms/provenance.md`: each baseline →
     source (repo / prior paper) → verification status (LQR gate, patient sanity,
     heuristic sanity) → notes. Becomes part of the reproducibility appendix.

## 5. Validation & smoke

5.1. `python -m compileall` (our dirs) clean; full existing suite green.
5.2. Smoke: run `umyo` and one existing heuristic on the 2-clinic patient config
     via the real `evaluate_formal` CLI; confirm it writes rows.
5.3. Update roadmap Phase 5 checkboxes.
