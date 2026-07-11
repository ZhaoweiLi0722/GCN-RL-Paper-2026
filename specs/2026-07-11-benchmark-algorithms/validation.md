# Validation — Benchmark Algorithms on the Patient-Condition Env

## Automated

- `python -m compileall` (our package dirs) exits clean.
- New tests pass:
  - **Existing heuristics on patient env:** MYO/ISO/MDL-1/MDL-2/F-MYO emit valid
    facility-net actions (shape, `[-1, 1]` bounds), are deterministic under seed,
    and beat a random policy on total cost.
  - **uMYO:** valid actions, deterministic, falls back to myopic on the base env,
    and — key — achieves a **higher eligibility rate than MYO** on a capacity-
    constrained patient config.
  - **RL patient sanity:** flat-DDPG trained briefly on a small patient config
    beats a random policy on evaluated cost.
- The **existing** suite still passes unchanged (no base-heuristic behaviour
  change): `python -m unittest discover -s tests`.

## Manual

- Run `umyo` and `myo` on `2_clinic_patient_condition.json` via
  `evaluate_formal`; confirm both write rows and `umyo`'s eligibility is >= MYO's.
- Spot-check on the 20-clinic patient config that `umyo` raises eligibility over
  MYO under supply disruption (the condition-aware advantage).
- Confirm the LQR-gate results are recorded in `provenance.md` (not re-run).

## Edge cases

- uMYO on the **base** (non-patient) env → behaves as plain myopic, no crash.
- Zero at-risk patients → uMYO surge term is zero (reduces to myopic).
- 2-clinic and 20-clinic patient configs both run.
- Determinism: same seed → identical action/cost sequences for every baseline.

## Definition of done

- All automated checks pass; existing suite unaffected.
- `umyo` is registered, verified, and beats condition-blind MYO on eligibility.
- Every baseline has a row in `provenance.md` with its verification status.
- No new dependencies (no Gurobi); English-only; base heuristics unchanged.
- Roadmap Phase 5 items checked off; `docs/research_howard/` not committed.
