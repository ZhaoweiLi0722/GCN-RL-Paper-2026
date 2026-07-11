# Validation — Patient-Condition & Product-Expiry Environment Layer

## Automated

- `python -m compileall .` (or the `PYTHONPYCACHEPREFIX` variant) exits clean.
- New unit tests pass:
  - **Patient model:** survival decreases monotonically while waiting; decay
    accelerates after the deterioration shock; eligibility flips below the
    threshold; identical results under a fixed seed.
  - **Expiry inventory:** items expire exactly at the shelf-life age; FIFO
    consumption; wasted quantity counted correctly.
  - **Viability hook inert by default:** with no `viability_fn` configured, the
    viability factor is exactly 1.0 and trajectories match the no-hook baseline;
    a stub `viability_fn` reduces the factor with transport time as expected.
  - **Env:** `reset(seed=)` deterministic; `step` returns
    `(next_state, reward, done, info)`; observation width equals the advertised
    `observation_size` and is constant across steps; graph node-feature shape is
    correct.
  - **Reward:** each new term (`patients_lost`, `material_wasted`,
    `urgency_weighted_shortfall`) moves the cost in a constructed scenario.
- The **existing** test suite still passes unchanged (base env untouched):
  `python -m unittest discover -s tests`.

## Manual

- Run the smoke: one learned agent (`flat_ddpg`) and one heuristic (`myo`) for a
  few steps on `20_clinic_patient_condition` (or the 2-clinic dev config).
  Confirm the run completes and `info` reports `patients_lost`,
  `material_wasted`, `eligibility_rate` alongside the existing diagnostics.
- Sanity-check dynamics by inspection:
  - With **zero effective capacity**, patients should deteriorate and be lost;
    `eligibility_rate` drops.
  - With **ample capacity**, most patients are served before the threshold;
    `patients_lost` ≈ 0.
  - Long production/transfer delays produce **expiry waste**.
- Confirm a condition-blind baseline (e.g. `myo`) leaves more patients lost than
  an urgency-aware run would — the qualitative signal the paper relies on (full
  quantification is Phases 7/9, not here).

## Edge cases

- No patients waiting (empty queue) — observation summary well-defined (zeros).
- All waiting patients expire in one epoch — no crash; costs/diagnostics correct.
- Single clinic and 20 clinics both build and step.
- Seeding: two runs with the same seed produce identical trajectories.

## Definition of done

- All automated checks above pass; existing suite unaffected.
- The new env builds from a committed config and runs end-to-end with both a
  learned agent and a heuristic.
- `info` exposes the new patient/expiry diagnostics without dropping existing
  ones; observation dimensionality is fixed and documented.
- No new dependencies; English-only; base env and its tests unchanged.
- Roadmap Phase 4 items checked off; `docs/research_howard/` not committed.
