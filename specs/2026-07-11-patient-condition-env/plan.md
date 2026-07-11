# Plan — Patient-Condition & Product-Expiry Environment Layer

Task groups are ordered but each is independently implementable and testable.
Validate after each group (`compileall` + that group's tests). Keep
`src/env/capacity_planning.py` and its tests untouched.

## 1. Patient survival model (`src/env/patient_condition.py`)

1.1. A `PatientState` (survival `S`, enrollment epoch, health index `H`,
     deterioration-shock epoch, status enum).
1.2. A `PatientConditionModel` with config-driven parameters: survival-curve
     bounds (upper/lower), health-index distribution, Weibull deterioration
     parameters, `eligibility_threshold`, per-epoch decay.
1.3. `enroll(rng) -> PatientState` and `advance(patient, epochs=1)` implementing
     decay + the post-shock accelerated decay.
1.4. `is_eligible(patient) -> bool` against the threshold.
1.5. Unit tests: survival monotonically decreases; decay accelerates after the
     shock; eligibility flips below threshold; deterministic under seed.

## 2. Aging, expiry-aware material inventory

2.1. An age-bucketed container for specimens/finished product (reuse the
     bioreactor-pipeline age-advancing array pattern), with a shelf-life window.
2.2. `add`, `advance` (age one epoch; items past shelf-life become **waste**),
     and `consume` (FIFO by age).
2.3. Track `material_wasted` per epoch for cost/diagnostics.
2.4. **Viability hook (disabled by default):** give `advance`/`consume` an
     optional `viability_fn(age, transport_time) -> factor` seam that defaults to
     a no-op (factor = 1.0). MVP ships with it off; enabling it later degrades
     product viability with cold-chain transport time — no re-architecting.
2.5. Unit tests: items expire at the right age; FIFO consumption; waste counted;
     with the default no-op hook, viability factor is exactly 1.0 (behaviour
     unchanged); a stub non-trivial `viability_fn` reduces the factor as expected.

## 3. Environment layer (`src/env/patient_capacity_planning.py`)

3.1. `PatientConditionCapacityEnv` that composes/subclasses `CapacityPlanningEnv`
     so base manufacturing dynamics, `facility_net` actions, and transfers are
     preserved; base env left unmodified.
3.2. `reset(seed=)` initializes base state + per-clinic patient queues + aging
     inventories; returns the extended observation.
3.3. `step(action)`: run base manufacturing step, then the patient layer —
     enroll arrivals, decay/deteriorate, apply eligibility gate (lose patients),
     start production for prioritized patients, age material, expire/waste,
     deliver. Return `(next_state, reward, done, info)`.
3.4. Priority: deteriorating patients (lower survival) ordered first when
     capacity is scarce (urgency-weighted).
3.5. **Transport-time plumbing for the viability hook:** when transfers/
     transshipment move specimens or product between clinics, thread the
     transfer lead time into the aging container's `viability_fn` seam (behind
     the same config flag, off by default). MVP behaviour is unchanged; the wiring
     exists so the pilot can enable cold-chain viability without new plumbing.

## 4. Reward coupling

4.1. Extend the cost with `w_lost·patients_lost + w_expiry·material_wasted +
     w_urgency·urgency_weighted_shortfall`; base cost unchanged; weights from
     config.
4.2. Add `patients_lost`, `material_wasted`, `eligibility_rate` (and keep
     existing service_level / waiting-time / utilization) to `info`.
4.3. Unit test: each new cost term responds to a constructed scenario.

## 5. Observation & graph features

5.1. Per-clinic fixed-width summary of the patient queue (waiting count, mean
     survival, count-at-risk, count-near-expiry, fixed survival/age histogram).
5.2. Concatenate onto the base flat observation; keep a documented, fixed
     `features_per_facility` so `FixedObservationScaler` and flat baselines work.
5.3. Add the same summary columns to graph node features; update the graph spec
     builder so GCN models see them.
5.4. Unit tests: observation width is fixed and matches the advertised
     `observation_size`; graph node-feature shape is correct.

## 6. Configs & scenarios

6.1. `experiments/configs/20_clinic_patient_condition.json` — the manuscript
     20-clinic setting plus the patient-condition/expiry parameters.
6.2. Disruption variants mirroring the existing `disruption_0_05/0_3/0_6`.
6.3. A tiny 2-clinic dev config for fast smoke tests.

## 7. Integration & smoke

7.1. Register the env so training/eval can build it from config (mirror how the
     base env is constructed).
7.2. Smoke: run one learned agent (e.g. `flat_ddpg`) and one heuristic for a few
     steps on the new env end-to-end; confirm `info` diagnostics populate.
7.3. `python -m compileall .`; full existing test suite still green.
7.4. (Deferred to Phases 5–6) wiring into the full benchmark manifest and the
     verification harness once algorithms target this env.
