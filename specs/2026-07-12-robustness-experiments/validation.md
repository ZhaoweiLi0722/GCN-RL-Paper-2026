# Robustness / stress experiments — Validation

## Invariant checks (every experiment must pass all three)

1. **Fair baseline present.** The result table includes `umyo` (condition-aware) and/or
   `fmyo` (forecast-aware), configured with the true `demand_rates` / shared forecast. The
   headline win statement names the *aware* heuristic beaten, not just a blind one.
2. **True OOD, labeled.** For any robustness/generalization claim, at least one test regime
   lies **outside** the DRL training support, and the table marks in-distribution vs OOD.
   A unit test confirms the randomization hook samples only within the training range and
   the eval path is fixed.
3. **Statistical rigor.** ≥3 seeds (5 for the headline experiment); IQM + bootstrap CIs
   reported; any "DRL wins" claim has non-overlapping CIs (state it explicitly when they
   overlap — no overclaiming).

## Automated
- Project tests stay green; new hook has a unit test (range coverage + fixed eval).
- Each runner validated at tiny budget (≤200 steps, 1 seed) before the real launch.
- Cite-key / config integrity unaffected.

## Manual
- Each experiment's config differs from nominal only in the intended dimension (diff the
  JSON) so the effect is isolated.
- Runs launched under `caffeinate`, resumable; a sleep-kill loses only the in-flight seed.

## Definition of done
- Experiments A (A1+A2), B, C, D each produce a committed IQM/CI table with the invariants
  satisfied, and a one-paragraph honest finding (which regimes DRL wins / ties / loses).
- `robust-experiment-design.md` status updated; results ready to feed Phase 10 (§5–§6).
- Roadmap Phase 9 items checked off as each lands.
