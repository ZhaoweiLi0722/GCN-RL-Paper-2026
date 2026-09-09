# P0: Public Information and Finite-Tree Mechanics

Date: 2026-09-09. Status before fixture execution: implementation and unit
checks, not a new scientific benchmark. Zhaowei explicitly requested continuing
after the audited residual screen. Howard remains unapproved; no signature or
joint endorsement is inferred.

## Scope and Authority

The preceding residual screen is complete at commit `6655e23`: R0 passed,
R1/R2 failed; prospective saving was 0.189611%. The current paper remains at
Stage E. Those results, costs, scenarios, forecasts, seeds and gates are not
changed or reclassified. There is no new training, holdout access, manuscript
performance claim, push or merge in this step.

The user's instruction authorizes bounded implementation/verification of the
next design. P0 introduces new opt-in modules and deterministic fixtures only.
Existing environment, heuristic, model, training and evaluation code remains
unchanged. The new forecaster is not connected to existing campaigns.

## What P0 Tests

1. Can a predictor depend only on observed arrivals and legitimately announced
   exposure schedules, with explicit issue time and exact restore?
2. Can a small repeated-decision planner enforce nonanticipation and distinguish
   a finite-grid clairvoyant optimum, a finite-grid known-distribution optimum
   using public history, and the best shared open-loop sequence?
3. Can those mechanics run against the existing patient simulator without
   changing the simulator's implementation?

None of these questions establishes an online policy-parameter learning gain.

## Information Timing

The existing environment exposes current arrivals before choosing action t.
The new forecaster therefore records those arrivals once at epoch t, and issues
a forecast for t+1 through t+horizon. Past counts are normalized by public
exposure and averaged over a fixed rolling window; future public exposure is
then applied. The frozen mode uses the fixed prior instead, while retaining
the same observations. This is an interpretable plumbing baseline, not a new
forecasting-method claim.

Callers are responsible for the provenance of an announced schedule. This API
cannot infer whether a caller has fraudulently labeled a latent future
multiplier as public. Real-scenario integration therefore requires a separate
observation adapter and comparator-access audit before scientific use.

Legacy forecast-derived error/history features are not reused. The actual
fixture disables legacy forecast, demand-history and demand-sequence feature
blocks, adds only the new public prediction, and retains physical patient and
resource observations. World IDs, RNG state, scenario labels and future tapes
are not included in the oracle's public information-set keys. Keys retain all
past actions, observed costs and observations, not just the latest state.

## Frozen Mechanics Fixture

Config: `experiments/configs/online_adaptation_information_mechanics.json`.

- Two facilities, seed 123 (test fixture only), horizon six decisions.
- Two equally weighted deterministic arrival tapes: identical zero arrivals at
  epoch zero, then a persistent left or right inflow beginning at epoch one.
  These are test vectors, not a proposed operational stochastic demand model.
- Four overtime requests: none, left, balanced and right. Other action slices
  use the existing no-op action. Original automatic production remains in place.
- Existing patient simulator, three-epoch production lead, two-epoch overtime
  commitment lead, persistence 0.5 and a shared budget of half the total headroom.
  The small environment settings follow the existing intertemporal mechanics
  tests. No clinical calibration or deployment validity is claimed.
- All four actions are enumerated at every decision. The maximum is
  `2 * (4 + 4^2 + ... + 4^6) = 10,920` distinct model transitions.
- Synthetic arrivals replace the simulator's sampled demand between steps.
  Other simulator dynamics remain unchanged. This tape adapter is fixture-only,
  not an alternative benchmark environment or a new scientific scenario.
- The planning diagnostic knows the finite distribution of possible tapes and
  simulator transitions. It is not learning that model online. The clairvoyant
  calculation additionally knows which tape will occur before deciding.
- Exactness is restricted to this grid, finite known-world distribution and
  horizon. No result is a continuous-action or real-system global optimum.

## Required Checks and Stopping Rule

- Identical current public histories with different unannounced futures give
  identical predictions. Both legacy forecast-source modes are tested as
  contrasts: they currently use future multipliers in scheduled-wave mode.
- Later arrivals can change new predictions, but never an already issued one.
- Repeated/skipped timestamps, hidden checkpoint fields, nonfinite values,
  changed estimator modes and invalid observation histories are rejected.
- On a hand-solvable two-step table: clairvoyant cost 0, nonanticipative cost 1,
  shared open-loop cost 2. Without an observable distinguishing signal,
  nonanticipative and open-loop values coincide. This is a mathematics fixture.
- The patient fixture has one root information set. Full public histories may
  branch only after differing information is available. Replaying the selected
  policy must reproduce the tree's value. Branching must not mutate its parent.
- Verify `clairvoyant <= nonanticipative <= best shared open-loop`, finite
  transition costs, exact declared transition count, and source/output hashes.
- Preserve a failed output root and investigate a mechanics error; never
  reinterpret fixture failures as scientific evidence. An existing output root
  cannot be reused. Even a pass authorizes no downstream research campaign.

## Execution

Commit this config, implementation and record before the six-step fixture.
The unit smoke test uses a temporary two-step variant with the same fixture
seed; it does not consume any research seed family.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache \
  '/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python' \
  -m evaluation.check_online_adaptation_mechanics \
  --output reports/2026-09-09-online-adaptation-mechanics/fixture
```

## Next Scientific Specification, Not Yet Executable

After P0, specify one independently motivated unknown operational process and
observable signal, an equal-information strong adaptive heuristic/planner,
and a credible control channel. Retain the 2x2 forecast/belief versus policy
update attribution design. Freeze calibration, cost normalization, information
access, query budgets, held-out process families and stopping rules before any
new performance screen or training. Use independent confirmation only after
method selection; no guaranteed improvement or publication is promised.
