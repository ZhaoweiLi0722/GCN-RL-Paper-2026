# Patient-Condition + Geography Residual Policy Probe

Date: 2026-07-17

## Scope

This probe re-evaluates the combined `patient_condition_geo` scenario after
adding real 20-clinic geography and continuous transfer-time information. The
goal is to test whether graph residual policies can improve on strong heuristic
anchors without sacrificing patient-facing service.

## 2026-07-17 Correction: True Transfer-Time Patient Scenario

During the post-geo-transfer-time audit, `patient_condition_geo` was found to
still use `transfer_lead_time: 0`, because `PatientConditionCapacityEnv` rejected
delayed transfers. That meant earlier `patient_condition_geo` results included
patient condition and geographic coordinates/cost features, but not a true
transfer-arrival pipeline.

The environment and config have now been corrected:

- `PatientConditionCapacityEnv` supports delayed reagent and idle-capacity
  transfers while continuing to block specimen pooling, since autologous patient
  specimens remain identity-bound.
- `20_clinic_patient_condition_geo.json` now uses `transfer_lead_time: 3`,
  `include_transfer_pipeline_state: true`, and distance thresholds
  `[500.0, 1500.0]`.
- Patient-condition transfer costs now use the geography/time-aware transfer
  cost helper, and evaluation metrics receive transfer arrays/costs from the
  patient environment.
- `patient_condition_geo` observation size increased from 300 to 360, and graph
  node features increased to 18 columns because pending transfer arrivals are
  now part of the state.

Corrected mini/targeted evidence should therefore supersede earlier
`patient_condition_geo` numbers that were run with immediate transfers.

## Transfer-Aware pMYO Residual TD3 Probe

A new `gcn_residual_pmyo_transfer_td3` arm was added because the replenishment-
only residuals were safe but could not express the main graph decision under
geographic lead times: small adjustments to pMYO's reagent and capacity transfer
requests. The new arm uses:

- pMYO anchor and zero-initialized GCN-TD3 actor,
- tiny centered residuals for `reagent_transfer` and `capacity_transfer`,
- positive-only replenishment residual,
- service/patient-risk-aware local-search distillation, and
- anchor fallback with deployment-scale candidates.

The first targeted run exposed an important anchor-reconstruction bug: residual
scale `0.0` was not exactly pMYO because `facility_net_action_from_state`
under-counted at-risk patients from the patient-summary histogram. That parser
now matches live pMYO on the patient+geo+pipeline state, and regression tests
cover both the heuristic helper and zero-output TD3 residual action.

After the fix, a corrected `mini_pilot` on `patient_condition_geo` showed:

| Seed | Local-search samples | Improved steps | Mean step improvement | Best nonzero cost gap | Best nonzero service gap | Deployed policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 27 | 9 | 218,362 | +0.162% | -0.0082 | anchor |
| 1 | 25 | 9 | 470,887 | +0.176% | -0.0121 | anchor |

Interpretation:

- Transfer-aware local search finds many short-horizon improvements around
  pMYO, unlike replenishment-only variants.
- Those improvements still do not generalize out-of-sample under the current
  10-episode mini budget; nonzero residual scales increase cost and reduce
  service on validation.
- The proposed method is currently safe/competitive through fallback, but it is
  not yet a genuine pMYO outperformer on the corrected patient+geo+lead-time
  setting.

The corrected `targeted_100` rerun now confirms the same conservative result at
a more meaningful budget:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget targeted_100 \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_transfer_td3 pmyo \
  --force
```

Aggregate result:

| Algorithm | Total cost mean | Service level mean | Average waiting time | Transshipment cost | Transshipment count |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN-residual-pMYO-transfer-TD3 | 1,014.48M | 0.4980 | 3.017 | 322,060 | 195.45 |
| pMYO | 1,014.48M | 0.4980 | 3.017 | 322,060 | 195.45 |

Seed-level formal evaluation:

| Seed | Algorithm | Total cost mean | Service level mean | Average wait | Transshipment cost | Transshipment count |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | GCN-residual-pMYO-transfer-TD3 | 1,001.53M | 0.5032 | 3.011 | 318,822 | 197.32 |
| 0 | pMYO | 1,001.53M | 0.5032 | 3.011 | 318,822 | 197.32 |
| 1 | GCN-residual-pMYO-transfer-TD3 | 1,027.44M | 0.4928 | 3.022 | 325,298 | 193.58 |
| 1 | pMYO | 1,027.44M | 0.4928 | 3.022 | 325,298 | 193.58 |

Corrected anchor-fallback diagnostics:

| Seed | Selected checkpoint | Selected residual scale | Best nonzero scale | Scale-0 cost gap | Best nonzero cost gap | Best nonzero service gap | Deployed policy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | episode 50 | 0.0 | 0.75 | +0.041% | +0.586% | -0.0099 | anchor |
| 1 | episode 50 | 0.0 | 0.75 | +0.124% | +0.566% | -0.0119 | anchor |

Local-search diagnostics:

| Seed | Samples | Improved steps | Anchor keep steps | Service-rejected steps | Mean step improvement | Local-search loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 398 | 223 | 175 | 5 | 129,313 | 0.0558 |
| 1 | 389 | 212 | 177 | 4 | 169,695 | 0.0823 |

Interpretation:

- The anchor reconstruction bug is largely fixed: residual scale `0.0` is now
  within about 0.04% to 0.12% of the live pMYO anchor on the fallback validation
  split, instead of the earlier near-1% mismatch.
- Transfer-aware local search finds many patient-risk-aware short-horizon
  improvements near pMYO, so the action space is not barren.
- The learned nonzero transfer residual still fails out-of-sample validation:
  the best nonzero residual scale increases cost and reduces service in both
  seeds.
- The deployed proposed method is therefore safe and competitive by fallback,
  but it is not yet a genuine pMYO outperformer in the corrected
  patient-condition plus geography plus lead-time scenario.

This also clarifies an apparent contradiction with earlier
`gcn_residual_pmyo` and `flat_residual_pmyo` summaries: those older residual
arms were replenishment-only and showed zero transshipment activity, so they are
not the right apples-to-apples comparison for the true transfer-time geography
scenario. The current apples-to-apples transfer-aware result is the
`gcn_residual_pmyo_transfer_td3` result above.

Next reliable algorithmic step: improve how the actor absorbs the local-search
signal, rather than running the full benchmark matrix. The strongest candidates
are more supervised local-search distillation epochs, a larger behavior-cloning
phase on local-search corrections, lower TD3 exploration noise, and candidate
features that explicitly encode pending transfer arrivals and transfer lead
time. A full matrix should wait until at least one transfer-aware seed deploys
a nonzero residual that clears the service gate.

## Transfer-Aware BC/Service-Margin Probe

A follow-up arm, `gcn_residual_pmyo_transfer_td3_bc`, was added to test whether
the transfer residual failure is mainly a supervised-distillation/generalization
problem. This arm keeps the same pMYO anchor and GCN-TD3 backend, but makes the
residual controller more conservative:

- transfer residual scales reduced from `0.01` to `0.005`,
- replenishment residual scale reduced from `0.02` to `0.015`,
- TD3 exploration and target-policy noise reduced,
- pMYO imitation pretraining increased to 30 episodes and 12 epochs,
- local-search distillation increased to 32 to 40 epochs depending on budget,
  and
- local-search candidates now account for pending transfer arrivals when
  computing resource and capacity pressure.

The candidate generator was also made transfer-pipeline-aware. If a clinic
already has reagent or capacity in transit, the local-search pressure pattern
now subtracts those pending arrivals before proposing additional transfer
corrections. This matters in the corrected geography scenario because transfer
lead time means a clinic can look under-supplied today even when a previous
decision has already scheduled a shipment.

Smoke validation passed end-to-end:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget smoke \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_transfer_td3_bc pmyo \
  --force
```

The first `mini_pilot` without an explicit positive service margin found a
useful failure mode:

| Seed | Best nonzero scale | Best nonzero cost gap | Best nonzero service gap | Deployed policy |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.25 | -0.201% | -0.0054 | anchor |
| 1 | 0.25 | +0.121% | -0.0043 | anchor |

Seed 0 found a residual that reduced validation cost slightly, but it still
reduced service. The fallback therefore correctly rejected it. This is exactly
the patient-facing tradeoff the paper should avoid claiming as an RL win.

The BC arm was then tightened with a positive local-search service margin
(`min_service_level_delta: 0.002`) and longer lookahead. The stricter gate
filtered many short-horizon candidates:

| Seed | Lookahead | Samples | Improved steps | Service-rejected steps | Mean step improvement | Local-search loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 22 | 2 | 18 | 846,770 | 0.0129 |
| 1 | 3 | 24 | 5 | 12 | 548,964 | 0.0319 |

Formal mini-pilot aggregate again equals pMYO because both seeds fall back to
the anchor:

| Algorithm | Total cost mean | Service level mean | Average waiting time | Transshipment cost | Transshipment count |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN-residual-pMYO-transfer-TD3-BC | 158.44M | 0.5055 | 2.463 | 136,201 | 52.10 |
| pMYO | 158.44M | 0.5055 | 2.463 | 136,201 | 52.10 |

Service-margin fallback diagnostics:

| Seed | Best nonzero scale | Best nonzero cost gap | Best nonzero service gap | Deployed policy |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.50 | +0.002% | -0.0067 | anchor |
| 1 | 0.25 | +0.160% | -0.0058 | anchor |

Interpretation:

- The stricter BC/local-search setup prevents the previous cost-saving but
  service-reducing seed-0 residual from being deployed.
- The remaining nonzero residuals are now very close to pMYO on cost, but they
  still reduce validation service by about 0.006.
- The next improvement should target service preservation directly, not just
  cost: either use service-positive demonstrations only, increase validation
  replications before checkpoint/scale selection, or learn a residual only for
  patient-risk-triggered replenishment while keeping transfer corrections as
  heuristic/post-decision local search rather than actor outputs.

The same BC/service-margin arm was then run at the full `targeted_100` budget:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget targeted_100 \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_transfer_td3_bc pmyo \
  --force
```

Aggregate targeted result:

| Algorithm | Total cost mean | Service level mean | Average waiting time | Transshipment cost | Transshipment count |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN-residual-pMYO-transfer-TD3-BC | 1,014.48M | 0.4980 | 3.017 | 322,060 | 195.45 |
| pMYO | 1,014.48M | 0.4980 | 3.017 | 322,060 | 195.45 |

Targeted local-search diagnostics:

| Seed | Lookahead | Samples | Improved steps | Anchor keep steps | Service-rejected steps | Mean step improvement | Local-search loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6 | 209 | 7 | 202 | 228 | 2,023,801 | 0.0160 |
| 1 | 6 | 211 | 10 | 201 | 224 | 2,141,861 | 0.0304 |

Targeted fallback diagnostics:

| Seed | Checkpoint selected | Best nonzero scale | Best nonzero cost gap | Best nonzero service gap | Deployed policy |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | episode 100 | 0.25 | +0.546% | -0.0132 | anchor |
| 1 | local search | 1.00 | +0.329% | -0.0003 | anchor |

Interpretation:

- At `targeted_100`, the deployed BC/service-margin policy remains exactly
  pMYO through anchor fallback.
- The stricter service-margin local search produces much fewer improved
  demonstrations than the earlier transfer-aware TD3 arm, but it filters many
  service-risky candidates and keeps the actor near the anchor.
- Seed 1 is now close on the service gate (`-0.0003`) but still worse on cost
  (`+0.329%`). Seed 0 remains worse on both cost and service for every nonzero
  residual scale.
- The current evidence supports a conservative manuscript claim: graph residual
  control can be made safe and competitive with pMYO under patient+geography
  lead times, but the learned transfer residual has not yet produced a robust
  out-of-sample improvement over the tuned pMYO heuristic.

The next targeted algorithmic experiment should separate transfer corrections
from the actor output. A promising design is a graph residual policy that learns
only patient-risk-triggered replenishment corrections while using pMYO's
transfer decisions unchanged, plus an optional online/rollout shield that can
accept learned transfer perturbations only when a short patient-facing
lookahead shows both cost and service non-inferiority. That would test whether
the graph model's value is in patient-risk inventory timing rather than direct
transfer rewiring.

## Patient-Risk Replenishment-Only Probe

To test whether transfer rewiring is the main source of residual-policy
instability, two transfer-free pMYO residual TD3 arms were added:

- `gcn_residual_pmyo_risk_replenish_td3`: only positive replenishment residuals
  at clinics with at-risk or near-expiry patients.
- `gcn_residual_pmyo_risk_pressure_td3`: only positive replenishment residuals,
  but candidate demonstrations combine patient-risk pressure with
  pending-transfer-aware resource pressure.

Both arms keep pMYO transfer decisions unchanged:

```json
"reagent_transfer": 0.0,
"capacity_transfer": 0.0,
"replenishment": 0.015
```

Smoke validation passed for both arms. The `mini_pilot` result was diagnostic
rather than positive.

Risk-only local search was too narrow:

| Algorithm | Seed | Samples | Improved steps | Anchor keep steps | Mean step improvement | Local-search loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| risk-replenish | 0 | 20 | 0 | 20 | 0 | 0.0000 |
| risk-replenish | 1 | 18 | 0 | 18 | 0 | 0.0000 |

The hybrid patient-risk/resource-pressure candidate did find a small local
signal:

| Algorithm | Seed | Samples | Improved steps | Anchor keep steps | Mean step improvement | Local-search loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| risk-pressure | 0 | 23 | 4 | 19 | 27,600 | 0.0041 |
| risk-pressure | 1 | 24 | 3 | 21 | 3,529 | 0.0001 |

However, both transfer-free arms still failed the validation fallback gate. The
formal mini-pilot aggregate equals pMYO because both seeds deploy the anchor:

| Algorithm | Total cost mean | Service level mean | Average waiting time | Transshipment cost | Transshipment count |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN-risk-replenish-TD3 | 158.44M | 0.5055 | 2.463 | 136,201 | 52.10 |
| GCN-risk-pressure-TD3 | 158.44M | 0.5055 | 2.463 | 136,201 | 52.10 |
| pMYO | 158.44M | 0.5055 | 2.463 | 136,201 | 52.10 |

Mini fallback diagnostics:

| Algorithm | Seed | Best nonzero scale | Best nonzero cost gap | Best nonzero service gap | Deployed policy |
| --- | ---: | ---: | ---: | ---: | --- |
| risk-replenish | 0 | 0.25 | +0.409% | -0.0031 | anchor |
| risk-replenish | 1 | 0.25 | +0.428% | -0.0046 | anchor |
| risk-pressure | 0 | 0.25 | +0.409% | -0.0031 | anchor |
| risk-pressure | 1 | 0.25 | +0.428% | -0.0046 | anchor |

Interpretation:

- Removing transfer residuals makes the action space safer but also removes
  most of the local improvement opportunity around pMYO.
- Pure patient-risk replenishment is too sparse; the hybrid risk/resource
  candidate creates some demonstrations, but not enough to generalize into a
  deployable residual.
- The next algorithmic step should not be another wider residual actor. The
  evidence now points to a shielded/post-decision design: use pMYO as the
  deployable policy, then run a short patient-facing rollout search over a small
  set of candidate corrections and accept a correction only when it is
  validation-safe. The graph network can then be used to learn when that shield
  is likely to approve a correction, instead of directly replacing pMYO's
  action.

## Current Evidence

Command:

```bash
.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase evaluate \
  --budget targeted_100 \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_mdl2 mdl2 pmyo iso myo \
  --force

.venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase aggregate \
  --budget targeted_100 \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_mdl2 mdl2 pmyo iso myo
```

Aggregate output:

| Algorithm | Total cost mean | Service level mean | Average waiting time |
| --- | ---: | ---: | ---: |
| ISO | 910.54M | 0.4810 | 2.737 |
| pMYO | 957.96M | 0.5337 | 2.923 |
| GCN-residual-MDL2 | 963.94M | 0.4710 | 2.949 |
| MDL-2 | 963.94M | 0.4710 | 2.949 |
| MYO | 971.70M | 0.5298 | 2.930 |

Interpretation:

- ISO has the lowest cost in this current reward/cost accounting, but it has
  weaker service than pMYO and MYO.
- pMYO remains the strongest patient-facing heuristic among the compared
  rules, with the highest service level.
- GCN-residual-MDL2 does not yet outperform the strongest heuristic. With the
  service-aware fallback, it conservatively equals its MDL-2 anchor.

## Fallback Diagnostics

The service-aware fallback now selects the learned residual only if it clears
both the cost gate and the service-level non-inferiority gate on a disjoint
validation split.

| Seed | Selected policy | Learned-anchor cost gap | Learned-anchor service gap |
| --- | --- | ---: | ---: |
| 0 | anchor | +0.185% | -0.0016 |
| 1 | anchor | -0.291% | -0.0216 |

Seed 1 is the important failure mode: the learned residual reduced validation
cost slightly, but it also reduced service level substantially. The new
fallback correctly rejects that policy. This is preferable for the manuscript:
the learned policy should not claim an operational win by trading away patient
service.

## pMYO Trust-Region Probe

A mini-pilot tested a narrower pMYO residual trust region plus service-aware
local-search filtering and ranking:

- pMYO residual group scales: 0.02 for reagent transfer, capacity transfer, and
  replenishment.
- L2 residual penalty: 0.05.
- Targeted local search epsilons: 0.005, 0.01, 0.02.
- Anchor keep probability/weight: 0.7 / 200000.
- Local-search candidates must be service-noninferior to the anchor over the
  lookahead rollout.
- Candidate ranking uses a service-level weight of 100000000.0, so local
  search ranks actions by a patient-facing scalarized score rather than cost
  alone.

Mini-pilot validation still selected the pMYO anchor:

| Algorithm/seed | Selected policy | Learned-anchor cost gap | Learned-anchor service gap |
| --- | --- | ---: | ---: |
| GCN-residual-pMYO seed 0 | anchor | +1.989% | -0.0106 |
| GCN-residual-pMYO seed 1 | anchor | +2.995% | -0.0108 |
| Flat-residual-pMYO seed 0 | anchor | +1.432% | -0.0072 |
| Flat-residual-pMYO seed 1 | anchor | +2.219% | -0.0131 |

The service-aware ranking changed the local-search objective and improved some
distillation losses, but the resulting learned policies still did not
generalize well enough to replace pMYO. This suggests pMYO is already a strong
anchor and that the next bottleneck is actor/distillation generalization, not
only local candidate selection.

An additional patient-risk-aware mini-pilot extended the local-search score
with eligibility, at-risk-unserved, and patients-lost terms:

- service-level weight: 100000000.0
- eligibility-rate weight: 100000000.0
- at-risk-unserved weight: 50000.0
- patients-lost weight: 500000.0

The patient-risk score did not make GCN-residual-pMYO pass fallback:

| Algorithm/seed | Selected policy | Learned-anchor cost gap | Learned-anchor service gap |
| --- | --- | ---: | ---: |
| GCN-residual-pMYO seed 0 | anchor | +1.989% | -0.0106 |
| GCN-residual-pMYO seed 1 | anchor | +2.995% | -0.0108 |
| Flat-residual-pMYO seed 0 | anchor | +0.930% | -0.0053 |
| Flat-residual-pMYO seed 1 | anchor | +2.219% | -0.0131 |

This suggests that scalarized patient-risk candidate scoring helps some flat
residual behavior but still does not solve the GCN residual generalization
problem around the strongest pMYO anchor.

## Residual Trust-Region Diagnostics

The pMYO residual action representation was tightened further so the learned
residual can no longer reduce replenishment relative to the pMYO anchor:

- `positive_only_groups: ["replenishment"]` for both flat and GCN pMYO residual
  policies.
- Functional, non-in-place tensor residual transforms, so PyTorch autograd can
  train the constrained residual actor safely.
- Validation-time residual deployment scale selection over fixed candidates:
  `0.0, 0.25, 0.5, 0.75, 1.0`.

The deployment-scale selector is a trust-region fallback: it tests partial
residual deployment on the validation split before formal evaluation. A nonzero
residual scale is deployed only if it clears the same cost and service gates as
the learned policy. If the best validation behavior is equivalent to a
zero-scale residual, the formal evaluation falls back to the pMYO anchor.

With reagent-transfer and capacity-transfer residuals still enabled, the
per-scale diagnostics showed that the nonzero corrections were directionally
bad:

| Budget | Seed | Best nonzero scale | Best nonzero cost gap | Best nonzero service gap |
| --- | ---: | ---: | ---: | ---: |
| `targeted_100` | 0 | 0.25 | +2.747% | -0.0381 |
| `targeted_100` | 1 | 0.25 | +2.471% | -0.0280 |

This was the key failure diagnosis: shrinking the deployment scale was not
enough, because even the smallest nonzero residual transfer correction increased
cost and reduced service. The pMYO residual action space was therefore narrowed
again to positive replenishment-only corrections:

- `reagent_transfer`: 0.0
- `capacity_transfer`: 0.0
- `replenishment`: 0.02
- `positive_only_groups: ["replenishment"]`

Smoke validation passed after this action-space narrowing. A `mini_pilot` on
`patient_condition_geo` showed a partial recovery:

| Seed | Deployed policy | Deployed residual scale | Best nonzero cost gap | Best nonzero service gap |
| --- | --- | ---: | ---: | ---: |
| 0 | learned | 0.25 | -0.592% | +0.0015 |
| 1 | anchor | 0.0 | +0.979% | -0.0062 |

The aggregate mini-pilot still did not beat pMYO because only one of the two
seeds deployed a learned residual and the formal evaluation mean was slightly
worse than pMYO:

| Algorithm | Total cost mean | Service level mean | Average waiting time |
| --- | ---: | ---: | ---: |
| GCN-residual-pMYO | 135.52M | 0.5701 | 2.279 |
| pMYO | 135.32M | 0.5716 | 2.275 |

This is still useful progress: the failure moved from "all nonzero scales hurt
both cost and service badly" to "one seed can pass the validation gate, but the
effect is not stable across seeds."

An additional state-gated replenishment-only mini-pilot tested whether learned
replenishment corrections should be applied only at clinics with positive
resource pressure:

- gate signal: demand + forecast/specimen/risk pressure minus reagent inventory,
- gated group: `replenishment`,
- threshold: 0.0.

This gate did not materially improve the mini-pilot:

| Seed | Deployed policy | Deployed residual scale | Best nonzero cost gap | Best nonzero service gap |
| --- | --- | ---: | ---: | ---: |
| 0 | learned | 0.25 | -0.593% | +0.0015 |
| 1 | anchor | 0.0 | +0.976% | -0.0062 |

The state gate slightly improved the failed seed's cost gap relative to the
ungated mini-pilot, but the change was too small to justify a full targeted run.
It also increased the local-search distillation loss. The main remaining issue
therefore appears to be seed-to-seed generalization of the residual actor, not
only action-space filtering.

## Targeted-100 GCN-pMYO Result

After the replenishment-only mini-pilot, `gcn_residual_pmyo` was re-run for the
full `targeted_100` budget on `patient_condition_geo` using:

- service-aware local-search distillation,
- positive replenishment-only residuals,
- service-aware checkpoint selection, and
- service-aware anchor fallback with validation-selected deployment scale.

Training/post-training diagnostics:

| Seed | Local-search samples | Improved steps | Anchor keep steps | Service-rejected steps | Local-search loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 406 | 237 | 169 | 3 | 0.0553 |
| 1 | 403 | 223 | 180 | 3 | 0.0425 |

Formal validation still selected the pMYO anchor for both seeds:

| Seed | Checkpoint selected | Best nonzero scale | Fallback selected | Best nonzero cost gap | Best nonzero service gap |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | local search | 1.0 | anchor | +0.720% | -0.0049 |
| 1 | local search | 1.0 | anchor | +0.938% | -0.0029 |

Aggregate result:

| Algorithm | Total cost mean | Service level mean | Average waiting time |
| --- | ---: | ---: | ---: |
| GCN-residual-pMYO | 957.96M | 0.5337 | 2.923 |
| pMYO | 957.96M | 0.5337 | 2.923 |

Because the learned policy failed the safety gate, the deployed
GCN-residual-pMYO policy is exactly the pMYO anchor. The important conclusion is
not that GCN-pMYO wins, but that the updated safeguards now make the proposed
method at least safe relative to pMYO while exposing a much smaller remaining
learned-vs-anchor validation gap than the earlier transfer-enabled pMYO
residual. Nonzero residual scales still did not clear the targeted validation
gates, so the current graph residual architecture is not yet a genuine
heuristic outperformer.

## Method Update

The benchmark runner now records learned-vs-anchor validation service and
supports a service-level safeguard in `anchor_fallback`. In the current
benchmark config, learned residual policies must be cost-noninferior and
service-noninferior before replacing their heuristic anchor.

Local-search distillation now inherits the same service non-inferiority gate by
default, so candidate residual demonstrations that reduce short-horizon service
are filtered out before actor distillation. For pMYO residual policies, the
action representation also prevents negative replenishment corrections, so the
learned residual cannot directly undo the anchor's patient-protective
replenishment choice.

Anchor fallback now includes residual deployment-scale selection. On the
validation split, the runner evaluates fixed residual scales such as
`0.0, 0.25, 0.5, 0.75, 1.0`; a nonzero residual scale is deployed only if it
beats the anchor while maintaining patient-facing service. A selected scale of
`0.0` is treated as anchor fallback, not as a learned-policy win.

This changes the proposed method from:

> residual RL with cost-only anchor fallback

to:

> graph residual RL with patient-facing residual distillation, positive-only
> pMYO replenishment-only corrections, and validation-selected trust-region
> anchor fallback

## TD3 Residual Backend Probe

Because the positive replenishment-only GCN-DDPG residual was still seed
unstable, a TD3-backed residual variant was added:

- algorithm: `gcn_residual_pmyo_td3`
- backbone: GCN-TD3 with clipped double-Q, target-policy smoothing, and delayed
  actor updates
- anchor: pMYO
- residual action space: positive replenishment-only correction
- safety controls: state-gated residuals, residual L2 penalty, validation-scale
  selection, and service-aware pMYO fallback
- benchmark role: stability probe, not a replacement for the pure `gcn_td3`
  baseline

Validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m unittest \
  tests.test_gcn_td3 tests.test_rl_utils tests.test_full_benchmark_runner

PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget smoke \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_td3 pmyo \
  --force
```

The smoke benchmark passed end-to-end: training, pMYO imitation pretraining,
local-search distillation, checkpoint writing, anchor fallback, evaluation,
aggregation, and plotting.

A `mini_pilot` run then compared `gcn_residual_pmyo_td3` against pMYO on
`patient_condition_geo`:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget mini_pilot \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_td3 pmyo \
  --force
```

Aggregate mini-pilot result:

| Algorithm | Total cost mean | Service level mean | Average waiting time |
| --- | ---: | ---: | ---: |
| GCN-residual-pMYO-TD3 | 135.52M | 0.5701 | 2.279 |
| pMYO | 135.32M | 0.5716 | 2.275 |

Per-seed fallback diagnostics:

| Seed | Deployed policy | Deployed residual scale | Best nonzero cost gap | Best nonzero service gap |
| --- | --- | ---: | ---: | ---: |
| 0 | learned | 0.25 | -0.589% | +0.0015 |
| 1 | anchor | 0.0 | +0.977% | -0.0062 |

Local-search diagnostics:

| Seed | Samples | Improved steps | Anchor keep steps | Service-rejected steps | Local-search loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | 1 | 16 | 0 | 0.0000 |
| 1 | 19 | 0 | 19 | 0 | 0.1094 |

Interpretation:

- TD3 residual reproduces the best positive signal from the DDPG residual
  mini-pilot: one seed passes the validation gate with a nonzero scale and
  improves both cost and service relative to pMYO.
- It does not yet solve the seed-instability problem. Seed 1 again fails both
  cost and service gates, so the aggregate deployed policy remains slightly
  worse than pMYO and must not be described as outperforming the heuristic.
- The aligned local-search signal is much sparser than the earlier unconstrained
  candidate search, which confirms that pMYO residual learning should only use
  candidates the actor can actually deploy.

The `targeted_100` run answered that question more conservatively:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget targeted_100 \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_td3 pmyo \
  --force
```

Aggregate targeted result:

| Algorithm | Total cost mean | Service level mean | Average waiting time |
| --- | ---: | ---: | ---: |
| GCN-residual-pMYO-TD3 | 957.96M | 0.5337 | 2.923 |
| pMYO | 957.96M | 0.5337 | 2.923 |

Both seeds fell back to pMYO, so the deployed TD3 residual policy is exactly the
anchor:

| Seed | Checkpoint selected | Best nonzero scale | Fallback selected | Best nonzero cost gap | Best nonzero service gap |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | episode 100 | 1.0 | anchor | +0.620% | -0.0049 |
| 1 | local search | 1.0 | anchor | +0.820% | -0.0027 |

Aligned local-search diagnostics:

| Seed | Samples | Improved steps | Anchor keep steps | Service-rejected steps | Local-search loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 291 | 101 | 190 | 0 | 0.1394 |
| 1 | 302 | 119 | 183 | 0 | 0.1551 |

Compared with targeted GCN-residual-pMYO-DDPG, TD3 slightly reduced the
validation gap for the best nonzero residual:

| Backbone | Seed 0 cost/service gap | Seed 1 cost/service gap |
| --- | ---: | ---: |
| DDPG residual | +0.720% / -0.0049 | +0.938% / -0.0029 |
| TD3 residual, aligned candidates | +0.620% / -0.0049 | +0.820% / -0.0027 |

This is a useful but insufficient improvement. The TD3 backbone narrows the
gap, but not enough to clear the safety gate. The bottleneck is therefore not
only DDPG training instability; the learned residual target still fails to
generalize as a service-noninferior correction around pMYO. Candidate alignment
also makes the evidence cleaner: the remaining gap is no longer explained by
unrepresentable transfer-action demonstrations.

## Candidate-Space Alignment Fix

The targeted TD3 residual probe exposed a mismatch between the local-search
candidate generator and the pMYO residual action space. The pMYO residual actor
can only express positive replenishment corrections, but the local-search
generator was still proposing:

- negative replenishment corrections,
- reagent-transfer corrections,
- capacity-transfer corrections, and
- combined transfer corrections.

Those candidates can look good in short-horizon local search but are not
representable by the actor after residual group masks are applied. This can
pollute supervised residual fitting with high-weight demonstrations that become
zero or partially contradictory targets.

The local-search generator now supports explicit candidate-space controls:

- `candidate_groups`
- `candidate_signs`

For pMYO residual policies, the candidate space is now aligned with the actor:

```json
"candidate_groups": ["replenishment_uniform", "replenishment_positive_pressure"],
"candidate_signs": [1.0]
```

The new `replenishment_positive_pressure` candidate adds replenishment only at
clinics with positive resource-pressure signal; it does not reduce
replenishment at other clinics. This matches the positive-only residual
constraint better than the older centered pressure pattern.

Validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m unittest \
  tests.test_full_benchmark_runner tests.test_gcn_td3 tests.test_rl_utils

PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --phase all \
  --budget smoke \
  --scenarios patient_condition_geo \
  --algorithms gcn_residual_pmyo_td3 pmyo \
  --force
```

Post-fix mini-pilot result for `gcn_residual_pmyo_td3`:

| Algorithm | Total cost mean | Service level mean | Average waiting time |
| --- | ---: | ---: | ---: |
| GCN-residual-pMYO-TD3 | 135.52M | 0.5701 | 2.279 |
| pMYO | 135.32M | 0.5716 | 2.275 |

Per-seed diagnostics:

| Seed | Local-search improved steps | Deployed policy | Best nonzero cost gap | Best nonzero service gap |
| --- | ---: | --- | ---: | ---: |
| 0 | 1 / 17 samples | learned scale 0.25 | -0.589% | +0.0015 |
| 1 | 0 / 19 samples | anchor | +0.977% | -0.0062 |

Interpretation:

- Candidate-space alignment confirms that many earlier local-search
  improvements were not actually learnable under the positive-only pMYO
  residual parameterization.
- The remaining learnable correction signal is sparse in the mini-pilot but
  present in the aligned targeted run. Even so, the targeted run still falls
  back to pMYO for both seeds.
- This makes the next algorithmic target sharper: improve the state-conditioned
  positive replenishment residual itself, instead of expanding candidate search
  into transfer actions that this policy cannot deploy.

## Post-Decision pMYO Shield Probe

A post-decision online shield was added as a diagnostic upper-bound direction
for patient-aware residual control. The policy starts from pMYO, evaluates a
small set of resource/capacity transfer and replenishment corrections on copied
short-horizon environments, and deploys a correction only when it clears the
patient-facing score and service-safety gate.

Code path:

- heuristic policy: `pmyo_shield`
- learned distillation arm: `gcn_residual_pmyo_shield_td3`
- scenario: `patient_condition_geo`
- budget: `mini_pilot`
- seeds: 0, 1
- MC evaluation replications: 5 per seed

Mini-pilot aggregate:

| Algorithm | Total cost mean | Service level | Avg. wait | Eligibility mean | Patients lost | At-risk unserved | Transfer cost | Transfer count | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pMYO | 158.44M | 0.5055 | 2.463 | 0.7489 | 337.2 | 520.3 | 136.20K | 52.1 | 0.22 ms |
| pMYO shield | 156.57M | 0.5210 | 2.422 | 0.7581 | 323.4 | 498.5 | 140.22K | 121.0 | 157 ms |
| GCN residual pMYO shield TD3 | 158.44M | 0.5055 | 2.463 | 0.7489 | 337.2 | 520.3 | 136.20K | 52.1 | 0.22 ms |

Interpretation:

- `pmyo_shield` is the first post-geography-transfer-time policy in this
  branch to beat pMYO on both operating cost and patient-facing metrics in the
  mini-pilot.
- The gain is small but coherent: about 1.18% lower cost, higher service,
  shorter average waiting time, higher eligibility retention, fewer lost
  patients, and fewer at-risk unserved patients.
- The tradeoff is clear: the shield uses many more transfers and is about three
  orders of magnitude slower at inference because it deep-copies the simulator
  and evaluates candidate corrections online.
- This makes the shield a useful teacher/oracle, but not yet the final proposed
  deployable method.

Three GCN-TD3 distillation variants were then tested:

1. `lookahead=3`, epsilons `[0.005, 0.01]`, 240 teacher samples.
2. `lookahead=2`, epsilon `[0.005]`, 240 teacher samples.
3. `lookahead=2`, epsilon `[0.005]`, 240 teacher samples, with transfer
   residual scales tightened from 0.020 to 0.005.

All variants used pMYO as the residual anchor and froze actor updates after
behavior-cloning pretraining. The formal deployment gate still selected pMYO
for both seeds. The best nonzero residual in the retained scale-tight
`lookahead=2` setting was:

| Seed | Best nonzero residual scale | Best nonzero cost gap | Best nonzero service gap |
| --- | ---: | ---: | ---: |
| 0 | 0.25 | +0.120% | -0.0065 |
| 1 | 0.25 | +0.336% | -0.0043 |

This says the current GCN residual can imitate the action pattern only
superficially. Small action errors in the transfer correction are enough to lose
service, so the fallback correctly keeps pMYO. The next learned-control step
should not be "more TD3 episodes"; it should be a service-aware shield
distillation design, likely a graph classifier/value model that learns when to
accept a candidate correction rather than regressing directly to tiny
continuous residual actions.

## Graph Shield Selector Around MDL2

The shield-distillation idea was converted from continuous residual regression
into a graph candidate classifier. Instead of asking TD3 to reproduce tiny
transfer residuals, the learned policy observes the graph state and selects one
candidate correction from the same discrete set used by the online shield. The
first selector used pMYO as the anchor; after the current geo-transfer-time
mini-pilot showed MDL2 as the strongest cost heuristic, the selector was
generalized to support an MDL2 anchor.

Code path:

- heuristic teacher: `mdl2_shield`
- learned student: `gcn_mdl2_shield_selector`
- anchor: `mdl2`
- scenario: `patient_condition_geo`
- budget: `mini_pilot`
- seeds: 0, 1
- MC evaluation replications: 5 per seed
- teacher-label states per training seed: 960

Current mini-pilot aggregate:

| Algorithm | Total cost mean | Service level | Eligibility mean | Patients lost | At-risk unserved | Transfer count | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDL2 shield | 149.21M | 0.5196 | 0.7469 | 336.0 | 514.8 | 99.0 | 156 ms |
| GCN MDL2 shield selector | 149.41M | 0.5198 | 0.7469 | 336.1 | 514.4 | 84.3 | 1.37 ms |
| MDL2 | 149.41M | 0.5197 | 0.7469 | 336.3 | 514.8 | 32.4 | 0.20 ms |
| ISO | 153.44M | 0.4756 | 0.7037 | 400.0 | 620.0 | 0.0 | 0.05 ms |
| pMYO shield | 156.57M | 0.5210 | 0.7581 | 323.4 | 498.5 | 121.0 | 159 ms |
| GCN pMYO shield selector | 157.59M | 0.5123 | 0.7528 | 332.3 | 513.2 | 189.4 | 1.40 ms |
| MYO | 158.29M | 0.5058 | 0.7490 | 337.4 | 519.9 | 52.0 | 0.19 ms |
| pMYO | 158.44M | 0.5055 | 0.7489 | 337.2 | 520.3 | 52.1 | 0.22 ms |
| uMYO | 187.16M | 0.5060 | 0.7486 | 338.8 | 520.6 | 50.7 | 0.22 ms |

Interpretation:

- The MDL2-anchored graph selector is now cost-competitive with the strongest
  non-shield heuristic: it is about 0.0035% lower cost than MDL2 in this
  mini-pilot, with slightly higher service and slightly lower at-risk unserved
  counts.
- The online MDL2 shield remains the lowest-cost policy in this small run, but
  it is roughly two orders of magnitude slower at decision time. The graph
  selector recovers most of its cost benefit while keeping inference around
  1-2 ms.
- The pMYO shield remains the best patient-facing policy in this mini-pilot:
  higher eligibility retention, fewer lost patients, and fewer at-risk unserved
  patients, but at a higher operating cost.
- The deployable learned-method story is therefore not yet "GCN beats every
  heuristic." The current evidence supports a more careful claim: a graph
  selector distilled from a service-aware shield can match or slightly improve
  the strongest fast heuristic, while online shield teachers still define the
  upper bound for cost or patient-facing metrics.

### Targeted-100 Validation

The key algorithms were then re-run under the larger targeted budget:

- budget: `targeted_100`
- training seeds: 0, 1
- training episodes: 100
- MC evaluation replications: 50 per seed
- scenario: `patient_condition_geo`

Aggregate targeted-100 result:

| Algorithm | Total cost mean | Cost SEM | Service level | Eligibility mean | Patients lost | At-risk unserved | Transfer count | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDL2 shield | 922.35M | 7.73M | 0.5241 | 0.6255 | 2473.6 | 3500.5 | 462.6 | 172 ms |
| MDL2 | 923.13M | 7.75M | 0.5227 | 0.6245 | 2481.7 | 3510.8 | 168.8 | 0.19 ms |
| GCN MDL2 shield selector | 923.31M | 7.75M | 0.5229 | 0.6246 | 2480.8 | 3509.6 | 179.1 | 1.30 ms |
| pMYO shield | 1012.59M | 8.15M | 0.5073 | 0.6243 | 2547.9 | 3649.4 | 466.8 | 176 ms |
| pMYO | 1014.48M | 8.20M | 0.4980 | 0.6143 | 2601.5 | 3718.7 | 195.5 | 0.23 ms |
| GCN pMYO shield selector | 1017.53M | 8.18M | 0.4967 | 0.6148 | 2607.7 | 3732.7 | 711.5 | 1.39 ms |

Targeted-100 interpretation:

- The mini-pilot result that the MDL2 selector slightly beats MDL2 does not
  fully hold under 50 evaluation replications. After confidence-gated
  calibration, the selector is 0.020% higher cost than MDL2 and 0.104% higher
  cost than the online MDL2 shield.
- The gap is small relative to evaluation uncertainty: the cost difference
  between MDL2 and the selector is about 0.18M, while the reported SEM is about
  7.7M.
- The selector remains deployable and fast: roughly 1.3 ms per decision versus
  172 ms for the online MDL2 shield. The online shield is therefore a useful
  teacher / upper bound, not a practical real-time controller.
- pMYO shield improves patient-facing metrics relative to pMYO, but both pMYO
  variants are much more expensive than the MDL2 family in this targeted run.
- Current best paper-safe statement: the GCN MDL2 shield selector is
  competitive with the strongest fast heuristic but has not yet beaten the
  online shield teacher.

Selector diagnostics:

| Algorithm | Seed | Teacher non-anchor labels | Student non-anchor predictions | Train accuracy |
| --- | ---: | ---: | ---: | ---: |
| GCN MDL2 shield selector | 0 | 18.4% | 49.0% | 52.2% |
| GCN MDL2 shield selector | 1 | 18.4% | 61.6% | 44.7% |
| GCN pMYO shield selector | 0 | 35.4% | 86.2% | 34.2% |
| GCN pMYO shield selector | 1 | 35.4% | 90.4% | 33.8% |

These diagnostics explain the original gap: the graph classifier accepts
non-anchor corrections much more often than the shield teacher. That produces
many more transfer actions and can erase the small cost/service gains available
from the online teacher.

A confidence-gated deployment rule was then added: non-anchor predictions are
accepted only if the classifier confidence exceeds 0.6; otherwise the policy
falls back to MDL2. Re-evaluating the same targeted-100 checkpoints with this
gate improved the selector from 924.47M to 923.31M and reduced transfer count
from 483.8 to 179.1. This closes most of the gap to MDL2 without retraining and
keeps inference at about 1.3 ms.

### Graph Residual TD3 Refocus

Because the manuscript target is an AI / RL journal venue, the method emphasis
was moved back from the non-RL selector toward graph-aware residual
actor-critic control. Two TD3-based variants were added and tested under the
same `patient_condition_geo` targeted-100 setting:

- `gcn_residual_mdl2_td3`: GCN-TD3 residual around the MDL2 anchor.
- `gcn_residual_mdl2_shield_td3`: GCN-TD3 residual around MDL2, warm-started
  from the online MDL2 shield and then fine-tuned with TD3.

Both use:

- MDL2 as the deterministic anchor action,
- bounded residual corrections over reagent transfer, capacity transfer, and
  replenishment,
- pressure-projected graph residuals so corrections align with resource and
  capacity pressure patterns,
- anchor-advantage actor loss with twin-min critic values,
- fine-grained validation-time residual scale selection
  (`0, 0.05, 0.1, 0.25, 0.5, 0.75, 1`),
- anchor fallback when the learned residual fails validation.

Runtime fixes were also added so longer TD3 experiments are feasible locally:

- cached graph node features for imitation regularization,
- cached node features and residual targets during supervised warm-start,
- configurable `update_frequency` / `updates_per_update` in the off-policy
  training loop.

The smoke runtime for `gcn_residual_mdl2_td3` dropped from about 60 seconds to
about 14 seconds after these changes.

Targeted-100 probe result:

| Algorithm | Total cost mean | Service level | Eligibility mean | Patients lost | At-risk unserved | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDL2 shield | 922.35M | 0.5241 | 0.6255 | 2473.6 | 3500.5 | 172 ms |
| MDL2 | 923.13M | 0.5227 | 0.6245 | 2481.7 | 3510.8 | 0.19 ms |
| GCN residual MDL2 TD3 | 923.13M | 0.5227 | 0.6245 | 2481.7 | 3510.8 | 0.21 ms |
| GCN residual MDL2-shield TD3 | 923.13M | 0.5227 | 0.6245 | 2481.7 | 3510.8 | 0.21 ms |
| GCN MDL2 shield selector | 923.31M | 0.5229 | 0.6246 | 2480.8 | 3509.6 | 1.30 ms |

Interpretation:

- The TD3 variants are now safe to deploy because validation fallback selects
  MDL2 when the residual does not improve the anchor.
- They do not yet outperform MDL2 or the online MDL2 shield. Their aggregate
  performance equals MDL2 because both targeted-100 seeds selected the anchor.
- Pure MDL2-warm-start TD3 learned residual directions that hurt validation:
  nonzero residual candidates increased cost by about 1.4-2.0% and reduced
  service by about 3.1-4.2 percentage points.
- MDL2-shield warm-start improved the residual direction substantially:
  nonzero residual candidates increased cost by about 0.8-1.0% and reduced
  service by about 0.7 percentage points. This is still not good enough to
  deploy, but it is the best evidence so far that shield demonstrations can
  make residual RL corrections less destructive.
- Current paper-safe RL statement: graph residual TD3 with trust-region
  fallback can match a strong fast heuristic safely, but the learned residual
  correction has not yet produced a statistically reliable improvement over
  MDL2 in the combined patient-condition/geography setting.

## Next Experiment

The next improvement should keep the paper's main line as graph-aware RL, while
using the shield selector as a supporting diagnostic:

1. Increase MDL2-shield warm-start data for `gcn_residual_mdl2_shield_td3`
   and test whether the 0.8-1.0% nonzero-residual validation gap closes.
2. Add an explicit service / eligibility term to the TD3 actor objective or
   critic target so residuals do not trade service away for noisy cost changes.
3. Restrict residual action groups further and run ablations:
   replenishment-only, transfer-only, and shield-candidate residuals.
4. Use validation-selected checkpoint and residual scale as the deployable
   policy definition; report the selected scale so fallback is transparent.
5. Keep the GCN MDL2 shield selector as a fast distilled-shield baseline, not
   as the main RL contribution.
6. Keep SAC/PPO as secondary benchmarks after the TD3 residual line is stable.

## Manuscript Narrative

Current manuscript-safe claim:

> In the combined patient-condition and geography scenario, direct continuous
> residual RL is difficult around strong heuristics. Adding graph pressure
> projection, anchor-advantage TD3 updates, fine-grained residual scale
> selection, and anchor fallback makes the graph residual policy safe: it
> matches the strongest fast heuristic when learned corrections fail
> validation. The MDL2-shield warm-start reduces the damage from nonzero
> residuals relative to pure MDL2 warm-start, but it has not yet delivered a
> deployable improvement over MDL2. The GCN MDL2 shield selector remains a fast
> distilled-shield baseline, while the main proposed method should be framed as
> graph-aware residual actor-critic control with validation-gated deployment.
