# Service-effort mechanics readout

Date: 2026-09-28. Classification: software mechanics passed; no scientific
headroom, clinical safety, GCN benefit or online DDPG benefit established.

## Completed work

- Added an isolated FIFO service-work primitive with continuous effort,
  indivisible completed orders, persistent partial progress, a shared effort
  budget, delayed activation and upfront effort/switching charges.
- Added a public-receipt-only response estimator that excludes zero-effort
  and backlog-limited observations. It is not an RL or control policy.
- Preserved the existing patient environment, overtime model and prior pilot
  evidence. The new primitive is not imported by those environment paths.
- Ran the prespecified three eight-step deterministic cases, producing 24
  transition records. No patient study, stochastic scientific seed, neural
  training, formal confirmation, remote push or main merge was performed.

Protocol and source execution commit:
`2634e1dd3344b25d22a7285a21d362e14b2b41a4`.
Raw artifact root:
[`reports/2026-09-28-service-effort-mechanics/fixture/`](../../reports/2026-09-28-service-effort-mechanics/fixture/).

## What the checks establish

| Check | Recorded result | Scope |
| --- | --- | --- |
| Work/job/effort accounting | Conservation checks passed at every step | Synthetic work units, not physical reactor or patient validation |
| Fractional action sensitivity | 0.250 versus 0.251 effort gives different remaining work, without falsely completing a fractional job | Unit test; not evidence of a useful clinical reward gradient |
| Commitment delay | One-step fixture and two-step unit test passed | The response at execution time applies to the matured effort |
| Information timing | Nominal and shifted worlds have equal public observations through decision epoch 3; first differ at epoch 4 | Shift occurs during service epoch 3; no foreknowledge supplied |
| Noiseless response identification | Final estimate (0.5, 1.25, 1.0) in the shifted case | A simple estimator is sufficient under these assumptions |
| Backlog censoring | Zero exact-response samples accepted in the small-backlog case | Empty queues are not interpreted as low productivity |
| Terminal accounting | Unfinished work and prepaid pending effort explicitly retained | No settled policy cost comparison is allowed |

Uncensored per-site sample counts were (7, 7, 7) for nominal, (7, 6, 7) for
persistent shift, and (0, 0, 0) for the small-backlog case. They are repeated
deterministic measurements, not independent statistical replications.

## Verification

- 63 relevant unittest tests passed, including 16 new service-effort tests
  and regressions for existing observation/overtime/disruption paths.
- Full repository `compileall` passed using the project virtual environment
  and an isolated cache directory.
- The separate row audit recomputed effort timing, charges, work/job balance,
  censoring and estimator updates without calling the environment transition.
  All 24 rows and 20 file hashes passed.
- Ten prior source/output files matched reference commit
  `5638587d843131c8fcda2e2b44442dcccd2899e4` before and after the fixture.
- The initial test command found pytest unavailable. Tests were converted to
  the repository's standard-library unittest convention; no dependency was
  installed and no scientific run was retried.

Reproduce from the integration worktree with the project interpreter:

```bash
python -m unittest tests.test_service_effort_mechanics -v
python -m evaluation.check_service_effort_mechanics --output NEW_EMPTY_OUTPUT_ROOT
python -m evaluation.check_service_effort_mechanics --output NEW_EMPTY_OUTPUT_ROOT --audit-only
```

The runner refuses a nonempty/existing output directory and requires its
source/config to match a local commit. Do not rerun over the retained fixture.

## Scientific interpretation

The result removes one software ambiguity: real-valued effort can advance
work without making completed orders fractional. It does not show that this is
an accurate model of a PRM operation, or that the existing clinical objective
has a learnable action-value signal.

More importantly, an unknown coefficient is not enough to establish a role
for DDPG. In the noiseless, directly measured fixture, an elementary estimator
identifies the response from one informative service observation. A capable
fixed controller supplied with that estimate may already adapt adequately.
The next study must retain this strong alternative. Do not add artificial
measurement noise or hide otherwise available information solely to defeat it.

The fixture contains only coupling through a shared effort budget. It has no
transport network or downstream patient-production coupling and therefore
cannot establish graph value. Joint operational coupling must be justified
before a network-level performance experiment.

## Next implementation decisions

1. Identify the eligible real operation: preparation, support service or
   staffing-dependent slot availability. Record which progress/effort/backlog
   measurements exist. Do not reinterpret overtime or biology silently.
2. Freeze settlement of incomplete work/patients and outstanding commitments,
   plus coherent equipment-arrival overflow semantics in a new environment
   version. Do not patch the historical benchmark in place.
3. Specify and implement a small repeated-decision baseline study with the
   same measurements and action budget for a frozen feedback rule, adaptive
   response-estimation rule, and identification-plus-MPC controller. Only
   after that evidence decide whether a bounded DDPG experiment is warranted.

Zhaowei has been asked which operations have defensible controllable personnel
hours. No response is recorded as of this readout; Howard's approval is not
inferred. Standalone software verification is complete, while engineering
calibration and scientific execution remain separate decisions.
