# S2 readout: control screen complete; planner adequacy not established

Date: 2026-09-29. Local, hypothetical development evidence only.
**Decision: do not launch DDPG from this screen.** The implementation and
accounting pass, but planning-budget stability and reproducible identification
benefit do not. This neither demonstrates nor disproves online DDPG value.

## What completed

- 288/288 prespecified episodes, six methods in 48 paired world/capacity/family/
  split cells. Only four world indices per family/capacity/split are independent
  sampling units; 288 is not the statistical sample size.
- 9,364 actual scalar transitions, all eight jobs and purchased requests settled
  in every episode. No failed episode was dropped or restarted.
- 39 episodes need a positive tail: 148 closure intervals in total, maximum six
  per episode. Their 182.4375 synthetic cost units remain in the comparisons.
- All 288 saved episodes replay through unchanged S1 scalar physics and receipt
  filters. One preselected fixed64 episode also replays every planner decision.
  All 60 paired contrast records are recomputed from persisted episodes.
- A separate calculation from raw holding/labor/switching rows reproduces every
  episode total and all 60 contrast means and four-world difference vectors.
- Six preselected adequacy probes, all nontrivial. Seven execution-file locks
  and 22 S1/historical locks pass before execution and on fresh-process audit.
- 123 relevant tests pass, including six-controller smoke fixtures; full Python
  compilation and diff checks pass. The recorded process exits zero in 24.9 s,
  below the fixed 900 s cap. No error file, retry or training occurred.

Controllers receive their own causal receipts, bookings and queue state.
Actual completion draws are paired across methods, capacities and families;
planning draws are separate, and discovery/replication streams are disjoint.
Fast-iid changes now use independently sampled orientations each interval,
not the predictable alternating engineering tape from S1.

The reservation rule, fixed-model planner and identified planner all have
the same action grid and information interface. Identification here updates a
fixed likelihood filter, **not neural weights**. The cost weights, scenarios,
methods and budgets were committed before the first recorded episode.

## Full prespecified mean contrasts

Values are left-method cost minus right-method cost in synthetic units;
**negative is lower total cost**. Each entry averages four paired worlds.
No confidence, significance, practical-benefit or clinical-safety claim follows.
`R` = reservation rule; `B` = booked rule; `F` = fixed-model rollout;
`I` = identified rollout; suffixes are forecast samples per candidate.

| Family | Slots | Split | R-B | F64-B | I64-F64 | F64-F16 | I64-I16 |
|---|---:|---|---:|---:|---:|---:|---:|
| Unchanged | 1 | Discovery | -6.203 | -2.961 | +2.648 | -13.461 | -5.766 |
| Unchanged | 1 | Replication | -6.758 | -4.508 | -0.945 | -2.609 | -0.484 |
| Unchanged | 4 | Discovery | -5.953 | -4.555 | +6.172 | -13.672 | -5.164 |
| Unchanged | 4 | Replication | -6.508 | -3.195 | +0.375 | +0.961 | +1.805 |
| Persistent | 1 | Discovery | +2.203 | +4.922 | +2.328 | -18.188 | +2.031 |
| Persistent | 1 | Replication | -3.109 | -4.664 | +4.984 | -9.992 | -4.461 |
| Persistent | 4 | Discovery | +2.203 | +15.680 | -8.109 | -4.469 | -3.156 |
| Persistent | 4 | Replication | -3.609 | -4.781 | +1.125 | -9.500 | -5.445 |
| Fast iid | 1 | Discovery | -5.703 | +6.578 | +0.594 | -5.672 | -2.633 |
| Fast iid | 1 | Replication | -6.586 | -4.719 | +5.383 | -4.422 | -0.313 |
| Fast iid | 4 | Discovery | -5.453 | +4.516 | +3.688 | -5.625 | +1.164 |
| Fast iid | 4 | Replication | -5.836 | -3.891 | -0.102 | -3.266 | -7.602 |

Every raw world pair, minimum/maximum and sign count is retained in
`reports/2026-09-29-completion-control-screen/run/contrasts.json`.

The apparent I64 advantage in persistent/slots4 discovery does **not** replicate:

| Persistent cell | I64-F64 for worlds 0,1,2,3 |
|---|---|
| Slots1 discovery | -0.84375, +1.65625, -0.18750, +8.68750 |
| Slots1 replication | -0.56250, +1.18750, +1.59375, +17.71875 |
| Slots4 discovery | -15.21875, +4.53125, +4.43750, -26.18750 |
| Slots4 replication | +0.93750, +0.09375, +3.03125, +0.43750 |

Do not promote the favorable discovery mean or its two favorable worlds into
a general adaptation claim. Nine of the twelve descriptive I64-F64 cell means
are positive; these dependent cells are not twelve independent trials.

## Planner adequacy

At the prespecified booked-rule epoch9 states in discovery world0:

| Family | Slots | Depth1/16 | Depth1/64 | Depth2/64 |
|---|---:|---|---|---|
| Unchanged | 1 | (0,0.75) | (0,0.75) | (0,0.75) |
| Unchanged | 4 | (0,0.75) | (0,0.75) | (0,0.75) |
| Persistent | 1 | (0,0.75) | (0.25,0.75) | (0.25,0.75) |
| Persistent | 4 | (0,0.75) | (0.25,0.75) | (0.25,0.75) |
| Fast iid | 1 | (0,0.75) | (0.25,0.75) | (0.25,0.75) |
| Fast iid | 4 | (0,0.75) | (0.25,0.75) | (0.25,0.75) |

Four of six probes disagree when sampling budget grows. They represent only
four unique public state/filter/capacity contexts: persistent and fast-iid have
identical observed histories at this early probe. Neither the six probe records
nor their repetitions across families are independent adequacy replications.

The depth1/64 and depth2/64 first requests agree locally. This does not certify
convergence, global optimality or accuracy elsewhere. Depth2 fixes a two-request
open-loop prefix, while depth1's second request uses the state-feedback fallback;
their continuation classes are **not nested**. A larger depth2 expected cost is
therefore not a computational error or evidence that planning is unnecessary.

All forecasts include complete settlement, but optimize only a limited prefix
and use a constant plug-in response estimate. They are restricted rollout
comparators, not strong full contingent-policy or belief-space MPC bounds.
Stored candidate marginal Monte Carlo standard errors are not paired-difference
standard errors and not effect confidence intervals.

Recorded planner work: 41,236,921 scalar-equivalent forecast transitions in the
episodes, plus 1,992,285 in the adequacy probes. Batched vectorized calls make
this feasible locally; these counts are **not actual environment interactions**.
They must not be hidden in future RL-versus-planning compute comparisons.

## Interpretation and limitations

Ordinary feedback can reduce costs without an online neural update. The
reservation rule costs less in all four recorded worlds of every unchanged
and fast-iid cell, but its persistent-change results reverse across splits.
This does not establish a strong frozen-policy bound or reliable adaptation.

As a descriptive accounting decomposition over all dependent cells, reservation
minus booked changes mean holding/labor/switching costs by approximately
**+1.792 / -7.167 / +1.099**. The net decrease is a paid-availability versus
waiting trade-off, not a dominance result on service outcomes. Synthetic total
cost savings cannot be called clinical improvement or noninferiority.

A post-hoc receipt count, not a prospective success endpoint, finds only six
post-change support completions per persistent I64 episode before the decision
cutoff, with 15-21 exposed site-intervals. The fixed eight-job model is a short
adaptation task. Sparse information, plug-in filter mismatch and noisy action
ranking are plausible limitations, **not causally separated explanations**.
The screen cannot distinguish them or prove that adding jobs would help RL.

The action grid is finite even though the underlying effort model is continuous.
There is no action-resolution convergence, calibrated staffing model, finite
intermediate-buffer blocking, patient outcome evaluation or many-site graph
ablation. Do not transfer these totals into the current paper's main results.
Earlier GCN evidence is neither re-estimated nor invalidated by this screen.

## Next bounded step, not executed here

Focus next on **reliable full-cost action comparisons**, not another DDPG run
or changes to costs chosen to manufacture an advantage:

1. Preserve this matrix. Freeze a separate validation protocol using all six
   archived probe contexts, including duplicates explicitly. Compare paired
   action-value differences on independent forecast blocks with budgets fixed
   before observing validation outcomes; retain raw per-sample paired costs.
2. Check ranking replication and continuation adequacy. A first-request result
   needs a credible feedback continuation, not merely a deeper fixed sequence.
   Any additional estimator/posterior-predictive comparison must be declared as
   a separate attribution question, not silently substituted into S2.
3. If credible action ranking and practical headroom remain unestablished, stop
   this channel or report that limitation. If they pass, the later comparator
   must be a competent frozen history-aware policy with the same receipt filter,
   versus tensor-identical online-updated forks, including exploration cost.

This is a proposed next study, not a new executed campaign. It requires the
usual explicit continuation and committed change control. No E1 site values,
new manufacturing scenario or Howard sign-off is inferred. Stage E stays closed.

## Provenance and read-only verification

- Initial protocol/config commit: `50199b0`.
- Recorded source commit: `b0b3e315a629b9fca44e742120075981d5187c01`.
- Immutable reference: `498ea59`; source/protocol code unchanged since execution.
- Reports: `reports/2026-09-29-completion-control-screen/run/`.
- Trace SHA256: `ed6167ca9cbbd384204a5ba35b00c3c784fe7b638f6c20ddd2ba762cdca94448`.
- Contrasts SHA256: `fd6701e716732b5f17e340e75557faee8ae80634aeea6327eca473ded0c34210`.
- Summary SHA256: `e74e61d552086636d54c60cc4904d716ca3eff5a3acaca3e94e70e6ef9e62051`.

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m evaluation.run_completion_control_screen \
  --output reports/2026-09-29-completion-control-screen/run --audit-only
"$PY" -m unittest tests.test_completion_rollout_control tests.test_completion_control_screen
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

The audit-only command verifies saved artifacts and replays the declared one
planner episode, without writing outputs or launching a new matrix. Runtime
and per-decision latency fields are measured and intentionally not claimed to
be byte-reproducible. Recorded source locks must match before verification.
