# Frozen Protocol and Pre-Registered Predictions

Frozen: 2026-08-29, **before** the confirmatory run was executed.

Exploration is closed. This file fixes every choice the confirmatory run will
make and records what I expect it to produce, so the run is a genuine test
rather than a description. Config:
`experiments/configs/continuous_overtime_confirmatory.json`.

## Environment (from E2b, unchanged)

```
enable_overtime_control     true
max_overtime_fraction       0.6
weight_overtime_linear      15,000
weight_overtime_quadratic   20,000
enable_overtime_fatigue     false
enable_production_throttle  false   (dormant, rejected at validation)
```

## Data (reserved, never touched during exploration)

```
scenario            routing_regional_drift
generation seeds    97100000-97100004      (45 states = 5 seeds x 9 epochs)
decision epochs     20,30,40,50,60,70,80,90,95
discovery seeds     97200000-97200002
validation seeds    97300000-97300004
action ladder       0.0 .. 1.0 in steps of 0.1 (11 rungs)
```

## Gates (unchanged from exploration; no re-tuning permitted)

| Stage | Criterion | Threshold |
| --- | --- | --- |
| Headroom | states with validated, clinically noninferior ≥1M saving | ≥ 0.30 |
| State-dependence | prospective value (discovery-selected, validation-scored) | ≥ 0.005 of anchor |
| | interior best-arm fraction | ≥ 0.30 |
| Label stability | best-action agreement, fresh streams | ≥ 0.70 |
| | pairwise cost-sign agreement | ≥ 0.80 |
| Ranking | pooled and worst-fold top-1 | ≥ 0.50 |
| | pairwise accuracy | ≥ 0.70 |
| | gain over state-blind, every fold | ≥ 0.05 |

**Ranking model, fixed:** ridge, α = 10, on the 15 aggregate decision-time
features. This was the best realized-cost configuration found in exploration
(−0.2655%, 41.2% of headroom). No alternative model, feature set, or α may be
tried on the confirmatory data — E4b and E4c established that neither
distributional shape, graph structure, nor nonlinearity improves on it, so
there is no honest reason to search again.

## Pre-registered predictions

Recorded before execution. Where these are wrong, the write-up says so.

| Quantity | Prediction | Basis |
| --- | --- | --- |
| Headroom gate | **PASS** | passed in all three exploratory scenarios |
| State-dependence value | **PASS**, +0.5% to +0.8% | E2b +0.6446%, E3 +0.6182% |
| Interior best-arm fraction | ≥ 0.80 | 0.926 in both E2b and E3 |
| Label stability, best-action | **PASS**, 0.85–0.98 | E3 0.926 fresh, 1.000 cross-family |
| Label stability, pairwise | **PASS**, ≥ 0.95 | E3 0.990 |
| Ranking top-1 | **FAIL**, 0.25–0.40 | E4 0.252, E4b ≤0.356, E4c ≤0.393 |
| Ranking worst-fold top-1 | **FAIL**, ≤ 0.35 | never exceeded 0.259 |
| Realized cost vs constant | −0.15% to −0.35% | −0.2655% best in exploration |
| Headroom captured | 30–50% | 41.2% best in exploration |
| **Overall verdict** | **channel fails E4; E5 not authorized** | |

Headline claim under test: *the overtime channel has real, replicable
state-dependent value, roughly 40% of which a simple model can capture out of
sample, but its optimum is not predictable from decision-time state to the
precision a policy would need.*

## What would falsify the exploratory conclusion

- Ranking top-1 ≥ 0.50 on the held-out scenario would mean learnability is
  regime-dependent and the exploratory negative does not generalize.
- A state-dependence value near zero would mean the E2b/E3 positive was
  specific to the exploratory scenarios.
- Label-stability agreement below 0.70 would mean E3's replication was a
  property of those scenarios rather than of the channel.

## Rules

1. The run executes **once**. Its result stands and is reported.
2. No gate, threshold, feature, model, or hyperparameter may change after it
   starts.
3. If it fails in an unanticipated way, that is a finding to report, not a
   reason to adjust and re-run.
4. Exploratory and confirmatory results are always reported together.
