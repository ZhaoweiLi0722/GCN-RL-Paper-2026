# Geography-aware environment dynamics

Date: 2026-07-14

## Objective

Move geography from graph metadata into operational dynamics so graph-aware
policies have a real network signal to exploit. The environment should remain
backward-compatible: non-geographic configs must preserve the previous behavior.

## Implemented Levers

- Geographic KNN default edges for information/specimen/resource sharing when
  `clinic_coordinates` are provided.
- Distance-aware transfer costs via `geographic_transfer_cost_scale`.
- Distance-bucketed transfer lead times via
  `transfer_lead_time_distance_thresholds`.
- Region-correlated supplier disruptions via
  `regional_supplier_disruption_probability`,
  `regional_supplier_disruption_duration`, and
  `regional_supplier_disruption_cluster_size`.
- Geographic demand-shock clusters when a demand shock config is active.

## Evaluation Role

This scenario is not meant to weaken the heuristics. It makes the benchmark
closer to the paper's network-coordination motivation:

```text
nearby clinics share faster and cheaper;
far clinics share slower and more expensively;
supplier failures can affect a regional cluster.
```

That creates a clean setting for testing whether residual GCN policies learn
when to deviate from MDL-2/ISO/MYO based on graph geography.

## Smoke Command

```bash
PYTHONPATH=. .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget smoke \
  --phase all \
  --algorithms gcn_residual_mdl2 mdl2 \
  --scenarios graph_dynamic_patient_forecast_geo \
  --force
```
