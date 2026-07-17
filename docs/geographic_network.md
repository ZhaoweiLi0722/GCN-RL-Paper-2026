# Geographic 20-Clinic Network

This repository includes a geography-aware version of the 20-clinic patient
forecast scenario:

```text
experiments/configs/20_clinic_graph_dynamic_patient_forecast_geo.json
```

The point-of-care treatment centers come from the B&B Paper 2026 final
submission, Appendix A Table A.1 and Figure 1. The paper provides the numbered
POC key, treatment-center addresses, and transportation distance/time/cost to
the California manufacturing site. The repository stores those fields in:

```text
data/bb_20_clinic_locations.json
```

The paper does not provide latitude/longitude values directly. The coordinates
used here are address/city-level coordinates derived from the treatment-center
addresses so that the simulation can construct a reproducible geographic graph.
They should be described as a geography-calibrated or address-based graph, not
as exact geocoded facility coordinates.

## POC Order

The clinic order follows the B&B Appendix A numbering:

1. Seattle Cancer Care Alliance
2. Oregon Health and Science University Hospital
3. Stanford Health Care
4. UCLA Health
5. UC San Diego Health
6. Mayo Clinic Arizona
7. Colorado Blood Cancer Institute
8. Baylor University Medical Center / Texas Oncology
9. Mayo Clinic
10. Siteman Cancer Center
11. Norton Cancer Institute
12. Cancer Treatment Centers of America Chicago
13. Massachusetts General Hospital Cancer Center
14. Memorial Sloan Kettering Cancer Center
15. Penn State Cancer Institute
16. MedStar Georgetown University Hospital
17. Levine Cancer Institute
18. Winship Cancer Institute of Emory University
19. Mayo Clinic Florida
20. Miami Cancer Institute

## How Geography Enters The Current Code

When an environment config provides `clinic_coordinates`, `CapacityPlanningEnv`
uses a symmetric geographic k-nearest-neighbor graph instead of the synthetic
ring graph for:

- `information_edges`
- `specimen_edges`
- `resource_edges`

The default geographic neighbor count is controlled by `geographic_neighbor_k`
and is set to `3` in the geographic patient-forecast config. Capacity edges
remain compatible with the current central capacity hub representation.

Demand-shock clusters also become geography-based in the geographic scenario:
the shock center is sampled randomly, and the affected cluster is selected from
the nearest clinics in great-circle distance. Without `clinic_coordinates`, the
existing ring-contiguous shock logic is unchanged.

The geography-aware config also lets geography affect operational dynamics:

- `geographic_transfer_cost_scale` makes inter-clinic transfer cost increase
  with great-circle distance.
- `geographic_transfer_speed_mph` and `geographic_transfer_fixed_hours` convert
  inter-clinic distance into a continuous transfer-time matrix in hours. This is
  exposed as `clinic_transfer_time_hours_matrix` in graph observations and can
  add a time-based transport surcharge through
  `geographic_transfer_time_cost_scale`.
- `transfer_lead_time_distance_thresholds` maps longer transfers into later
  arrival buckets within the existing transfer pipeline.
- `regional_supplier_disruption_probability`,
  `regional_supplier_disruption_duration`, and
  `regional_supplier_disruption_cluster_size` create correlated supplier shocks
  across nearby clinics.

All three levers are opt-in. Existing non-geographic configs keep the previous
ring/dense graph behavior, fixed transfer lead time, and independent supplier
availability.

## Running The Geo Scenario

Dry-run the patient forecast benchmark with only the geography-aware scenario:

```bash
python -m evaluation.run_full_benchmark \
  --plan experiments/configs/patient_forecast_benchmark.json \
  --phase dry-run \
  --budget smoke \
  --scenarios graph_dynamic_patient_forecast_geo
```

Run a tiny smoke pass:

```bash
python -m evaluation.run_full_benchmark \
  --plan experiments/configs/patient_forecast_benchmark.json \
  --phase all \
  --budget smoke \
  --scenarios graph_dynamic_patient_forecast_geo
```

Use this scenario as a robustness/extension experiment until the geography
assumptions are finalized in the manuscript.

## Transfer Time Interpretation

The main patient-condition geography scenario uses weekly decision epochs, but
the B&B/Wiley transportation data are measured in hours. For that reason,
`experiments/configs/20_clinic_patient_condition_geo.json` keeps
`transfer_lead_time` at `0`: sub-week transport does not become a full one-week
pipeline delay. Instead, geography enters through graph edges, distance-based
transfer costs, regional clusters, and the continuous
`clinic_transfer_time_hours_matrix`.

The older integer `transfer_lead_time` settings should be interpreted as
discrete-time sensitivity experiments or as a replication of the prior
two-facility weekly-transshipment abstraction, not as the main realistic
20-clinic geography assumption.
