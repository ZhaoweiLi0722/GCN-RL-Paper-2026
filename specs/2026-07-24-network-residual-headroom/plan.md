# Network-Residual Headroom and Matched Representation Test

## Question

Can a clinically guarded residual policy improve on MDL-2 by correcting network
transfer and replenishment decisions, and does a graph representation improve
those corrections relative to a matched flat-state policy?

## Scenario

- `patient_condition_geo_demand_drift`
- 20 clinics with geographic transfer times and costs
- calibrated waiting and in-process patient deterioration
- MDL-2 receives the estimated demand model while realized demand follows the
  drifted process

## Pre-Registered Headroom Gate

Advance to learned-policy training only when the online look-ahead teacher:

- reduces mean episode cost by at least 1% relative to MDL-2;
- has a paired 95% bootstrap confidence interval entirely below zero; and
- finds a clinically feasible correction in at least 5% of probed states.

Candidate corrections may modify reagent transfer, capacity transfer, and
replenishment, but not specimen transfer. A candidate is rejected when it lowers
completion service, increases patient loss, or increases in-process patient
ineligibility relative to the MDL-2 continuation.

## Matched Learned Policies

1. `gcn_residual_mdl2_network_ddpg_afd`
2. `gcn_residual_mdl2_network_td3_afd`
3. `flat_residual_mdl2_network_ddpg_afd`

All policies use the same MDL-2 anchor, action bounds, teacher demonstrations,
training episodes, random seeds, validation streams, and fallback rules. The
GCN-DDPG versus flat-DDPG comparison isolates the graph representation. The
GCN-DDPG versus GCN-TD3 comparison isolates the off-policy optimizer.

## Pilot

- 100 training episodes
- 52 decisions per episode
- training seeds 0, 1, and 2
- 100 paired Monte Carlo evaluation replications per seed
- separate 30-replication fallback-validation stream

Primary comparisons use a paired two-level bootstrap over training seeds and
Monte Carlo replications:

- each learned policy versus MDL-2;
- GCN-DDPG versus flat-DDPG;
- GCN-TD3 versus GCN-DDPG.

The manuscript will not be updated until this pilot produces a stable,
statistically interpretable stage result.
