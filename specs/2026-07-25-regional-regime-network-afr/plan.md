# Regional-Regime Network AFR Experiment Plan

## Research question

Can a graph-aware anchored residual RL policy exploit spatially heterogeneous
demand shifts and geographic resource-sharing opportunities that a fixed-prior
MDL-2 rule does not adapt to, while preserving patient-service safeguards?

## Fairness principle

All deployable methods receive the same online observation. The forecast is
computed from the fixed pre-deployment demand estimate and does not reveal the
active simulator demand regime. Policies may infer changes from current demand,
the 12-period demand history, inventories, transfer pipelines, supplier state,
patient-risk summaries, and geographic neighbors.

## Scenarios

- `patient_condition_geo_regional_drift`: gradual geographically heterogeneous
  drift; used as a smooth-shift validation case.
- `patient_condition_geo_abrupt_regime_shift`: regional demand hotspots reverse
  at week 26; primary training and attribution case.
- `patient_condition_geo_compound_regional_stress`: regional drift, clustered
  demand shocks, supplier disruptions, patient deterioration, geography, and
  distance-dependent transfer lead times; external stress validation.

The existing nominal integrated scenario remains the non-shift reference.

## Stage 1: headroom gate

Anchor: fixed-prior MDL-2.

Teacher action set:

- MDL-2 anchor;
- signed reagent transfer;
- signed reagent and idle-bioreactor transfer;
- signed combined transfer and replenishment;
- regional replenishment correction;
- uniform replenishment reduction.

Each candidate is evaluated to the end of the 52-week episode using common
random numbers. Clinical feasibility requires no decrease in completion service,
no increase in patients lost, and no increase in manufacturing ineligibility.

Advance criteria:

- paired episode cost improvement of at least 1%;
- upper confidence bound below zero;
- clinically noninferior episode metrics;
- corrections in at least 5% of states;
- transfer-involving actions in at least 15% of corrected states.

The mini screen uses two trajectories per scenario only for scenario selection.
The primary abrupt-shift cache uses 40 trajectories and three CRN continuations.

## Stage 2: matched representation attribution

Train from the identical 40-trajectory teacher cache:

- Network AFR-GCN-DDPG;
- Flat AFR-DDPG.

Hold constant the MDL-2 anchor, action heads, correction labels, training seeds,
network widths where structurally comparable, residual scale screen, correction
gate, and evaluation CRNs. The graph model may use geographic adjacency and
edge attributes; the flat model receives the same observable facility state.
The flat actor/critic/gate widths are parameter matched to the graph model
(approximately 528k versus 525k trainable parameters in total).

Initial screen:

- three training seeds;
- 30 paired validation replications per seed to choose one global residual
  scale and gate threshold;
- 100 disjoint paired holdout replications per seed with the selected deployment
  settings;
- MDL-2, rMDL-2, and forecast-aware MDL-2 comparators;
- cost, completion service, patients lost, manufacturing ineligibility,
  residual deployment rate, and transfer volume.

Advance criteria:

- Network AFR improves on fixed-prior MDL-2 by at least 1% with a 95% paired
  confidence interval below zero;
- completion service and patients lost are noninferior;
- Network AFR outperforms matched Flat AFR under identical CRNs;
- at least two of three seeds deploy nonzero residual actions.

## Stage 3: robustness and final study

Only after Stage 2 passes:

- evaluate gradual drift, abrupt shift, compound stress, and nominal integrated
  scenarios;
- add rolling-estimate/tuned robust MDL-2 and a clearly labeled oracle bound;
- compare pure GCN-DDPG, Network AFR-GCN-DDPG, Network AFR-GCN-TD3, flat
  residual DDPG, ISO, MYO, pMYO, and MDL-2;
- run five training seeds and 500 paired Monte Carlo replications;
- report hierarchical confidence intervals and multiplicity-aware comparisons.

No manuscript claim of graph or RL superiority is made from the mini screen or
teacher headroom alone.
