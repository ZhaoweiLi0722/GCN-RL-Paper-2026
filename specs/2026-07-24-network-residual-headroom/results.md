# Stage Results

## Narrow Residual Probe

Configuration: `experiments/configs/network_residual_headroom.json`

- 8 anchor-trajectory state probes, 416 states
- 20 paired online-teacher replications, 1,040 decisions
- clinically feasible correction rate: 98.46%
- teacher wins versus MDL-2: 19/20 episodes
- mean total-cost difference: -17,826,174
- mean total-cost gap: -0.4623%
- paired 95% bootstrap CI: [-22,045,814, -13,576,702]
- completion-service difference: +0.002873
- mean patients-lost difference: -19.55
- in-process patient-ineligibility-rate difference: -0.001311

The benefit is statistically clear and clinically aligned, but it does not pass
the pre-registered 1% episode-cost materiality gate. Therefore the 100 x 3
learned-policy screen is not yet justified under this candidate set.

## Boundary Diagnostic

Of 411 non-anchor state-probe selections, 372 selected the maximum allowed
epsilon of 0.02. This indicates a truncated action search rather than exhausted
headroom. A second-stage wide probe is therefore being run with epsilon values
0.02, 0.04, and 0.08 while preserving the same clinical guards and 1% gate.

## Current Decision

Do not reinterpret the 0.4623% result as passing the original gate. Proceed with
the explicitly labeled wide-amplitude sensitivity. Launch matched GCN-DDPG,
GCN-TD3, and flat-DDPG training only if the wider teacher demonstrates material
headroom, or otherwise treat any learned-policy run as exploratory.

## Full Network Residual And History Screen

The end-to-go compact teacher around MDL-2 established material full-action
headroom:

- 20 trajectories, 1,040 decisions, 5 CRNs per candidate
- teacher total-cost gap versus MDL-2: -1.5548%
- correction rate: 37.8%
- completion-service difference: +0.0177
- patients-lost difference: -162.1

A true network residual actor was then implemented with separate replenishment,
reagent-edge, and capacity-edge heads. The edge heads conserve flow exactly and
the specimen-transfer residual is fixed at zero. A grouped deployment gate was
required because capacity-transfer labels were too rare to deploy safely.

With 12-week causal demand history, a fixed gate
`(reagent=0.5, capacity=closed, replenishment=0.4)`, and scale 1.0, three
pretrain-only GCN seeds produced:

- total-cost gap versus MDL-2: -0.19998%
- hierarchical paired 95% CI: [-9,292,078, -5,729,583]
- 81 wins in 90 CRN pairs
- completion-service difference: 0
- patients-lost difference: 0

This is a reproducible learned improvement, but it captures only about 13% of
the teacher headroom and remains below the pre-registered 1% materiality gate.
An 8-week history and a smaller GCN did not improve trajectory-level gate AUC.

## Information-Matched Heuristic Audit

`fMDL-2` was added as an explicit forecast-aware baseline. In the current
environment the forecast reveals the active true demand rate, so fMDL-2 should
be interpreted as an oracle-style information upper bound rather than an
ordinary fixed-prior heuristic.

On 100 independent CRN replications:

- fMDL-2 versus MDL-2 total-cost gap: -0.57290%
- paired 95% CI: [-29,334,898, -14,082,577]
- completion-service difference: +0.00311
- patients-lost difference: -12.18

The history-12 AFR-GCN-DDPG remained 0.12344% more expensive than fMDL-2 on the
matched 90-pair screen; the CI crossed zero. A separately tuned shrinkage
rolling baseline (`rMDL-2`, validation-selected history weight 0.05) did not
beat MDL-2 on its independent test set: +0.06071% cost and +4.41 patients lost.

## Forecast-Anchor Headroom

A 16-week look-ahead teacher around fMDL-2 reduced patient losses but increased
full-episode cost by 1.70382%. This is terminal-horizon bias: delayed expiry and
patient-loss consequences extend beyond a fixed 16-week rollout. Those teacher
labels must not be used for policy training.

The end-to-go fMDL-2 teacher removed that bias:

- 10 trajectories, 520 decisions, 5 CRNs per candidate
- total-cost gap versus fMDL-2: -0.89299%
- paired 95% CI: [-64,841,320, -11,624,291]
- completion-service difference: +0.01273
- patients-lost difference: -125.9
- correction rate: 37.5%

The CI is clearly below zero, but the mean narrowly misses the 1% advance gate.
Only 23 of 195 corrections involved transfer; most selected replenishment.
With only 520 states, the grouped gate did not generalize well. A diagnostic
fMDL-2-anchor actor either fully fell back at safe thresholds or achieved only
-0.00164% at a lower threshold, with a CI crossing zero.

## Updated Decision

Do not launch longer online DDPG/TD3 training from the current checkpoints.
The next evidence-generating stage is:

1. add pre-registered regional drift and compound disruption scenarios that
   create genuine edge-transfer value;
2. keep fMDL-2 as an oracle-style upper bound and MDL-2/rMDL-2 as deployable
   heuristics;
3. expand end-to-go teacher coverage to at least 40 trajectories for the first
   screen, then 60-100 if gate generalization continues to improve;
4. require a matched flat residual ablation and nonzero transfer deployment
   before claiming graph value;
5. enter 100-episode online training only after the learned pretrain policy
   clears the clinical guards and shows a materially larger held-out gain.
