# Service-effort decision fixture

Status: local, deterministic software validation, 2026-09-29. The user requested
continued progress on the roadmap. No collaborator approval, engineering
calibration or authorization for a new neural/patient campaign is inferred.

## Purpose and bounded scope

Build the decision comparator layer before attempting DDPG. Reuse the isolated
service-effort primitive and the existing finite-scenario-tree exact solver;
do not change either historical implementation. This is a small test problem,
not the headroom experiment for the manuscript.

The committed config specifies two sites, two unit-work orders per site, five
decision epochs, a one-interval commitment delay, four feasible allocations,
and a shared hour budget. One nominal world has known response (1,1). A separate
two-world family is nominal for two intervals, then persistently changes to
(0.5,1.5) or (1.5,0.5), each with probability one half. There is no measurement
noise, arrival process, patient biology, transport graph or scientific RNG.

Cost is effort plus switching cost plus remaining-work holding cost. At the
end, all pending paid commitments execute with zero new commitments, holding
cost continues, and a fixed synthetic external-service charge closes every
remaining work unit. No obligation disappears at the horizon. External service
is a mathematical closure convention, not validated clinical outsourcing or a
patient-release assumption. All numbers are fixed test inputs; do not optimize
them for a favorable result or interpret their cost scale economically.

## Baselines

1. Fixed equal allocation, including its wasted effort when backlog empties.
2. Backlog feedback rule, which can change actions without learning parameters.
3. Adaptive backlog rule using remaining work divided by estimated response.
4. Fixed-model MPC with the nominal response, current observed state, and the
   full remaining decision horizon plus settlement.
5. Identification-plus-MPC with the same planner and horizons, replacing only
   the response estimate using uncensored, already observed service receipts.

All five use the same four-action grid and decision-time public observations.
MPC predicts from aggregate workload, homogeneous known job size, and pending
hours. It never clones the latent simulator or reads future coefficients. All
planning transition queries are counted. The adaptive estimator has alpha 1
because this is a noiseless mechanics test, not a selected learning rate.

Also solve the known finite-world law exactly for three diagnostic quantities:
best open-loop sequence, optimal nonanticipative feedback policy, and a
clairvoyant controller that knows the world. The exact solver knows the entire
two-world law, unlike the constant-response MPC model. It is a model-privileged
diagnostic, not an equally informed practical baseline. Its optimum is exact
only on this action grid, horizon and closure rule; it is not a bound on the
full continuous-action clinical problem.

## Prospective software checks

- Public-state MPC prediction equals actual execution when its response model
  is correct and the homogeneous unit-job assumption holds.
- Worlds with equal histories remain indistinguishable until actual feedback
  differs; no world labels or future change times enter controller inputs.
- Replanning does not mutate a parent state or read private simulator state.
- Commitments mature and are charged even beyond the final decision epoch.
- All 15 baseline episodes, 75 decision rows, 15 drain rows, and 4,092 exact
  tree transition rows exist with no duplicate identities.
- Clairvoyant optimum <= nonanticipative optimum <= best open loop, and each
  implemented baseline is no better than the exact same-grid optimum.
- Under no change, fixed-model and identification MPC agree and attain the
  same-grid optimum. No required sign is imposed on identification benefit
  in the changed family.

These are deterministic test cases, not 15 independent statistical trials.
Passing them advances comparator software readiness only. Even if the adaptive
controller wins, it is system-identification evidence in a toy problem, not
online DDPG evidence or a clinical finding. Preserve a null or adverse result.

## Execution and immutability

Commit source/config/protocol before the recorded run. The runner refuses an
existing output root and records source and output hashes. Preserve the prior
patient/environment sources and both September 28 fixture/pilot outputs against
commit `1aae687c567eec4f8145b07eb95abd89f40aa32f`.

Output: `reports/2026-09-29-service-effort-decisions/fixture/`.
Do not start DDPG, alter the formal holdout, push, or merge main in this packet.
