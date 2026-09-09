# User-Authorized Residual Screen: Readout

Completed and independently audited on 2026-09-09. Development evidence;
Howard review remains pending. Zhaowei authorized this bounded execution;
no Howard approval, joint endorsement, or formal confirmation is claimed.

## Decision

Execution succeeded (`completed`, exit 0). The scientific decision is
`budget_neutral_residual_support_failed`: R0 mechanics passed, R1 optimistic
headroom and R2 prospective support failed. No ranking learner or DDPG training
was launched or authorized by this result. No rerun, new seeds, threshold
change, new scenario, or formal-holdout use followed it.

This was a policy-free screen, NOT an online DDPG experiment. It tests whether
a finite library of local, continuous overtime reallocations offers enough
reproducible benefit around the strongest frozen J2 graph-forecast comparator.

## Results

Each of 54 states had 81 actions: the baseline plus 20 facilities x two
directions x two transfer magnitudes. Discovery used two independent worlds
per state; validation used five fresh worlds, with paired randomness across
the 81 actions in each world. Positive saving means lower 12-step total cost.

| Criterion | Observed | Prespecified gate | Outcome |
| --- | ---: | ---: | --- |
| Noncollapsed residuals | 4,198/4,320 = 97.1759% | At least 90% | Pass |
| Facility/direction coverage | 40/40 | 40/40 | Pass |
| Maximum budget error | 2.13e-14 | At most 1e-8 | Pass |
| Optimistic validation-selected saving | 0.240906% | At least 0.5% | Fail |
| Discovery-selected validation saving | 0.189611% | At least 0.5% | Fail |
| Material, clinically noninferior states, either selector | 0/54 | At least 30% | Fail |
| Aggregate clinical noninferiority | Both selectors pass | Required | Pass |
| Prospective saving positive in each scenario | 3/3 | Required | Pass |

The best state-level validation saving was 0.446542%, below the materiality
threshold. The discovery-selected action had a positive validation cost
saving in 53/54 states; this is not a significance test or evidence that DDPG
learned the action. All selections here are finite-library lookups.

| Scenario | Prospective saving | Optimistic saving |
| --- | ---: | ---: |
| Scheduled rotation, forward | 0.216086% | 0.268648% |
| Scheduled rotation, reverse | 0.145670% | 0.206097% |
| Scheduled rotation, slower/noisier | 0.206869% | 0.248533% |

## Stability Is Not Established by a Zero Denominator

No discovery-selected action achieved a 0.5% saving. The summary serializes
the corresponding validation-win fraction as 0.0, but its denominator is
zero: it is **not an observed 0% success rate**.

Only four candidate-state comparisons had an absolute discovery difference
of at least 0.5%; all four were harmful, not beneficial, and their signs
replicated. Thus the reported 100% material sign agreement is 4/4 harmful
comparisons, not stable labels for useful positive actions. Exact best-action
agreement was 16/54 = 29.63%; with small differences this alone cannot diagnose
whether useful ranking labels exist.

## Interpretation and Limits

1. This time the action channel was usually active and budget-neutral. Integer
   rounding is not an adequate explanation of this screen's negative gate.
2. The observed local savings are small relative to the frozen 0.5% gate,
   although not literally zero. That gate is a project decision rule, not an
   EAAI acceptance criterion or a universal definition of operational value.
3. The optimistic selector uses validation outcomes to choose among 81 actions.
   It is not deployable, not an unbiased performance estimate, and not a bound
   on the full policy class or global optimum.
4. Each rollout changes only its first action, then uses the frozen comparator
   for continuation. This does not test repeated coordinated residual actions,
   a full-episode learned controller, or adaptation under unknown dynamics.
5. The comparator is the strongest frozen graph-forecast heuristic, not the
   paper's pretrained neural policy. These results cannot establish that the
   pretrained policy is globally near-optimal.
6. The current scenario convention supplies forecasts based on future scheduled
   demand and regime multipliers. That can be legitimate for announced demand;
   it cannot simply be relabeled as learning an unannounced future change.

For the current manuscript, preserve the distinction between package-level
performance and the unestablished marginal contribution of online updates.
This screen supports a scoped feasibility/limitations discussion. It does not
justify claiming an online RL gain or weakening the gate after seeing data.

## Next Design Work

Close this particular local-support screen. The next useful design question is
whether a credible operating problem contains **persistent uncertainty and a
consequential intertemporal decision** that a strong informed/reactive baseline
has not already solved. Do not merely make demand larger until RL wins.

The companion [information-boundary memo](next_study_information_contract.md)
specifies the first design deliverables: public information at decision time,
unknown-versus-announced changes, equal-information baselines, and a 2x2
ablation separating forecast/belief adaptation from policy-parameter updates.
A small faithful solvable case should establish controllable headroom before
another neural-training campaign. That is a new study specification, not an
automatic continuation or authorization granted by this readout.

## Reproducibility and Validation

- Isolated branch: `codex/residual-headroom-user-authorized`.
- Execution commit: `400d64f764b4e96a0d93d0c4a2721c4f68b2f08b`.
- Original PR #12 source: `118f9978f50433d25de1e392e242cd1655101b43`.
- Output: `results/intertemporal_residual_allocation_headroom_development/r0_zhaowei_20260908`.
- [Generated summary](../../results/intertemporal_residual_allocation_headroom_development/r0_zhaowei_20260908/summary.json),
  [raw rows](../../results/intertemporal_residual_allocation_headroom_development/r0_zhaowei_20260908/residual_headroom_rows.csv),
  [artifact inventory](../../results/intertemporal_residual_allocation_headroom_development/r0_zhaowei_20260908/artifact_inventory.json).
- Exactly 30,618 unique finite rows: 8,748 discovery and 21,870 validation.
  All 378 paired worlds have 81 actions and matching final RNG hashes.
- State/scenario/epoch/seed/action Cartesian products reconcile; mechanics do
  not change with replication; all 17 inventory hashes match, and each input
  also matches its bytes in the execution commit.
- Independent standard-library recomputation of state selections, savings,
  clinical guards, materiality, stability, mechanics and R0-R2 matches the
  generated summary. No simulator import, state generation, or rollout occurs
  in [audit_result.py](audit_result.py). Run from the repository root:

  ```bash
  python3 specs/2026-09-08-user-authorized-residual-screen/audit_result.py
  ```

- Original PR #12 worktree remains clean at its unchanged commit; the original
  config's output root remains absent. Source config, protocol and parent J2
  evidence match their frozen hashes.
- Evaluator PID 55681 and its PID-bound `caffeinate` helper exited with code 0.
  A read-only process scan found no related training/evaluation process.
- Redirected stderr is empty; stdout/stderr contain no traceback, native
  crash, OOM or nonfinite-metric signature. The evaluator was supervised by the
  current tool session, not a new detached automation.

Timing caveat: UTC start/end timestamps differ by 9,721.50 seconds, whereas
the recorded monotonic elapsed time is 3,641.24 seconds. The discrepancy is
retained, not repaired; its cause was not established. Do not use this run for
wall-time benchmarking. It does not change row cardinality, hashes or the
independently reproduced scientific calculations.
