# R3: design ready for review, scientific execution not ready

2026-09-29. Protocol, config and synthetic tests committed as `1318149`.
No new environment trajectory, checkpoint inference, fitting or performance
evaluation was run. The proposed pilot has not been implemented or launched.

## Work completed

- Adopted the original 52-step finite-window cost endpoint for this draft,
  with unchanged weights, absolute negative cost, one scale 1e-9 and gamma=1.
  A cutoff at step 52 terminates this objective, not real patient obligations.
- Separated first-action evaluation under MDL-2 continuation from value under
  a frozen actor. Current inspected G1 code calls the former; DDPG bootstrap
  uses the target actor. Neither defines the other automatically. This adds
  a target-definition requirement, not proof of the cause of old null results.
- Preserved the historical actor/gate as the intended baseline. N4-N7 use
  different engineering heads and cannot stand in for pretrained policies.
  The historical loader has partial gate-weight transfer fallbacks, so a
  strict key/shape/hash/output-parity check is required before reuse.
- Specified a small, fixed data-only proposal: 12 states, at most six unique
  actions, 8+8 draws, at most 1,152 continuation records / 37,596 environment
  transitions / 3,600 seconds. These are proposed caps, not consumed budget.
  No actor/critic fitting and no automatic escalation after any result.
- Made the frozen actor's first action the primary reference and kept its
  continuation identical for all choices. MDL-2's first action followed by
  that actor is a separate secondary reference, not a full MDL-2 policy.
- Required replication-level costs, clinical metrics, cutoff obligations and
  executed-action reachability. Neither G1 mean-only data nor exact argmax
  accuracy alone can answer that prospective question.

## Validation actually performed

Eleven new synthetic tests plus 48 related tests passed (59 total). They check
52-step undiscounted returns against independent suffix sums, one-step TD,
equal weighting of early/late costs, objective-terminal masking, earlier
truncation, counterfactual lineage and distinct continuation IDs. The invented
two-step example confirms only the logical possibility that two continuations
rank identical first actions differently; it is not a PRM experiment.

Budget arithmetic independently reproduces 1,152, 37,596 and 195 maximum seed
starts. Eight inspected source locks remain unchanged. Whole-repository
compileall and diff checks pass. A fresh process check found no matching
research Python process, only the check itself. No background job or automation
was created. Tests use no environment, learner or historical model checkpoint.

```bash
python -m unittest tests.test_fixed_window_objective tests.test_validated_returns tests.test_g1_decision_cost tests.test_reward_objective_bridge -q
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python -m compileall -q .
```

Use the project virtualenv. These are engineering tests, not a scientific pilot
command. The JSON is explicitly non-executable and has six null launch inputs.

## Actual data dependency

The authoritative G1 summary references these F1 control frozen-pretrain files:

`results/patient_indexed_specimen_routing_ddpg_online_paired_advantage_development/control/training/patient_indexed_specimen_routing_ddpg_online_paired_advantage_control_100/gcn_residual_mdl2_network_ddpg_afd/seed<60|61|62>/checkpoints/gcn_residual_mdl2_network_ddpg_afd_seed<60|61|62>_pretrain.pt`

Need the three checkpoints plus effective per-run `config.json`, `summary.json`
and corresponding `training_manifest.json`. Historical actor-only tensor hashes
are available in the G1 summary but are not whole-checkpoint/gate hashes.
The original training source/version and a complete file inventory must also
be resolved before inference. Do not supply only a final actor or replace seeds.

Observed in this session:

| Scoped location | Read-only observation |
| --- | --- |
| Current integration worktree's relative F1 result tree | Does not exist |
| `/private/tmp/gcnrl-stage-f-ddpg-audit` | Does not exist |
| `/Users/lizhaowei/gcnrl-patient-routing-persistent/results` | Targeted file scan found no seed60-62 pretrain checkpoint or F1 training manifest |
| `/Users/lizhaowei/gcnrl-rl-attribution-refinement/results` | Same targeted file scan found no match |
| RTX host mentioned by the old coauthor draft | Not connected or verified; not treated as current fact |

This is not a whole-machine/cloud search and does not establish data loss.
No other worktree or remote service was modified. An asynchronous question has
asked Zhaowei for the current local/backup location; no response is recorded yet.
Do not silently substitute the formal-holdout actors or a newly randomized one.

Follow-up: `artifact-recovery.md` records the expanded local/archive search,
its permission-denied coverage limits, retained historical Git source, exact
ten-file minimum recovery package plus authenticated hash mapping, and the
environment-source compatibility gap. No F1 weights were recovered or loaded.

## Exact next action

When the artifact location is supplied, first perform read-only file/source
provenance and strict actor/gate compatibility checks. Then complete the
collector/analysis implementation, allocate fresh streams with collision
checks, run bounded engineering acceptance, and request authorization for the
specific capped data-only pilot before collecting new scientific outcomes.
Do not declare the pilot executable just because these arithmetic tests pass.

The R2 descriptive cost finding is unchanged. R3 adds no positive online RL
evidence. It narrows the next question to value under the correct continuation
and a preserved pretrained actor, without tuning costs to obtain improvement.
