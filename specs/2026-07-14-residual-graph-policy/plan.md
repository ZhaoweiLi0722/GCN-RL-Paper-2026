# Residual graph policy plan

Date: 2026-07-14

## Motivation

The current evidence says graph learning improves learned control over flat
learning, but tuned heuristics still dominate pure learned policies. The next
methodological step is therefore not to train a larger black-box actor from
scratch. Instead, we evaluate a learning-augmented controller:

```text
action = heuristic_base_action + bounded_gcn_residual
```

The heuristic handles the base inventory/capacity rule. The GCN actor learns
small network-aware corrections around that strong operational policy.

## Stage 1 Scope

Implemented experiment arms:

- `gcn_residual_mdl2`: MDL-2 anchor plus bounded GCN correction.
- `gcn_residual_iso`: isolated local policy anchor plus bounded GCN correction.
- `gcn_residual_myo`: myopic sharing anchor plus bounded GCN correction.
- `gcn_pure_ddpg`: same GCN-DDPG architecture without a residual anchor.
- `gcn_td3`: stable graph TD3 reference.
- `mdl2`, `iso`, `myo`, `umyo`: deterministic heuristic references.

The residual action scales are deliberately conservative. Specimen transfer is
disabled because autologous specimens are identity-bound in the patient model.
Reagent transfer and capacity transfer receive small residual scales; the
replenishment residual is larger but centered across clinics so the policy
mostly reallocates purchasing pressure rather than globally increasing orders.

## Commands

Dry run:

```bash
PYTHONPATH=. .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget smoke \
  --phase dry-run
```

Single-arm smoke:

```bash
PYTHONPATH=. .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget smoke \
  --phase all \
  --algorithms gcn_residual_mdl2 mdl2 \
  --scenarios graph_dynamic_transfer_delay \
  --force
```

Pilot, once smoke is clean:

```bash
PYTHONPATH=. caffeinate -i .venv/bin/python -m evaluation.run_full_benchmark \
  --plan experiments/configs/residual_policy_benchmark.json \
  --budget pilot \
  --phase all \
  --primary-only
```

## Decision Gate

Advance to a full benchmark only if the pilot shows at least one residual GCN
arm that either:

- beats its own heuristic anchor on cost without lowering service/eligibility,
  or
- reduces the cost gap to the best tuned heuristic by at least 25% relative to
  pure `gcn_pure_ddpg` / `gcn_td3`.

If neither condition holds, the next method change should be environment-side:
geography-dependent transfer cost/delay and waiting-time-dependent patient
deterioration, not more pure compute.
