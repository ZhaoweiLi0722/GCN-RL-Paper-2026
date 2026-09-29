# Routing-primary method-to-execution reconciliation

## Scope and decision

This is a reporting correction, not a new method, training run, holdout test,
or explanation proven to cause the null online increment. Historical source,
configurations, models, outputs and primary results are unchanged. The local
manuscript now describes the executed regularized, anchor-relative procedure
instead of substituting generic one-step DDPG equations for it.

The ten-run training manifest matches its historical SHA256:
`0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2`.
All ten current effective configs (GCN/flat, seeds 10--14) match the explicit
settings checked in `evaluation/audit_formal_method_contract.py`. Their paths
and identities agree with the verified manifest. Current config hashes are
recorded but are not independent proof of historical byte identity.

Evidence: `reports/2026-09-29-formal-method-contract/contract.json`.
It contains historical source excerpts, line bounds and hashes, current
config values and hashes, and the two historical fixed-evaluation configs.
Its `passed` status means these static contract checks passed, not that the
entire historical campaign was reexecuted or every scientific claim validated.

Training source: `9c0718b0da49d34f7f878ed2f036e96bf7bb3693`.
Evaluation source: `62a344e7c8f501862c6ae1dcf3be5ef59964ea0d`.

## Corrections made

| Previous generic description | Executed routing-primary contract | Historical source |
| --- | --- | --- |
| Shared learned graph encoder | Separate actor and critic encoders and optimizers; shared graph specification | `src/models/gcn_ddpg.py:964-1045`; `src/models/gcn.py:974-1018` |
| Learned corrections in all four channels | Specimen residual scale 0.10; capacity, reagent and replenishment residual scales zero | Ten effective configs, `residual_action.group_scales` |
| Independent Gaussian noise after action composition | OU noise on network output before residual composition, gating and clipping | `src/models/gcn_ddpg.py:1142-1162` |
| Replay stores final feasible patient allocations | Replay stores the bounded normalized request supplied to `env.step`; the environment matches legal events | `src/rl/experiment.py:283-312`; `src/rl/action_projection.py:41-66` |
| One-step negative-cost Bellman target | Online items contain up to four discounted one-step anchor-relative rewards, scaled by 1e-9, with effective gamma^n bootstrapping and terminal masking | `src/models/gcn_ddpg.py:1164-1299,1816-1876` |
| Critic target receives raw actor residual | Target action is composed with the anchor, hard gated and request-grid quantized | `src/models/gcn_ddpg.py:1850-1876` |
| Unregularized DDPG actor/critic update | Teacher ranking plus AFD, frozen-reference, self-imitation and residual penalties | `src/models/gcn_ddpg.py:1878-2055` |
| Differentiable feasibility projection | Straight-through request-grid quantization; no derivative through actual discrete event matching | `src/rl/action_projection.py:105-149`; `src/models/gcn_ddpg.py:1420-1435` |
| Validation chooses deployed checkpoint and fallback | Formal final/pretrain variants and scale 1.0 fixed; checkpoint group thresholds retained; no automatic anchor fallback | Historical `*_confirmation_100_{final,pretrain}_eval.json` and training configs |

The manuscript's critic target is written for generic serialized replay items
using their stored return and effective discount. Only newly collected online
items are identified as four-step relative returns. It does not incorrectly
assign that transformation to all offline teacher/replay data.

For each actual visited state, the reference reward comes from a separate
one-step MDL-2 transition in an exact pre-step clone. Summing these differences
over four actual transitions is not the same as comparing two autonomous
four-step trajectories. No policy-invariance theorem for this transformation
is established here; the undiscounted evaluation cost remains a distinct
reported outcome.

The historical defaults also matter: behavior and target actions use hard
gates, but actor optimization uses detached soft gate scores. Frozen-reference
regularization defaults to uniform MSE in network-output space, released on
online samples satisfying the self-imitation gate. Eligibility requires a
full four-step online return and a first-step return, each above 0.0005 after
scaling. Coefficients 500, 3, 1 and 0.05 must not be compared as if the losses
or gradient norms had equal units and magnitude.

Source locations for these defaults and masks:
`src/models/gcn_ddpg.py:356-366,664-668,2079-2120,2173-2267,4410-4471`.
The JSON's manual source findings are reviewer interpretations, clearly
separated from automated config equality checks.

Correction from September queue packet G1: the **critic and correction gate**
flatten facility embeddings in a fixed order (`src/models/gcn.py:242-250`).
The formal actor instead uses shared network-residual edge heads. M1's earlier
phrase "primary readout" was too broad. Whole-policy permutation equivariance
or network-size generalization still does not follow from graph convolutions.
The completed input/head/gate audit and narrower package-level attribution are
in `specs/2026-09-29-formal-graph-contract/readout.md`.

## Reproduction

From the persistent integration worktree, use the project's Python interpreter:

```bash
python -m evaluation.audit_formal_method_contract \
  --training-root /Users/lizhaowei/gcnrl-patient-routing-persistent/results/patient_indexed_specimen_routing_mac_mps_primary/confirmation/ddpg_routing_primary_100/patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100 \
  --output /private/tmp/formal-method-contract-new-output.json
python -m unittest tests.test_formal_method_contract
```

The output path must not already exist. The audit reads immutable Git objects
and JSON only, checks source configs again before returning, and imports no
model or simulator. It launches zero training and zero evaluation jobs.

## Remaining boundaries

- M2 must trace teacher-cache metadata and the exact offline/online replay
  semantics, including the relationship between calibration and Bellman
  supervision. Static coefficients cannot prove which loss dominated.
- G1 must finish full graph/flat feature parity and topology-claim checks.
- The edits do not establish online-RL benefit, explain it away, certify the
  original simulator as globally optimal, or change the GCN cost contrasts.
- No TeX engine is installed in this environment. Source-level checks do not
  replace PDF compilation and visual layout review.
- No remote push, PR, main merge, message or formal campaign was initiated.

Validation: 87 focused tests, full Python compilation and `git diff --check`
passed. A second read-only audit reproduced `contract.json` byte for byte.
The tests cover the new explicit config checks and existing cost-audit,
statistics, disruption and synthetic mechanism modules; they are not a
reexecution of the historical training pipeline. PDF layout is unverified.
